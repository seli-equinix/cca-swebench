# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Dual-model orchestrator — 8B researches, 80B synthesizes.

The 8B is a silent research agent: it orchestrates tools (web_search,
fetch_url, etc.) and builds context through analysis. Its text output
is preserved in memory as context for the 80B but never streamed to
the user. The 80B always generates the final user-facing response.

Architecture:
    _process_messages() overrides the Anthropic iteration loop to add a
    third branch: when the 8B finishes research (no more tool calls),
    force one more iteration with the 80B for synthesis.

    get_llm_params() is called ONCE per iteration (llm.py:179). A fresh
    chat object is created each time, so switching models between
    iterations is seamless.

    _num_iterations is incremented in base.py:221 AFTER get_root_tag()
    returns, so during get_llm_params(), _num_iterations == 0 on the
    first iteration.

Context preservation:
    on_llm_output() returning "" would lose text from memory (since
    _process_plain_text stores the post-on_llm_output text). So we
    explicitly save 8B text to memory BEFORE returning "" for display.
    The 80B then sees all research context in get_memory_by_visibility().

Stall detection:
    The 8B can research for as many iterations as it needs — there is no
    fixed cap. Instead, we detect when the 8B is stuck by hashing its
    output. If the 8B produces the same analysis text twice in a row,
    it's looping, not thinking. We force 80B synthesis and reset so the
    80B can delegate back to the 8B if it needs more research.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr

from ..core import types as cf
from ..core.analect import AnalectRunContext
from ..core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..core.llm_manager import LLMParams
from ..core.memory import CfMessage
from opentelemetry.trace import StatusCode

from ..core.tracing import get_tracer, OPENINFERENCE_SPAN_KIND, OUTPUT_VALUE
from ..orchestrator.anthropic import AnthropicLLMOrchestrator
from ..orchestrator.exceptions import OrchestratorInterruption

logger = logging.getLogger(__name__)

# Max consecutive 8B iterations with tool errors before escalating to 80B
MAX_FAST_CONSECUTIVE_FAILURES = 3

# Global error circuit breaker — works for ALL models (not just 8B→80B).
# After HINT threshold: inject a recovery message suggesting different approach.
# After STOP threshold: force the LLM to respond without tools.
ERROR_HINT_THRESHOLD = 3
ERROR_STOP_THRESHOLD = 5

# How many times the 8B can produce identical output before we consider
# it stuck. 2 means: first time is fine, second identical output = stalled.
STALL_REPEAT_THRESHOLD = 2

# Max consecutive 8B research iterations before forcing 80B synthesis.
# Prevents the 8B from looping through web_search indefinitely when
# results vary slightly (evading the hash-based stall detector).
# This is a backstop, not a normal limit — good research may need many iterations.
MAX_CONSECUTIVE_8B_RESEARCH = 8

# Max 8B→80B research cycles before forcing a final answer.
# Each cycle = 8B does research → 80B evaluates → 80B calls more tools → repeat.
# After this many cycles, a "stop researching" message is injected.
# This is a backstop — the 80B's thinking should decide when to stop naturally.
MAX_RESEARCH_CYCLES = 8

# Tools that only READ data — safe for the 8B to orchestrate.
# Any tool NOT in this set triggers 80B on the next iteration.
RESEARCH_TOOLS = frozenset({
    # Web tools
    "web_search",
    "fetch_url_content",
    # Knowledge search tools
    "search_codebase",
    "search_knowledge",
    "search_documents",
    "list_session_docs",
    # Graph tools (read-only queries)
    "query_call_graph",
    "find_orphan_functions",
    "analyze_dependencies",
    # Note-taker tools (read-only)
    "search_notes",
    "get_trajectory",
    "list_recent_sessions",
})


class DualModelOrchestrator(AnthropicLLMOrchestrator):
    """Orchestrator that switches between fast 8B and reasoning 80B per iteration.

    The 8B is a research agent — it gathers data via tools and builds
    context through analysis. Its text is preserved in memory (for the
    80B) but never streamed to the user. The 80B always synthesizes
    the final response.

    Key behaviours:
    - 8B text → memory (context for 80B), not streamed to user
    - When 8B finishes research → force 80B synthesis iteration
    - Stall detection: if 8B repeats itself, force synthesis
    - Quality gate escalates to 80B after consecutive 8B failures
    """

    _tool_orch_params: LLMParams | None = PrivateAttr(default=None)
    _last_tool_names: list[str] = PrivateAttr(default_factory=list)
    _using_fast_model: bool = PrivateAttr(default=False)
    _model_reason: str = PrivateAttr(default="initial planning")
    _consecutive_fast_failures: int = PrivateAttr(default=0)
    _force_primary: bool = PrivateAttr(default=False)
    _last_queue_had_error: bool = PrivateAttr(default=False)
    # Stall detection: track 8B output hashes to detect repetition
    _last_fast_context_hash: int = PrivateAttr(default=0)
    _repeated_context_count: int = PrivateAttr(default=0)
    # Post-completion synthesis: track whether tools ran and synthesis done
    _had_tool_iterations: bool = PrivateAttr(default=False)
    _synthesis_done: bool = PrivateAttr(default=False)
    # Tool-nudge: re-prompt once if model describes intent but doesn't call tools
    _tool_nudge_done: bool = PrivateAttr(default=False)
    # Global error circuit breaker (works for 80B too, not just 8B→80B)
    _total_consecutive_errors: int = PrivateAttr(default=0)
    _error_hint_injected: bool = PrivateAttr(default=False)
    # Complexity hint from router — controls nudge behavior
    _estimated_steps: int = PrivateAttr(default=10)
    # When True, this route REQUIRES tool use (CODER/INFRA).
    # When False (USER/SEARCH), inline responses without tools are acceptable.
    # Controls whether the smart nudge skips on is_simple tasks.
    _requires_tool_use: bool = PrivateAttr(default=True)
    # Tracking flag for monitoring: was the tool nudge skipped?
    _nudge_skipped: bool = PrivateAttr(default=False)
    # Research depth limiting: count consecutive 8B iterations and total cycles
    _consecutive_8b_iters: int = PrivateAttr(default=0)
    _research_cycle_count: int = PrivateAttr(default=0)
    # Research Brief Protocol tracking
    _original_query: str = PrivateAttr(default="")          # captured from first human msg
    _research_brief: str = PrivateAttr(default="")          # 8B's latest research text
    _research_executor_injected: bool = PrivateAttr(default=False)  # executor context sent once
    _search_call_count: int = PrivateAttr(default=0)        # total research tool calls made
    # Duplication guard: True when the primary (80B) already streamed text
    # containing code alongside a tool call.  If set, synthesis must be
    # skipped — otherwise the 80B rewrites its entire response a second time.
    _primary_streamed_code: bool = PrivateAttr(default=False)
    # Dynamic tool escalation (Phase 2): pool of disabled extensions that
    # can be selectively enabled mid-loop by the Functionary tool selector.
    _tool_pool: dict = PrivateAttr(default_factory=dict)
    _escalation_count: int = PrivateAttr(default=0)
    _escalated_groups: list = PrivateAttr(default_factory=list)
    _tool_router_config: object = PrivateAttr(default=None)
    _route_name: str = PrivateAttr(default="unknown")

    # ── Model selection ──────────────────────────────────────────

    def _overlay_fast_params(self, params: LLMParams) -> LLMParams:
        """Replace model-level fields with fast model, keeping tools.

        super().get_llm_params() adds tools to additional_kwargs["tools"]
        and Claude beta tags. The fast model's additional_kwargs has
        base_url (pointing to Spark1:8400). Merging via update() preserves
        tools and swaps the endpoint.
        """
        fast = self._tool_orch_params
        assert fast is not None
        params.model = fast.model
        if fast.temperature is not None:
            params.temperature = fast.temperature
        if fast.max_tokens is not None:
            params.max_tokens = fast.max_tokens
        if fast.initial_max_tokens is not None:
            params.initial_max_tokens = fast.initial_max_tokens
        # Merge additional_kwargs: keeps "tools" from super, adds "base_url".
        # CRITICAL: copy the dict first — AnthropicLLMOrchestrator.get_llm_params()
        # does a shallow .copy() of LLMParams, so additional_kwargs is a shared
        # reference to self.llm_params[0]'s dict. Mutating in-place would
        # permanently contaminate the 80B's base_url with the 8B's endpoint.
        if fast.additional_kwargs:
            params.additional_kwargs = dict(params.additional_kwargs or {})
            params.additional_kwargs.update(fast.additional_kwargs)
        return params

    def _is_8b_stalled(self) -> bool:
        """Check if the 8B is stuck producing the same output."""
        return self._repeated_context_count >= STALL_REPEAT_THRESHOLD

    def _should_use_fast_model(self) -> bool:
        """Decide whether this iteration should use the 8B model."""
        # No tool_orchestrator configured
        if self._tool_orch_params is None:
            return False
        # Quality gate triggered — force 80B for rest of request
        if self._force_primary:
            return False
        # First iteration — always 80B (needs to understand task)
        if self._num_iterations == 0:
            return False
        # After synthesis injection (coding routes only), never switch back to 8B.
        # Prevents a post-synthesis tool call (e.g. web_search) from
        # triggering a wasteful 8B research cycle after final answer.
        # Research routes (_research_cycle_count > 0) use evaluation checkpoints
        # instead — the 80B's thinking decides when research is done naturally.
        if self._synthesis_done and self._research_cycle_count == 0:
            return False
        # After tool execution: use 8B only if ALL tools were research tools
        if self._last_tool_names:
            if all(t in RESEARCH_TOOLS for t in self._last_tool_names):
                # Stall detection: if 8B is repeating itself, force synthesis
                if self._is_8b_stalled():
                    logger.info(
                        "Dual-model: 8B stalled (repeated output %d times) "
                        "— forcing 80B synthesis",
                        self._repeated_context_count,
                    )
                    self._consecutive_8b_iters = 0
                    return False
                # Depth cap: prevent endless research loops (evades stall detection
                # when results vary slightly between searches)
                self._consecutive_8b_iters += 1
                if self._consecutive_8b_iters > MAX_CONSECUTIVE_8B_RESEARCH:
                    logger.info(
                        "Dual-model: 8B research depth limit (%d consecutive) "
                        "— forcing 80B synthesis",
                        self._consecutive_8b_iters,
                    )
                    self._consecutive_8b_iters = 0
                    return False
                return True
        # Non-research tools or no tools → 80B; reset consecutive counter
        self._consecutive_8b_iters = 0
        return False

    async def get_llm_params(self) -> LLMParams:
        """Pick model based on iteration context.

        Always calls super() first to get tool-decorated params (with tools,
        beta tags, etc.), then overlays the fast model settings when needed.
        """
        params = await super().get_llm_params()

        if self._should_use_fast_model():
            self._using_fast_model = True
            self._model_reason = f"research: {self._last_tool_names}"
            logger.info(
                "Dual-model: iter %d → 8B (after research tools: %s)",
                self._num_iterations,
                self._last_tool_names,
            )
            return self._overlay_fast_params(params)

        self._using_fast_model = False
        self._model_reason = (
            "initial planning"
            if self._num_iterations == 0
            else "quality gate"
            if self._force_primary
            else "8B stalled"
            if self._is_8b_stalled()
            else f"creation: {self._last_tool_names}"
            if self._last_tool_names
            else "final synthesis"
        )
        logger.info(
            "Dual-model: iter %d → 80B (%s)",
            self._num_iterations,
            self._model_reason,
        )
        return params

    # ── Phoenix tracing enrichment ──────────────────────────────

    async def _invoke_claude_impl(
        self,
        chat: BaseChatModel,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> str:
        """Override to enrich Phoenix spans with dual-model context."""
        tracer = get_tracer()
        iteration = getattr(self, '_iteration_count', 0) + 1
        object.__setattr__(self, '_iteration_count', iteration)

        model_name = getattr(chat, 'model', 'unknown')
        with tracer.start_as_current_span("cca.llm.invoke") as span:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute("llm.model_name", str(model_name))
            span.set_attribute("cca.llm.iteration", iteration)
            span.set_attribute("cca.llm.message_count", len(messages))

            # Dual-model tracing attributes
            span.set_attribute("cca.dual_model.using_fast", self._using_fast_model)
            span.set_attribute("cca.dual_model.reason", self._model_reason)
            span.set_attribute("cca.dual_model.quality_gate_active", self._force_primary)
            span.set_attribute(
                "cca.dual_model.consecutive_failures",
                self._consecutive_fast_failures,
            )
            span.set_attribute(
                "cca.dual_model.repeated_context_count",
                self._repeated_context_count,
            )
            span.set_attribute(
                "cca.dual_model.total_consecutive_errors",
                self._total_consecutive_errors,
            )
            if self._last_tool_names:
                span.set_attribute(
                    "cca.dual_model.last_tools",
                    ",".join(self._last_tool_names),
                )
            span.set_attribute("cca.research.cycles", self._research_cycle_count)
            span.set_attribute("cca.research.search_count", self._search_call_count)
            span.set_attribute(
                "cca.research.brief_length", len(self._research_brief)
            )

            try:
                response = await context.invoke(chat, messages)
                response = await self.on_llm_response(response, context)
                result = await self._process_response(response, context)

                span.set_attribute(OUTPUT_VALUE, str(result)[:500] if result else "")
                span.set_status(StatusCode.OK)
                return result
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e)[:500])
                raise

    # ── 8B context preservation + stall detection ─────────────────

    async def on_llm_output(
        self, text: str, context: AnalectRunContext
    ) -> str:
        """8B text → memory (context for 80B), not streamed to user.

        on_llm_output() returning "" would lose the text from memory
        because _process_plain_text() stores the post-on_llm_output
        text. We explicitly save 8B text to memory BEFORE returning ""
        so the 80B sees the 8B's research analysis in subsequent
        iterations via get_memory_by_visibility().

        Also tracks output hashes for stall detection — if the 8B
        produces the same analysis twice in a row, it's stuck.
        """
        if self._using_fast_model:
            if text.strip():
                # Preserve 8B's research context in memory for 80B
                context.memory_manager.add_messages([
                    CfMessage(type=cf.MessageType.AI, content=text)
                ])

                # Capture as research brief for the 80B evaluation checkpoint
                self._research_brief = text

                # Stall detection: hash the output to detect repetition
                context_hash = hash(text.strip())
                if context_hash == self._last_fast_context_hash:
                    self._repeated_context_count += 1
                    logger.info(
                        "Dual-model: 8B repeated output (%d chars, "
                        "repeat #%d)",
                        len(text),
                        self._repeated_context_count,
                    )
                else:
                    self._repeated_context_count = 0
                    self._last_fast_context_hash = context_hash

                logger.info(
                    "Dual-model: 8B research brief (%d chars) → memory for 80B",
                    len(text),
                )
            return ""  # Don't stream to user — only 80B speaks

        # Primary (80B) path — track whether it streams code before a tool call.
        # If it does, synthesis must be skipped to prevent duplicate output.
        if text.strip() and not self._primary_streamed_code:
            code_signals = ["```", "def ", "class ", "import ", "return "]
            if any(s in text for s in code_signals):
                self._primary_streamed_code = True

        return await super().on_llm_output(text, context)

    # ── Dynamic tool escalation ───────────────────────────────────

    async def _select_and_escalate(
        self, context: AnalectRunContext
    ) -> list[str]:
        """Ask Functionary which tool groups to enable and activate them.

        Returns list of tool names that were newly enabled (for injection
        into the nudge/continuation message). Returns empty list if
        Functionary says no additional tools are needed or if the pool
        is empty.
        """
        if not self._tool_pool or self._tool_router_config is None:
            return []

        # Collect the agent's last assistant text from memory
        last_text = ""
        msgs = context.memory_manager.get_session_memory().messages
        for msg in reversed(msgs):
            if msg.type == cf.MessageType.AI and msg.content:
                content = msg.content
                if isinstance(content, str):
                    last_text = content
                elif isinstance(content, list):
                    last_text = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                break

        if not last_text:
            return []

        # Collect current tool names from enabled extensions
        current_tool_names: list[str] = []
        for ext in self._enabled_tool_use_extensions:
            try:
                for tool in await ext.tools:
                    name = getattr(tool, 'name', None)
                    if name:
                        current_tool_names.append(name)
            except Exception:
                pass

        # Call Functionary tool selector
        from .expert_router import select_tools_for_escalation
        requested_groups = await select_tools_for_escalation(
            agent_output=last_text[:2000],
            current_route=getattr(self, '_route_name', 'unknown'),
            current_tools=current_tool_names,
            config=self._tool_router_config,
        )

        if not requested_groups:
            return []

        # Map group names → ToolGroup enum and enable matching pool extensions
        from .tool_groups import ToolGroup
        enabled_names: list[str] = []

        for group_name in requested_groups:
            # Find matching ToolGroup
            try:
                tg = ToolGroup(group_name)
            except ValueError:
                logger.warning("Unknown tool group from selector: %s", group_name)
                continue

            ext = self._tool_pool.pop(tg, None)
            if ext is None:
                continue

            # Enable the extension
            if hasattr(ext, 'enable_tool_use'):
                ext.enable_tool_use = True
            if hasattr(ext, 'included_in_system_prompt'):
                ext.included_in_system_prompt = True

            # Collect enabled tool names for the injection message
            try:
                for tool in await ext.tools:
                    name = getattr(tool, 'name', None)
                    if name:
                        enabled_names.append(name)
            except Exception:
                enabled_names.append(tg.value)

            self._escalated_groups.append(tg.value)
            logger.info(
                "Escalation: enabled %s (%d tools)",
                tg.value, len(enabled_names),
            )

        if enabled_names:
            self._escalation_count += 1
            # Safety valve: after 3 escalations, enable all remaining
            if self._escalation_count >= 3 and self._tool_pool:
                logger.info(
                    "Escalation safety valve: enabling all %d remaining pool groups",
                    len(self._tool_pool),
                )
                for tg, ext in list(self._tool_pool.items()):
                    if hasattr(ext, 'enable_tool_use'):
                        ext.enable_tool_use = True
                    if hasattr(ext, 'included_in_system_prompt'):
                        ext.included_in_system_prompt = True
                    self._escalated_groups.append(tg.value)
                    try:
                        for tool in await ext.tools:
                            name = getattr(tool, 'name', None)
                            if name:
                                enabled_names.append(name)
                    except Exception:
                        enabled_names.append(tg.value)
                self._tool_pool.clear()

            # Bump max_iterations to give the agent room to use new tools
            if self.max_iterations < 15:
                logger.info(
                    "Escalation: bumping max_iterations %d → 20",
                    self.max_iterations,
                )
                self.max_iterations = 20

        return enabled_names

    # ── Iteration loop override ───────────────────────────────────

    async def _process_messages(
        self, task: str, context: AnalectRunContext
    ) -> None:
        """Iterative loop with synthesis branches (no recursion).

        Branches checked after each LLM call:
        1. tool_use_queue → process tools → continue
        2. 8B done → evaluation checkpoint for 80B → continue
        3. No tools called → nudge → continue
        4. Tool work done, no synthesis (coding routes only) → synthesis prompt → continue
        5. Done → run extension hooks → break

        Research Brief Protocol (SEARCH route):
        - 80B plans searches and calls web_search 3-5 times in one response
        - 8B executes all searches, writes a structured research brief
        - 80B receives brief + evaluation checkpoint: decides if done or needs more
        - If done: writes final answer → completion branch
        - If more needed: calls web_search again → another research cycle
        - Backstop limits (MAX_RESEARCH_CYCLES, MAX_CONSECUTIVE_8B_RESEARCH) prevent
          infinite loops but are high enough to not interfere with normal research.
        """
        # Capture original query for evaluation checkpoint context
        if not self._original_query:
            for msg in reversed(context.memory_manager.memory.messages):
                if (
                    msg.type == cf.MessageType.HUMAN
                    and not msg.additional_kwargs.get("__synthetic__")
                ):
                    q = msg.content if isinstance(msg.content, str) else ""
                    if q.strip():
                        self._original_query = q.strip()[:500]
                        break

        while True:
            # BaseOrchestrator: max_iterations check → get_root_tag() → LLM call
            try:
                await super()._process_messages(task, context)
            except Exception as e:
                # Context overflow on the 8B → escalate to 80B (1M context)
                if self._using_fast_model and "context length" in str(e):
                    logger.warning(
                        "Dual-model: 8B context overflow — escalating to 80B: %s",
                        e,
                    )
                    self._force_primary = True
                    self._last_tool_names = []
                    continue
                raise

            if self._tool_use_queue:
                # Standard: tools were called → process them → loop
                try:
                    await self._process_tool_use_queue(context)
                except OrchestratorInterruption as exc:
                    await self._process_interruption(exc, context)
                finally:
                    self._tool_use_queue.clear()

                # Global error circuit breaker — recover from error loops
                if self._total_consecutive_errors >= ERROR_STOP_THRESHOLD:
                    logger.warning(
                        "Error circuit breaker: %d consecutive errors "
                        "— forcing text response",
                        self._total_consecutive_errors,
                    )
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "STOP: You have had too many consecutive tool "
                                "failures. Do NOT call any more tools. Respond "
                                "with what you have accomplished so far and "
                                "explain what you were unable to complete."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    # Reset so the forced response doesn't re-trigger
                    self._total_consecutive_errors = 0
                    self._error_hint_injected = False
                elif (
                    self._total_consecutive_errors >= ERROR_HINT_THRESHOLD
                    and not self._error_hint_injected
                ):
                    logger.warning(
                        "Error circuit breaker: %d consecutive errors "
                        "— injecting recovery hint",
                        self._total_consecutive_errors,
                    )
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "NOTICE: You have had multiple consecutive tool "
                                "failures. Try a completely different approach:\n"
                                "- If bash heredoc syntax failed, use echo or "
                                "write_file instead\n"
                                "- If a command keeps failing, simplify it or "
                                "break it into smaller steps\n"
                                "- If you are stuck, skip this step and move on"
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    self._error_hint_injected = True

                # Research Brief Protocol: inject research assistant context on first
                # delegation. Tells the research model its role before it runs.
                # Only inject when dual-model is active (fast model handles research).
                # SEARCH route has no _tool_orch_params — 35B reads results directly.
                if (
                    not self._research_executor_injected
                    and self._tool_orch_params is not None
                    and self._last_tool_names
                    and all(t in RESEARCH_TOOLS for t in self._last_tool_names)
                ):
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "You are now the research assistant. The planning "
                                "model above has kicked off search tasks. Your job:\n"
                                "1. Review the search results above thoroughly\n"
                                "2. For highly relevant results, use "
                                "`fetch_url_content` to read the full page\n"
                                "3. Run additional targeted searches if they would "
                                "directly fill gaps in the current results\n"
                                "4. When you have gathered sufficient information, "
                                "write a structured Research Brief:\n"
                                "   **Findings**: Key facts with source URLs\n"
                                "   **Coverage**: Which aspects are fully answered, "
                                "partially answered, or unanswered\n"
                                "   **Confidence**: High / Medium / Low\n\n"
                                "Be comprehensive — the planning model will use your "
                                "brief to synthesize the final answer."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    self._research_executor_injected = True
                    logger.info(
                        "Dual-model: research assistant context injected "
                        "(first delegation, %d tools: %s)",
                        len(self._last_tool_names),
                        self._last_tool_names,
                    )

                # SEARCH loop guard: fire when model does a single web_search
                # after the first parallel batch (>= 3 searches done, iter >= 2).
                # Root cause: the model wants full page content but web_search
                # only returns 500-char snippets. It loops on site:docs.python.org
                # queries indefinitely. The correct tool is fetch_url_content.
                # Hard-block web_search only — fetch_url_content stays open so
                # the model can read the full page it's been searching for.
                if (
                    self._tool_orch_params is None  # SEARCH route, no fast model
                    and not self._requires_tool_use  # double-check: SEARCH only
                    and self._search_call_count >= 3   # first batch complete
                    and len(self._last_tool_names) == 1  # single-search loop
                    and self._last_tool_names[0] == "web_search"  # not fetch_url
                    and self._num_iterations >= 2  # past the first batch iter
                    and not self._synthesis_done
                ):
                    logger.info(
                        "Search loop guard: single web_search at iter %d "
                        "(%d searches done) — blocking web_search, fetch_url_content still open",
                        self._num_iterations,
                        self._search_call_count,
                    )
                    # Hard-block web_search only. fetch_url_content remains available.
                    from .utility_tools import UtilityToolsExtension
                    for ext in self.extensions:
                        if isinstance(ext, UtilityToolsExtension):
                            ext.block_searches()
                            break
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "web_search is now blocked — you have already done "
                                f"{self._search_call_count} searches and web_search "
                                "only returns 500-character snippets regardless. "
                                "To get full page content, call fetch_url_content "
                                "with a URL you found in your earlier results "
                                "(e.g. https://docs.python.org/3.13/whatsnew/3.13.html). "
                                "Then write your final answer."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    self._synthesis_done = True

                continue

            elif self._using_fast_model:
                # 8B finished research (no more tools) → force 80B synthesis.
                self._research_cycle_count += 1
                self._consecutive_8b_iters = 0
                self._last_tool_names = []
                self._repeated_context_count = 0
                self._last_fast_context_hash = 0

                if self._research_cycle_count >= MAX_RESEARCH_CYCLES:
                    # Too many research cycles — inject final synthesis prompt
                    # and prevent more 8B research so we don't loop forever.
                    logger.info(
                        "Dual-model: research cycle limit (%d cycles) "
                        "— injecting stop message, forcing final synthesis",
                        self._research_cycle_count,
                    )
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "You have done sufficient research across multiple "
                                "rounds. Write your final answer NOW using only "
                                "the information already gathered. "
                                "Do NOT call any more search tools."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    self._force_primary = True  # No more 8B after this
                else:
                    # Evaluation checkpoint: 80B decides if research is complete
                    # or if more searches are needed. Its thinking handles this.
                    logger.info(
                        "Dual-model: 8B research complete (cycle %d/%d, "
                        "brief=%d chars) — injecting evaluation checkpoint",
                        self._research_cycle_count,
                        MAX_RESEARCH_CYCLES,
                        len(self._research_brief),
                    )
                    original_q = self._original_query or "the user's question"
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                f"Research phase {self._research_cycle_count} "
                                f"complete. The research above was gathered to "
                                f"answer: \"{original_q}\"\n\n"
                                "Evaluate the research brief and decide:\n"
                                "- **If sufficient**: write your complete, "
                                "well-cited final answer now. Do NOT call any "
                                "tools.\n"
                                "- **If critical information is missing**: call "
                                "`web_search` with up to 3 specific, targeted "
                                "queries to fill the gaps. The research executor "
                                "will handle them and report back.\n\n"
                                "Think carefully — if you have ~80% of what you "
                                "need, synthesize now rather than searching more."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                continue

            elif (
                not self._had_tool_iterations
                and not self._tool_nudge_done
                and self._num_iterations <= 2
            ):
                # Model described intent but didn't call tools on first turn.
                # Qwen3 sometimes generates markdown code blocks instead of
                # making tool calls.  Nudge it once to actually use tools —
                # BUT skip if the route allows inline responses AND the model
                # either already gave code or the task is simple.
                #
                # Route awareness:
                # - CODER/INFRA/USER (_requires_tool_use=True): skip only if
                #   has_code (model gave a usable inline answer with real code).
                #   "is_simple" alone is not enough — these routes should call
                #   tools even for short tasks (file edits, profile deletions).
                # - SEARCH (_requires_tool_use=False): skip if has_code
                #   OR is_simple (inline answers are fine for informational
                #   queries; research tools are run by the 8B automatically).
                self._tool_nudge_done = True
                has_code = self._last_assistant_has_code(context)
                is_simple = self._estimated_steps <= 3
                if has_code or (is_simple and not self._requires_tool_use):
                    logger.info(
                        "Dual-model: skipping tool nudge — %s",
                        "response has code"
                        if has_code
                        else "simple task (non-creation route)",
                    )
                    self._nudge_skipped = True
                    # Fall through to completion branch (don't continue)
                else:
                    # Try dynamic tool escalation before standard nudge
                    if self._tool_pool:
                        enabled = await self._select_and_escalate(context)
                        if enabled:
                            tools_str = ", ".join(enabled)
                            logger.info(
                                "Dual-model: escalated tools [%s] — nudging with new tools",
                                tools_str,
                            )
                            context.memory_manager.add_messages([
                                CfMessage(
                                    type=cf.MessageType.HUMAN,
                                    content=(
                                        f"You now have additional tools available: "
                                        f"{tools_str}. Use them to complete the task. "
                                        f"Don't describe what you'll do — call the "
                                        f"appropriate tool to execute it now."
                                    ),
                                    additional_kwargs={"__synthetic__": True},
                                )
                            ])
                            continue

                    logger.info(
                        "Dual-model: no tools called after %d iterations "
                        "— nudging model to use tools",
                        self._num_iterations,
                    )
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "You have tools available to perform this action. "
                                "Don't just describe what you'll do — call the "
                                "appropriate tool to actually execute it now."
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    continue

            elif (
                self._had_tool_iterations
                and not self._synthesis_done
                and self._research_cycle_count == 0
                and self._requires_tool_use
            ):
                # 80B produced text after tool work (coding/non-research routes).
                # Normally its response was an internal working draft — force one
                # more iteration to produce a single, clean, consolidated answer.
                #
                # Exception: if the primary (80B) already streamed text WITH CODE
                # alongside a lightweight tool call (e.g. remember_user_fact),
                # synthesis would make it rewrite the exact same response a second
                # time. Skip synthesis in that case — the streamed response is
                # already the complete, correct answer.
                if self._primary_streamed_code:
                    logger.info(
                        "Dual-model: skipping synthesis — primary already "
                        "streamed a code response (would duplicate output)"
                    )
                    # Run the same completion sequence as the normal end branch.
                    # Cannot fall through: we're inside an elif, so the else:break
                    # branch never fires — we must break explicitly here.
                    self._synthesis_done = True
                    try:
                        await self._on_process_tool_use_queue_complete(context)
                    except OrchestratorInterruption as exc:
                        await self._process_interruption(exc, context)
                        continue
                    self._strip_synthetic_messages(context)
                    break
                elif self._tool_pool:
                    # Before forcing synthesis, try escalation — the agent
                    # may need tools it doesn't have yet (e.g., USER route
                    # agent finished user management but can't write files).
                    enabled = await self._select_and_escalate(context)
                    if enabled:
                        tools_str = ", ".join(enabled)
                        logger.info(
                            "Dual-model: escalated tools [%s] — continuing instead of synthesis",
                            tools_str,
                        )
                        context.memory_manager.add_messages([
                            CfMessage(
                                type=cf.MessageType.HUMAN,
                                content=(
                                    f"You now have additional tools available: "
                                    f"{tools_str}. Use them to complete the "
                                    f"remaining parts of the task."
                                ),
                                additional_kwargs={"__synthetic__": True},
                            )
                        ])
                        continue
                    # Functionary said no_additional_tools — proceed with synthesis
                    logger.info(
                        "Dual-model: tool selector says no additional tools needed — "
                        "proceeding with synthesis"
                    )
                    self._synthesis_done = True
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "Your previous response was an internal draft. Now "
                                "produce your FINAL response for the user. Rules:\n"
                                "- **Do NOT call any tools** — write plain text only\n"
                                "- Give ONE consolidated answer (do NOT repeat code "
                                "or explanations that already appeared above)\n"
                                "- If you already showed code, just reference it — "
                                "don't show it again\n"
                                "- Keep it concise: what was done, what the result "
                                "was, and any next steps\n"
                                "- Do NOT start with 'Summary of Accomplishments' "
                                "or similar headings"
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    continue
                else:
                    logger.info(
                        "Dual-model: tool work complete — forcing consolidated response"
                    )
                    self._synthesis_done = True
                    context.memory_manager.add_messages([
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=(
                                "Your previous response was an internal draft. Now "
                                "produce your FINAL response for the user. Rules:\n"
                                "- **Do NOT call any tools** — write plain text only\n"
                                "- Give ONE consolidated answer (do NOT repeat code "
                                "or explanations that already appeared above)\n"
                                "- If you already showed code, just reference it — "
                                "don't show it again\n"
                                "- Keep it concise: what was done, what the result "
                                "was, and any next steps\n"
                                "- Do NOT start with 'Summary of Accomplishments' "
                                "or similar headings"
                            ),
                            additional_kwargs={"__synthetic__": True},
                        )
                    ])
                    continue

            else:
                # 80B finished — normal end, run extension hooks
                try:
                    await self._on_process_tool_use_queue_complete(context)
                except OrchestratorInterruption as exc:
                    await self._process_interruption(exc, context)
                    continue
                # Clean up synthetic messages before session persists
                self._strip_synthetic_messages(context)
                break

    def _strip_synthetic_messages(self, context: AnalectRunContext) -> None:
        """Remove synthetic HUMAN messages so they don't persist to future requests.

        The tool nudge and synthesis prompts are injected as HUMAN messages
        (required for the LLM to process them), but they should not appear
        in the next HTTP request's conversation history.
        """
        memory = context.memory_manager.memory
        original = len(memory.messages)
        memory.messages = [
            m for m in memory.messages
            if not m.additional_kwargs.get("__synthetic__")
        ]
        removed = original - len(memory.messages)
        if removed:
            logger.debug("Stripped %d synthetic messages from memory", removed)

    # ── Tool tracking + quality gate ─────────────────────────────

    async def _process_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        all_tool_names: set[str],
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult | None:
        """Override to detect tool errors for the quality gate."""
        result = await super()._process_tool_use(tool_use, all_tool_names, context)
        if result is not None and getattr(result, "is_error", False):
            self._last_queue_had_error = True
        return result

    async def _process_tool_use_queue(
        self, context: AnalectRunContext
    ) -> None:
        """Override to track which tools were used and detect failures."""
        # Capture tool names BEFORE processing
        # (queue is cleared in _process_messages finally block)
        self._last_tool_names = [tu.name for tu in self._tool_use_queue]
        self._last_queue_had_error = False
        self._had_tool_iterations = True

        # Count research tool calls for monitoring
        self._search_call_count += sum(
            1 for n in self._last_tool_names if n in RESEARCH_TOOLS
        )

        await super()._process_tool_use_queue(context)

        # Global error circuit breaker (all models)
        if self._last_queue_had_error:
            self._total_consecutive_errors += 1
        else:
            self._total_consecutive_errors = 0
            self._error_hint_injected = False

        # Quality gate: check for errors after processing (8B→80B)
        self._update_quality_gate()

    def _last_assistant_has_code(self, context: AnalectRunContext) -> bool:
        """Check if the last assistant message already contains code."""
        for msg in reversed(context.memory_manager.memory.messages):
            if msg.type == cf.MessageType.AI:
                text = msg.content if isinstance(msg.content, str) else ""
                if not text:
                    continue
                if "```" in text:
                    return True
                code_patterns = [
                    "def ", "class ", "import ", "from ", "lambda ",
                    "print(", "return ", "async def ", "await ",
                ]
                return any(p in text for p in code_patterns)
        return False

    def _update_quality_gate(self) -> None:
        """Escalate to 80B if 8B has too many consecutive failures."""
        if not self._using_fast_model:
            self._consecutive_fast_failures = 0
            return

        if self._last_queue_had_error:
            self._consecutive_fast_failures += 1
            logger.warning(
                "Dual-model: 8B tool failure %d/%d (tools: %s)",
                self._consecutive_fast_failures,
                MAX_FAST_CONSECUTIVE_FAILURES,
                self._last_tool_names,
            )
            if self._consecutive_fast_failures >= MAX_FAST_CONSECUTIVE_FAILURES:
                self._force_primary = True
                logger.warning(
                    "Dual-model: quality gate triggered — escalating to 80B "
                    "after %d consecutive 8B failures",
                    self._consecutive_fast_failures,
                )
        else:
            self._consecutive_fast_failures = 0
