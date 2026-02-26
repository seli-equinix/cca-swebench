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
from ..core.tracing import get_tracer, OPENINFERENCE_SPAN_KIND, OUTPUT_VALUE
from ..orchestrator.anthropic import AnthropicLLMOrchestrator
from ..orchestrator.exceptions import OrchestratorInterruption

logger = logging.getLogger(__name__)

# Max consecutive 8B iterations with tool errors before escalating to 80B
MAX_FAST_CONSECUTIVE_FAILURES = 3

# How many times the 8B can produce identical output before we consider
# it stuck. 2 means: first time is fine, second identical output = stalled.
STALL_REPEAT_THRESHOLD = 2

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
                    return False
                return True
        # No tools in last iteration (final synthesis) — 80B
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
            if self._last_tool_names:
                span.set_attribute(
                    "cca.dual_model.last_tools",
                    ",".join(self._last_tool_names),
                )

            response = await context.invoke(chat, messages)
            response = await self.on_llm_response(response, context)
            result = await self._process_response(response, context)

            span.set_attribute(OUTPUT_VALUE, str(result)[:500] if result else "")
            return result

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
                    "Dual-model: 8B context (%d chars) → memory for 80B",
                    len(text),
                )
            return ""  # Don't stream to user — only 80B speaks
        return await super().on_llm_output(text, context)

    # ── Iteration loop override ───────────────────────────────────

    async def _process_messages(
        self, task: str, context: AnalectRunContext
    ) -> None:
        """Override with two extra synthesis branches:

        1. 8B done → force 80B synthesis (existing: research → summary)
        2. 80B done after tools → force summary (new: ensures the user
           sees what files were created, code written, and test results
           instead of just running commentary from tool iterations)

        Branch order: tool_use_queue → 8B done → tool-work summary → normal end
        """
        # BaseOrchestrator: max_iterations check → get_root_tag() → LLM call
        try:
            await super(AnthropicLLMOrchestrator, self)._process_messages(
                task, context
            )
        except Exception as e:
            # Context overflow on the 8B → escalate to 80B (1M context)
            if self._using_fast_model and "context length" in str(e):
                logger.warning(
                    "Dual-model: 8B context overflow — escalating to 80B: %s",
                    e,
                )
                self._force_primary = True
                self._last_tool_names = []
                await self._process_messages(task, context)
                return
            raise

        if self._tool_use_queue:
            # Standard: tools were called → process them → recurse
            try:
                await self._process_tool_use_queue(context)
            except OrchestratorInterruption as exc:
                await self._process_interruption(exc, context)
            finally:
                self._tool_use_queue.clear()
            await self._process_messages(task, context)

        elif self._using_fast_model:
            # 8B finished research (no more tools) → force 80B synthesis.
            # Reset stall detection so 80B can delegate back to 8B.
            logger.info(
                "Dual-model: 8B research complete — forcing 80B synthesis"
            )
            self._last_tool_names = []
            self._repeated_context_count = 0
            self._last_fast_context_hash = 0
            await self._process_messages(task, context)

        elif (
            not self._had_tool_iterations
            and not self._tool_nudge_done
            and self._num_iterations <= 2
        ):
            # Model described intent but didn't call tools on first turn.
            # Qwen3 sometimes generates markdown code blocks instead of
            # making tool calls.  Nudge it once to actually use tools.
            logger.info(
                "Dual-model: no tools called after %d iterations "
                "— nudging model to use tools",
                self._num_iterations,
            )
            self._tool_nudge_done = True
            context.memory_manager.add_messages([
                CfMessage(
                    type=cf.MessageType.HUMAN,
                    content=(
                        "You have bash and file editing tools available. "
                        "Don't just describe what you'll do — use the tools "
                        "to actually execute the code. Start now."
                    ),
                )
            ])
            await self._process_messages(task, context)

        elif self._had_tool_iterations and not self._synthesis_done:
            # 80B finished after tool work but hasn't summarized yet.
            # Force one more 80B call to produce a clear summary of
            # what was accomplished (files created, code written, test results).
            logger.info(
                "Dual-model: tool work complete — forcing synthesis summary"
            )
            self._synthesis_done = True
            context.memory_manager.add_messages([
                CfMessage(
                    type=cf.MessageType.HUMAN,
                    content=(
                        "Now give a clear summary of what you accomplished: "
                        "what files you created or modified, the key code, "
                        "and the results of any tests. Show the final code."
                    ),
                )
            ])
            await self._process_messages(task, context)

        else:
            # 80B finished — normal end, run extension hooks
            try:
                await self._on_process_tool_use_queue_complete(context)
            except OrchestratorInterruption as exc:
                await self._process_interruption(exc, context)
                await self._process_messages(task, context)

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

        await super()._process_tool_use_queue(context)

        # Quality gate: check for errors after processing
        self._update_quality_gate()

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
