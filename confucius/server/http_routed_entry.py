# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Dynamic route-based entry for the CCA server.

Unified entry that builds its extension list dynamically from the route
classification.  The Functionary router classifies each request, and this
entry uses tool_groups.py to select exactly the right tools.

Route → ToolGroups → Extensions → Orchestrator

This means the LLM only ever sees the tools relevant to the current task:
- USER route: 7 tools (user management + notes)
- CODER route: ~28 tools (file, shell, memory, web, code, graph, docs, notes, rules)
- INFRA route: ~28 tools (same as coder, wider shell allowlist)
- SEARCH route: ~23 tools (web, file, memory, code, graph, docs, notes)
- PLANNER route: ~12 tools (planner, memory, code search, web, notes)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import PrivateAttr

from ..core import types as cf
from ..core.analect import Analect, AnalectRunContext
from ..core.config import CCAConfigError, get_llm_params
from ..core.entry.base import EntryInput, EntryOutput
from ..core.entry.decorators import public
from ..core.entry.mixin import EntryAnalectMixin
from ..core.memory import CfMessage
from ..orchestrator.extensions import Extension
from .dual_model_orchestrator import DualModelOrchestrator
from ..orchestrator.types import OrchestratorInput

from ..analects.code.tasks import get_task_definition, get_search_task_definition, get_planner_task_definition
from ..analects.infrastructure.tasks import get_infra_task_definition
from .expert_router import ExpertType, RouteDecision
from ..core.config import get_tool_router_config
from .tool_groups import (
    build_extensions_for_route,
    build_tool_pool,
    get_max_iterations,
)
from .user.user_context import get_user_task_definition

logger = logging.getLogger(__name__)


def _get_complexity_guidance(expert: ExpertType) -> str:
    """Build complexity guidance for complex tasks (estimated_steps >= 8)."""
    guidance = (
        "## Complex Task — Follow These Guidelines\n"
        "1. **Plan first**: Use `write_memory` to create a step-by-step plan "
        "before editing any files. Track progress as you work.\n"
        "2. **Explore thoroughly**: Read ALL relevant files before editing. "
        "Trace call chains, imports, and dependencies.\n"
        "3. **Change incrementally**: One logical change at a time. Verify each step.\n"
        "4. **Track progress**: Update your plan in memory as you complete steps.\n"
        "5. **Review your work**: Re-read all modified files before finishing.\n"
    )
    if expert == ExpertType.CODER:
        guidance += (
            "6. A **code reviewer** will check your work after file edits — "
            "address any issues it raises.\n"
            "7. **Test suggestions** may appear after creating new files — "
            "consider implementing them.\n"
        )
    elif expert == ExpertType.INFRASTRUCTURE:
        guidance += (
            "6. A **code reviewer** will check your changes — "
            "address any issues it raises.\n"
            "7. **Verify on all affected nodes** — don't assume one node means all are correct.\n"
        )
    return guidance


# Route → task definition builder
_ROUTE_TASK_DEFS = {
    ExpertType.USER: get_user_task_definition,
    ExpertType.CODER: get_task_definition,
    ExpertType.INFRASTRUCTURE: get_infra_task_definition,
    ExpertType.SEARCH: get_search_task_definition,  # dedicated search prompt
    ExpertType.PLANNER: get_planner_task_definition,
}


@public
class HttpRoutedEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Dynamic route-based entry — one class for all expert routes.

    Uses the RouteDecision from the Functionary router to:
    1. Select the correct system prompt (task definition)
    2. Build the extension list from the route's tool groups
    3. Set route-appropriate max_iterations

    Unified entry for all routed requests.
    """

    # Populated after impl() runs — exposes orchestrator metrics
    _tool_iterations: int = PrivateAttr(default=0)
    _tools_escalated: bool = PrivateAttr(default=False)
    _escalated_groups: list = PrivateAttr(default_factory=list)

    def __init__(
        self,
        route: RouteDecision,
        user_context: str = "",
        user_extension: Optional[Extension] = None,
        backend_clients: Optional[Any] = None,
        session_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._route = route
        self._user_context = user_context
        self._user_extension = user_extension
        self._backend_clients = backend_clients
        self._session_id = session_id
        self._user_id = user_id

    @classmethod
    def display_name(cls) -> str:
        return "HttpRouted"

    @classmethod
    def description(cls) -> str:
        return "Dynamic route-based assistant with expert-specific tools"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="What files are in the current directory?")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        expert = self._route.expert
        current_time = datetime.now().isoformat(timespec="seconds")

        # 1. Get route-specific system prompt
        task_def_fn = _ROUTE_TASK_DEFS.get(expert, get_task_definition)
        task_def: str = task_def_fn(current_time=current_time)

        # 2. Inject user personalization context (before task def)
        if self._user_context:
            task_def = self._user_context + "\n\n" + task_def

        # 3. Inject routing context (task summary from classifier)
        route_header = self._route.to_context_header()
        if route_header:
            task_def = task_def + "\n\n" + route_header

        # 3b. Inject complexity guidance for complex tasks
        if self._route.is_complex:
            task_def += "\n\n" + _get_complexity_guidance(expert)

        # 4. Build extensions from route's tool groups
        extensions = build_extensions_for_route(
            route=self._route,
            user_extension=self._user_extension,
            backend_clients=self._backend_clients,
            session_id=self._session_id,
            user_id=self._user_id,
        )

        # 5. Compute dynamic iteration limit from estimated steps
        max_iterations = get_max_iterations(self._route)

        logger.info(
            f"HttpRoutedEntry: route={expert.value}, "
            f"estimated_steps={self._route.estimated_steps}, "
            f"extensions={len(extensions)}, "
            f"max_iter={max_iterations}, "
            f"summary={self._route.task_summary[:60]}"
        )

        # 6. Create orchestrator and run
        # SEARCH uses a dedicated 35B profile (search) as primary — lower temp, factual synthesis.
        # CODER/INFRA/USER use the standard coder profile.
        primary_role = "search" if expert == ExpertType.SEARCH else "coder"
        orchestrator = DualModelOrchestrator(
            llm_params=[
                get_llm_params(primary_role),
            ],
            extensions=extensions,
            raw_output_parser=None,
            max_iterations=max_iterations,
        )
        # Inject research model for tool iterations (CODER/INFRA only).
        # SEARCH: no tool_orch_params — primary 35B handles web_search directly.
        #   The dual-model handoff pattern causes hallucination loops for web search
        #   regardless of model size: the "brief writer" invents failures, the primary
        #   believes it, and the loop continues. Direct search is reliable.
        #   search_researcher profile exists in config for future dedicated hardware.
        # CODER/INFRA: uses tool_orchestrator (8B on Spark1) — fast for deterministic
        #   tool calls (bash, file edits) where output is always ground truth.
        if expert != ExpertType.SEARCH:
            try:
                orchestrator._tool_orch_params = get_llm_params("tool_orchestrator")
            except CCAConfigError:
                pass  # Falls back to primary model for all iterations
        # Pass router complexity estimate to orchestrator (controls nudge behavior)
        orchestrator._estimated_steps = self._route.estimated_steps
        orchestrator._route_name = expert.value

        # Dynamic tool escalation: build pool of disabled extensions
        tool_router_config = get_tool_router_config()
        if tool_router_config.enabled:
            pool = build_tool_pool(
                route=self._route,
                user_extension=self._user_extension,
                backend_clients=self._backend_clients,
                session_id=self._session_id,
                user_id=self._user_id,
            )
            if pool:
                # Add pool extensions to the orchestrator's extension list
                # (disabled — invisible until escalation enables them)
                orchestrator.extensions.extend(pool.values())
                orchestrator._tool_pool = pool
                orchestrator._tool_router_config = tool_router_config
                logger.info(
                    "Tool escalation pool: %d groups for %s route",
                    len(pool), expert.value,
                )
        # SEARCH allows inline responses (informational queries have no side effects).
        # CODER/INFRA/USER all require tool use — nudge fires when no tools called
        # and model just describes actions instead of executing them.
        orchestrator._requires_tool_use = expert != ExpertType.SEARCH

        try:
            await context.invoke_analect(
                orchestrator,
                OrchestratorInput(
                    messages=[
                        CfMessage(
                            type=cf.MessageType.HUMAN,
                            content=inp.question,
                            attachments=inp.attachments,
                        )
                    ],
                    task=task_def,
                ),
            )
        finally:
            # Always capture orchestrator metrics even if an exception is raised
            # (e.g. MaxIterationsReachedError).  Without finally, the metadata
            # callback reads _tool_iterations=0 on the exception path.
            iters = orchestrator._num_iterations
            self._tool_iterations = max(0, iters - 1) if iters > 0 else 0
            self._nudge_skipped = orchestrator._nudge_skipped
            self._circuit_breaker_fired = orchestrator._error_hint_injected
            self._tools_escalated = orchestrator._escalation_count > 0
            self._escalated_groups = list(orchestrator._escalated_groups)

        return EntryOutput()
