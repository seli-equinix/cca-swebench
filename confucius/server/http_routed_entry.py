# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Dynamic route-based entry for the CCA server.

Replaces per-route entry classes (HttpCodeAssistEntry, HttpInfraEntry) with
a single entry that builds its extension list dynamically from the route
classification.  The Functionary router classifies each request, and this
entry uses tool_groups.py to select exactly the right tools.

Route → ToolGroups → Extensions → Orchestrator

This means the LLM only ever sees the tools relevant to the current task:
- USER route: 6 tools (user management only)
- CODER route: ~10 tools (file, shell, memory, web)
- INFRA route: ~10 tools (shell, file, memory, web)
- SEARCH route: ~8 tools (web, file, memory)
- PLANNER route: ~6 tools (memory only)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from ..core import types as cf
from ..core.analect import Analect, AnalectRunContext
from ..core.config import get_llm_params
from ..core.entry.base import EntryInput, EntryOutput
from ..core.entry.decorators import public
from ..core.entry.mixin import EntryAnalectMixin
from ..core.memory import CfMessage
from ..orchestrator.anthropic import AnthropicLLMOrchestrator
from ..orchestrator.extensions import Extension
from ..orchestrator.types import OrchestratorInput

from ..analects.code.tasks import get_task_definition
from ..analects.infrastructure.tasks import get_infra_task_definition
from .expert_router import ExpertType, RouteDecision
from .tool_groups import (
    ROUTE_MAX_ITERATIONS,
    build_extensions_for_route,
)
from .user.user_context import get_user_task_definition

logger = logging.getLogger(__name__)


# Route → task definition builder
_ROUTE_TASK_DEFS = {
    ExpertType.USER: get_user_task_definition,
    ExpertType.CODER: get_task_definition,
    ExpertType.INFRASTRUCTURE: get_infra_task_definition,
    ExpertType.SEARCH: get_task_definition,       # reuse coder for now
    ExpertType.PLANNER: get_task_definition,      # reuse coder for now
}


@public
class HttpRoutedEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Dynamic route-based entry — one class for all expert routes.

    Uses the RouteDecision from the Functionary router to:
    1. Select the correct system prompt (task definition)
    2. Build the extension list from the route's tool groups
    3. Set route-appropriate max_iterations

    Replaces HttpCodeAssistEntry and HttpInfraEntry.
    """

    def __init__(
        self,
        route: RouteDecision,
        user_context: str = "",
        user_extension: Optional[Extension] = None,
    ) -> None:
        super().__init__()
        self._route = route
        self._user_context = user_context
        self._user_extension = user_extension

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

        # 4. Build extensions from route's tool groups
        extensions = build_extensions_for_route(
            expert=expert,
            user_extension=self._user_extension,
        )

        # 5. Get route-specific iteration limit
        max_iterations = ROUTE_MAX_ITERATIONS.get(expert, 20)

        logger.info(
            f"HttpRoutedEntry: route={expert.value}, "
            f"extensions={len(extensions)}, "
            f"max_iter={max_iterations}, "
            f"summary={self._route.task_summary[:60]}"
        )

        # 6. Create orchestrator and run
        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                get_llm_params("coder"),
            ],
            extensions=extensions,
            raw_output_parser=None,
            max_iterations=max_iterations,
        )

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

        return EntryOutput()
