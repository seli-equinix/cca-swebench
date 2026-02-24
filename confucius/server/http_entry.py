# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""HTTP-aware Code Assist entry for the CCA server.

Extends CodeAssistEntry to:
1. Inject user context into the system prompt (task_def)
2. Add UserToolsExtension to the extensions list
3. Everything else (orchestrator, extensions, LLM) stays identical to CLI mode.

This is the key integration point — user-awareness reaches the orchestrator
without modifying any existing CCA code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from ..core import types as cf
from ..core.analect import AnalectRunContext
from ..core.config import get_llm_params
from ..core.entry.base import EntryInput, EntryOutput
from ..core.entry.decorators import public
from ..core.entry.mixin import EntryAnalectMixin
from ..core.analect import Analect
from ..core.memory import CfMessage
from ..orchestrator.anthropic import AnthropicLLMOrchestrator
from ..orchestrator.extensions import Extension
from ..orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ..orchestrator.extensions.command_line.base import CommandLineExtension
from ..orchestrator.extensions.file.edit import FileEditExtension
from ..orchestrator.extensions.function import FunctionExtension
from ..orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ..orchestrator.extensions.plain_text import PlainTextExtension
from ..orchestrator.extensions.plan.llm import LLMCodingArchitectExtension
from ..orchestrator.extensions.expert.reviewer import CodeReviewerExtension
from ..orchestrator.extensions.expert.test_gen import TestGeneratorExtension
## SoloModeExtension is CLI-only (forces progress-tracking loop).
## HTTP AAAM uses a bounded max_iterations instead.
from ..orchestrator.types import OrchestratorInput
from ..analects.code.commands import get_allowed_commands
from ..analects.code.tasks import get_task_definition
from .utility_tools import UtilityToolsExtension


def _get_functions() -> list[Callable[..., Any]]:
    """Placeholder for future function-call tools."""
    return []


@public
class HttpCodeAssistEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Code assist entry with user awareness for HTTP mode.

    Extends the standard CodeAssistEntry to:
    1. Prepend user personalization context to the system prompt
    2. Append UserToolsExtension to the extensions list

    The orchestrator, all other extensions, LLM selection, and tool execution
    are IDENTICAL to CLI mode.
    """

    def __init__(
        self,
        user_context: str = "",
        user_extension: Optional[Extension] = None,
        route_context: str = "",
    ) -> None:
        super().__init__()
        self._user_context = user_context
        self._user_extension = user_extension
        self._route_context = route_context

    @classmethod
    def display_name(cls) -> str:
        return "HttpCode"

    @classmethod
    def description(cls) -> str:
        return "HTTP-mode coding assistant with user awareness"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="What files are in the current directory?")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        # Build task/system prompt (SAME template as CLI CodeAssistEntry)
        task_def: str = get_task_definition(
            current_time=datetime.now().isoformat(timespec="seconds")
        )

        # INJECT: User personalization context before the task definition
        if self._user_context:
            task_def = self._user_context + "\n\n" + task_def

        # INJECT: Routing context (task summary from classifier)
        if self._route_context:
            task_def = task_def + "\n\n" + self._route_context

        # Build extensions (SAME list as CLI CodeAssistEntry)
        extensions: list[Extension] = [
            LLMCodingArchitectExtension(),
            FileEditExtension(
                max_output_lines=500,
                enable_tool_use=True,
            ),
            CommandLineExtension(
                allowed_commands=get_allowed_commands(),
                max_output_lines=300,
                allow_bash_script=True,
                enable_tool_use=True,
            ),
            FunctionExtension(functions=_get_functions(), enable_tool_use=True),
            CodeReviewerExtension(),
            TestGeneratorExtension(),
            PlainTextExtension(),
            HierarchicalMemoryExtension(),
            AnthropicPromptCaching(),
        ]

        # INJECT: User tools extension (identify_user, remember_user_fact, etc.)
        if self._user_extension is not None:
            extensions.append(self._user_extension)

        # INJECT: Utility tools (web_search, fetch_url_content)
        extensions.append(UtilityToolsExtension())

        # HTTP AAAM: bounded iterations (not the CLI default of 1000)
        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                get_llm_params("coder"),
            ],
            extensions=extensions,
            raw_output_parser=None,
            max_iterations=20,
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
