# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from ...core import types as cf
from ...core.analect import Analect, AnalectRunContext
from ...core.entry.base import EntryInput, EntryOutput
from ...core.entry.decorators import public
from ...core.entry.mixin import EntryAnalectMixin
from ...core.memory import CfMessage
from ...orchestrator.anthropic import AnthropicLLMOrchestrator
from ...orchestrator.extensions import Extension
from ...orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ...orchestrator.extensions.command_line.base import CommandLineExtension
from ...orchestrator.extensions.file.edit import FileEditExtension
from ...orchestrator.extensions.function import FunctionExtension
from ...orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.extensions.plan.llm import LLMCodingArchitectExtension
from ...orchestrator.extensions.expert.reviewer import CodeReviewerExtension
from ...orchestrator.extensions.expert.test_gen import TestGeneratorExtension
from ...orchestrator.extensions.solo import SoloModeExtension
from ...orchestrator.types import OrchestratorInput
from .commands import get_allowed_commands
from ...core.config import get_llm_params
from .tasks import get_task_definition


def get_functions() -> list[Callable[..., Any]]:
    """Placeholder for future function-call tools."""
    return []


@public
class CodeAssistEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Coding Assist Analect

    This analect wires an LLM-based orchestrator with planning, file editing,
    command execution, and thinking extensions to assist with coding tasks.
    """

    @classmethod
    def display_name(cls) -> str:
        return "Code"

    @classmethod
    def description(cls) -> str:
        return "LLM-powered coding assistant with planning, file editing, and CLI tools"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Refactor this module and add tests.")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:

        # Build task/system prompt from template
        task_def: str = get_task_definition(
            current_time=datetime.now().isoformat(timespec="seconds")
        )

        # Prepare extensions per spec
        extensions: list[Extension] = [
            LLMCodingArchitectExtension(),  # planning (config-aware via config_role="planner")
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
            FunctionExtension(functions=get_functions(), enable_tool_use=True),
            CodeReviewerExtension(),    # auto-disabled if "reviewer" not in config
            TestGeneratorExtension(),   # auto-disabled if "tester" not in config
            PlainTextExtension(),
            HierarchicalMemoryExtension(),
            AnthropicPromptCaching(),
            SoloModeExtension(),
        ]

        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                get_llm_params("coder"),
            ],
            extensions=extensions,
            raw_output_parser=None,
        )

        # Use OrchestratorInput to run
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

        # No need to extract messages from memory; the orchestrator and IO handle output display.
        return EntryOutput()
