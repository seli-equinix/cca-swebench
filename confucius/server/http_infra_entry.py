# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""HTTP-aware Infrastructure entry for the CCA server.

Follows the same pattern as HttpCodeAssistEntry but uses:
- Infrastructure-specific system prompt (cluster info, SSH access, etc.)
- Expanded command allowlist (docker, ssh, systemctl, etc.)
- Same orchestrator, same LLM, same session management

Created by the expert router when a request is classified as infrastructure.
"""

from __future__ import annotations

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
from ..orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ..orchestrator.extensions.command_line.base import CommandLineExtension
from ..orchestrator.extensions.file.edit import FileEditExtension
from ..orchestrator.extensions.function import FunctionExtension
from ..orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ..orchestrator.extensions.plain_text import PlainTextExtension
from ..orchestrator.extensions.plan.llm import LLMCodingArchitectExtension
from ..orchestrator.types import OrchestratorInput

from ..analects.infrastructure.commands import get_infra_commands
from ..analects.infrastructure.tasks import get_infra_task_definition
from .utility_tools import UtilityToolsExtension


def _get_functions() -> list[Callable[..., Any]]:
    """Placeholder for future function-call tools."""
    return []


@public
class HttpInfraEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Infrastructure entry with expanded tools for HTTP mode.

    Differences from HttpCodeAssistEntry:
    1. Uses infrastructure-specific system prompt (cluster info, SSH, etc.)
    2. Expanded command allowlist (docker, ssh, systemctl, sshpass, etc.)
    3. No code review or test generation experts (not relevant)
    4. Higher max_iterations (infra tasks often have more steps)
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
        return "HttpInfra"

    @classmethod
    def description(cls) -> str:
        return "HTTP-mode infrastructure assistant with DevOps tools"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Check the status of all Docker Swarm services.")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        # Build infrastructure-specific system prompt
        task_def: str = get_infra_task_definition(
            current_time=datetime.now().isoformat(timespec="seconds")
        )

        # Inject user personalization context
        if self._user_context:
            task_def = self._user_context + "\n\n" + task_def

        # Inject routing context (task summary from classifier)
        if self._route_context:
            task_def = task_def + "\n\n" + self._route_context

        # Build extensions — infra-focused
        extensions: list[Extension] = [
            LLMCodingArchitectExtension(),  # planning still useful
            FileEditExtension(
                max_output_lines=500,
                enable_tool_use=True,
            ),
            CommandLineExtension(
                allowed_commands=get_infra_commands(),  # EXPANDED allowlist
                max_output_lines=500,  # infra output can be verbose
                allow_bash_script=True,
                enable_tool_use=True,
            ),
            FunctionExtension(functions=_get_functions(), enable_tool_use=True),
            # No CodeReviewerExtension — not relevant for infra
            # No TestGeneratorExtension — not relevant for infra
            PlainTextExtension(),
            HierarchicalMemoryExtension(),
            AnthropicPromptCaching(),
        ]

        # User tools (identify_user, remember_user_fact, etc.)
        if self._user_extension is not None:
            extensions.append(self._user_extension)

        # Utility tools (web_search, fetch_url_content)
        extensions.append(UtilityToolsExtension())

        # Use coder model for infra too (same 80B model, different prompt)
        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                get_llm_params("coder"),
            ],
            extensions=extensions,
            raw_output_parser=None,
            max_iterations=30,  # Infra tasks often need more steps
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
