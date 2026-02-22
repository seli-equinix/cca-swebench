# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""Test Generator expert extension.

Observes file creation via ToolUseObserver and suggests tests for newly
created files. Output is advisory — injected into memory for the main
LLM to see on its next iteration.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, PrivateAttr

from ....core.analect import AnalectRunContext
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from .base import ExpertExtension
from .test_gen_prompts import TEST_GENERATOR_PROMPT

logger: logging.Logger = logging.getLogger(__name__)


class TestGeneratorExtension(ExpertExtension):
    """Suggests tests after the main LLM creates new files."""

    name: str = "test_generator"
    config_role: str = "tester"
    prompt: ChatPromptTemplate = Field(default=TEST_GENERATOR_PROMPT)
    output_tag: str = "test_suggestions"

    _new_files_created: int = PrivateAttr(default=0)

    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        """Count new file creations."""
        if not self.enabled:
            return
        if tool_use.name == "str_replace_editor":
            command = (tool_use.input or {}).get("command", "")
            if command == "create":
                self._new_files_created += 1

    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """Fire test suggestions when new files have been created."""
        if not self._can_invoke():
            return
        if self._new_files_created == 0:
            return

        self._new_files_created = 0
        await self._run_expert(context)
