# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""Code Reviewer expert extension.

Observes file edits via ToolUseObserver and fires a code review after
a configurable number of edits. The review is advisory — injected into
memory for the main LLM to see on its next iteration.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, PrivateAttr

from ....core.analect import AnalectRunContext
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from .base import ExpertExtension
from .reviewer_prompts import CODE_REVIEWER_PROMPT

logger: logging.Logger = logging.getLogger(__name__)


class CodeReviewerExtension(ExpertExtension):
    """Reviews code changes after the main LLM makes file edits."""

    name: str = "code_reviewer"
    config_role: str = "reviewer"
    prompt: ChatPromptTemplate = Field(default=CODE_REVIEWER_PROMPT)
    output_tag: str = "code_review"
    review_threshold: int = Field(
        default=3,
        description="Number of file edits before triggering a review",
    )

    _edits_since_last_review: int = PrivateAttr(default=0)

    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        """Count file edit operations (CREATE, STR_REPLACE, INSERT)."""
        if not self.enabled:
            return
        if tool_use.name == "str_replace_editor":
            command = (tool_use.input or {}).get("command", "")
            if command in ("create", "str_replace", "insert"):
                self._edits_since_last_review += 1

    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """Fire review if edit threshold is reached."""
        if not self._can_invoke():
            return
        if self._edits_since_last_review < self.review_threshold:
            return

        self._edits_since_last_review = 0
        await self._run_expert(context)
