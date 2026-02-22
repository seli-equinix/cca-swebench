# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""Base class for LLM expert extensions.

Expert extensions make independent LLM calls within the orchestrator loop
to provide advisory feedback (code review, test suggestions, etc.).
They observe tool use via ToolUseObserver and fire at
on_process_messages_complete() after the main LLM's tool execution.

Experts are optional: if the config role is missing, the expert is
silently disabled (all hooks become no-ops). This is the ONE place
where catching CCAConfigError is acceptable.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, PrivateAttr, model_validator

from ....core import types as cf
from ....core.analect import AnalectRunContext
from ....core.config import CCAConfigError, get_llm_params
from ....core.llm_manager import LLMParams
from ....core.memory import CfMessage
from ..plan.utils import prompt_to_convo_tag
from ..tool_use import ToolUseObserver

logger: logging.Logger = logging.getLogger(__name__)


class ExpertExtension(ToolUseObserver):
    """Base class for advisory LLM expert extensions.

    Subclasses must set:
        config_role  — role name in config.toml (e.g. "reviewer")
        prompt       — ChatPromptTemplate for the expert's system prompt
        output_tag   — XML tag wrapping output in memory (e.g. "code_review")

    If the config role is absent, the expert is disabled: all hooks are
    no-ops and no LLM calls are made.
    """

    # --- Configuration ---
    config_role: str = Field(
        ..., description="Role name in config.toml [active] section"
    )
    prompt: ChatPromptTemplate = Field(
        ..., description="System prompt template for this expert"
    )
    output_tag: str = Field(
        ..., description="XML tag wrapping expert output in memory"
    )

    # --- Behavior ---
    max_input_tokens: int = Field(
        default=50000,
        description="Max tokens of conversation context to send to the expert",
    )
    max_invocations: int | None = Field(
        default=None,
        description="Max expert invocations per session (None = unlimited)",
    )
    parallel_safe: bool = Field(
        default=True,
        description="Safe for concurrent execution via asyncio.gather()",
    )

    # --- Resolved at init ---
    llm_params: LLMParams | None = Field(
        default=None,
        description="Resolved from config. None = expert disabled.",
    )

    # --- Private state ---
    _invocation_count: int = PrivateAttr(default=0)

    # Extension base fields
    included_in_system_prompt: bool = False

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _load_config(self) -> ExpertExtension:
        """Load LLM params from config. Disable expert if role is missing."""
        try:
            self.llm_params = get_llm_params(self.config_role)
            logger.info(
                f"Expert '{self.config_role}' enabled: model={self.llm_params.model}"
            )
        except CCAConfigError:
            self.llm_params = None
            logger.info(
                f"Expert '{self.config_role}' disabled (role not in config)"
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether this expert is active (has valid config)."""
        return self.llm_params is not None

    def _can_invoke(self) -> bool:
        """Check if the expert can fire (enabled + under invocation limit)."""
        if not self.enabled:
            return False
        if (
            self.max_invocations is not None
            and self._invocation_count >= self.max_invocations
        ):
            return False
        return True

    async def _call_expert(
        self, context: AnalectRunContext
    ) -> str | None:
        """Make an LLM call with this expert's prompt and conversation context.

        Returns the expert's response text, or None on failure.
        """
        if not self._can_invoke():
            return None

        assert self.llm_params is not None

        # Snapshot recent conversation from memory
        messages = context.memory_manager.get_messages()
        lc_messages = []
        for msg in messages[-50:]:  # Last 50 messages as context
            lc_messages.extend(await msg.to_lc_messages())

        # Format: system prompt + conversation as XML
        convo_tag_messages = prompt_to_convo_tag(lc_messages)
        full_prompt = self.prompt + convo_tag_messages

        # Get chat model and invoke
        chat = context.llm_manager._get_chat(params=self.llm_params)

        try:
            await context.io.system(
                f"Running {self.config_role} expert...",
                run_label=f"{self.config_role.title()} Expert",
            )
            response = await context.invoke(chat, full_prompt.format_messages())
            content = response.content
            text = (
                content
                if isinstance(content, str)
                else "\n".join(
                    ct.get("text", "") if isinstance(ct, dict) else str(ct)
                    for ct in content
                )
            )
            self._invocation_count += 1
            await context.io.system(
                f"{self.config_role.title()} expert complete.",
                run_status=cf.RunStatus.COMPLETED,
                run_label=f"{self.config_role.title()} Expert",
            )
            return text

        except Exception as exc:
            logger.warning(
                f"Expert '{self.config_role}' failed: {type(exc).__name__}: {exc}"
            )
            await context.io.system(
                f"{self.config_role.title()} expert failed: {exc}",
                run_status=cf.RunStatus.FAILED,
                run_label=f"{self.config_role.title()} Expert",
            )
            return None

    async def _inject_into_memory(
        self, text: str, context: AnalectRunContext
    ) -> None:
        """Wrap expert output in XML tag and inject as HUMAN message into memory.

        Uses HUMAN type (not AI) to avoid requiring thinking blocks when
        thinking is enabled — same pattern as LLMPlannerExtension (line 223).
        """
        tagged_text = f"<{self.output_tag}>\n{text}\n</{self.output_tag}>"
        msg = CfMessage(content=tagged_text, type=cf.MessageType.HUMAN)
        context.memory_manager.add_messages([msg])

    async def _run_expert(self, context: AnalectRunContext) -> None:
        """Call expert and inject result into memory. Full pipeline."""
        text = await self._call_expert(context)
        if text:
            await self._inject_into_memory(text, context)
