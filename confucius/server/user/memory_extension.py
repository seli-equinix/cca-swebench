# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Lightweight user memory extension for non-USER routes.

Provides 3 tools for the 80B LLM on CODER/INFRA/SEARCH routes:
- get_user_context: Read current user profile and session status
- remember_user_fact: Store facts discovered during work
- update_user_preference: Update preferences mentioned during work

Unlike the full UserToolsExtension (6 tools), this does NOT include:
- identify_user (handled server-side by the Functionary router)
- infer_user (semantic matching — router handles this)
- manage_user_profile (CRUD ops — USER route only)

This keeps the tool count low on coding routes while still allowing
the 80B to store/read user context mid-loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

from .session_manager import Session, UserSessionManager

logger = logging.getLogger(__name__)


class UserMemoryExtension(ToolUseExtension):
    """Lightweight user memory for non-USER routes.

    3 tools: get_user_context (read), remember_user_fact (write),
    update_user_preference (write).

    Shares the same session_mgr and session as UserToolsExtension,
    reusing its handler logic for the 3 tools it exposes.
    """

    name: str = "UserMemoryExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _session_mgr: UserSessionManager
    _session: Session
    _critical_facts: Any  # Optional[CriticalFactsExtractor]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        session_mgr: UserSessionManager,
        session: Session,
        critical_facts: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_session_mgr", session_mgr)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_critical_facts", critical_facts)

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="get_user_context",
                description=(
                    "Get information about the current session and identified user. "
                    "Returns session status, full user profile from Qdrant, "
                    "and critical infrastructure facts (IPs, passwords, hostnames) "
                    "from Redis. Use to check who you're talking to and what "
                    "infrastructure context is available."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                },
            ),
            ant.Tool(
                name="remember_user_fact",
                description=(
                    "Save an important fact about the current user. "
                    "Facts persist across sessions and are included in future context.\n\n"
                    "Standard keys: employer, main_project, team, role, "
                    "infrastructure, registry, deployment, tool, preference\n\n"
                    "Special keys for structured profile data:\n"
                    "- key='skill', value='Python' → adds skill to profile skills list\n"
                    "- key='alias', value='seli' → adds alias/nickname to profile\n"
                    "- key='remove_skill', value='Java' → removes skill from profile\n"
                    "- key='remove_alias', value='old_name' → removes alias from profile\n\n"
                    "Call immediately when the user mentions personal details, "
                    "infrastructure setup, tools, or anything they want remembered."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "Fact category. Use 'skill' to add a skill, "
                                "'alias' to add a nickname, 'remove_skill' to "
                                "remove a skill, 'remove_alias' to remove a nickname, "
                                "or any other key for general facts."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": "The fact value",
                        },
                    },
                    "required": ["key", "value"],
                },
            ),
            ant.Tool(
                name="update_user_preference",
                description=(
                    "Update a user preference. Preferences affect how you respond. "
                    "Examples: verbosity=detailed, code_style=python, explanation_level=expert"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Preference name",
                        },
                        "value": {
                            "type": "string",
                            "description": "Preference value",
                        },
                    },
                    "required": ["key", "value"],
                },
            ),
        ]

    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """Dispatch tool calls to the appropriate handler."""
        name = tool_use.name
        inp = tool_use.input or {}

        try:
            if name == "get_user_context":
                result = await self._handle_get_context()
            elif name == "remember_user_fact":
                result = await self._handle_remember_fact(inp)
            elif name == "update_user_preference":
                result = await self._handle_update_preference(inp)
            else:
                result = f"Unknown tool: {name}"

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error(f"UserMemory tool '{name}' failed: {e}")
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    async def _handle_get_context(self) -> str:
        """Fetch current user profile + critical facts."""
        critical_facts = {}
        if self._critical_facts:
            critical_facts = await self._critical_facts.get_facts(
                self._session.session_id
            )
        result = await self._session_mgr.get_session_context(
            self._session, critical_facts
        )
        return json.dumps(result)

    # Keys that route to structured profile operations
    _SKILL_KEYS = {"skill", "skills", "add_skill"}
    _ALIAS_KEYS = {"alias", "aliases", "add_alias", "nickname"}
    _REMOVE_SKILL_KEYS = {"remove_skill", "remove_skills"}
    _REMOVE_ALIAS_KEYS = {"remove_alias", "remove_aliases", "remove_nickname"}

    async def _handle_remember_fact(self, inp: dict) -> str:
        key = inp.get("key", "").strip()
        value = inp.get("value", "").strip()
        if not key or not value:
            return "Error: both key and value are required"

        key_lower = key.lower()

        # Route structured profile operations to proper methods
        if self._session.identified and self._session.user_id:
            if key_lower in self._SKILL_KEYS:
                result = await self._session_mgr.add_user_skill(
                    self._session.user_id, value
                )
                logger.info("Added skill '%s' for user %s", value, self._session.user_id)
                return json.dumps(result)

            if key_lower in self._ALIAS_KEYS:
                result = await self._session_mgr.add_user_alias(
                    self._session.user_id, value
                )
                logger.info("Added alias '%s' for user %s", value, self._session.user_id)
                return json.dumps(result)

            if key_lower in self._REMOVE_SKILL_KEYS:
                result = await self._session_mgr.remove_user_skill(
                    self._session.user_id, value
                )
                logger.info("Removed skill '%s' for user %s", value, self._session.user_id)
                return json.dumps(result)

            if key_lower in self._REMOVE_ALIAS_KEYS:
                result = await self._session_mgr.remove_user_alias(
                    self._session.user_id, value
                )
                logger.info("Removed alias '%s' for user %s", value, self._session.user_id)
                return json.dumps(result)

        # Default: store as generic fact
        success = await self._session_mgr.remember_user_fact(
            self._session, key, value
        )
        if success:
            if self._session.identified:
                return f"Remembered: {key} = {value}"
            return (
                f"Noted: {key} = {value} "
                "(will be saved once user is identified)"
            )
        return "Failed to store fact"

    async def _handle_update_preference(self, inp: dict) -> str:
        key = inp.get("key", "").strip()
        value = inp.get("value", "").strip()
        if not key or not value:
            return "Error: both key and value are required"

        success = await self._session_mgr.update_user_preference(
            self._session, key, value
        )
        if success:
            return f"Preference updated: {key} = {value}"
        return "Failed to update preference"
