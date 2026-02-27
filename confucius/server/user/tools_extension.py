# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""CCA extension providing user identification and memory tools.

Registers LLM-callable tools so the agent can identify users, store facts,
and update preferences during a conversation.  Follows CCA's native
ToolUseExtension pattern (same as FileEditExtension, CommandLineExtension).

Tools provided:
- identify_user(name) -> Link session to user
- remember_user_fact(key, value) -> Store fact about user
- update_user_preference(key, value) -> Update user preference
- infer_user(message) -> Semantic user matching
- get_user_context() -> Get current session/user info (enhanced: Redis + Qdrant)
- manage_user_profile(action, ...) -> Full CRUD for user profiles
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import Field

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

from .session_manager import Session, UserSessionManager

logger = logging.getLogger(__name__)


class UserToolsExtension(ToolUseExtension):
    """CCA extension that provides user identification and memory tools.

    The LLM receives these as callable tools and can invoke them
    to identify users, store facts, and update preferences.
    """

    name: str = "UserToolsExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    # These must NOT be declared as pydantic fields — store as private attrs
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
        # Store on the instance directly (bypass pydantic)
        object.__setattr__(self, "_session_mgr", session_mgr)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_critical_facts", critical_facts)

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="identify_user",
                description=(
                    "Link the current session to a user by name. "
                    "Use when the user says their name (e.g. 'I'm Sean'). "
                    "Returns the user profile if found or created."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The user's name or alias",
                        }
                    },
                    "required": ["name"],
                },
            ),
            ant.Tool(
                name="remember_user_fact",
                description=(
                    "Save an important fact about the current user. "
                    "Facts persist across sessions and are included in future context. "
                    "Use for things like: employer, main project, server names, etc."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Fact category (e.g. 'employer', 'main_project')",
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
            ant.Tool(
                name="infer_user",
                description=(
                    "Check if the current message matches a known user via semantic search. "
                    "Use when the user's identity is unclear but their message "
                    "might contain clues (projects, tools, patterns they've mentioned before)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The user's message to match against known profiles",
                        }
                    },
                    "required": ["message"],
                },
            ),
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
                name="manage_user_profile",
                description=(
                    "Manage user profiles — view, update, or delete user data.\n\n"
                    "Actions:\n"
                    "- view: See all stored data for a user (facts, preferences, skills, aliases, session history)\n"
                    "- update_facts: Add/update facts (pass data={key: value, ...})\n"
                    "- remove_facts: Remove a fact (pass key=\"fact_key\")\n"
                    "- update_preferences: Add/update preferences (pass data={key: value, ...})\n"
                    "- remove_preferences: Remove a preference (pass key=\"pref_key\")\n"
                    "- add_skill: Add a skill (pass value=\"Python\")\n"
                    "- remove_skill: Remove a skill (pass value=\"Python\")\n"
                    "- add_alias: Add an alias name (pass value=\"seli\")\n"
                    "- remove_alias: Remove an alias (pass value=\"old_name\")\n"
                    "- delete_profile: Permanently delete a user profile. ALWAYS set confirm_delete=true when the user says they want to delete or confirms deletion.\n"
                    "- list_all: List all known user profiles (summary view)"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to perform",
                            "enum": [
                                "view", "update_facts", "remove_facts",
                                "update_preferences", "remove_preferences",
                                "add_skill", "remove_skill",
                                "add_alias", "remove_alias",
                                "delete_profile", "list_all",
                            ],
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Target user ID (defaults to current session's user)",
                        },
                        "data": {
                            "type": "object",
                            "description": "Key-value data for update actions",
                        },
                        "key": {
                            "type": "string",
                            "description": "Single key for remove actions",
                        },
                        "value": {
                            "type": "string",
                            "description": "Single value for add/remove operations",
                        },
                        "confirm_delete": {
                            "type": "boolean",
                            "description": "Set to true when user confirms they want deletion. Required for delete_profile action.",
                        },
                    },
                    "required": ["action"],
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
            if name == "identify_user":
                result = await self._handle_identify_user(inp)
            elif name == "remember_user_fact":
                result = await self._handle_remember_fact(inp)
            elif name == "update_user_preference":
                result = await self._handle_update_preference(inp)
            elif name == "infer_user":
                result = await self._handle_infer_user(inp)
            elif name == "get_user_context":
                result = await self._handle_get_context()
            elif name == "manage_user_profile":
                result = await self._handle_manage_profile(inp)
            else:
                result = f"Unknown tool: {name}"

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error(f"User tool '{name}' failed: {e}")
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ==================== Tool Handlers ====================

    async def _handle_identify_user(self, inp: dict) -> str:
        """Thin dispatch to session_manager.identify_user()."""
        name = inp.get("name", "").strip()
        if not name:
            return json.dumps({"error": "name is required"})
        result = await self._session_mgr.identify_user(
            name, self._session
        )
        return json.dumps(result)

    # Keys that route to structured profile operations (shared with UserMemoryExtension)
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
                return json.dumps(result)

            if key_lower in self._ALIAS_KEYS:
                result = await self._session_mgr.add_user_alias(
                    self._session.user_id, value
                )
                return json.dumps(result)

            if key_lower in self._REMOVE_SKILL_KEYS:
                result = await self._session_mgr.remove_user_skill(
                    self._session.user_id, value
                )
                return json.dumps(result)

            if key_lower in self._REMOVE_ALIAS_KEYS:
                result = await self._session_mgr.remove_user_alias(
                    self._session.user_id, value
                )
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

    async def _handle_infer_user(self, inp: dict) -> str:
        """Thin dispatch to session_manager.infer_and_link_user()."""
        message = inp.get("message", "").strip()
        if not message:
            return json.dumps({"error": "message is required"})
        result = await self._session_mgr.infer_and_link_user(
            message, self._session
        )
        return json.dumps(result)

    async def _handle_get_context(self) -> str:
        """Enhanced: fetches critical facts from Redis + full profile from Qdrant."""
        # Fetch critical facts from Redis (async)
        critical_facts = {}
        if self._critical_facts:
            critical_facts = await self._critical_facts.get_facts(
                self._session.session_id
            )
        # Delegate to session_manager (queries Qdrant for full profile)
        result = await self._session_mgr.get_session_context(
            self._session, critical_facts
        )
        return json.dumps(result)

    async def _handle_manage_profile(self, inp: dict) -> str:
        """Dispatch manage_user_profile actions to session_manager methods."""
        action = inp.get("action", "").strip()
        user_id = inp.get("user_id") or (
            self._session.user_id if self._session.identified else None
        )
        mgr = self._session_mgr

        if action == "list_all":
            return json.dumps(await mgr.list_all_user_summaries())

        if not user_id:
            return json.dumps({
                "error": "No user identified. Use identify_user first or specify user_id."
            })

        if action == "view":
            return json.dumps(await mgr.view_user_profile(user_id))
        elif action == "update_facts":
            return json.dumps(
                await mgr.update_user_facts_batch(user_id, inp.get("data", {}))
            )
        elif action == "remove_facts":
            return json.dumps(
                await mgr.remove_user_fact(user_id, inp.get("key", "").strip())
            )
        elif action == "update_preferences":
            return json.dumps(
                await mgr.update_user_preferences_batch(user_id, inp.get("data", {}))
            )
        elif action == "remove_preferences":
            return json.dumps(
                await mgr.remove_user_preference(user_id, inp.get("key", "").strip())
            )
        elif action == "add_skill":
            return json.dumps(
                await mgr.add_user_skill(user_id, inp.get("value", "").strip())
            )
        elif action == "remove_skill":
            return json.dumps(
                await mgr.remove_user_skill(user_id, inp.get("value", "").strip())
            )
        elif action == "add_alias":
            return json.dumps(
                await mgr.add_user_alias(user_id, inp.get("value", "").strip())
            )
        elif action == "remove_alias":
            return json.dumps(
                await mgr.remove_user_alias(user_id, inp.get("value", "").strip())
            )
        elif action == "delete_profile":
            confirm = inp.get("confirm_delete", False)
            result = await mgr.delete_user_profile(
                user_id, confirm_delete=confirm
            )
            if result.get("deleted") and self._session.user_id == user_id:
                self._session.identified = False
                self._session.user_id = None
            return json.dumps(result)

        return json.dumps({"error": f"Unknown action: {action}"})
