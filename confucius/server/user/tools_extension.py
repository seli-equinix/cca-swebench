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
- get_user_context() -> Get current session/user info
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

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        session_mgr: UserSessionManager,
        session: Session,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Store on the instance directly (bypass pydantic)
        object.__setattr__(self, "_session_mgr", session_mgr)
        object.__setattr__(self, "_session", session)

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
                    "Returns session status, user profile, and preferences if identified."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
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
        name = inp.get("name", "").strip()
        if not name:
            return "Error: name is required"

        mgr = self._session_mgr
        session = self._session

        # Check alias consolidation first
        consolidation = mgr.detect_alias_consolidation(name)
        if consolidation:
            result = await mgr._handle_alias_consolidation(
                consolidation, session
            )
            user = result.get("user")
            if user:
                return (
                    f"Identified as {user.display_name} "
                    f"(action: {result.get('action')}). "
                    f"Session count: {user.session_count}"
                )
            return result.get("message", "Alias consolidation completed")

        # Try to find or create user
        existing = await mgr.find_user_by_name(name)
        if existing:
            await mgr.link_session_to_user(session, existing)
            return (
                f"Welcome back, {existing.display_name}! "
                f"Session #{existing.session_count}. "
                f"Known facts: {json.dumps(existing.facts) if existing.facts else 'none yet'}. "
                f"Preferences: {json.dumps(existing.preferences) if existing.preferences else 'none yet'}."
            )

        # Create new profile
        import uuid

        from .session_manager import UserProfile

        profile = UserProfile(
            user_id=str(uuid.uuid4()),
            display_name=name.capitalize(),
            aliases=[name.lower()],
            session_count=1,
        )
        await mgr.save_user_profile(profile)
        await mgr.link_session_to_user(session, profile)
        return (
            f"Nice to meet you, {profile.display_name}! "
            "I've created your profile and will remember you."
        )

    async def _handle_remember_fact(self, inp: dict) -> str:
        key = inp.get("key", "").strip()
        value = inp.get("value", "").strip()
        if not key or not value:
            return "Error: both key and value are required"

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
        message = inp.get("message", "").strip()
        if not message:
            return "Error: message is required"

        result = await self._session_mgr._infer_user_from_message(
            message, self._session
        )
        if result and result.get("user"):
            user = result["user"]
            confidence = result["confidence"]
            action = result["action"]

            if action == "auto_link":
                await self._session_mgr.link_session_to_user(
                    self._session, user
                )
                return (
                    f"Auto-identified as {user.display_name} "
                    f"(confidence: {confidence:.0%}). Session linked."
                )
            elif action == "ask":
                return (
                    f"Possible match: {user.display_name} "
                    f"(confidence: {confidence:.0%}). "
                    "Ask the user to confirm with identify_user."
                )

        return "No matching user found from message content."

    async def _handle_get_context(self) -> str:
        session = self._session
        parts = [
            f"Session ID: {session.session_id[:16]}...",
            f"Identified: {session.identified}",
            f"Messages: {session.message_count}",
        ]

        if session.identified and session.user_id:
            profile = await self._session_mgr._get_user_profile(
                session.user_id
            )
            if profile:
                parts.extend([
                    f"User: {profile.display_name}",
                    f"Session count: {profile.session_count}",
                    f"Facts: {json.dumps(profile.facts) if profile.facts else 'none'}",
                    f"Preferences: {json.dumps(profile.preferences) if profile.preferences else 'none'}",
                    f"Skills: {', '.join(profile.skills) if profile.skills else 'none'}",
                ])
        else:
            parts.append(
                "User not yet identified. Use identify_user or infer_user."
            )

        return "\n".join(parts)
