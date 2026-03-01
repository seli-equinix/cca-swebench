# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""RulesToolsExtension — 4 LLM-callable tools for behavior rule management.

Tools:
  create_rule    — Create a persistent behavior rule (always/auto_attached/agent_requested/manual)
  request_rule   — Retrieve rules by name, file path, or semantic search
  list_rules     — List all rules with optional type filter
  delete_rule    — Delete a rule (ownership check)

ToolGroup: RULES
Routes: CODER, INFRASTRUCTURE

Shares the `mcp_rules` Qdrant collection with the MCP server.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension
from .rules_store import Rule, RulesStore, VALID_RULE_TYPES

logger = logging.getLogger(__name__)


class RulesToolsExtension(ToolUseExtension):
    """CCA extension providing behavior rule management tools."""

    name: str = "RulesToolsExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _store: Any  # RulesStore
    _user_id: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        backend_clients: Any,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_store", RulesStore(backend_clients))
        object.__setattr__(self, "_user_id", user_id)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="create_rule",
                description=(
                    "Create a persistent behavior rule that guides your responses.\n\n"
                    "Rule types:\n"
                    "- **always**: Auto-applied every conversation "
                    "(e.g. 'Always use type hints in Python')\n"
                    "- **auto_attached**: Applied when working with matching files "
                    "(requires globs or regex)\n"
                    "- **agent_requested**: You can find it via semantic search "
                    "(requires description)\n"
                    "- **manual**: Only applied when the user explicitly mentions it by name\n\n"
                    "Use this when the user says things like 'always remember to...', "
                    "'whenever I work with Python files...', or 'create a rule for...'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short rule name (e.g. 'python-style', 'api-errors')",
                        },
                        "rule": {
                            "type": "string",
                            "description": "The rule content — instructions to follow",
                        },
                        "rule_type": {
                            "type": "string",
                            "enum": ["always", "auto_attached", "agent_requested", "manual"],
                            "description": "When this rule should be applied",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description for agent_requested semantic matching",
                        },
                        "globs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File glob patterns for auto_attached (e.g. ['*.py', 'src/**/*.ts'])",
                        },
                        "regex": {
                            "type": "string",
                            "description": "Regex pattern for auto_attached file matching",
                        },
                        "global_rule": {
                            "type": "boolean",
                            "description": "If true, rule applies to all users (default: false)",
                        },
                    },
                    "required": ["name", "rule", "rule_type"],
                },
            ),
            ant.Tool(
                name="request_rule",
                description=(
                    "Retrieve rules by exact name, file path match, or semantic search.\n\n"
                    "- **name**: Get a specific rule by exact name\n"
                    "- **file_path**: Find auto_attached rules that match a file path\n"
                    "- **query**: Semantic search over rule descriptions\n"
                    "- No parameters: return all rules for the current user\n\n"
                    "Use this to check existing rules before creating duplicates, "
                    "or to find relevant guidelines for the current task."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact rule name to look up",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "File path to match against auto_attached rules",
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query for semantic rule search",
                        },
                        "include_global": {
                            "type": "boolean",
                            "description": "Include global rules (default: true)",
                        },
                    },
                },
            ),
            ant.Tool(
                name="list_rules",
                description=(
                    "List all rules with optional filtering by type.\n"
                    "Returns rule names, types, and descriptions (not full content).\n"
                    "Use request_rule with a name to get the full rule content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rule_type": {
                            "type": "string",
                            "enum": ["always", "auto_attached", "agent_requested", "manual"],
                            "description": "Filter by rule type",
                        },
                        "include_global": {
                            "type": "boolean",
                            "description": "Include global rules (default: true)",
                        },
                    },
                },
            ),
            ant.Tool(
                name="delete_rule",
                description=(
                    "Delete a rule by ID or name. You can only delete your own rules.\n"
                    "Use list_rules first to find the rule ID."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "rule_id": {
                            "type": "string",
                            "description": "Rule ID to delete",
                        },
                        "name": {
                            "type": "string",
                            "description": "Rule name to delete (alternative to rule_id)",
                        },
                    },
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        name = tool_use.name
        inp = tool_use.input or {}

        try:
            if name == "create_rule":
                result = await self._handle_create(inp)
            elif name == "request_rule":
                result = await self._handle_request(inp)
            elif name == "list_rules":
                result = await self._handle_list(inp)
            elif name == "delete_rule":
                result = await self._handle_delete(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("Rules tool '%s' failed: %s", name, e)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_create(self, inp: dict[str, Any]) -> str:
        rule_name = inp.get("name", "").strip()
        rule_content = inp.get("rule", "").strip()
        rule_type = inp.get("rule_type", "manual")
        description = inp.get("description", "").strip() or None
        globs = inp.get("globs", [])
        regex = inp.get("regex")
        global_rule = inp.get("global_rule", False)

        if not rule_name:
            return json.dumps({"error": "name is required"})
        if not rule_content:
            return json.dumps({"error": "rule content is required"})
        if rule_type not in VALID_RULE_TYPES:
            return json.dumps({
                "error": f"Invalid rule_type: {rule_type}. "
                f"Must be: {', '.join(sorted(VALID_RULE_TYPES))}"
            })
        if rule_type == "agent_requested" and not description:
            return json.dumps({
                "error": "agent_requested rules require a description for semantic matching"
            })
        if rule_type == "auto_attached" and not (globs or regex):
            return json.dumps({
                "error": "auto_attached rules require globs or regex patterns"
            })

        user_id = "global" if global_rule else (self._user_id or "anonymous")

        rule = Rule(
            id=str(uuid.uuid4()),
            name=rule_name,
            rule_type=rule_type,
            rule=rule_content,
            description=description,
            globs=globs if isinstance(globs, list) else [],
            regex=regex,
            user_id=user_id,
        )

        if await self._store.store_rule(rule):
            return json.dumps({
                "rule_id": rule.id,
                "name": rule_name,
                "rule_type": rule_type,
                "user_id": user_id,
                "message": f"Rule '{rule_name}' created successfully",
            })
        return json.dumps({"error": "Failed to store rule"})

    async def _handle_request(self, inp: dict[str, Any]) -> str:
        name = inp.get("name", "").strip() or None
        file_path = inp.get("file_path", "").strip() or None
        query = inp.get("query", "").strip() or None
        include_global = inp.get("include_global", True)

        user_id = self._user_id or "anonymous"
        store = self._store

        rules: list[Rule] = []
        if name:
            rule = await store.get_rule_by_name(
                name, user_id if not include_global else None
            )
            if rule:
                rules = [rule]
        elif file_path:
            rules = await store.get_rules_for_file(file_path, user_id)
        elif query:
            rules = await store.search_rules_semantic(query, user_id, n_results=5)
        else:
            rules = await store.get_all_rules(
                user_id, include_global=include_global
            )

        return json.dumps({
            "rules": [r.to_dict() for r in rules],
            "count": len(rules),
            "search_criteria": {
                "name": name,
                "file_path": file_path,
                "query": query,
            },
        })

    async def _handle_list(self, inp: dict[str, Any]) -> str:
        rule_type = inp.get("rule_type")
        include_global = inp.get("include_global", True)
        user_id = self._user_id or "anonymous"

        rules = await self._store.get_all_rules(
            user_id=user_id,
            rule_type=rule_type,
            include_global=include_global,
        )

        by_type: dict[str, list[dict]] = {
            "always": [], "auto_attached": [], "agent_requested": [], "manual": [],
        }
        for rule in rules:
            if rule.rule_type in by_type:
                by_type[rule.rule_type].append({
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "user_id": rule.user_id,
                    "globs": rule.globs if rule.rule_type == "auto_attached" else None,
                })

        return json.dumps({
            "total_count": len(rules),
            "by_type": by_type,
            "filter": {"rule_type": rule_type, "include_global": include_global},
        })

    async def _handle_delete(self, inp: dict[str, Any]) -> str:
        rule_id = inp.get("rule_id", "").strip() or None
        name = inp.get("name", "").strip() or None

        if not rule_id and not name:
            return json.dumps({"error": "Either rule_id or name is required"})

        user_id = self._user_id or "anonymous"
        store = self._store

        # Resolve name → rule_id with ownership check
        if not rule_id and name:
            rule = await store.get_rule_by_name(name, user_id)
            if not rule:
                return json.dumps({"error": f"Rule not found: {name}"})
            if rule.user_id != user_id and rule.user_id != "global":
                return json.dumps({"error": "Cannot delete another user's rule"})
            rule_id = rule.id

        if await store.delete_rule(rule_id):
            return json.dumps({
                "deleted": True,
                "rule_id": rule_id,
                "name": name,
            })
        return json.dumps({"error": "Failed to delete rule"})
