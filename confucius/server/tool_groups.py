# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tool group definitions and route-based extension builder.

Maps ExpertType routes to ToolGroup sets, then builds the correct
extension list for each route.  Adding a new tool group is:
  1. Add the ToolGroup enum value
  2. Register the extension factory in TOOL_GROUP_FACTORIES
  3. Add it to the appropriate routes in ROUTE_TOOL_GROUPS
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from ..orchestrator.extensions import Extension
from ..orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ..orchestrator.extensions.command_line.base import CommandLineExtension
from ..orchestrator.extensions.file.edit import FileEditExtension
from ..orchestrator.extensions.function import FunctionExtension
from ..orchestrator.extensions.memory.hierarchical import HierarchicalMemoryExtension
from ..orchestrator.extensions.plain_text import PlainTextExtension
from ..orchestrator.extensions.plan.llm import LLMCodingArchitectExtension

from ..analects.code.commands import get_allowed_commands
from ..analects.infrastructure.commands import get_infra_commands
from .expert_router import ExpertType, RouteDecision
from ..orchestrator.extensions.expert.reviewer import CodeReviewerExtension
from ..orchestrator.extensions.expert.test_gen import TestGeneratorExtension
from .user.memory_extension import UserMemoryExtension
from .utility_tools import UtilityToolsExtension

logger = logging.getLogger(__name__)


# ========================= Tool Groups =========================


class ToolGroup(str, Enum):
    """Logical groupings of tools.  Each maps to one extension class."""
    USER = "user"           # UserToolsExtension (6 tools)
    USER_MEMORY = "user_memory"  # UserMemoryExtension (3 tools: get_context, remember_fact, update_pref)
    WEB = "web"             # UtilityToolsExtension (2 tools)
    FILE = "file"           # FileEditExtension (1 tool: TextEditor)
    SHELL = "shell"         # CommandLineExtension (1 tool: BashTool)
    MEMORY = "memory"       # HierarchicalMemoryExtension (6 tools)
    PLANNER = "planner"     # LLMCodingArchitectExtension (0 tools, prompt-based)
    CODE_SEARCH = "code_search"  # CodeSearchExtension (3 tools: search_codebase, search_knowledge, index_workspace)
    GRAPH = "graph"         # GraphToolsExtension (3 tools: query_call_graph, find_orphan_functions, analyze_dependencies)
    DOCUMENT = "document"   # DocumentToolsExtension (4 tools: upload_document, search_documents, list_session_docs, promote_doc_to_knowledge)
    NOTES = "notes"         # NoteSearchExtension (1 tool: search_notes)
    RULES = "rules"         # RulesToolsExtension (4 tools: create_rule, request_rule, list_rules, delete_rule)
    # Future groups — uncomment as extensions are ported:
    # VISION = "vision"     # VisionToolsExtension (1 tool)
    # GIT = "git"           # GitToolsExtension (1 tool)


# ========================= Route → Tool Groups =========================

# Which tool groups each route needs.
# Order matters: extensions are presented to the LLM in this order.

ROUTE_TOOL_GROUPS: Dict[ExpertType, List[ToolGroup]] = {
    ExpertType.USER: [
        ToolGroup.USER,
        ToolGroup.NOTES,
    ],
    ExpertType.CODER: [
        ToolGroup.PLANNER,
        ToolGroup.FILE,
        ToolGroup.SHELL,
        ToolGroup.MEMORY,
        ToolGroup.WEB,
        ToolGroup.USER_MEMORY,
        ToolGroup.CODE_SEARCH,
        ToolGroup.GRAPH,
        ToolGroup.DOCUMENT,
        ToolGroup.NOTES,
        ToolGroup.RULES,
    ],
    ExpertType.INFRASTRUCTURE: [
        ToolGroup.PLANNER,
        ToolGroup.FILE,
        ToolGroup.SHELL,
        ToolGroup.MEMORY,
        ToolGroup.WEB,
        ToolGroup.USER_MEMORY,
        ToolGroup.CODE_SEARCH,
        ToolGroup.GRAPH,
        ToolGroup.DOCUMENT,
        ToolGroup.NOTES,
        ToolGroup.RULES,
    ],
    ExpertType.SEARCH: [
        ToolGroup.WEB,
        ToolGroup.FILE,
        ToolGroup.MEMORY,
        ToolGroup.USER_MEMORY,
        ToolGroup.CODE_SEARCH,
        ToolGroup.GRAPH,
        ToolGroup.DOCUMENT,
        ToolGroup.NOTES,
    ],
    ExpertType.PLANNER: [
        ToolGroup.PLANNER,
        ToolGroup.MEMORY,
        ToolGroup.CODE_SEARCH,
        ToolGroup.WEB,
        ToolGroup.NOTES,
    ],
}


# ========================= Route Settings =========================


_BASE_MAX_ITERATIONS: Dict[ExpertType, int] = {
    ExpertType.USER: 10,
    ExpertType.CODER: 20,
    ExpertType.INFRASTRUCTURE: 30,
    ExpertType.SEARCH: 15,
    ExpertType.PLANNER: 10,
}


_SIMPLE_MAX_ITERATIONS = 8


def get_max_iterations(route: RouteDecision) -> int:
    """Compute max iterations from route's estimated steps.

    Simple tasks (estimated_steps <= 3) get a low cap to prevent
    over-iteration on one-liners and short functions.
    Formula: max(base, min(estimated_steps * 2, 200)).
    The base per-route value acts as a floor for that route type.
    """
    if route.is_simple:  # estimated_steps <= 3
        return _SIMPLE_MAX_ITERATIONS
    base = _BASE_MAX_ITERATIONS.get(route.expert, 20)
    from_steps = route.estimated_steps * 2
    return max(base, min(from_steps, 200))


def _get_commands_for_route(expert: ExpertType) -> Optional[Dict[str, str]]:
    """Return the command allowlist for a route (lazy evaluation)."""
    if expert == ExpertType.INFRASTRUCTURE:
        return get_infra_commands()
    elif expert in (ExpertType.CODER, ExpertType.SEARCH):
        return get_allowed_commands()
    return None


# ========================= Extension Builder =========================


def build_extensions_for_route(
    route: RouteDecision,
    user_extension: Optional[Extension] = None,
    backend_clients: Optional[Any] = None,
    session_id: str = "",
    user_id: Optional[str] = None,
) -> List[Extension]:
    """Build the extension list for a given route.

    Only includes extensions whose tool group is in the route's
    ROUTE_TOOL_GROUPS mapping.  For complex tasks (estimated_steps >= 8),
    conditionally adds CodeReviewerExtension and TestGeneratorExtension.
    Always appends PlainTextExtension and AnthropicPromptCaching at the end.

    Args:
        route: The RouteDecision from the Functionary router.
        user_extension: Pre-built UserToolsExtension (session-bound).
        backend_clients: Shared BackendClients for code intelligence extensions.
        session_id: Current session ID (for document scoping).
        user_id: Current user ID (for user knowledge collections).

    Returns:
        Ordered list of extensions for the orchestrator.
    """
    expert = route.expert
    groups = ROUTE_TOOL_GROUPS.get(expert, ROUTE_TOOL_GROUPS[ExpertType.CODER])
    commands = _get_commands_for_route(expert)
    extensions: List[Extension] = []

    for group in groups:
        if group == ToolGroup.PLANNER:
            extensions.append(LLMCodingArchitectExtension())

        elif group == ToolGroup.FILE:
            extensions.append(FileEditExtension(
                max_output_lines=500,
                enable_tool_use=True,
            ))

        elif group == ToolGroup.SHELL:
            if commands is not None:
                extensions.append(CommandLineExtension(
                    allowed_commands=commands,
                    max_output_lines=500 if expert == ExpertType.INFRASTRUCTURE else 300,
                    allow_bash_script=True,
                    enable_tool_use=True,
                ))

        elif group == ToolGroup.MEMORY:
            extensions.append(HierarchicalMemoryExtension())

        elif group == ToolGroup.USER:
            if user_extension is not None:
                extensions.append(user_extension)

        elif group == ToolGroup.WEB:
            extensions.append(UtilityToolsExtension())

        elif group == ToolGroup.USER_MEMORY:
            # Lightweight 3-tool user memory for non-USER routes.
            # Extract session_mgr/session/critical_facts from the
            # pre-built UserToolsExtension (same session context).
            if user_extension is not None:
                from .user.tools_extension import UserToolsExtension
                if isinstance(user_extension, UserToolsExtension):
                    extensions.append(UserMemoryExtension(
                        session_mgr=user_extension._session_mgr,
                        session=user_extension._session,
                        critical_facts=user_extension._critical_facts,
                    ))

        elif group == ToolGroup.CODE_SEARCH:
            if backend_clients is not None:
                from .code_intelligence.search_extension import CodeSearchExtension
                extensions.append(CodeSearchExtension(
                    backend_clients=backend_clients,
                    session_id=session_id,
                    user_id=user_id,
                ))

        elif group == ToolGroup.GRAPH:
            if backend_clients is not None and backend_clients.memgraph is not None:
                from .code_intelligence.graph_extension import GraphToolsExtension
                extensions.append(GraphToolsExtension(
                    backend_clients=backend_clients,
                ))

        elif group == ToolGroup.DOCUMENT:
            if backend_clients is not None:
                from .code_intelligence.document_extension import DocumentToolsExtension
                extensions.append(DocumentToolsExtension(
                    backend_clients=backend_clients,
                    session_id=session_id,
                    user_id=user_id,
                ))

        elif group == ToolGroup.NOTES:
            if backend_clients is not None and user_id:
                from .note_search_extension import NoteSearchExtension
                extensions.append(NoteSearchExtension(
                    backend_clients=backend_clients,
                    user_id=user_id,
                ))

        elif group == ToolGroup.RULES:
            if backend_clients is not None:
                from .code_intelligence.rules_extension import RulesToolsExtension
                extensions.append(RulesToolsExtension(
                    backend_clients=backend_clients,
                    user_id=user_id,
                ))

    # Conditionally add expert extensions for complex tasks
    if route.is_complex:  # estimated_steps >= 8
        if expert in (ExpertType.CODER, ExpertType.INFRASTRUCTURE):
            reviewer = CodeReviewerExtension(review_threshold=2)
            if reviewer.enabled:
                extensions.append(reviewer)
                logger.info("Added CodeReviewerExtension (threshold=2)")

            if expert == ExpertType.CODER:
                tester = TestGeneratorExtension()
                if tester.enabled:
                    extensions.append(tester)
                    logger.info("Added TestGeneratorExtension")

    # Always include these non-tool extensions
    extensions.append(PlainTextExtension())
    extensions.append(AnthropicPromptCaching())

    tool_count = sum(
        1 for ext in extensions
        if hasattr(ext, 'enable_tool_use') and ext.enable_tool_use
    )
    logger.info(
        f"Built {len(extensions)} extensions ({tool_count} tool-providing) "
        f"for route {expert.value} (estimated_steps={route.estimated_steps}): "
        f"groups={[g.value for g in groups]}"
    )

    return extensions
