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
        ToolGroup.WEB,          # web_search + fetch_url_content
        ToolGroup.USER_MEMORY,  # remember_user_fact + get_user_context
        ToolGroup.NOTES,        # search_notes (past session knowledge)
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

    SEARCH route uses direct 35B (no 8B research phase). Correct flow:
    iter 0 (parallel web_search) → iter 1 (fetch_url_content) → iter 2
    (write final answer). Cap at 8 — buffer for complexity without
    allowing the runaway search loops that higher caps permitted.
    """
    base = _BASE_MAX_ITERATIONS.get(route.expert, 20)
    if route.expert == ExpertType.SEARCH:
        # Direct 35B needs at most 8 iterations: search batch + fetch + answer.
        # Old comment about "17+ 8B iterations" is obsolete — 8B removed.
        return 8
    if route.is_simple:  # estimated_steps <= 3
        return _SIMPLE_MAX_ITERATIONS
    from_steps = route.estimated_steps * 2
    return max(base, min(from_steps, 200))


def _get_commands_for_route(expert: ExpertType) -> Optional[Dict[str, str]]:
    """Return the command allowlist for a route (lazy evaluation)."""
    if expert == ExpertType.INFRASTRUCTURE:
        return get_infra_commands()
    elif expert in (ExpertType.CODER, ExpertType.SEARCH):
        return get_allowed_commands()
    return None


# ========================= Pool Groups =========================

# Tool groups eligible for dynamic escalation (pool building).
# Excludes USER (session-bound, already on USER route) and PLANNER
# (prompt-only, no tool_use). These are the groups that can be
# selectively enabled mid-loop by the Functionary tool selector.
POOLABLE_GROUPS: frozenset[ToolGroup] = frozenset({
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
})


# ========================= Extension Builder =========================


def _build_extension_for_group(
    group: ToolGroup,
    expert: ExpertType,
    commands: Optional[Dict[str, str]],
    user_extension: Optional[Extension],
    backend_clients: Optional[Any],
    session_id: str,
    user_id: Optional[str],
) -> Optional[Extension]:
    """Build a single extension for a tool group.

    Returns None if the group's prerequisites aren't met (e.g., no
    backend_clients for CODE_SEARCH, no user_extension for USER).
    """
    if group == ToolGroup.PLANNER:
        return LLMCodingArchitectExtension()

    elif group == ToolGroup.FILE:
        return FileEditExtension(
            max_output_lines=500,
            enable_tool_use=True,
        )

    elif group == ToolGroup.SHELL:
        if commands is not None:
            return CommandLineExtension(
                allowed_commands=commands,
                max_output_lines=500 if expert == ExpertType.INFRASTRUCTURE else 300,
                allow_bash_script=True,
                enable_tool_use=True,
            )

    elif group == ToolGroup.MEMORY:
        return HierarchicalMemoryExtension()

    elif group == ToolGroup.USER:
        if user_extension is not None:
            return user_extension

    elif group == ToolGroup.WEB:
        return UtilityToolsExtension()

    elif group == ToolGroup.USER_MEMORY:
        if user_extension is not None:
            from .user.tools_extension import UserToolsExtension
            if isinstance(user_extension, UserToolsExtension):
                return UserMemoryExtension(
                    session_mgr=user_extension._session_mgr,
                    session=user_extension._session,
                    critical_facts=user_extension._critical_facts,
                )

    elif group == ToolGroup.CODE_SEARCH:
        if backend_clients is not None:
            from .code_intelligence.search_extension import CodeSearchExtension
            return CodeSearchExtension(
                backend_clients=backend_clients,
                session_id=session_id,
                user_id=user_id,
            )

    elif group == ToolGroup.GRAPH:
        if backend_clients is not None and backend_clients.memgraph is not None:
            from .code_intelligence.graph_extension import GraphToolsExtension
            return GraphToolsExtension(
                backend_clients=backend_clients,
            )

    elif group == ToolGroup.DOCUMENT:
        if backend_clients is not None:
            from .code_intelligence.document_extension import DocumentToolsExtension
            return DocumentToolsExtension(
                backend_clients=backend_clients,
                session_id=session_id,
                user_id=user_id,
            )

    elif group == ToolGroup.NOTES:
        if backend_clients is not None and user_id:
            from .note_search_extension import NoteSearchExtension
            return NoteSearchExtension(
                backend_clients=backend_clients,
                user_id=user_id,
            )

    elif group == ToolGroup.RULES:
        if backend_clients is not None:
            from .code_intelligence.rules_extension import RulesToolsExtension
            return RulesToolsExtension(
                backend_clients=backend_clients,
                user_id=user_id,
            )

    return None


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
        ext = _build_extension_for_group(
            group, expert, commands,
            user_extension, backend_clients,
            session_id, user_id,
        )
        if ext is not None:
            extensions.append(ext)

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


def build_tool_pool(
    route: RouteDecision,
    user_extension: Optional[Extension] = None,
    backend_clients: Optional[Any] = None,
    session_id: str = "",
    user_id: Optional[str] = None,
) -> Dict[ToolGroup, Extension]:
    """Build a pool of disabled extensions for dynamic escalation.

    Returns extensions for tool groups that the CODER route has but the
    current route does not. Pool extensions are built with tool_use
    disabled — they're invisible to the LLM until escalation enables them.

    CODER and INFRASTRUCTURE routes already have all tools, so this
    returns an empty dict for those routes (zero overhead).

    Args:
        route: The RouteDecision from the Functionary router.
        user_extension: Pre-built UserToolsExtension (session-bound).
        backend_clients: Shared BackendClients for code intelligence extensions.
        session_id: Current session ID (for document scoping).
        user_id: Current user ID (for user knowledge collections).

    Returns:
        Dict mapping ToolGroup → disabled Extension for selective enablement.
    """
    expert = route.expert

    # CODER and INFRASTRUCTURE already have everything — empty pool
    if expert in (ExpertType.CODER, ExpertType.INFRASTRUCTURE):
        return {}

    current_groups = set(ROUTE_TOOL_GROUPS.get(expert, []))
    coder_groups = set(ROUTE_TOOL_GROUPS[ExpertType.CODER])

    # Pool = CODER groups minus current groups, filtered to poolable only
    pool_groups = (coder_groups - current_groups) & POOLABLE_GROUPS

    if not pool_groups:
        return {}

    # Pool SHELL always uses CODER's command allowlist (conservative)
    pool_commands = get_allowed_commands()
    pool: Dict[ToolGroup, Extension] = {}

    for group in pool_groups:
        ext = _build_extension_for_group(
            group, ExpertType.CODER, pool_commands,
            user_extension, backend_clients,
            session_id, user_id,
        )
        if ext is not None:
            # Disable the extension — invisible to the LLM until escalation
            if hasattr(ext, 'enable_tool_use'):
                ext.enable_tool_use = False
            if hasattr(ext, 'included_in_system_prompt'):
                ext.included_in_system_prompt = False
            pool[group] = ext

    if pool:
        logger.info(
            f"Built tool pool with {len(pool)} groups for {expert.value}: "
            f"{[g.value for g in pool]}"
        )

    return pool
