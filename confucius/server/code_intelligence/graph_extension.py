# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""GraphToolsExtension — 3 LLM-callable tools for code knowledge graph.

Tools:
  query_call_graph       — Find callers/callees/call chains for a function
  find_orphan_functions  — Find functions with no callers
  analyze_dependencies   — File-level and function-level dependency analysis

ToolGroup: GRAPH
Routes: CODER, INFRASTRUCTURE
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)


class GraphToolsExtension(ToolUseExtension):
    """CCA extension providing code knowledge graph query tools."""

    name: str = "GraphToolsExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _backend_clients: Any

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, backend_clients: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_backend_clients", backend_clients)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="query_call_graph",
                description=(
                    "Query the code knowledge graph for function call relationships. "
                    "Uses a pre-indexed AST-parsed call graph with complete cross-file "
                    "relationships — more accurate than grep for tracing calls.\n"
                    "- callers: find ALL functions that call the given function "
                    "(including indirect, cross-file callers grep would miss)\n"
                    "- callees: find ALL functions called by the given function\n"
                    "- call_chain: traverse the call graph transitively (up to depth)\n"
                    "Start here for any 'who calls X' or 'what does X call' question."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Function name to query (e.g. 'Backup-VM')",
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["callers", "callees", "call_chain"],
                            "description": "Type of query to run",
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Disambiguate same-name functions by file",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Max traversal depth for call_chain (1-10, default 3)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 20)",
                        },
                    },
                    "required": ["function_name", "query_type"],
                },
            ),
            ant.Tool(
                name="find_orphan_functions",
                description=(
                    "Find functions with no inbound callers in the code knowledge graph. "
                    "The graph tracks ALL call relationships from AST parsing, so this "
                    "finds truly unused functions that grep-based searches would miss.\n"
                    "Results may be dead code, entry points, or event handlers.\n"
                    "Excludes: main, __init__, __main__, _-prefixed functions.\n"
                    "Start here for any 'find unused code' or 'find dead functions' request."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Filter by project name",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 50)",
                        },
                    },
                    "required": [],
                },
            ),
            ant.Tool(
                name="analyze_dependencies",
                description=(
                    "Analyze code dependencies at the file or function level using the "
                    "pre-indexed knowledge graph.\n"
                    "- File-level: lists all functions in a file + their cross-file callers "
                    "(shows coupling and change impact across the codebase)\n"
                    "- Function-level: shows callers, callees, and file breakdown\n"
                    "Start here for any 'what are the dependencies' or 'impact of changes' question."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Analyze dependencies for a file",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Analyze dependencies for a specific function",
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results per query (default 20)",
                        },
                    },
                    "required": [],
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
            if name == "query_call_graph":
                result = await self._handle_query_call_graph(inp)
            elif name == "find_orphan_functions":
                result = await self._handle_find_orphans(inp)
            elif name == "analyze_dependencies":
                result = await self._handle_analyze_deps(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("Graph tool '%s' failed: %s", name, e)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    _cached_graph: Any = None

    def _get_graph(self) -> Any:
        """Get MemgraphClient from BackendClients (cached)."""
        if self._cached_graph is not None:
            return self._cached_graph
        memgraph = self._backend_clients.memgraph
        if not memgraph:
            return None
        from .memgraph_client import MemgraphClient
        self._cached_graph = MemgraphClient(memgraph)
        return self._cached_graph

    async def _handle_query_call_graph(self, inp: dict[str, Any]) -> str:
        """Query callers, callees, or call chains."""
        func_name = inp.get("function_name", "").strip()
        query_type = inp.get("query_type", "callers")
        if not func_name:
            return json.dumps({"error": "function_name is required"})

        graph = self._get_graph()
        if not graph:
            return json.dumps({"error": "Memgraph not available"})

        project = inp.get("project")
        file_path = inp.get("file_path")
        depth = min(max(inp.get("depth", 3), 1), 10)
        limit = min(max(inp.get("limit", 20), 1), 100)

        if query_type == "callers":
            results = await graph.get_callers(
                func_name, project=project, file_path=file_path, limit=limit
            )
            return json.dumps({
                "function": func_name,
                "query_type": "callers",
                "callers": results,
                "count": len(results),
            })

        elif query_type == "callees":
            results = await graph.get_callees(
                func_name, project=project, file_path=file_path, limit=limit
            )
            return json.dumps({
                "function": func_name,
                "query_type": "callees",
                "callees": results,
                "count": len(results),
            })

        elif query_type == "call_chain":
            # Both directions
            outbound = await graph.get_call_chain(
                func_name, depth=depth, direction="out",
                project=project, limit=limit,
            )
            inbound = await graph.get_call_chain(
                func_name, depth=depth, direction="in",
                project=project, limit=limit,
            )
            return json.dumps({
                "function": func_name,
                "query_type": "call_chain",
                "depth": depth,
                "outbound_calls": outbound,
                "inbound_calls": inbound,
                "outbound_count": len(outbound),
                "inbound_count": len(inbound),
            })

        else:
            return json.dumps({"error": f"Invalid query_type: {query_type}"})

    async def _handle_find_orphans(self, inp: dict[str, Any]) -> str:
        """Find functions with no callers."""
        graph = self._get_graph()
        if not graph:
            return json.dumps({"error": "Memgraph not available"})

        project = inp.get("project")
        language = inp.get("language")
        limit = min(max(inp.get("limit", 50), 1), 200)

        results = await graph.find_orphan_functions(
            project=project, language=language, limit=limit
        )
        return json.dumps({
            "orphan_functions": results,
            "count": len(results),
            "filters": {"project": project, "language": language},
        })

    async def _handle_analyze_deps(self, inp: dict[str, Any]) -> str:
        """Analyze file-level or function-level dependencies."""
        graph = self._get_graph()
        if not graph:
            return json.dumps({"error": "Memgraph not available"})

        file_path = inp.get("file_path", "").strip()
        func_name = inp.get("function_name", "").strip()
        project = inp.get("project")
        limit = min(max(inp.get("limit", 20), 1), 100)

        if file_path:
            # File-level analysis
            functions = await graph.get_file_functions(file_path)
            cross_deps = await graph.get_cross_file_deps(file_path, limit=limit)
            return json.dumps({
                "analysis_type": "file",
                "file_path": file_path,
                "functions": functions,
                "function_count": len(functions),
                "cross_file_dependents": cross_deps,
                "dependent_file_count": len(cross_deps),
            })

        elif func_name:
            # Function-level analysis
            callers = await graph.get_callers(
                func_name, project=project, limit=limit
            )
            callees = await graph.get_callees(
                func_name, project=project, limit=limit
            )

            # Group by file
            caller_files: dict[str, list[str]] = {}
            for c in callers:
                fp = c.get("file_path", "unknown")
                caller_files.setdefault(fp, []).append(c.get("name", ""))

            callee_files: dict[str, list[str]] = {}
            for c in callees:
                fp = c.get("file_path", "unknown")
                callee_files.setdefault(fp, []).append(c.get("name", ""))

            return json.dumps({
                "analysis_type": "function",
                "function_name": func_name,
                "callers": callers,
                "caller_count": len(callers),
                "caller_files": caller_files,
                "callees": callees,
                "callee_count": len(callees),
                "callee_files": callee_files,
            })

        else:
            # Graph stats overview
            stats = await graph.get_stats()
            return json.dumps({
                "analysis_type": "overview",
                "graph_stats": stats,
            })
