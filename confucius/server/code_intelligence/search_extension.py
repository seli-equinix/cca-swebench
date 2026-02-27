# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""CodeSearchExtension — 3 LLM-callable tools for code intelligence.

Tools:
  search_codebase    — Semantic search over codebase_files Qdrant collection
  search_knowledge   — Unified search across 3 tiers (ephemeral → user → project)
  index_workspace    — Walk filesystem, AST-parse, embed, upsert to Qdrant + Memgraph

ToolGroup: CODE_SEARCH
Routes: CODER, INFRASTRUCTURE, SEARCH
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)


class CodeSearchExtension(ToolUseExtension):
    """CCA extension providing code search and workspace indexing tools."""

    name: str = "CodeSearchExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    # Injected state (set via object.__setattr__ in __init__)
    _backend_clients: Any
    _session_id: str
    _user_id: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        backend_clients: Any,
        session_id: str = "",
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_backend_clients", backend_clients)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_user_id", user_id)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="search_codebase",
                description=(
                    "Semantic search over the indexed codebase. Finds functions, "
                    "classes, sections, and code chunks by meaning — not just keywords.\n"
                    "Results include file path, language, project, function signature, "
                    "and graph context (callers/callees from the knowledge graph).\n"
                    "Use this when the user asks about specific code, functions, or "
                    "implementations in their projects."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Filter by language: python, powershell, bash, "
                                "yaml, markdown"
                            ),
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name (e.g. EVA, OASIS)",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (1-20, default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ant.Tool(
                name="search_knowledge",
                description=(
                    "Unified search across all knowledge tiers:\n"
                    "1. Ephemeral docs (current session uploads, highest priority)\n"
                    "2. User knowledge (personal saved documents)\n"
                    "3. Project knowledge (shared codebase)\n"
                    "Results are priority-ranked: session docs boosted +0.15, "
                    "user docs +0.08, project docs at base score.\n"
                    "Code results are auto-enriched with graph context (callers/callees).\n"
                    "Use this for broad searches across all available knowledge."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Max results per tier (1-20, default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ant.Tool(
                name="index_workspace",
                description=(
                    "Index the workspace filesystem into the code knowledge base.\n"
                    "Walks the specified path (or all configured paths), extracts "
                    "functions/classes via tree-sitter AST parsing, embeds them, "
                    "and stores in the shared Qdrant + Memgraph backends.\n"
                    "Uses SHA256 change detection — unchanged files are skipped.\n"
                    "Use force=true to re-index everything.\n"
                    "Returns stats: files scanned, indexed, skipped, errors."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Directory to index (default: all configured paths). "
                                "Example: /workspace or /home/seli/docker-swarm-stacks"
                            ),
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force re-index all files (default false)",
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
            if name == "search_codebase":
                result = await self._handle_search_codebase(inp)
            elif name == "search_knowledge":
                result = await self._handle_search_knowledge(inp)
            elif name == "index_workspace":
                result = await self._handle_index_workspace(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("CodeSearch tool '%s' failed: %s", name, e)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_search_codebase(self, inp: dict[str, Any]) -> str:
        """Semantic search over codebase_files collection."""
        query = inp.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        n_results = min(max(inp.get("n_results", 5), 1), 20)
        language = inp.get("language")
        project = inp.get("project")

        qdrant = self._backend_clients.qdrant
        if not qdrant or not self._backend_clients.available:
            return json.dumps({"error": "Qdrant/Embedding not available"})

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Embed the query
            vectors = await self._backend_clients.embed([query])
            query_vector = vectors[0]

            # Build filter
            must_conditions = []
            if language:
                must_conditions.append(
                    FieldCondition(key="language", match=MatchValue(value=language))
                )
            if project:
                must_conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )

            search_filter = Filter(must=must_conditions) if must_conditions else None

            # Search
            results = await qdrant.query_points(
                collection_name="codebase_files",
                query=query_vector,
                query_filter=search_filter,
                limit=n_results,
                with_payload=True,
            )

            # Format results
            formatted = []
            for pt in results.points:
                payload = pt.payload or {}
                entry = {
                    "name": payload.get("name", ""),
                    "file_path": payload.get("file_path", ""),
                    "language": payload.get("language", ""),
                    "project": payload.get("project", ""),
                    "type": payload.get("type", ""),
                    "signature": payload.get("signature", ""),
                    "score": round(pt.score, 3),
                    "content": payload.get("_content", "")[:3000],
                }
                # Add AST metadata if present
                for field in ("line_start", "line_end", "is_async", "class_name"):
                    if payload.get(field):
                        entry[field] = payload[field]
                formatted.append(entry)

            # Enrich with graph context
            await self._enrich_with_graph(formatted, project)

            return json.dumps({
                "results": formatted,
                "count": len(formatted),
                "query": query,
            })

        except Exception as e:
            logger.error("search_codebase error: %s", e)
            return json.dumps({"error": str(e)})

    async def _handle_search_knowledge(self, inp: dict[str, Any]) -> str:
        """Unified search across all knowledge tiers."""
        query = inp.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        n_results = min(max(inp.get("n_results", 5), 1), 20)
        language = inp.get("language")
        project = inp.get("project")

        qdrant = self._backend_clients.qdrant
        if not qdrant or not self._backend_clients.available:
            return json.dumps({"error": "Qdrant/Embedding not available"})

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vectors = await self._backend_clients.embed([query])
            query_vector = vectors[0]

            all_results = []
            sources_searched = []

            # Tier 1: Ephemeral docs (session-scoped, +0.15 boost)
            if self._session_id:
                try:
                    ephem_results = await qdrant.query_points(
                        collection_name="ephemeral_docs",
                        query=query_vector,
                        query_filter=Filter(must=[
                            FieldCondition(
                                key="session_id",
                                match=MatchValue(value=self._session_id),
                            )
                        ]),
                        limit=n_results,
                        with_payload=True,
                    )
                    for pt in ephem_results.points:
                        payload = pt.payload or {}
                        all_results.append({
                            "source": "ephemeral",
                            "score": round(min(1.0, pt.score + 0.15), 3),
                            "name": payload.get("doc_name", ""),
                            "content": payload.get("_content", "")[:2000],
                            "file_path": payload.get("file_path", ""),
                        })
                    sources_searched.append("ephemeral")
                except Exception:
                    pass  # Collection may not exist

            # Tier 2: User knowledge (+0.08 boost)
            if self._user_id:
                user_collection = f"user_{self._user_id}_knowledge"
                try:
                    user_results = await qdrant.query_points(
                        collection_name=user_collection,
                        query=query_vector,
                        limit=n_results,
                        with_payload=True,
                    )
                    for pt in user_results.points:
                        payload = pt.payload or {}
                        all_results.append({
                            "source": "user",
                            "score": round(min(1.0, pt.score + 0.08), 3),
                            "name": payload.get("doc_name", payload.get("name", "")),
                            "content": payload.get("_content", "")[:2000],
                            "file_path": payload.get("file_path", ""),
                        })
                    sources_searched.append("user")
                except Exception:
                    pass  # Collection may not exist

            # Tier 3: Project knowledge (base score)
            must_conditions = []
            if language:
                must_conditions.append(
                    FieldCondition(key="language", match=MatchValue(value=language))
                )
            if project:
                must_conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )
            search_filter = Filter(must=must_conditions) if must_conditions else None

            proj_results = await qdrant.query_points(
                collection_name="codebase_files",
                query=query_vector,
                query_filter=search_filter,
                limit=n_results,
                with_payload=True,
            )
            for pt in proj_results.points:
                payload = pt.payload or {}
                all_results.append({
                    "source": "project",
                    "score": round(pt.score, 3),
                    "name": payload.get("name", ""),
                    "file_path": payload.get("file_path", ""),
                    "language": payload.get("language", ""),
                    "project": payload.get("project", ""),
                    "type": payload.get("type", ""),
                    "signature": payload.get("signature", ""),
                    "content": payload.get("_content", "")[:2000],
                })
            sources_searched.append("project")

            # Enrich project results with graph context
            project_results = [r for r in all_results if r.get("source") == "project"]
            await self._enrich_with_graph(project_results, project)

            # Sort by score
            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)

            return json.dumps({
                "results": all_results[:n_results * 3],
                "count": len(all_results),
                "sources_searched": sources_searched,
                "query": query,
            })

        except Exception as e:
            logger.error("search_knowledge error: %s", e)
            return json.dumps({"error": str(e)})

    async def _handle_index_workspace(self, inp: dict[str, Any]) -> str:
        """Index workspace filesystem."""
        from .workspace_indexer import WorkspaceIndexer

        path = inp.get("path", "").strip()
        force = inp.get("force", False)

        # Determine paths to index
        if path:
            paths = [path]
        else:
            paths = self._get_configured_paths()

        indexer = WorkspaceIndexer(self._backend_clients)
        stats = await indexer.index_paths(paths, force=force)
        return json.dumps(stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_configured_paths(self) -> list[str]:
        """Get index paths from config.toml or defaults."""
        try:
            from ...core.config import get_config
            config = get_config()
            raw = config.get("indexer", {})
            paths = raw.get("paths", ["/workspace"])
            if isinstance(paths, list):
                return paths
        except Exception:
            pass
        return ["/workspace"]

    async def _enrich_with_graph(
        self,
        results: list[dict],
        project_filter: Optional[str] = None,
    ) -> None:
        """Enrich search results with callers/callees from Memgraph."""
        memgraph = self._backend_clients.memgraph
        if not memgraph:
            return

        try:
            from .memgraph_client import MemgraphClient
            graph = MemgraphClient(memgraph)
        except Exception:
            return

        for result in results:
            func_name = result.get("name", "")
            file_path = result.get("file_path", "")
            if not func_name or " " in func_name or "/" in func_name:
                continue

            try:
                callers = await graph.get_callers(
                    func_name,
                    project=project_filter or result.get("project"),
                    file_path=file_path,
                    limit=5,
                )
                callees = await graph.get_callees(
                    func_name,
                    project=project_filter or result.get("project"),
                    file_path=file_path,
                    limit=5,
                )

                if callers or callees:
                    result["graph_context"] = {
                        "callers": [
                            {"name": c.get("name", ""), "file": c.get("file_path", "")}
                            for c in callers
                        ],
                        "callees": [
                            {"name": c.get("name", ""), "file": c.get("file_path", "")}
                            for c in callees
                        ],
                    }
            except Exception as e:
                logger.debug("Graph enrichment failed for %s: %s", func_name, e)
