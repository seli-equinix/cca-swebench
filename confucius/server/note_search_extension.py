# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""NoteSearchExtension — read-only LLM-callable tool for searching past notes.

Tools:
  search_notes — Semantic search over the cca_notes Qdrant collection

ToolGroup: NOTES
Routes: CODER, INFRASTRUCTURE, SEARCH
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..core.analect import AnalectRunContext
from ..core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)

NOTES_COLLECTION = "cca_notes"
MIN_SCORE = 0.15  # Qwen3-Embedding-8B cosine scores are low


class NoteSearchExtension(ToolUseExtension):
    """CCA extension providing read-only access to past session notes."""

    name: str = "NoteSearchExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _backend_clients: Any
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
        object.__setattr__(self, "_backend_clients", backend_clients)
        object.__setattr__(self, "_user_id", user_id)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="search_notes",
                description=(
                    "Search your past session notes for previously learned knowledge.\n"
                    "Notes contain insights, solutions, configurations, and facts "
                    "extracted from previous conversations with this user.\n"
                    "Use this BEFORE web_search when the question might have been "
                    "answered in a previous session (e.g. infrastructure details, "
                    "project configurations, debugging solutions).\n"
                    "Returns matching notes ranked by relevance."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (1-10, default 5)",
                        },
                    },
                    "required": ["query"],
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
            if name == "search_notes":
                result = await self._handle_search_notes(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("NoteSearch tool '%s' failed: %s", name, e)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    async def _handle_search_notes(self, inp: dict[str, Any]) -> str:
        """Semantic search over cca_notes collection."""
        query = inp.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        if not self._user_id:
            return json.dumps({"error": "No user identified — cannot search notes"})

        n_results = min(max(inp.get("n_results", 5), 1), 10)

        qdrant = self._backend_clients.qdrant
        if not qdrant or not self._backend_clients.available:
            return json.dumps({"error": "Qdrant/Embedding not available"})

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vectors = await self._backend_clients.embed([query])
            query_vector = vectors[0]

            results = await qdrant.query_points(
                collection_name=NOTES_COLLECTION,
                query=query_vector,
                query_filter=Filter(must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=self._user_id),
                    )
                ]),
                limit=n_results,
                with_payload=True,
            )

            formatted = []
            for pt in results.points:
                if pt.score < MIN_SCORE:
                    continue
                payload = pt.payload or {}
                formatted.append({
                    "note": payload.get("note", ""),
                    "category": payload.get("category", ""),
                    "session_id": payload.get("session_id", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "score": round(pt.score, 3),
                })

            return json.dumps({
                "results": formatted,
                "count": len(formatted),
                "query": query,
            })

        except Exception as e:
            logger.error("search_notes error: %s", e)
            return json.dumps({"error": str(e)})
