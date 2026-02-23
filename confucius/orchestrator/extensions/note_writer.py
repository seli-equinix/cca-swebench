# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Qdrant-backed note storage extension for the CCA note-taker LLM.

Provides LLM-callable tools so the deep note-taker can store insights
directly to Qdrant (instead of creating markdown files on disk).
Also provides trajectory retrieval from Redis and note dedup search.

Used by the refactored CCANoteTakerEntry for deep/periodic analysis.
The per-request NoteObserver uses direct API calls instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)

# Collection and key constants (shared with note_observer.py)
NOTES_COLLECTION: str = "cca_notes"
EMBEDDING_DIMS: int = 4096
TRAJECTORY_KEY_PREFIX: str = "cca:trajectory:"


def _stable_note_id(content: str, session_id: str) -> str:
    raw = f"{session_id}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _point_id_from_str(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16) % (2**63)


class NoteWriterExtension(ToolUseExtension):
    """Extension providing Qdrant-backed note storage tools.

    The deep note-taker LLM uses these tools to:
    - store_note: Save an insight to Qdrant (with embedding)
    - search_notes: Semantic search for dedup / context
    - get_trajectory: Read a session trajectory from Redis
    - list_recent_sessions: List available trajectories
    """

    name: str = "NoteWriterExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    # Not declared as pydantic fields — stored directly on instance
    _qdrant: Any  # qdrant_client.QdrantClient
    _redis: Any  # redis.asyncio.Redis
    _embedding_func: Any  # async callable(texts) -> embeddings
    _session_id: str
    _user_id: Optional[str]
    _user_name: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        qdrant: Any,
        redis_client: Any,
        embedding_func: Any,
        session_id: str = "",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_qdrant", qdrant)
        object.__setattr__(self, "_redis", redis_client)
        object.__setattr__(self, "_embedding_func", embedding_func)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_user_id", user_id)
        object.__setattr__(self, "_user_name", user_name)

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="store_note",
                description=(
                    "Save an insight or observation to the knowledge base. "
                    "Notes are embedded and stored for semantic search in future sessions. "
                    "Use for non-obvious insights that required effort to discover."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The insight (1-3 sentences, specific and actionable)",
                        },
                        "note_type": {
                            "type": "string",
                            "description": "Category of insight",
                            "enum": [
                                "insight",
                                "pattern",
                                "pitfall",
                                "technique",
                                "fact",
                            ],
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relevant keywords for searchability",
                        },
                    },
                    "required": ["content", "note_type"],
                },
            ),
            ant.Tool(
                name="search_notes",
                description=(
                    "Search existing notes in the knowledge base by semantic similarity. "
                    "Use before storing a new note to check for duplicates."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Max results to return (default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ant.Tool(
                name="get_trajectory",
                description=(
                    "Retrieve a conversation trajectory from storage by session ID. "
                    "Returns the full message history for analysis."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to retrieve (use list_recent_sessions to find IDs)",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            ant.Tool(
                name="list_recent_sessions",
                description=(
                    "List recent session IDs that have trajectories available for analysis. "
                    "Use to find sessions to analyze with get_trajectory."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max sessions to return (default 20)",
                        },
                    },
                },
            ),
        ]

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Dispatch tool calls to handler methods."""
        params = tool_use.input or {}
        name = tool_use.name

        try:
            if name == "store_note":
                result = await self._handle_store_note(params)
            elif name == "search_notes":
                result = await self._handle_search_notes(params)
            elif name == "get_trajectory":
                result = await self._handle_get_trajectory(params)
            elif name == "list_recent_sessions":
                result = await self._handle_list_recent_sessions(params)
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            logger.error("NoteWriterExtension tool error (%s): %s", name, e)
            result = f"Error: {e}"

        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content=result if isinstance(result, str) else json.dumps(result),
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_store_note(self, params: Dict[str, Any]) -> str:
        """Store a note to Qdrant with embedding."""
        content = params.get("content", "")
        note_type = params.get("note_type", "insight")
        tags = params.get("tags", [])

        if not content:
            return "Error: content is required"

        if self._qdrant is None:
            return "Error: Qdrant not available"

        # Generate embedding
        embeddings = await self._embedding_func([content])
        if not embeddings:
            return "Error: failed to generate embedding"

        try:
            from qdrant_client.http.models import PointStruct

            note_id = _stable_note_id(content, self._session_id)
            now = datetime.now(timezone.utc).isoformat()

            point = PointStruct(
                id=_point_id_from_str(note_id),
                vector=embeddings[0],
                payload={
                    "note_id": note_id,
                    "session_id": self._session_id,
                    "user_id": self._user_id,
                    "user_name": self._user_name,
                    "timestamp": now,
                    "note_type": note_type,
                    "content": content,
                    "tags": tags,
                    "source": "deep_analysis",
                    "project_context": None,
                },
            )

            self._qdrant.upsert(
                collection_name=NOTES_COLLECTION,
                points=[point],
            )

            return f"Stored note (id={note_id}, type={note_type}, tags={tags})"

        except Exception as e:
            return f"Error storing note: {e}"

    async def _handle_search_notes(self, params: Dict[str, Any]) -> str:
        """Semantic search over existing notes."""
        query = params.get("query", "")
        n_results = params.get("n_results", 5)

        if not query:
            return "Error: query is required"
        if self._qdrant is None:
            return "Error: Qdrant not available"

        embeddings = await self._embedding_func([query])
        if not embeddings:
            return "Error: failed to generate embedding"

        try:
            results = self._qdrant.search(
                collection_name=NOTES_COLLECTION,
                query_vector=embeddings[0],
                limit=n_results,
                with_payload=True,
            )

            notes: List[Dict[str, Any]] = []
            for hit in results:
                if hit.payload:
                    notes.append(
                        {
                            "score": round(hit.score, 4),
                            "content": hit.payload.get("content", ""),
                            "type": hit.payload.get("note_type", ""),
                            "tags": hit.payload.get("tags", []),
                            "session_id": hit.payload.get("session_id", ""),
                            "user": hit.payload.get("user_name") or hit.payload.get("user_id", "anonymous"),
                        }
                    )

            return json.dumps(notes, indent=2)

        except Exception as e:
            return f"Error searching notes: {e}"

    async def _handle_get_trajectory(self, params: Dict[str, Any]) -> str:
        """Retrieve trajectory from Redis."""
        session_id = params.get("session_id", "")
        if not session_id:
            return "Error: session_id is required"
        if self._redis is None:
            return "Error: Redis not available"

        try:
            key = f"{TRAJECTORY_KEY_PREFIX}{session_id}"
            data = await self._redis.get(key)
            if data:
                messages = json.loads(data)
                # Truncate very long trajectories for the LLM
                if len(json.dumps(messages)) > 50000:
                    # Take first and last portions
                    return json.dumps(
                        {
                            "session_id": session_id,
                            "message_count": len(messages),
                            "messages": messages[:20] + [{"type": "...", "content": f"[{len(messages) - 40} messages omitted]"}] + messages[-20:],
                            "truncated": True,
                        },
                        indent=2,
                    )
                return json.dumps(
                    {
                        "session_id": session_id,
                        "message_count": len(messages),
                        "messages": messages,
                    },
                    indent=2,
                )
            return f"No trajectory found for session {session_id}"
        except Exception as e:
            return f"Error retrieving trajectory: {e}"

    async def _handle_list_recent_sessions(self, params: Dict[str, Any]) -> str:
        """List recent session IDs from Redis trajectory keys."""
        limit = params.get("limit", 20)
        if self._redis is None:
            return "Error: Redis not available"

        try:
            session_ids: List[str] = []
            cursor = 0
            while len(session_ids) < limit:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match=f"{TRAJECTORY_KEY_PREFIX}*",
                    count=100,
                )
                for key in keys:
                    sid = key.replace(TRAJECTORY_KEY_PREFIX, "")
                    session_ids.append(sid)
                if cursor == 0:
                    break

            return json.dumps(
                {
                    "count": len(session_ids[:limit]),
                    "session_ids": session_ids[:limit],
                },
                indent=2,
            )
        except Exception as e:
            return f"Error listing sessions: {e}"
