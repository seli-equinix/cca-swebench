# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""DocumentToolsExtension — 4 LLM-callable tools for document management.

Tools:
  upload_document             — Chunk, embed, store in ephemeral_docs
  search_documents            — Search within current session's ephemeral docs
  list_session_docs           — List all docs in current session
  promote_doc_to_knowledge    — Copy doc to user or project knowledge collection

ToolGroup: DOCUMENT
Routes: CODER, INFRASTRUCTURE, SEARCH
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from ...core.analect import AnalectRunContext
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)

# Chunking settings (match MCP document_processor.py)
CHUNK_SIZE_CHARS = 4000
CHUNK_OVERLAP_CHARS = 400
EPHEMERAL_TTL_HOURS = 24
MAX_DOC_SIZE = 10 * 1024 * 1024  # 10 MB
EMBEDDING_DIMS = 4096

EPHEMERAL_COLLECTION = "ephemeral_docs"


def _chunk_text(content: str) -> list[str]:
    """Split content into overlapping chunks with natural boundary detection."""
    if len(content) <= CHUNK_SIZE_CHARS:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = start + CHUNK_SIZE_CHARS

        if end < len(content):
            for boundary in ["\n\n", "\n", ". ", " "]:
                pos = content.rfind(boundary, start + CHUNK_SIZE_CHARS // 2, end)
                if pos > start:
                    end = pos + len(boundary)
                    break

        chunks.append(content[start:end])
        new_start = end - CHUNK_OVERLAP_CHARS
        start = new_start if new_start > start else end

    return chunks


def _make_doc_id(session_id: str, name: str) -> str:
    """Generate deterministic document ID."""
    ts = datetime.now().isoformat()
    return hashlib.md5(f"{session_id}:{name}:{ts}".encode()).hexdigest()


class DocumentToolsExtension(ToolUseExtension):
    """CCA extension providing document upload, search, and promotion tools."""

    name: str = "DocumentToolsExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

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
                name="upload_document",
                description=(
                    "Upload a document to the session's ephemeral storage.\n"
                    "Content is chunked, embedded, and stored for 24 hours.\n"
                    "Use this when the user pastes code, config, or text they "
                    "want to analyze or reference later.\n"
                    "Returns the document ID for later reference."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Document content (text, code, config, etc.)",
                        },
                        "name": {
                            "type": "string",
                            "description": "Document name/title",
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "MIME type (default: text/plain)",
                        },
                    },
                    "required": ["content", "name"],
                },
            ),
            ant.Tool(
                name="search_documents",
                description=(
                    "Search documents uploaded in the current session.\n"
                    "Uses semantic search to find relevant chunks.\n"
                    "Only searches within the current session's ephemeral docs."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Max results (1-20, default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ant.Tool(
                name="list_session_docs",
                description=(
                    "List all documents uploaded in the current session.\n"
                    "Returns document names, sizes, and chunk counts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            ant.Tool(
                name="promote_doc_to_knowledge",
                description=(
                    "Promote a session document to permanent knowledge storage.\n"
                    "- target='user': saves to your personal knowledge collection\n"
                    "- target='project': saves to the shared project codebase\n"
                    "The document's embeddings are reused (no re-embedding).\n"
                    "Use this when a document should be permanently searchable."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID (from upload_document result)",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["user", "project"],
                            "description": "Where to promote: 'user' or 'project'",
                        },
                        "project": {
                            "type": "string",
                            "description": "Project name (required for target='project')",
                        },
                    },
                    "required": ["doc_id", "target"],
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
            if name == "upload_document":
                result = await self._handle_upload(inp)
            elif name == "search_documents":
                result = await self._handle_search(inp)
            elif name == "list_session_docs":
                result = await self._handle_list(inp)
            elif name == "promote_doc_to_knowledge":
                result = await self._handle_promote(inp)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error("Document tool '%s' failed: %s", name, e)
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_upload(self, inp: dict[str, Any]) -> str:
        """Upload a document to ephemeral storage."""
        content = inp.get("content", "")
        name = inp.get("name", "untitled").strip()
        mime_type = inp.get("mime_type", "text/plain")

        if not content:
            return json.dumps({"error": "content is required"})
        if len(content) > MAX_DOC_SIZE:
            return json.dumps({"error": f"Document too large (max {MAX_DOC_SIZE // 1024 // 1024}MB)"})

        qdrant = self._backend_clients.qdrant
        if not qdrant or not self._backend_clients.available:
            return json.dumps({"error": "Qdrant/Embedding not available"})

        doc_id = _make_doc_id(self._session_id, name)
        chunks = _chunk_text(content)
        now = datetime.now()
        expires_at = (now + timedelta(hours=EPHEMERAL_TTL_HOURS)).isoformat()

        # Embed all chunks
        try:
            vectors = await self._backend_clients.embed(chunks)
        except Exception as e:
            return json.dumps({"error": f"Embedding failed: {e}"})

        # Ensure collection exists
        await self._ensure_ephemeral_collection(qdrant)

        # Build points
        from qdrant_client.models import PointStruct

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = hashlib.md5(f"{doc_id}:{i}".encode()).hexdigest()
            payload = {
                "doc_id": doc_id,
                "doc_name": name,
                "mime_type": mime_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "session_id": self._session_id,
                "user_id": self._user_id or "",
                "processed_at": now.isoformat(),
                "expires_at": expires_at,
                "is_ephemeral": True,
                "_content": chunk[:5000],
            }
            points.append(PointStruct(id=chunk_id, vector=vec, payload=payload))

        # Upsert
        await qdrant.upsert(collection_name=EPHEMERAL_COLLECTION, points=points)

        return json.dumps({
            "doc_id": doc_id,
            "name": name,
            "chunks": len(chunks),
            "expires_at": expires_at,
        })

    async def _handle_search(self, inp: dict[str, Any]) -> str:
        """Search within session's ephemeral docs."""
        query = inp.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        n_results = min(max(inp.get("n_results", 5), 1), 20)

        qdrant = self._backend_clients.qdrant
        if not qdrant or not self._backend_clients.available:
            return json.dumps({"error": "Qdrant/Embedding not available"})

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vectors = await self._backend_clients.embed([query])
            query_vector = vectors[0]

            results = await qdrant.query_points(
                collection_name=EPHEMERAL_COLLECTION,
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

            formatted = []
            for pt in results.points:
                payload = pt.payload or {}
                formatted.append({
                    "doc_name": payload.get("doc_name", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 0),
                    "score": round(pt.score, 3),
                    "content": payload.get("_content", "")[:2000],
                })

            return json.dumps({
                "results": formatted,
                "count": len(formatted),
                "query": query,
            })

        except Exception as e:
            logger.error("search_documents error: %s", e)
            return json.dumps({"error": str(e)})

    async def _handle_list(self, inp: dict[str, Any]) -> str:
        """List all documents in current session."""
        qdrant = self._backend_clients.qdrant
        if not qdrant:
            return json.dumps({"error": "Qdrant not available"})

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Scroll all points for this session
            docs: dict[str, dict] = {}
            offset = None
            while True:
                results = await qdrant.scroll(
                    collection_name=EPHEMERAL_COLLECTION,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=self._session_id),
                        )
                    ]),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = results

                for pt in points:
                    payload = pt.payload or {}
                    doc_id = payload.get("doc_id", "")
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "doc_id": doc_id,
                            "name": payload.get("doc_name", ""),
                            "mime_type": payload.get("mime_type", ""),
                            "chunks": payload.get("total_chunks", 0),
                            "uploaded_at": payload.get("processed_at", ""),
                            "expires_at": payload.get("expires_at", ""),
                        }

                if next_offset is None:
                    break
                offset = next_offset

            return json.dumps({
                "documents": list(docs.values()),
                "count": len(docs),
                "session_id": self._session_id,
            })

        except Exception as e:
            logger.error("list_session_docs error: %s", e)
            return json.dumps({"error": str(e)})

    async def _handle_promote(self, inp: dict[str, Any]) -> str:
        """Promote an ephemeral document to permanent knowledge."""
        doc_id = inp.get("doc_id", "").strip()
        target = inp.get("target", "").strip()
        project = inp.get("project", "default").strip()

        if not doc_id:
            return json.dumps({"error": "doc_id is required"})
        if target not in ("user", "project"):
            return json.dumps({"error": "target must be 'user' or 'project'"})
        if target == "user" and not self._user_id:
            return json.dumps({"error": "No user_id — cannot promote to user knowledge"})

        qdrant = self._backend_clients.qdrant
        if not qdrant:
            return json.dumps({"error": "Qdrant not available"})

        try:
            from qdrant_client.models import (
                Filter, FieldCondition, MatchValue,
                PointStruct, VectorParams, Distance,
            )

            # Retrieve all chunks for this document (with vectors)
            source_points = []
            offset = None
            while True:
                results = await qdrant.scroll(
                    collection_name=EPHEMERAL_COLLECTION,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=self._session_id),
                        ),
                    ]),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                points, next_offset = results
                source_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            if not source_points:
                return json.dumps({"error": f"No document found with doc_id={doc_id}"})

            # Determine target collection
            if target == "user":
                target_collection = f"user_{self._user_id}_knowledge"
            else:
                target_collection = "codebase_files"

            # Ensure target collection exists
            try:
                await qdrant.get_collection(target_collection)
            except Exception:
                await qdrant.create_collection(
                    collection_name=target_collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMS,
                        distance=Distance.COSINE,
                    ),
                )

            # Build promoted points (reuse embeddings)
            now = datetime.now().isoformat()
            promoted_points = []
            for pt in source_points:
                payload = dict(pt.payload or {})
                # Update metadata for permanent storage
                payload["source"] = "promoted_from_ephemeral"
                payload["promoted_at"] = now
                payload["promoted_by"] = self._user_id or ""
                payload["is_ephemeral"] = False
                payload.pop("expires_at", None)
                payload.pop("session_id", None)

                if target == "project":
                    payload["project"] = project

                new_id = hashlib.md5(
                    f"promoted:{doc_id}:{pt.id}".encode()
                ).hexdigest()

                promoted_points.append(PointStruct(
                    id=new_id,
                    vector=pt.vector,
                    payload=payload,
                ))

            # Upsert to target
            await qdrant.upsert(
                collection_name=target_collection,
                points=promoted_points,
            )

            doc_name = source_points[0].payload.get("doc_name", doc_id) if source_points else doc_id

            return json.dumps({
                "promoted": True,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "target": target,
                "target_collection": target_collection,
                "chunks_promoted": len(promoted_points),
            })

        except Exception as e:
            logger.error("promote_doc_to_knowledge error: %s", e)
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_ephemeral_collection(self, qdrant: Any) -> None:
        """Ensure the ephemeral_docs collection exists."""
        try:
            await qdrant.get_collection(EPHEMERAL_COLLECTION)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            await qdrant.create_collection(
                collection_name=EPHEMERAL_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMS,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", EPHEMERAL_COLLECTION)
