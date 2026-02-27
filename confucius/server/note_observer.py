# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Lightweight per-request note extraction service.

Fires asynchronously after every HTTP request to extract insights from the
coder's conversation trajectory.  Notes are stored in Qdrant for semantic
search; raw trajectories are stored in Redis for the deep note-taker.

Infrastructure (all on Spark1, already running):
- Qdrant:     192.168.4.205:6333  (cca_notes collection)
- Embedding:  192.168.4.205:8200  (Qwen3-Embedding-8B, 4096 dims)
- Redis:      192.168.4.205:6379  (trajectory storage, 24h TTL)
- vLLM:       192.168.4.205:8400  (Qwen3-8B-FP8 for extraction)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.tracing import (
    get_tracer,
    OPENINFERENCE_SPAN_KIND,
    INPUT_VALUE,
    OUTPUT_VALUE,
    LLM_MODEL_NAME,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_COMPLETION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via env or constructor)
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://192.168.4.205:6333")
DEFAULT_EMBEDDING_URL: str = os.getenv("EMBEDDING_URL", "http://192.168.4.205:8200")
DEFAULT_REDIS_URL: str = os.getenv(
    "REDIS_URL", "redis://:Loveme-sex64@192.168.4.205:6379/0"
)

NOTES_COLLECTION: str = "cca_notes"
EMBEDDING_DIMS: int = 4096
TRAJECTORY_KEY_PREFIX: str = "cca:trajectory:"
TRAJECTORY_TTL: int = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Extraction prompt (lightweight — no tool use, just JSON output)
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT: str = """\
You are a coding session observer. Extract key insights from this conversation.

Return ONLY a JSON array of notes. Each note has:
- "content": The insight (1-3 sentences, specific and actionable)
- "type": One of "insight", "pattern", "pitfall", "technique", "fact"
- "tags": List of relevant keywords

Rules:
- Only extract non-obvious insights that required effort to discover
- Skip trivial observations (standard patterns, documented behavior)
- 0-5 notes per conversation (empty array [] if nothing noteworthy)
- Be specific — include file names, function names, error messages
- If the user revealed personal/project info, create a "fact" type note

Return valid JSON only, no markdown fences."""


def _stable_note_id(content: str, session_id: str) -> str:
    """Generate a stable ID for deduplication."""
    raw = f"{session_id}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _point_id_from_str(s: str) -> int:
    """Convert a hex string to a stable Qdrant point ID (positive int)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16) % (2**63)


class NoteObserver:
    """Lightweight per-request note extraction service.

    Fires as an async background task after the coder orchestrator completes.
    Extracts structured insights via Spark1 Qwen3-8B and stores them in
    Qdrant for semantic search.  Also persists raw trajectories to Redis.

    All errors are caught and logged — note-taking must never crash the
    main request flow.
    """

    def __init__(
        self,
        *,
        llm_url: str,
        llm_model: str,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
        redis_url: str = DEFAULT_REDIS_URL,
        redis_client: Any = None,
        temperature: float = 0.3,
        max_trajectory_messages: int = 50,
    ) -> None:
        self._llm_url = llm_url.rstrip("/")
        self._llm_model = llm_model
        self._qdrant_url = qdrant_url
        self._embedding_url = embedding_url
        self._redis_url = redis_url
        self._temperature = temperature
        self._max_messages = max_trajectory_messages

        # Backends (set during initialize())
        self._redis: Any = redis_client  # redis.asyncio.Redis (may be pre-injected)
        self._qdrant: Any = None  # qdrant_client.AsyncQdrantClient
        self._http_client: Any = None  # httpx.AsyncClient (for embeddings/non-LLM)
        self._openai_client: Any = None  # openai.AsyncOpenAI (for LLM — auto-traced)
        self._embedding_model: Optional[str] = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Qdrant, Redis, embedding server.  Safe to call multiple times."""
        if self._initialized:
            return

        import httpx
        from openai import AsyncOpenAI

        self._http_client = httpx.AsyncClient(timeout=60.0)

        # AsyncOpenAI client for LLM calls — auto-traced by OpenAIInstrumentor
        self._openai_client = AsyncOpenAI(
            base_url=self._llm_url,
            api_key="not-needed",  # local vLLM, no auth
            timeout=60.0,
        )

        # ---- Redis (reuse if injected, else connect) --------------------
        if self._redis is None:
            try:
                import redis.asyncio as redis_async

                self._redis = redis_async.from_url(
                    self._redis_url, decode_responses=True
                )
                await self._redis.ping()
                logger.info("NoteObserver connected to Redis")
            except Exception as e:
                logger.warning("NoteObserver: Redis unavailable: %s", e)
                self._redis = None

        # ---- Qdrant -----------------------------------------------------
        try:
            from qdrant_client import AsyncQdrantClient

            self._qdrant = AsyncQdrantClient(url=self._qdrant_url)
            await self._ensure_collection()
            logger.info("NoteObserver connected to Qdrant (%s)", self._qdrant_url)
        except Exception as e:
            logger.warning("NoteObserver: Qdrant unavailable: %s", e)
            self._qdrant = None

        # ---- Discover embedding model -----------------------------------
        try:
            resp = await self._http_client.get(f"{self._embedding_url}/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                self._embedding_model = models[0].get("id")
                logger.info(
                    "NoteObserver embedding model: %s", self._embedding_model
                )
        except Exception as e:
            logger.warning("NoteObserver: embedding discovery failed: %s", e)

        self._initialized = True

    async def close(self) -> None:
        """Clean shutdown."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        self._qdrant = None

    async def _ensure_collection(self) -> None:
        """Create the cca_notes collection if it doesn't exist."""
        if self._qdrant is None:
            return
        try:
            from qdrant_client.http import models

            collections = (await self._qdrant.get_collections()).collections
            names = [c.name for c in collections]
            if NOTES_COLLECTION not in names:
                await self._qdrant.create_collection(
                    collection_name=NOTES_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMS,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", NOTES_COLLECTION)
        except Exception as e:
            logger.error("Failed to ensure %s collection: %s", NOTES_COLLECTION, e)

    # ------------------------------------------------------------------
    # Main entry point (called from app.py as background task)
    # ------------------------------------------------------------------

    async def process(
        self,
        messages: List[Any],
        session_id: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        project_context: Optional[str] = None,
    ) -> None:
        """Extract notes from a completed conversation trajectory.

        This is fire-and-forget — all exceptions are caught and logged.
        """
        tracer = get_tracer()

        with tracer.start_as_current_span("cca.note_observer") as span:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
            span.set_attribute("cca.note.session_id", session_id)
            span.set_attribute("cca.note.user_id", user_id or "anonymous")
            span.set_attribute("cca.note.message_count", len(messages))

            try:
                if not messages:
                    span.set_attribute("cca.note.status", "empty")
                    return

                # 1. Store raw trajectory to Redis
                with tracer.start_as_current_span("cca.note.store_trajectory") as redis_span:
                    redis_span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
                    await self._store_trajectory_to_redis(messages, session_id)
                    redis_span.set_attribute("cca.note.status", "stored")

                # 2. Extract notes via Spark1 Qwen3-8B
                with tracer.start_as_current_span("cca.note.extract_llm") as llm_span:
                    llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
                    llm_span.set_attribute(LLM_MODEL_NAME, self._llm_model)
                    llm_span.set_attribute("cca.note.llm_url", self._llm_url)
                    notes, llm_usage = await self._extract_notes(messages)
                    llm_span.set_attribute(
                        "cca.note.notes_extracted",
                        len(notes) if notes else 0,
                    )
                    if llm_usage:
                        llm_span.set_attribute(
                            LLM_TOKEN_COUNT_PROMPT,
                            llm_usage.get("prompt_tokens", 0),
                        )
                        llm_span.set_attribute(
                            LLM_TOKEN_COUNT_COMPLETION,
                            llm_usage.get("completion_tokens", 0),
                        )
                        llm_span.set_attribute(
                            OUTPUT_VALUE,
                            llm_usage.get("content", "")[:500],
                        )

                if not notes:
                    logger.debug(
                        "NoteObserver: no notes extracted for session %s", session_id
                    )
                    span.set_attribute("cca.note.status", "no_notes")
                    return

                # 3. Store notes to Qdrant
                with tracer.start_as_current_span("cca.note.store_qdrant") as qdrant_span:
                    qdrant_span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
                    qdrant_span.set_attribute("cca.note.notes_count", len(notes))
                    await self._store_notes_to_qdrant(
                        notes=notes,
                        session_id=session_id,
                        user_id=user_id,
                        user_name=user_name,
                        project_context=project_context,
                    )
                    qdrant_span.set_attribute("cca.note.status", "stored")

                span.set_attribute("cca.note.status", "success")
                span.set_attribute("cca.note.notes_stored", len(notes))
                logger.info(
                    "NoteObserver: stored %d notes for session %s (user=%s)",
                    len(notes),
                    session_id,
                    user_id or "anonymous",
                )

            except Exception as e:
                span.set_attribute("cca.note.status", "error")
                span.set_attribute("cca.note.error", str(e)[:200])
                logger.error("NoteObserver.process() failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Redis trajectory storage
    # ------------------------------------------------------------------

    async def _store_trajectory_to_redis(
        self, messages: List[Any], session_id: str
    ) -> None:
        """Serialize and store the full trajectory to Redis with TTL."""
        if self._redis is None:
            return

        try:
            serialized = []
            for msg in messages:
                entry: Dict[str, Any] = {}
                if hasattr(msg, "type"):
                    entry["type"] = (
                        msg.type.value if hasattr(msg.type, "value") else str(msg.type)
                    )
                if hasattr(msg, "content"):
                    # content may be a string or a list of dicts (tool use)
                    if isinstance(msg.content, str):
                        entry["content"] = msg.content
                    elif isinstance(msg.content, list):
                        entry["content"] = json.dumps(msg.content, default=str)
                    else:
                        entry["content"] = str(msg.content)
                serialized.append(entry)

            key = f"{TRAJECTORY_KEY_PREFIX}{session_id}"
            await self._redis.setex(
                key, TRAJECTORY_TTL, json.dumps(serialized, default=str)
            )
            logger.debug(
                "Stored trajectory (%d messages) to Redis: %s", len(serialized), key
            )
        except Exception as e:
            logger.warning("Failed to store trajectory to Redis: %s", e)

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _extract_notes(
        self, messages: List[Any]
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Call Spark1 Qwen3-8B to extract structured notes.

        Uses AsyncOpenAI client so the call is auto-traced by
        OpenAIInstrumentor (ChatCompletion span with model, tokens).

        Returns (notes, usage_info) where usage_info contains token
        counts and raw content for span enrichment.
        """
        if self._openai_client is None:
            return [], None

        # Build a condensed view of the conversation
        trajectory_text = self._format_trajectory(messages)
        if not trajectory_text.strip():
            return [], None

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Analyze this conversation and extract "
                            f"insights:\n\n{trajectory_text}"
                        ),
                    },
                ],
                temperature=self._temperature,
                max_tokens=1024,
            )

            # Extract the LLM's response text
            content = response.choices[0].message.content or ""

            # Build usage info for span enrichment
            usage_info: Dict[str, Any] = {"content": content}
            if response.usage:
                usage_info["prompt_tokens"] = response.usage.prompt_tokens
                usage_info["completion_tokens"] = (
                    response.usage.completion_tokens
                )

            # Strip thinking tags if present (Qwen3 may include them)
            content = self._strip_thinking(content)

            # Parse JSON from the response
            return self._parse_notes_json(content), usage_info

        except Exception as e:
            logger.warning("Note extraction LLM call failed: %s", e)
            return [], None

    def _format_trajectory(self, messages: List[Any]) -> str:
        """Format recent messages into a readable trajectory string."""
        # Take last N messages to stay within context window
        recent = messages[-self._max_messages :]
        lines: List[str] = []

        for msg in recent:
            msg_type = "unknown"
            if hasattr(msg, "type"):
                msg_type = (
                    msg.type.value if hasattr(msg.type, "value") else str(msg.type)
                )

            content = ""
            if hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # Tool use blocks — summarize
                    parts = []
                    for block in msg.content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                parts.append(
                                    f"[tool: {block.get('name', '?')}]"
                                )
                            elif block.get("type") == "tool_result":
                                result_content = block.get("content", "")
                                if len(str(result_content)) > 200:
                                    result_content = str(result_content)[:200] + "..."
                                parts.append(f"[result: {result_content}]")
                            elif block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            else:
                                parts.append(f"[{block.get('type', 'block')}]")
                        else:
                            parts.append(str(block))
                    content = " ".join(parts)
                else:
                    content = str(msg.content)

            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."

            if content.strip():
                lines.append(f"[{msg_type}] {content}")

        return "\n".join(lines)

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Remove <think>...</think> tags from Qwen3 output."""
        import re

        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    @staticmethod
    def _parse_notes_json(content: str) -> List[Dict[str, Any]]:
        """Parse JSON array from LLM output, handling common issues."""
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        if not content:
            return []

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # Validate each note has required fields
                valid = []
                for note in parsed:
                    if isinstance(note, dict) and "content" in note:
                        valid.append(
                            {
                                "content": str(note["content"]),
                                "type": str(
                                    note.get("type", "insight")
                                ),
                                "tags": list(note.get("tags", [])),
                            }
                        )
                return valid
            return []
        except json.JSONDecodeError:
            logger.debug("Failed to parse notes JSON: %s", content[:200])
            return []

    # ------------------------------------------------------------------
    # Qdrant note storage
    # ------------------------------------------------------------------

    async def _store_notes_to_qdrant(
        self,
        notes: List[Dict[str, Any]],
        session_id: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        project_context: Optional[str] = None,
    ) -> None:
        """Embed and upsert notes to the cca_notes collection."""
        if self._qdrant is None or self._http_client is None:
            return
        if not notes:
            return

        # Generate embeddings for all note contents
        contents = [n["content"] for n in notes]
        embeddings = await self._embed(contents)
        if not embeddings or len(embeddings) != len(notes):
            logger.warning(
                "Embedding count mismatch: %d notes, %d embeddings",
                len(notes),
                len(embeddings) if embeddings else 0,
            )
            return

        try:
            from qdrant_client.http.models import PointStruct

            now = datetime.now(timezone.utc).isoformat()
            points = []
            for note, embedding in zip(notes, embeddings):
                note_id = _stable_note_id(note["content"], session_id)
                point = PointStruct(
                    id=_point_id_from_str(note_id),
                    vector=embedding,
                    payload={
                        "note_id": note_id,
                        "session_id": session_id,
                        "user_id": user_id,
                        "user_name": user_name,
                        "timestamp": now,
                        "note_type": note.get("type", "insight"),
                        "content": note["content"],
                        "tags": note.get("tags", []),
                        "source": "per_request",
                        "project_context": project_context,
                    },
                )
                points.append(point)

            await self._qdrant.upsert(
                collection_name=NOTES_COLLECTION,
                points=points,
            )
        except Exception as e:
            logger.error("Failed to store notes to Qdrant: %s", e)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via the embedding server."""
        if not texts or not self._embedding_model or not self._http_client:
            return []

        try:
            resp = await self._http_client.post(
                f"{self._embedding_url}/v1/embeddings",
                json={"model": self._embedding_model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.warning("Embedding call failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Search (used by context enrichment and API endpoints)
    # ------------------------------------------------------------------

    async def search_notes(
        self,
        query: str,
        user_id: Optional[str] = None,
        n_results: int = 5,
        min_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Semantic search over extracted notes.

        Returns a list of note payloads sorted by relevance.
        """
        if self._qdrant is None:
            return []

        embeddings = await self._embed([query])
        if not embeddings:
            return []

        try:
            from qdrant_client.http.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            # Build optional filter
            q_filter = None
            if user_id:
                q_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                )

            response = await self._qdrant.query_points(
                collection_name=NOTES_COLLECTION,
                query=embeddings[0],
                limit=n_results,
                query_filter=q_filter,
                with_payload=True,
            )

            notes = []
            for hit in response.points:
                if hit.score >= min_score and hit.payload:
                    note = dict(hit.payload)
                    note["score"] = round(hit.score, 4)
                    notes.append(note)

            return notes

        except Exception as e:
            logger.error("Note search failed: %s", e)
            return []

    async def get_trajectory(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve a trajectory from Redis by session ID."""
        if self._redis is None:
            return None
        try:
            key = f"{TRAJECTORY_KEY_PREFIX}{session_id}"
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning("Failed to retrieve trajectory: %s", e)
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the notes collection."""
        stats: Dict[str, Any] = {
            "collection": NOTES_COLLECTION,
            "qdrant_connected": self._qdrant is not None,
            "redis_connected": self._redis is not None,
            "embedding_model": self._embedding_model,
        }

        if self._qdrant is not None:
            try:
                info = await self._qdrant.get_collection(NOTES_COLLECTION)
                stats["total_notes"] = info.points_count

                # Scroll all notes to compute breakdowns
                notes_by_type: Dict[str, int] = {}
                notes_by_user: Dict[str, int] = {}
                offset = None
                while True:
                    results, next_offset = await self._qdrant.scroll(
                        collection_name=NOTES_COLLECTION,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for pt in results:
                        if pt.payload:
                            nt = pt.payload.get("note_type", "unknown")
                            notes_by_type[nt] = notes_by_type.get(nt, 0) + 1
                            uid = pt.payload.get("user_id") or "anonymous"
                            notes_by_user[uid] = notes_by_user.get(uid, 0) + 1
                    if next_offset is None:
                        break
                    offset = next_offset

                stats["notes_by_type"] = notes_by_type
                stats["notes_by_user"] = notes_by_user
            except Exception as e:
                stats["error"] = str(e)

        if self._redis is not None:
            try:
                cursor = 0
                count = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=f"{TRAJECTORY_KEY_PREFIX}*",
                        count=100,
                    )
                    count += len(keys)
                    if cursor == 0:
                        break
                stats["recent_trajectories_in_redis"] = count
            except Exception as e:
                stats["redis_scan_error"] = str(e)

        return stats


def format_notes_for_prompt(notes: List[Dict[str, Any]]) -> str:
    """Format notes for injection into the system prompt."""
    if not notes:
        return ""

    lines = ["<past_insights>", "From your previous sessions:"]
    for note in notes:
        note_type = note.get("note_type", "insight")
        content = note.get("content", "")
        lines.append(f"- [{note_type}] {content}")
    lines.append("</past_insights>")
    return "\n".join(lines)
