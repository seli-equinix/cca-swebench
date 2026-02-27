# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""User session management and identification system.

Ported from MCP server session_manager.py. Provides:
- Smart auto-identification from messages ("I'm Sean", "my name is Sean")
- Confidence-based user linking (0.80 auto, 0.60 ask)
- Alias consolidation ("I am both seli and sean")
- User profiles persisted in Qdrant
- Session storage in Redis with TTL
- Graceful degradation (local fallback if infrastructure unavailable)

Shared infrastructure with MCP server on Spark1:
- Redis: 192.168.4.205:6379 (session storage)
- Qdrant: 192.168.4.205:6333 (user profiles)
- Embedding: 192.168.4.205:8200 (semantic matching)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default connection URLs (overridable via constructor or env vars)
# ---------------------------------------------------------------------------
DEFAULT_REDIS_URL: str = os.getenv(
    "REDIS_URL", "redis://:Loveme-sex64@192.168.4.205:6379/0"
)
DEFAULT_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://192.168.4.205:6333")
DEFAULT_EMBEDDING_URL: str = os.getenv("EMBEDDING_URL", "http://192.168.4.205:8200")

# Session / profile TTL defaults
SESSION_TTL: int = int(os.getenv("SESSION_TTL", "86400"))  # 24 hours
USER_PROFILE_TTL: int = int(os.getenv("USER_PROFILE_TTL", "2592000"))  # 30 days

# Qdrant collection names (shared with MCP server)
PROFILES_COLLECTION: str = "user_profiles"
CONTEXTS_COLLECTION: str = "user_contexts"

# Embedding dimensions (Qwen3-Embedding-8B)
EMBEDDING_DIMS: int = 4096

# Redis key prefixes (namespaced to avoid collision with MCP sessions)
SESSION_KEY_PREFIX: str = "cca:session:"


def _stable_point_id(user_id: str) -> int:
    """Deterministic Qdrant point ID from user_id.

    Uses SHA256 so the same user_id always maps to the same point ID,
    regardless of Python process (hash() is randomized per PYTHONHASHSEED).
    """
    return int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:8], "big")


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class Session:
    """Active CCA session.

    Sessions are ephemeral and link to permanent UserProfiles.
    Multiple sessions can belong to the same user (different clients).
    """

    session_id: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    identified: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    client_type: Optional[str] = None

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "identified": self.identified,
            "context": self.context,
            "client_type": self.client_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            message_count=data.get("message_count", 0),
            identified=data.get("identified", False),
            context=data.get("context", {}),
            client_type=data.get("client_type"),
        )


@dataclass
class UserProfile:
    """Permanent user profile stored in Qdrant.

    User-Centric Design:
    - User profiles are permanent and follow the user across clients
    - Sessions link TO users, not the other way around
    - Facts/preferences are stored on the USER, not the session
    - Session history tracks which sessions belonged to this user
    """

    user_id: str
    display_name: str
    email: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    session_count: int = 0
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    session_history: List[str] = field(default_factory=list)
    known_clients: List[str] = field(default_factory=list)

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        # Handle missing fields for backwards compatibility
        if "aliases" not in data:
            data["aliases"] = []
        if "session_history" not in data:
            data["session_history"] = []
        if "known_clients" not in data:
            data["known_clients"] = []
        if "facts" not in data:
            data["facts"] = {}
        if "preferences" not in data:
            data["preferences"] = {}
        if "skills" not in data:
            data["skills"] = []
        # Filter to only known fields to avoid TypeError on extra keys
        known_fields = {
            "user_id",
            "display_name",
            "email",
            "aliases",
            "facts",
            "preferences",
            "skills",
            "session_count",
            "last_seen",
            "created_at",
            "session_history",
            "known_clients",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ============================================================================
# Embedding helper (used for profile / context vector search)
# ============================================================================


class _ProfileEmbeddingFunc:
    """Embedding function that talks to an OpenAI-compatible /v1/embeddings endpoint.

    Features:
    - Dynamic model discovery from /v1/models on first use
    - Model name caching
    - Auto-retry with fresh model lookup on 404
    - Graceful degradation with clear logging
    """

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._model_name: Optional[str] = None
        self._model_checked: bool = False
        self._client: Any = None  # lazily created httpx.AsyncClient

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _discover_model(self) -> Optional[str]:
        """Dynamically discover the embedding model name from vLLM /v1/models."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id")
                logger.info("Discovered embedding model: %s", model_id)
                return model_id
            logger.warning("No models found in /v1/models response")
            return None
        except Exception as e:
            logger.warning("Failed to discover embedding model: %s", e)
            return None

    async def _get_model_name(self, force_refresh: bool = False) -> Optional[str]:
        if self._model_name is None or force_refresh:
            self._model_name = await self._discover_model()
            self._model_checked = True
        return self._model_name

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via /v1/embeddings (OpenAI-compatible)."""
        if not texts:
            return []

        model = await self._get_model_name()
        if not model:
            logger.warning("No embedding model available")
            return []

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.url}/v1/embeddings",
                json={"model": model, "input": texts},
            )

            # Handle model not found - try refreshing model name
            if response.status_code == 404:
                data = response.json()
                if "model" in str(data.get("error", {}).get("param", "")):
                    logger.info("Model not found, refreshing model name...")
                    model = await self._get_model_name(force_refresh=True)
                    if model:
                        response = await client.post(
                            f"{self.url}/v1/embeddings",
                            json={"model": model, "input": texts},
                        )

            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
        except Exception as e:
            logger.warning("Embedding call failed: %s", e)
            return []

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================================
# UserSessionManager
# ============================================================================


class UserSessionManager:
    """Manages CCA sessions with smart user identification.

    Architecture:
    - Sessions stored in Redis (fast, TTL-based)
    - User profiles stored in Qdrant (permanent, semantic search)
    - Graceful degradation: falls back to in-memory dicts when
      Redis or Qdrant are unavailable.

    Connection Parameters:
        redis_url:     Full Redis URL (default from REDIS_URL env / hardcoded)
        qdrant_url:    Qdrant REST URL (default from QDRANT_URL env / hardcoded)
        embedding_url: Embedding server URL (default from EMBEDDING_URL env / hardcoded)
    """

    # ------------------------------------------------------------------
    # Confidence thresholds for auto-identification
    # These are appropriate for Qwen3-Embedding-8B (4096 dims, cosine).
    # ------------------------------------------------------------------
    AUTO_LINK_THRESHOLD: float = 0.80  # Auto-link session to user
    ASK_THRESHOLD: float = 0.60  # LLM should ask via identify_user tool
    # Below ASK_THRESHOLD -> stay anonymous

    @staticmethod
    def detect_client_type(headers: Dict[str, str]) -> Optional[str]:
        """Detect client type from HTTP request headers."""
        user_agent = headers.get("user-agent", "").lower()
        if "curl" in user_agent:
            return "curl"
        elif "continue" in user_agent or headers.get("x-continue-client"):
            return "continue"
        elif "vscode" in user_agent or "vs code" in user_agent:
            return "vscode"
        elif "python" in user_agent or "httpx" in user_agent:
            return "python"
        elif any(b in user_agent for b in ("chrome", "firefox", "safari", "edge")):
            return "web"
        return None

    # ------------------------------------------------------------------
    # Name extraction regex patterns (25+)
    #
    # Each pattern captures a single group: the candidate user name.
    # Patterns are tried in order; first valid match wins.
    # ------------------------------------------------------------------
    NAME_PATTERNS: List[str] = [
        # 1. "Hi/Hey/Hello, I'm <name>"
        r"(?:^|\s)(?:hi|hey|hello|howdy|greetings)[,!.\s]+(?:i'?m|i am|this is|it'?s|my name is|call me)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 2. "I'm <name>" / "I am <name>" / "this is <name>" / "call me <name>"
        r"(?:^|\s)(?:i'?m|i am|this is|it'?s|my name is|call me)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 3. "<name> here" (e.g. "Sean here")
        r"(?:^|\s)([A-Za-z][A-Za-z0-9_-]{1,20})\s+here(?:\s|$|[.,!?])",
        # 4. Signature style: "-- <name>" (must be near end of message)
        r"(?:\u2014|--)\s*([A-Za-z][A-Za-z0-9_-]{1,20})\s*$",
        # 5. "they call me <name>"
        r"(?:^|\s)(?:they|people|everyone|folks)\s+call\s+me\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 6. "the name's <name>" / "name's <name>"
        r"(?:^|\s)(?:the\s+)?name'?s\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 7. "you can call me <name>"
        r"(?:^|\s)you\s+can\s+call\s+me\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 8. "I go by <name>"
        r"(?:^|\s)i\s+go\s+by\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 9. "it's me, <name>" / "its me <name>"
        r"(?:^|\s)it'?s\s+me[,\s]+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 10. "yo, <name> here"
        r"(?:^|\s)yo[,!.\s]+([A-Za-z][A-Za-z0-9_-]{1,20})\s+here(?:\s|$|[.,!?])",
        # 11. "sup, I'm <name>"
        r"(?:^|\s)(?:sup|what'?s\s+up|whaddup)[,!.\s]+(?:i'?m|i am)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 12. "my handle is <name>"
        r"(?:^|\s)my\s+(?:handle|nick|nickname|username|user\s*name)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 13. "known as <name>"
        r"(?:^|\s)(?:also\s+)?known\s+as\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 14. "name: <name>" (form-style)
        r"(?:^|\s)name\s*:\s*([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 15. "user: <name>" / "user <name>"
        r"(?:^|\s)user\s*:\s*([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 16. "<name> speaking" / "<name> talking"
        r"(?:^|\s)([A-Za-z][A-Za-z0-9_-]{1,20})\s+(?:speaking|talking|typing)(?:\s|$|[.,!?])",
        # 17. "just <name>" (short intro)
        r"^(?:just\s+)?([A-Za-z][A-Za-z0-9_-]{1,20})\s+checking\s+in(?:\s|$|[.,!?])",
        # 18. "hi there, <name> here"
        r"(?:hi|hey|hello)\s+there[,!.\s]+([A-Za-z][A-Za-z0-9_-]{1,20})\s+here(?:\s|$|[.,!?])",
        # 19. "<name> reporting in"
        r"(?:^|\s)([A-Za-z][A-Za-z0-9_-]{1,20})\s+reporting\s+in(?:\s|$|[.,!?])",
        # 20. "signed, <name>" / "regards, <name>"
        r"(?:signed|regards|cheers|thanks|sincerely|best)[,\s]+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])?$",
        # 21. "remember me? I'm <name>"
        r"remember\s+me\??\s+(?:i'?m|i am)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 22. "back again, <name> here"
        r"(?:back|i'?m\s+back)[,!.\s]+([A-Za-z][A-Za-z0-9_-]{1,20})\s+here(?:\s|$|[.,!?])",
        # 23. "FYI it's <name>"
        r"(?:fyi|btw|by the way)[,\s]+(?:it'?s|i'?m|i am)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 24. "<name> at your service"
        r"(?:^|\s)([A-Za-z][A-Za-z0-9_-]{1,20})\s+at\s+your\s+service(?:\s|$|[.,!?])",
        # 25. "hey, it's <name> again"
        r"(?:hey|hi|hello)[,!.\s]+it'?s\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s+again(?:\s|$|[.,!?])",
        # 26. "pleased to meet you, I'm <name>"
        r"pleased\s+to\s+meet\s+you[,!.\s]+(?:i'?m|i am|call me)\s+([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 27. "introduce myself - <name>"
        r"introduce\s+myself[,.\s:]+(?:i'?m\s+)?([A-Za-z][A-Za-z0-9_-]{1,20})(?:\s|$|[.,!?])",
        # 28. "Hey <name>." / "Hi <name>," (direct greeting with name)
        r"^(?:hey|hi|hello|yo)[,!.\s]+([A-Za-z][A-Za-z0-9_-]{1,20})[.,!]",
        # 29. "Hi <name> here" (already exists but let's add "Hi <name>, I" for follow-up)
        r"^(?:hey|hi|hello)[,!.\s]+([A-Za-z][A-Za-z0-9_-]{1,20})[,.\s]+(?:I|please|can|could|would)",
    ]

    # ------------------------------------------------------------------
    # Alias / consolidation regex patterns (9+)
    #
    # Each pattern captures TWO groups: name1 and name2.
    # ------------------------------------------------------------------
    ALIAS_PATTERNS: List[str] = [
        # 1. "I am both seli and sean" / "I'm both X and Y"
        r"(?:i'?m|i am)\s+both\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s+and\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 2. "seli is my computer name but I am sean"
        r"([A-Za-z][A-Za-z0-9_-]{1,20})\s+is\s+my\s+(?:computer|username|user|account|login|handle|nick|nickname)\s*(?:name)?\s*(?:,|but|and|however)?\s*(?:i'?m|i am|my\s+(?:real\s+)?name\s+is|call\s+me)?\s*([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 3. "my username is seli but my name is sean"
        r"my\s+(?:username|user|account|login|handle|nick|nickname)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s*(?:,|but|and|however)\s*my\s+(?:real\s+)?name\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 4. "I go by both seli and sean" / "people call me X or Y"
        r"(?:i\s+go\s+by|people\s+call\s+me|you\s+can\s+call\s+me)\s+(?:both\s+)?([A-Za-z][A-Za-z0-9_-]{1,20})\s+(?:and|or)\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 5. "seli and sean are both me"
        r"([A-Za-z][A-Za-z0-9_-]{1,20})\s+and\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s+are\s+(?:both\s+)?(?:me|the\s+same|the\s+same\s+person|one\s+person)",
        # 6. "I'm sean, also known as seli"
        r"(?:i'?m|i am|my\s+name\s+is)\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s*,?\s*(?:also\s+known\s+as|aka|a\.?k\.?a\.?|or)\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 7. "call me sean or seli"
        r"(?:call\s+me|use)\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s+or\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 8. "merge seli and sean" (explicit merge request)
        r"(?:merge|combine|consolidate|unify|link)\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s+(?:and|with|into)\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 9. "seli is also sean"
        r"([A-Za-z][A-Za-z0-9_-]{1,20})\s+is\s+also\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 10. "seli here, but you might know me as sean"
        r"([A-Za-z][A-Za-z0-9_-]{1,20})\s+here\s*,?\s*(?:but\s+)?(?:you\s+)?(?:might|may|could)?\s*(?:also\s+)?know\s+me\s+as\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
        # 11. "<name1> / <name2> - same person"
        r"([A-Za-z][A-Za-z0-9_-]{1,20})\s*/\s*([A-Za-z][A-Za-z0-9_-]{1,20})\s*[-:=]\s*same\s+person",
        # 12. "I'm <name1>, but [my] friends call me <name2>"
        r"(?:i'?m|i am|my\s+name\s+is)\s+([A-Za-z][A-Za-z0-9_-]{1,20})\s*,?\s*(?:but\s+)?(?:my\s+)?(?:friends|everyone|people|they)\s+call\s+me\s+([A-Za-z][A-Za-z0-9_-]{1,20})",
    ]

    # ------------------------------------------------------------------
    # Words that look like names but are NOT valid names.
    # ------------------------------------------------------------------
    NOT_NAMES: set[str] = {
        # greetings / filler
        "hi",
        "hello",
        "hey",
        "help",
        "thanks",
        "please",
        "yes",
        "no",
        "ok",
        "okay",
        "sure",
        "fine",
        "good",
        "great",
        "nice",
        "cool",
        "awesome",
        "wonderful",
        # common English words
        "the",
        "and",
        "but",
        "for",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "does",
        "did",
        "doing",
        "would",
        "could",
        "should",
        "might",
        "must",
        "shall",
        "will",
        "can",
        "may",
        "need",
        "just",
        "also",
        "very",
        "really",
        "actually",
        "probably",
        "maybe",
        "here",
        "there",
        "where",
        "when",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "such",
        "only",
        "same",
        "so",
        "than",
        "too",
        "now",
        "then",
        "once",
        # CS / tech terms
        "user",
        "admin",
        "root",
        "test",
        "guest",
        "anonymous",
        "unknown",
        "true",
        "false",
        "null",
        "none",
        "undefined",
        "nan",
        "code",
        "file",
        "function",
        "class",
        "method",
        "variable",
        "error",
        "python",
        "javascript",
        "java",
        "rust",
        "golang",
        "bash",
        "shell",
        # misc
        "back",
        "again",
        "new",
        "old",
        "not",
        "your",
        "with",
        "from",
        "about",
        "into",
        "over",
        # action verbs (commonly appear after em dashes)
        "remove",
        "delete",
        "update",
        "create",
        "add",
        "edit",
        "fix",
        "show",
        "list",
        "write",
        "read",
        "run",
        "stop",
        "start",
        "check",
        "get",
        "set",
        "find",
        "search",
        "install",
        "build",
        "deploy",
        "send",
        "make",
        "use",
        "try",
        "open",
        "close",
        "save",
        "load",
        "merge",
        "split",
        "sort",
        "copy",
        "move",
        "print",
        "help",
        "also",
        "forget",
        "give",
        "take",
        "tell",
        "know",
        "need",
        "want",
        "like",
        "look",
        "keep",
        "turn",
        "put",
        "pull",
        "push",
        "call",
        "ask",
        "work",
        "link",
        "view",
        # common adjectives/nouns extracted as names
        "key",
        "trying",
        "sure",
        "good",
        "best",
        "just",
        "well",
        "much",
        "even",
        "last",
        "next",
        "long",
        "great",
        "high",
        "right",
        "still",
        "down",
        "should",
        "would",
        "could",
        "might",
        "shall",
        "must",
        "will",
        "been",
        "being",
        "have",
        "having",
        "does",
        "doing",
        "done",
    }

    # ======================================================================
    # Construction / initialisation
    # ======================================================================

    def __init__(
        self,
        redis_url: str = DEFAULT_REDIS_URL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
    ) -> None:
        self._redis_url = redis_url
        self._qdrant_url = qdrant_url
        self._embedding_url = embedding_url

        # Backends (set during initialize())
        self._redis: Any = None  # redis.asyncio.Redis
        self._qdrant: Any = None  # qdrant_client.QdrantClient
        self._embedding_func: Optional[_ProfileEmbeddingFunc] = None

        # Local fallback stores
        self._local_sessions: Dict[str, Session] = {}
        self._local_profiles: Dict[str, UserProfile] = {}

        self._initialized: bool = False

    async def initialize(self) -> None:
        """Connect to Redis and Qdrant.  Safe to call multiple times."""
        if self._initialized:
            return

        # ---- Redis ----------------------------------------------------------
        try:
            import redis.asyncio as redis_async

            self._redis = redis_async.from_url(
                self._redis_url, decode_responses=True
            )
            await self._redis.ping()
            logger.info(
                "Session manager connected to Redis at %s", self._redis_url
            )
        except Exception as e:
            logger.warning(
                "Redis unavailable for sessions, using local storage: %s", e
            )
            self._redis = None

        # ---- Qdrant ---------------------------------------------------------
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.http import models

            self._qdrant = AsyncQdrantClient(url=self._qdrant_url)

            # Ensure user_profiles collection exists
            collections = (await self._qdrant.get_collections()).collections
            collection_names = [c.name for c in collections]

            if PROFILES_COLLECTION not in collection_names:
                await self._qdrant.create_collection(
                    collection_name=PROFILES_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMS,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created %s collection in Qdrant", PROFILES_COLLECTION
                )
            else:
                logger.info(
                    "User profiles collection exists in Qdrant"
                )

            # Initialise embedding function
            self._embedding_func = _ProfileEmbeddingFunc(self._embedding_url)
            logger.info(
                "Profile embedding function initialised (%s)",
                self._embedding_url,
            )
        except Exception as e:
            logger.warning(
                "Qdrant unavailable for profiles, using local storage: %s", e
            )
            self._qdrant = None

        self._initialized = True

    async def close(self) -> None:
        """Shut down connections cleanly."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._embedding_func is not None:
            await self._embedding_func.close()
            self._embedding_func = None
        if self._qdrant is not None:
            await self._qdrant.close()
            self._qdrant = None
        self._initialized = False

    # ======================================================================
    # Session CRUD
    # ======================================================================

    async def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> Session:
        """Get an existing session or create a new one.

        Args:
            session_id: Existing session ID, or ``None`` to create new.

        Returns:
            ``Session`` instance.
        """
        if session_id:
            session = await self._get_session(session_id)
            if session:
                return session

        new_id = session_id or str(uuid.uuid4())
        session = Session(session_id=new_id)
        await self.save_session(session)
        logger.info("Created new session: %s", new_id)
        return session

    async def _get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID (Redis first, then local fallback)."""
        if self._redis is not None:
            try:
                data = await self._redis.get(
                    f"{SESSION_KEY_PREFIX}{session_id}"
                )
                if data:
                    return Session.from_dict(json.loads(data))
            except Exception as e:
                logger.warning("Redis get session failed: %s", e)

        return self._local_sessions.get(session_id)

    async def save_session(self, session: Session) -> None:
        """Persist a session to Redis (with TTL) or local fallback."""
        session.last_activity = time.time()

        if self._redis is not None:
            try:
                await self._redis.setex(
                    f"{SESSION_KEY_PREFIX}{session.session_id}",
                    SESSION_TTL,
                    json.dumps(session.to_dict()),
                )
                return
            except Exception as e:
                logger.warning("Redis save session failed: %s", e)

        # Fallback
        self._local_sessions[session.session_id] = session

    async def delete_session(self, session_id: str) -> None:
        """Remove a session from all stores."""
        if self._redis is not None:
            try:
                await self._redis.delete(
                    f"{SESSION_KEY_PREFIX}{session_id}"
                )
            except Exception as e:
                logger.warning("Redis delete session failed: %s", e)

        self._local_sessions.pop(session_id, None)

    # ======================================================================
    # User-centric session linking
    # ======================================================================

    async def get_user_for_session(
        self, session: Session
    ) -> Optional[UserProfile]:
        """Return the ``UserProfile`` linked to *session*, or ``None``."""
        if not session.user_id:
            return None
        return await self._get_user_profile(session.user_id)

    async def link_session_to_user(
        self,
        session: Session,
        user: UserProfile,
        client_type: Optional[str] = None,
    ) -> Session:
        """Link a session to a user profile.

        - Session gets ``user_id`` and ``identified=True``
        - User gets session added to ``session_history``
        - Pending facts/preferences on session context are flushed to user
        """
        session.user_id = user.user_id
        session.identified = True
        if client_type:
            session.client_type = client_type

        # Update user profile
        user.last_seen = time.time()
        user.session_count += 1

        # Add session to history (keep last 50)
        if session.session_id not in user.session_history:
            user.session_history.append(session.session_id)
            if len(user.session_history) > 50:
                user.session_history = user.session_history[-50:]

        # Track client type
        if client_type and client_type not in user.known_clients:
            user.known_clients.append(client_type)

        # Flush pending facts from session context to user
        pending_facts: Dict[str, str] = session.context.pop(
            "pending_facts", {}
        )
        if pending_facts:
            user.facts.update(pending_facts)
            logger.info(
                "Applied %d pending facts to user %s",
                len(pending_facts),
                user.display_name,
            )

        # Flush pending preferences
        pending_prefs: Dict[str, Any] = session.context.pop(
            "pending_preferences", {}
        )
        if pending_prefs:
            user.preferences.update(pending_prefs)
            logger.info(
                "Applied %d pending preferences to user %s",
                len(pending_prefs),
                user.display_name,
            )

        await self.save_session(session)
        await self.save_user_profile(user)

        logger.info(
            "Linked session %s... to user %s (client: %s)",
            session.session_id[:8],
            user.display_name,
            client_type,
        )
        return session

    # ======================================================================
    # Smart auto-identification (main entry point)
    # ======================================================================

    async def smart_identify_on_first_message(
        self,
        message: str,
        session: Session,
        client_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform smart auto-identification on the first message of a session.

        Flow:
            0. Check for alias consolidation ("I am both seli and sean")
            1. Try to extract a name from the message
               a. If user exists with high confidence -> auto-link
               b. If user exists with medium confidence -> ``action="asking"``
               c. If user does NOT exist -> ``action="asking_new"``
            2. If no name extracted, try semantic inference from message content
            3. If nothing works -> stay anonymous

        Returns:
            Dict with keys:
            - ``identified``: bool
            - ``user``: ``UserProfile`` or None
            - ``action``: one of ``"already_identified"``, ``"auto_linked"``,
              ``"asking"``, ``"asking_new"``, ``"merged"``, ``"anonymous"``
            - ``confidence``: float (when applicable)
            - ``extracted_name``: str (when applicable)
            - ``message``: optional human-readable string
        """
        # Already identified?
        if session.identified:
            user = await self.get_user_for_session(session)
            return {
                "identified": True,
                "user": user,
                "action": "already_identified",
            }

        # ---- Step 0: alias consolidation -----------------------------------
        consolidation = self.detect_alias_consolidation(message)
        if consolidation:
            logger.info(
                "Smart ID: Detected alias consolidation: %s <-> %s",
                consolidation["name1"],
                consolidation["name2"],
            )
            result = await self._handle_alias_consolidation(
                consolidation, session, client_type
            )
            return {
                "identified": True,
                "user": result.get("user"),
                "action": result.get("action", "merged"),
                "message": result.get("message"),
                "consolidation": consolidation,
            }

        # ---- Step 1: extract name from message -----------------------------
        extracted_name = self.extract_name_from_message(message)

        if extracted_name:
            logger.info(
                "Smart ID: Extracted name '%s' from message", extracted_name
            )
            existing_user = await self.find_user_by_name(extracted_name)

            if existing_user:
                score = self._calculate_name_match_score(
                    extracted_name,
                    existing_user.display_name,
                    existing_user.aliases,
                )
                logger.info(
                    "Smart ID: Found existing user '%s' (score: %.2f)",
                    existing_user.display_name,
                    score,
                )

                if score >= self.AUTO_LINK_THRESHOLD:
                    session = await self.link_session_to_user(
                        session, existing_user, client_type
                    )
                    return {
                        "identified": True,
                        "user": existing_user,
                        "action": "auto_linked",
                        "confidence": score,
                        "extracted_name": extracted_name,
                        "message": f"Welcome back, {existing_user.display_name}!",
                    }
                elif score >= 0.30:
                    # Medium confidence - caller should ask for confirmation
                    session.context["potential_user_id"] = existing_user.user_id
                    session.context["potential_user_confidence"] = score
                    session.context["extracted_name"] = extracted_name
                    await self.save_session(session)

                    return {
                        "identified": False,
                        "user": None,
                        "action": "asking",
                        "confidence": score,
                        "potential_user": existing_user.display_name,
                        "extracted_name": extracted_name,
                    }
                else:
                    # Score too low — semantic search returned a false
                    # positive (e.g. "InferKnown_abc" vs "InferKnown_xyz").
                    # Treat as new user.
                    logger.info(
                        "Smart ID: Semantic match '%s' rejected "
                        "(name score: %.2f < 0.30) — treating as new user",
                        existing_user.display_name,
                        score,
                    )
                    existing_user = None  # fall through to new-user branch

            if not existing_user:
                # New user detected - caller should ask if they want to be remembered
                logger.info(
                    "Smart ID: New user detected - '%s'", extracted_name
                )
                session.context["pending_new_user_name"] = extracted_name
                session.context["pending_client_type"] = client_type
                await self.save_session(session)

                return {
                    "identified": False,
                    "user": None,
                    "action": "asking_new",
                    "extracted_name": extracted_name,
                    "message": (
                        f"I noticed you introduced yourself as {extracted_name}. "
                        "Would you like me to remember you?"
                    ),
                }

        # ---- Step 2: semantic inference (no name extracted) ----------------
        inference = await self._infer_user_from_message(message, session)

        if inference and inference.get("user"):
            user = inference["user"]
            confidence = inference["confidence"]
            action = inference["action"]

            if action == "auto_link":
                session = await self.link_session_to_user(
                    session, user, client_type
                )
                return {
                    "identified": True,
                    "user": user,
                    "action": "auto_linked",
                    "confidence": confidence,
                    "message": (
                        f"Welcome back, {user.display_name}! "
                        "I recognised you from your message."
                    ),
                }
            elif action == "ask":
                session.context["potential_user_id"] = user.user_id
                session.context["potential_user_confidence"] = confidence
                await self.save_session(session)

                return {
                    "identified": False,
                    "user": None,
                    "action": "asking",
                    "confidence": confidence,
                    "potential_user": user.display_name,
                }

        # ---- Step 3: nothing worked ----------------------------------------
        return {
            "identified": False,
            "user": None,
            "action": "anonymous",
        }

    # ======================================================================
    # Router-based identification
    # ======================================================================

    async def identify_from_router(
        self,
        name: str,
        facts: List[str],
        session: Session,
        client_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Identify user from router-extracted info (no LLM needed).

        Called when the Functionary router detects a user introduction
        alongside its routing classification. Uses the same Qdrant lookup
        as smart_identify_on_first_message but skips regex extraction
        (already done by the router).

        Args:
            name: User name extracted by the router.
            facts: Personal facts extracted by the router.
            session: Current session.
            client_type: Client type (e.g. "open-webui").

        Returns:
            Dict with keys: identified, user, action, confidence, extracted_name.
        """
        if session.identified:
            user = await self.get_user_for_session(session)
            return {
                "identified": True,
                "user": user,
                "action": "already_identified",
                "id_source": "router",
            }

        name = name.strip()
        if not name or not self._is_valid_name(name):
            return {
                "identified": False,
                "user": None,
                "action": "invalid_name",
                "extracted_name": name,
                "id_source": "router",
            }

        logger.info("Router ID: extracted name '%s', %d facts", name, len(facts))

        existing_user = await self.find_user_by_name(name)

        if existing_user:
            score = self._calculate_name_match_score(
                name, existing_user.display_name, existing_user.aliases
            )
            logger.info(
                "Router ID: found existing user '%s' (score: %.2f)",
                existing_user.display_name,
                score,
            )

            if score >= self.AUTO_LINK_THRESHOLD:
                session = await self.link_session_to_user(
                    session, existing_user, client_type
                )
                # Store any new facts extracted by the router
                if facts:
                    await self._store_facts_from_extraction(
                        existing_user, facts
                    )
                return {
                    "identified": True,
                    "user": existing_user,
                    "action": "auto_linked",
                    "confidence": score,
                    "extracted_name": name,
                    "id_source": "router",
                    "message": f"Welcome back, {existing_user.display_name}!",
                }
            elif score >= 0.30:
                # Medium confidence — store context and let caller decide
                session.context["potential_user_id"] = existing_user.user_id
                session.context["potential_user_confidence"] = score
                session.context["extracted_name"] = name
                await self.save_session(session)

                return {
                    "identified": False,
                    "user": None,
                    "action": "asking",
                    "confidence": score,
                    "potential_user": existing_user.display_name,
                    "extracted_name": name,
                    "id_source": "router",
                }
            else:
                # Score too low — false semantic match, treat as new user
                logger.info(
                    "Router ID: Semantic match '%s' rejected "
                    "(name score: %.2f < 0.30) — treating as new user",
                    existing_user.display_name,
                    score,
                )
                existing_user = None  # fall through to new-user branch

        if not existing_user:
            # New user — auto-create (same as app.py auto-create flow)
            logger.info("Router ID: new user detected — '%s'", name)
            create_result = await self.identify_user(
                name, session, client_type
            )
            if create_result.get("status") in (
                "new_user", "welcome_back", "identified",
            ):
                user = await self.get_user_for_session(session)
                if facts and user:
                    await self._store_facts_from_extraction(user, facts)
                return {
                    "identified": True,
                    "user": user,
                    "action": "router_identified",
                    "extracted_name": name,
                    "id_source": "router",
                }
            return {
                "identified": False,
                "user": None,
                "action": "anonymous",
                "extracted_name": name,
                "id_source": "router",
            }

    async def _store_facts_from_extraction(
        self,
        user: "UserProfile",
        facts: List[str],
    ) -> None:
        """Store facts extracted by the router as user facts.

        Parses natural language fact strings into key/value pairs.
        Examples:
            "I work at Equinix" → key="employer", value="Equinix"
            "I prefer Python"   → key="preference_language", value="Python"
            "my project uses Docker" → key="fact", value="project uses Docker"
        """
        # Simple heuristic parsing — extract key/value from common patterns
        FACT_PATTERNS = [
            (r"(?:I\s+)?work(?:s)?\s+(?:at|for)\s+(.+)", "employer"),
            (r"(?:I\s+)?prefer\s+(.+)", "preference"),
            (r"(?:I(?:'m|\s+am)\s+(?:a|an)\s+)(.+)", "role"),
            (r"(?:my\s+)?(?:main\s+)?project\s+(?:is|uses?)\s+(.+)", "project"),
            (r"(?:I\s+)?use\s+(.+)", "tool"),
        ]
        for fact_str in facts[:10]:  # cap at 10 to prevent abuse
            fact_str = fact_str.strip()
            if not fact_str:
                continue

            key = "fact"
            value = fact_str

            for pattern, fact_key in FACT_PATTERNS:
                match = re.match(pattern, fact_str, re.IGNORECASE)
                if match:
                    key = fact_key
                    value = match.group(1).strip()
                    break

            try:
                # Create a temporary session-like context for remember_user_fact
                temp_session = Session(
                    session_id=f"router-extract-{user.user_id}",
                    user_id=user.user_id,
                    identified=True,
                )
                await self.remember_user_fact(temp_session, key, value)
                logger.debug(
                    "Router ID: stored fact %s=%s for user %s",
                    key, value, user.user_id,
                )
            except Exception as e:
                logger.warning(
                    "Router ID: failed to store fact '%s' for user %s: %s",
                    fact_str, user.user_id, e,
                )

    # ======================================================================
    # Name extraction
    # ======================================================================

    def extract_name_from_message(self, message: str) -> Optional[str]:
        """Extract a user's name from their message.

        Handles patterns like:
        - "hi this is sean"
        - "I'm seli"
        - "my name is John"
        - "Sean here"
        - "-- Sean"

        Returns:
            Extracted name or ``None`` if no valid name found.
        """
        for pattern in self.NAME_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    logger.info(
                        "Extracted name '%s' from message using pattern",
                        name,
                    )
                    return name
        return None

    def _is_valid_name(self, name: str) -> bool:
        """Return ``True`` if *name* looks like a genuine user name."""
        if not name or len(name) < 2:
            return False
        if not name[0].isalpha():
            return False
        if name.lower() in self.NOT_NAMES:
            return False
        if name.isdigit():
            return False
        if len(name) > 20:
            return False
        return True

    # ======================================================================
    # Alias / consolidation detection
    # ======================================================================

    def detect_alias_consolidation(
        self, message: str
    ) -> Optional[Dict[str, Any]]:
        """Detect if a user is indicating they have multiple names/aliases.

        Handles patterns like:
        - "I am both seli and sean"
        - "seli is my computer name but I am sean"
        - "merge seli and sean"
        - "seli and sean are both me"

        Returns:
            Dict with ``name1``, ``name2``, ``primary``, ``alias``, and
            ``pattern_matched``; or ``None`` if no consolidation detected.
        """
        for pattern in self.ALIAS_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name1 = match.group(1).strip()
                name2 = match.group(2).strip()

                if not self._is_valid_name(name1) or not self._is_valid_name(
                    name2
                ):
                    continue
                if name1.lower() == name2.lower():
                    continue

                # Determine primary vs alias
                message_lower = message.lower()

                if (
                    "real name" in message_lower
                    or "my name is" in message_lower
                ):
                    primary, alias = name2, name1
                elif any(
                    w in message_lower
                    for w in [
                        "username",
                        "computer",
                        "account",
                        "login",
                        "handle",
                        "nick",
                    ]
                ):
                    primary, alias = name2, name1
                elif "merge" in message_lower or "combine" in message_lower:
                    primary, alias = name1, name2
                elif any(
                    w in message_lower
                    for w in [
                        "also known as",
                        " aka ",
                        "a.k.a",
                        "call me",
                        "friends call me",
                        "people call me",
                    ]
                ):
                    # "I'm X, also known as Y" → X is primary, Y is alias
                    primary, alias = name1, name2
                else:
                    primary, alias = name1, name2

                logger.info(
                    "Detected alias consolidation: '%s' and '%s' "
                    "(primary: %s, alias: %s)",
                    name1,
                    name2,
                    primary,
                    alias,
                )

                return {
                    "name1": name1,
                    "name2": name2,
                    "primary": primary,
                    "alias": alias,
                    "pattern_matched": pattern,
                }
        return None

    # ======================================================================
    # User lookup
    # ======================================================================

    async def find_user_by_name(self, name: str) -> Optional[UserProfile]:
        """Find a user by name -- exact match first, then semantic search.

        This prevents creating duplicate profiles for similar names.
        """
        clean_name = name.strip().lower()

        if self._qdrant is not None:
            try:
                from qdrant_client.http import models

                # Step 1: exact match (scroll all profiles, compare names)
                results = await self._qdrant.scroll(
                    collection_name=PROFILES_COLLECTION,
                    limit=100,
                    with_payload=True,
                )

                if results[0]:
                    for point in results[0]:
                        payload = point.payload
                        display = payload.get("display_name", "").lower()
                        aliases = [
                            a.lower() for a in payload.get("aliases", [])
                        ]
                        if clean_name == display or clean_name in aliases:
                            logger.info(
                                "Exact match found: %s",
                                payload.get("display_name"),
                            )
                            return UserProfile.from_dict(payload)

                # Step 2: semantic / fuzzy search via embedding
                if self._embedding_func is not None:
                    embeddings = await self._embedding_func([name])
                    if embeddings:
                        results = (await self._qdrant.query_points(
                            collection_name=PROFILES_COLLECTION,
                            query=embeddings[0],
                            limit=1,
                            score_threshold=0.85,
                        )).points
                        if results:
                            candidate = UserProfile.from_dict(
                                results[0].payload
                            )
                            name_score = self._calculate_name_match_score(
                                name,
                                candidate.display_name,
                                candidate.aliases,
                            )
                            logger.info(
                                "Semantic match found: %s (embed: %.3f, "
                                "name: %.2f)",
                                candidate.display_name,
                                results[0].score,
                                name_score,
                            )
                            if name_score >= 0.30:
                                return candidate
                            logger.info(
                                "Semantic match rejected — name score "
                                "%.2f < 0.30 (different user)",
                                name_score,
                            )

                logger.info("No matching user found for: %s", name)
            except Exception as e:
                logger.warning("User search failed: %s", e)

        # Fallback: local storage exact match
        for profile in self._local_profiles.values():
            if profile.display_name.lower() == clean_name:
                return profile
            if clean_name in [a.lower() for a in profile.aliases]:
                return profile

        return None

    # ======================================================================
    # Name match scoring
    # ======================================================================

    def _calculate_name_match_score(
        self,
        extracted_name: str,
        display_name: str,
        aliases: List[str],
    ) -> float:
        """Calculate how well *extracted_name* matches a user profile.

        Returns a score from 0.0 to 1.0.
        """
        extracted_lower = extracted_name.lower().strip()
        display_lower = display_name.lower().strip()

        # Exact match with display name
        if extracted_lower == display_lower:
            return 1.0

        # Exact match with any alias
        for alias in aliases:
            if extracted_lower == alias.lower().strip():
                return 1.0

        # Partial containment
        if extracted_lower in display_lower or display_lower in extracted_lower:
            return 0.85
        for alias in aliases:
            alias_lower = alias.lower().strip()
            if extracted_lower in alias_lower or alias_lower in extracted_lower:
                return 0.85

        # First-name match (first word of display name)
        display_first = display_lower.split()[0] if display_lower else ""
        if extracted_lower == display_first:
            return 0.90

        # Simple Levenshtein-like similarity for typos
        if abs(len(extracted_lower) - len(display_lower)) <= 2:
            common = sum(
                1 for a, b in zip(extracted_lower, display_lower) if a == b
            )
            max_len = max(len(extracted_lower), len(display_lower))
            if max_len > 0 and common / max_len >= 0.8:
                return 0.75

        return 0.0

    # ======================================================================
    # Profile merge
    # ======================================================================

    async def merge_user_profiles(
        self,
        primary_user: UserProfile,
        secondary_user: UserProfile,
        session: Optional[Session] = None,
    ) -> UserProfile:
        """Merge *secondary_user* into *primary_user*.

        The primary profile is kept and enriched; the secondary profile is
        deleted from Qdrant.
        """
        logger.info(
            "Merging user profiles: %s -> %s",
            secondary_user.display_name,
            primary_user.display_name,
        )

        # Merge aliases
        secondary_aliases = [secondary_user.display_name.lower()]
        secondary_aliases.extend(a.lower() for a in secondary_user.aliases)
        existing_lower = {a.lower() for a in primary_user.aliases}
        existing_lower.add(primary_user.display_name.lower())
        for alias in secondary_aliases:
            if alias not in existing_lower:
                primary_user.aliases.append(alias)
                existing_lower.add(alias)

        # Merge facts (prefix on conflict)
        for key, value in secondary_user.facts.items():
            if key not in primary_user.facts:
                primary_user.facts[key] = value
            elif primary_user.facts[key] != value:
                primary_user.facts[f"merged_{key}"] = value

        # Merge preferences (primary wins)
        for key, value in secondary_user.preferences.items():
            if key not in primary_user.preferences:
                primary_user.preferences[key] = value

        # Merge skills (union)
        existing_skills = set(primary_user.skills)
        for skill in secondary_user.skills:
            if skill not in existing_skills:
                primary_user.skills.append(skill)
                existing_skills.add(skill)

        # Merge known clients (union)
        existing_clients = set(primary_user.known_clients)
        for client in secondary_user.known_clients:
            if client not in existing_clients:
                primary_user.known_clients.append(client)
                existing_clients.add(client)

        # Merge session history (union)
        existing_sessions = set(primary_user.session_history)
        for sess_id in secondary_user.session_history:
            if sess_id not in existing_sessions:
                primary_user.session_history.append(sess_id)
                existing_sessions.add(sess_id)

        # Combine session counts
        primary_user.session_count += secondary_user.session_count

        # Use earliest created_at, latest last_seen
        if secondary_user.created_at < primary_user.created_at:
            primary_user.created_at = secondary_user.created_at
        if secondary_user.last_seen > primary_user.last_seen:
            primary_user.last_seen = secondary_user.last_seen

        # Save merged profile
        await self.save_user_profile(primary_user)

        # Delete secondary from Qdrant
        if self._qdrant is not None:
            try:
                from qdrant_client.http import models

                for coll in (PROFILES_COLLECTION, CONTEXTS_COLLECTION):
                    try:
                        await self._qdrant.delete(
                            collection_name=coll,
                            points_selector=models.FilterSelector(
                                filter=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="user_id",
                                            match=models.MatchValue(
                                                value=secondary_user.user_id
                                            ),
                                        )
                                    ]
                                )
                            ),
                        )
                    except Exception:
                        pass  # collection may not have this user
                logger.info(
                    "Deleted secondary profile: %s (%s)",
                    secondary_user.display_name,
                    secondary_user.user_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete secondary profile from Qdrant: %s", e
                )

        # Remove from local cache
        self._local_profiles.pop(secondary_user.user_id, None)

        # Link current session to merged profile
        if session is not None and not session.identified:
            session.user_id = primary_user.user_id
            session.identified = True
            await self.save_session(session)

        logger.info(
            "Profile merge complete: %s now has aliases: %s",
            primary_user.display_name,
            primary_user.aliases,
        )
        return primary_user

    # ======================================================================
    # User profile CRUD
    # ======================================================================

    async def save_user_profile(
        self,
        profile: UserProfile,
        update_context_embedding: bool = True,
    ) -> None:
        """Save a user profile to Qdrant (or local fallback).

        When *update_context_embedding* is ``True``, also refreshes the
        user's context embedding in the ``user_contexts`` collection for
        semantic auto-identification.
        """
        if self._qdrant is not None and self._embedding_func is not None:
            try:
                from qdrant_client.http import models

                # Embed profile for name-matching searches
                profile_text = (
                    f"{profile.display_name} "
                    f"{profile.email or ''} "
                    f"{' '.join(profile.skills)}"
                )
                embeddings = await self._embedding_func([profile_text])

                if embeddings:
                    point_id = _stable_point_id(profile.user_id)
                    await self._qdrant.upsert(
                        collection_name=PROFILES_COLLECTION,
                        points=[
                            models.PointStruct(
                                id=point_id,
                                vector=embeddings[0],
                                payload=profile.to_dict(),
                            )
                        ],
                    )
                    logger.info(
                        "Saved user profile to Qdrant: %s",
                        profile.display_name,
                    )

                    if update_context_embedding:
                        await self._update_user_context_embedding(profile)
                    return
            except Exception as e:
                logger.warning("Qdrant save profile failed: %s", e)

        # Fallback
        self._local_profiles[profile.user_id] = profile

    async def _get_user_profile(
        self, user_id: str
    ) -> Optional[UserProfile]:
        """Look up a user profile by ID."""
        if self._qdrant is not None:
            try:
                from qdrant_client.http import models

                results = await self._qdrant.scroll(
                    collection_name=PROFILES_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )
                if results[0]:
                    return UserProfile.from_dict(results[0][0].payload)
            except Exception as e:
                logger.warning("Qdrant get profile failed: %s", e)

        return self._local_profiles.get(user_id)

    async def get_all_users(self) -> List[UserProfile]:
        """Return all known user profiles (deduplicated by user_id)."""
        seen: Dict[str, UserProfile] = {}

        if self._qdrant is not None:
            try:
                results = await self._qdrant.scroll(
                    collection_name=PROFILES_COLLECTION,
                    limit=1000,
                    with_payload=True,
                )
                if results[0]:
                    for point in results[0]:
                        profile = UserProfile.from_dict(point.payload)
                        seen[profile.user_id] = profile
            except Exception as e:
                logger.warning("Failed to get users from Qdrant: %s", e)

        # Include local profiles not already present
        for profile in self._local_profiles.values():
            if profile.user_id not in seen:
                seen[profile.user_id] = profile

        return list(seen.values())

    # ======================================================================
    # User facts & preferences (session-aware)
    # ======================================================================

    async def remember_user_fact(
        self, session: Session, fact_key: str, fact_value: str
    ) -> bool:
        """Store a fact about the user linked to *session*.

        If the session is not yet identified, the fact is stashed in
        ``session.context["pending_facts"]`` and will be flushed to the
        user profile once identification occurs.

        Returns ``True`` if the fact was persisted (or stashed) successfully.
        """
        if session.identified and session.user_id:
            profile = await self._get_user_profile(session.user_id)
            if profile:
                profile.facts[fact_key] = fact_value
                profile.last_seen = time.time()
                await self.save_user_profile(profile)
                logger.info(
                    "Remembered fact '%s' for user %s",
                    fact_key,
                    profile.display_name,
                )
                return True

        # Stash on session for later
        pending = session.context.setdefault("pending_facts", {})
        pending[fact_key] = fact_value
        await self.save_session(session)
        logger.info(
            "Stashed pending fact '%s' on session %s",
            fact_key,
            session.session_id[:8],
        )
        return True

    async def update_user_preference(
        self, session: Session, pref_key: str, pref_value: Any
    ) -> bool:
        """Store a preference for the user linked to *session*.

        Follows the same stash-then-flush pattern as ``remember_user_fact``.
        """
        if session.identified and session.user_id:
            profile = await self._get_user_profile(session.user_id)
            if profile:
                profile.preferences[pref_key] = pref_value
                profile.last_seen = time.time()
                await self.save_user_profile(profile)
                logger.info(
                    "Updated preference '%s' for user %s",
                    pref_key,
                    profile.display_name,
                )
                return True

        # Stash
        pending = session.context.setdefault("pending_preferences", {})
        pending[pref_key] = pref_value
        await self.save_session(session)
        logger.info(
            "Stashed pending preference '%s' on session %s",
            pref_key,
            session.session_id[:8],
        )
        return True

    # ======================================================================
    # Context-building helpers
    # ======================================================================

    def build_user_context_string(self, user: UserProfile) -> str:
        """Build a searchable text blob from a user profile.

        This is embedded and stored in ``user_contexts`` so that incoming
        messages can be compared semantically to known users.
        """
        parts: List[str] = []
        parts.append(f"User: {user.display_name}")
        if user.aliases:
            parts.append(f"Also known as: {', '.join(user.aliases)}")
        if user.facts:
            facts_str = "; ".join(
                f"{k}: {v}" for k, v in user.facts.items()
            )
            parts.append(f"Facts: {facts_str}")
        if user.preferences:
            prefs_str = "; ".join(
                f"{k}: {v}" for k, v in user.preferences.items()
            )
            parts.append(f"Preferences: {prefs_str}")
        if user.skills:
            parts.append(f"Skills: {', '.join(user.skills)}")
        if user.known_clients:
            parts.append(f"Uses: {', '.join(user.known_clients)}")
        return "\n".join(parts)

    async def build_session_context(
        self, session: Session
    ) -> Dict[str, Any]:
        """Build an LLM-injectable context dict for a session.

        Returns dict with:
        - ``is_identified``: bool
        - ``display_name``, ``preferences``, ``facts``, etc. if identified
        """
        context: Dict[str, Any] = {
            "is_identified": session.identified,
            "session_id": session.session_id,
            "session_messages": session.message_count,
            "session_started": datetime.fromtimestamp(
                session.created_at
            ).isoformat(),
        }

        if session.user_id:
            profile = await self._get_user_profile(session.user_id)
            if profile:
                context.update(
                    {
                        "display_name": profile.display_name,
                        "email": profile.email,
                        "preferences": profile.preferences,
                        "facts": profile.facts,
                        "skills": profile.skills,
                        "total_sessions": profile.session_count,
                        "first_seen": datetime.fromtimestamp(
                            profile.created_at
                        ).isoformat(),
                        "last_seen": datetime.fromtimestamp(
                            profile.last_seen
                        ).isoformat(),
                    }
                )

        return context

    # ======================================================================
    # Stats
    # ======================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the session manager."""
        stats: Dict[str, Any] = {
            "redis_connected": self._redis is not None,
            "qdrant_connected": self._qdrant is not None,
            "local_sessions": len(self._local_sessions),
            "local_profiles": len(self._local_profiles),
        }

        if self._redis is not None:
            try:
                cursor = 0
                session_count = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match=f"{SESSION_KEY_PREFIX}*", count=100
                    )
                    session_count += len(keys)
                    if cursor == 0:
                        break
                stats["redis_sessions"] = session_count
            except Exception as e:
                stats["redis_error"] = str(e)

        if self._qdrant is not None:
            try:
                info = await self._qdrant.get_collection(PROFILES_COLLECTION)
                stats["qdrant_profiles"] = info.points_count
            except Exception as e:
                stats["qdrant_error"] = str(e)

        return stats

    # ======================================================================
    # Internal helpers
    # ======================================================================

    async def _update_user_context_embedding(
        self, user: UserProfile
    ) -> bool:
        """Update / upsert the user's context embedding in Qdrant.

        The ``user_contexts`` collection stores one vector per user
        representing everything we know about them.  Incoming messages are
        compared against these vectors for semantic auto-identification.
        """
        if self._qdrant is None or self._embedding_func is None:
            return False

        try:
            from qdrant_client.http import models

            context_str = self.build_user_context_string(user)
            embeddings = await self._embedding_func([context_str])
            if not embeddings:
                logger.warning(
                    "Failed to embed context for user %s",
                    user.display_name,
                )
                return False

            embedding = embeddings[0]

            # Ensure collection exists
            collections = (await self._qdrant.get_collections()).collections
            if CONTEXTS_COLLECTION not in [c.name for c in collections]:
                await self._qdrant.create_collection(
                    collection_name=CONTEXTS_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMS,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created %s collection in Qdrant",
                    CONTEXTS_COLLECTION,
                )

            point_id = _stable_point_id(user.user_id)
            await self._qdrant.upsert(
                collection_name=CONTEXTS_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "user_id": user.user_id,
                            "display_name": user.display_name,
                            "context_string": context_str,
                            "updated_at": time.time(),
                        },
                    )
                ],
            )
            logger.info(
                "Updated context embedding for user %s (%d dims)",
                user.display_name,
                len(embedding),
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to update user context embedding: %s", e
            )
            return False

    async def _infer_user_from_message(
        self, message: str, session: Session
    ) -> Optional[Dict[str, Any]]:
        """Infer user identity by embedding the message and comparing
        against all user context vectors.

        Returns:
            Dict with ``user``, ``confidence``, ``action``
            (``"auto_link"`` | ``"ask"`` | ``"anonymous"``), or ``None``.
        """
        if session.identified:
            return None
        if self._qdrant is None or self._embedding_func is None:
            return None

        try:
            from qdrant_client.http import models

            # Check collection exists
            collections = (await self._qdrant.get_collections()).collections
            if CONTEXTS_COLLECTION not in [c.name for c in collections]:
                return None

            embeddings = await self._embedding_func([message])
            if not embeddings:
                return None

            msg_embedding = embeddings[0]

            # Search with threshold
            results = (await self._qdrant.query_points(
                collection_name=CONTEXTS_COLLECTION,
                query=msg_embedding,
                limit=3,
                with_payload=True,
                score_threshold=self.ASK_THRESHOLD,
            )).points

            logger.info(
                "User inference query: %d matches above threshold %.2f",
                len(results),
                self.ASK_THRESHOLD,
            )

            if not results:
                # Peek without threshold for debug logging
                all_results = (await self._qdrant.query_points(
                    collection_name=CONTEXTS_COLLECTION,
                    query=msg_embedding,
                    limit=3,
                    with_payload=True,
                )).points
                if all_results:
                    logger.info(
                        "User inference: Best match score %.3f "
                        "(below threshold)",
                        all_results[0].score,
                    )
                return {
                    "user": None,
                    "confidence": 0.0,
                    "action": "anonymous",
                }

            best = results[0]
            confidence = best.score
            user_id = best.payload.get("user_id")
            display_name = best.payload.get("display_name")

            logger.info(
                "User inference: %s (confidence: %.2f)",
                display_name,
                confidence,
            )

            user = await self._get_user_profile(user_id)
            if not user:
                return {
                    "user": None,
                    "confidence": 0.0,
                    "action": "anonymous",
                }

            if confidence >= self.AUTO_LINK_THRESHOLD:
                action = "auto_link"
            elif confidence >= self.ASK_THRESHOLD:
                action = "ask"
            else:
                action = "anonymous"

            return {
                "user": user,
                "confidence": confidence,
                "action": action,
            }
        except Exception as e:
            logger.error("User inference failed: %s", e)
            return None

    async def _handle_alias_consolidation(
        self,
        consolidation: Dict[str, Any],
        session: Session,
        client_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a detected alias-consolidation request.

        Called when ``detect_alias_consolidation()`` finds a pattern
        indicating two names should be merged.
        """
        primary_name: str = consolidation["primary"]
        alias_name: str = consolidation["alias"]

        primary_user = await self.find_user_by_name(primary_name)
        alias_user = await self.find_user_by_name(alias_name)

        if primary_user and alias_user:
            if primary_user.user_id == alias_user.user_id:
                # Already the same user
                await self.link_session_to_user(
                    session, primary_user, client_type
                )
                return {
                    "action": "already_same",
                    "user": primary_user,
                    "message": (
                        f"Got it! {primary_name} and {alias_name} are "
                        "already linked to the same profile."
                    ),
                }
            # Merge
            merged = await self.merge_user_profiles(
                primary_user, alias_user, session
            )
            return {
                "action": "merged",
                "user": merged,
                "message": (
                    f"I've merged the profiles! You're now "
                    f"{merged.display_name}, and I'll recognise you "
                    "by either name."
                ),
            }

        elif primary_user:
            if alias_name.lower() not in [
                a.lower() for a in primary_user.aliases
            ]:
                primary_user.aliases.append(alias_name.lower())
                await self.save_user_profile(primary_user)
            await self.link_session_to_user(
                session, primary_user, client_type
            )
            return {
                "action": "alias_added",
                "user": primary_user,
                "message": (
                    f"Got it! I've added '{alias_name}' as an alias. "
                    f"I'll recognise you as {primary_user.display_name} "
                    "either way."
                ),
            }

        elif alias_user:
            old_name = alias_user.display_name
            alias_user.display_name = primary_name.capitalize()
            if alias_name.lower() not in [
                a.lower() for a in alias_user.aliases
            ]:
                alias_user.aliases.append(alias_name.lower())
            if old_name.lower() not in [
                a.lower() for a in alias_user.aliases
            ]:
                alias_user.aliases.append(old_name.lower())
            await self.save_user_profile(alias_user)
            await self.link_session_to_user(
                session, alias_user, client_type
            )
            return {
                "action": "renamed",
                "user": alias_user,
                "message": (
                    f"Got it! I've updated your profile. You're now "
                    f"{alias_user.display_name}, and I'll still "
                    f"recognise '{alias_name}'."
                ),
            }

        else:
            # Neither exists -- create new
            profile = UserProfile(
                user_id=str(uuid.uuid4()),
                display_name=primary_name.capitalize(),
                aliases=[alias_name.lower(), primary_name.lower()],
                known_clients=[client_type] if client_type else [],
                session_count=1,
            )
            await self.save_user_profile(profile)
            await self.link_session_to_user(session, profile, client_type)
            return {
                "action": "created",
                "user": profile,
                "message": (
                    f"Nice to meet you, {profile.display_name}! "
                    "I'll remember you by either name."
                ),
            }

    # ======================================================================
    # High-level user operations (consolidated from tools_extension)
    # ======================================================================

    async def identify_user(
        self,
        name: str,
        session: Session,
        client_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Identify user by name — alias check, find-or-create, link session.

        Consolidates logic previously in tools_extension._handle_identify_user.
        Returns a dict suitable for JSON serialisation back to the LLM.
        """
        if not name:
            return {"error": "name is required"}

        # Check alias consolidation first
        consolidation = self.detect_alias_consolidation(name)
        if consolidation:
            result = await self._handle_alias_consolidation(
                consolidation, session, client_type
            )
            user = result.get("user")
            if user:
                return {
                    "status": "identified",
                    "display_name": user.display_name,
                    "action": result.get("action"),
                    "session_count": user.session_count,
                }
            return {
                "status": "consolidation",
                "message": result.get(
                    "message", "Alias consolidation completed"
                ),
            }

        # Try to find existing user
        existing = await self.find_user_by_name(name)
        if existing:
            await self.link_session_to_user(session, existing, client_type)
            return {
                "status": "welcome_back",
                "display_name": existing.display_name,
                "session_count": existing.session_count,
                "facts": existing.facts,
                "preferences": existing.preferences,
            }

        # Create new profile
        profile = UserProfile(
            user_id=str(uuid.uuid4()),
            display_name=name.capitalize(),
            aliases=[name.lower()],
            known_clients=[client_type] if client_type else [],
            session_count=1,
        )
        await self.save_user_profile(profile)
        await self.link_session_to_user(session, profile, client_type)
        return {
            "status": "new_user",
            "display_name": profile.display_name,
            "user_id": profile.user_id,
        }

    async def infer_and_link_user(
        self, message: str, session: Session
    ) -> Dict[str, Any]:
        """Semantic user matching + auto-link if high confidence.

        Consolidates logic previously in tools_extension._handle_infer_user.
        """
        if not message:
            return {"error": "message is required"}

        result = await self._infer_user_from_message(message, session)
        if result and result.get("user"):
            user = result["user"]
            confidence = result["confidence"]
            action = result["action"]

            if action == "auto_link":
                await self.link_session_to_user(session, user)
                return {
                    "status": "auto_linked",
                    "display_name": user.display_name,
                    "confidence": confidence,
                }
            elif action == "ask":
                return {
                    "status": "possible_match",
                    "display_name": user.display_name,
                    "confidence": confidence,
                }

        return {"status": "no_match"}

    async def get_session_context(
        self,
        session: Session,
        critical_facts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build comprehensive session + user context.

        Queries Qdrant for authoritative profile (not just local cache).
        Accepts critical facts from Redis for infrastructure context.
        """
        result: Dict[str, Any] = {
            "session": {
                "session_id": session.session_id[:16] + "...",
                "identified": session.identified,
                "message_count": session.message_count,
                "client_type": getattr(session, "client_type", None),
            }
        }

        if session.identified and session.user_id:
            # Pull from Qdrant (authoritative) → falls back to local cache
            profile = await self._get_user_profile(session.user_id)
            if profile:
                result["user"] = {
                    "display_name": profile.display_name,
                    "user_id": profile.user_id,
                    "session_count": profile.session_count,
                    "facts": profile.facts or {},
                    "preferences": profile.preferences or {},
                    "skills": profile.skills or [],
                    "aliases": profile.aliases or [],
                    "known_clients": profile.known_clients or [],
                    "last_seen": profile.last_seen,
                }
        else:
            result["user"] = None
            result["hint"] = (
                "User not yet identified. "
                "Use identify_user or infer_user."
            )

        # Include critical facts if provided
        if critical_facts:
            result["critical_facts"] = critical_facts

        return result

    # ======================================================================
    # Profile CRUD for manage_user_profile tool
    # ======================================================================

    async def view_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Return full profile as dict."""
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        return {
            "user_id": profile.user_id,
            "display_name": profile.display_name,
            "email": profile.email,
            "aliases": profile.aliases,
            "facts": profile.facts,
            "preferences": profile.preferences,
            "skills": profile.skills,
            "session_count": profile.session_count,
            "known_clients": profile.known_clients,
            "created_at": profile.created_at,
            "last_seen": profile.last_seen,
            "session_history_count": len(profile.session_history),
        }

    async def update_user_facts_batch(
        self, user_id: str, data: Dict[str, str]
    ) -> Dict[str, Any]:
        """Batch update multiple facts."""
        if not data:
            return {"error": "data is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        profile.facts.update(data)
        profile.last_seen = time.time()
        await self.save_user_profile(profile)
        return {"updated": data, "all_facts": profile.facts}

    async def remove_user_fact(
        self, user_id: str, key: str
    ) -> Dict[str, Any]:
        """Remove a single fact by key."""
        if not key:
            return {"error": "key is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        removed = profile.facts.pop(key, None)
        if removed is None:
            return {
                "error": f"Fact '{key}' not found",
                "available_keys": list(profile.facts.keys()),
            }
        await self.save_user_profile(profile)
        return {
            "removed": key,
            "was": removed,
            "remaining_facts": profile.facts,
        }

    async def update_user_preferences_batch(
        self, user_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Batch update multiple preferences."""
        if not data:
            return {"error": "data is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        profile.preferences.update(data)
        profile.last_seen = time.time()
        await self.save_user_profile(profile)
        return {"updated": data, "all_preferences": profile.preferences}

    async def remove_user_preference(
        self, user_id: str, key: str
    ) -> Dict[str, Any]:
        """Remove a single preference by key."""
        if not key:
            return {"error": "key is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        removed = profile.preferences.pop(key, None)
        if removed is None:
            return {
                "error": f"Preference '{key}' not found",
                "available_keys": list(profile.preferences.keys()),
            }
        await self.save_user_profile(profile)
        return {
            "removed": key,
            "was": removed,
            "remaining_preferences": profile.preferences,
        }

    async def add_user_skill(
        self, user_id: str, value: str
    ) -> Dict[str, Any]:
        """Add a skill to the user's profile."""
        if not value:
            return {"error": "value is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        if value not in profile.skills:
            profile.skills.append(value)
            await self.save_user_profile(profile)
        return {"skills": profile.skills}

    async def remove_user_skill(
        self, user_id: str, value: str
    ) -> Dict[str, Any]:
        """Remove a skill from the user's profile."""
        if not value:
            return {"error": "value is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        if value in profile.skills:
            profile.skills.remove(value)
            await self.save_user_profile(profile)
            return {"removed": value, "skills": profile.skills}
        return {
            "error": f"Skill '{value}' not found",
            "available": profile.skills,
        }

    async def add_user_alias(
        self, user_id: str, value: str
    ) -> Dict[str, Any]:
        """Add alias — re-embeds the context vector."""
        if not value:
            return {"error": "value is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        if value.lower() not in [a.lower() for a in profile.aliases]:
            profile.aliases.append(value.lower())
            await self.save_user_profile(
                profile, update_context_embedding=True
            )
        return {"aliases": profile.aliases}

    async def remove_user_alias(
        self, user_id: str, value: str
    ) -> Dict[str, Any]:
        """Remove alias — re-embeds the context vector."""
        if not value:
            return {"error": "value is required"}
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
        lower_aliases = [a.lower() for a in profile.aliases]
        if value.lower() in lower_aliases:
            idx = lower_aliases.index(value.lower())
            profile.aliases.pop(idx)
            await self.save_user_profile(
                profile, update_context_embedding=True
            )
            return {"removed": value, "aliases": profile.aliases}
        return {
            "error": f"Alias '{value}' not found",
            "available": profile.aliases,
        }

    async def delete_user_profile(
        self, user_id: str, confirm_delete: bool = False
    ) -> Dict[str, Any]:
        """Delete user from Qdrant + local cache.

        Requires confirm_delete=True to actually delete.
        """
        profile = await self._get_user_profile(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}

        if not confirm_delete:
            return {
                "error": (
                    "Set confirm_delete=true to permanently "
                    "delete this profile"
                ),
                "user": profile.display_name,
                "data_to_lose": {
                    "facts": len(profile.facts),
                    "preferences": len(profile.preferences),
                    "skills": len(profile.skills),
                    "sessions": profile.session_count,
                },
            }

        # Delete from Qdrant
        if self._qdrant is not None:
            from qdrant_client.http import models

            for coll in (PROFILES_COLLECTION, CONTEXTS_COLLECTION, "cca_notes"):
                try:
                    await self._qdrant.delete(
                        collection_name=coll,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="user_id",
                                        match=models.MatchValue(
                                            value=user_id
                                        ),
                                    )
                                ]
                            )
                        ),
                    )
                except Exception:
                    pass

        self._local_profiles.pop(user_id, None)
        logger.info(
            "Deleted user profile: %s (%s)",
            user_id,
            profile.display_name,
        )
        return {
            "deleted": True,
            "user_id": user_id,
            "display_name": profile.display_name,
        }

    async def list_all_user_summaries(self) -> Dict[str, Any]:
        """List all user profiles (summary view)."""
        users = await self.get_all_users()
        return {
            "count": len(users),
            "users": [
                {
                    "user_id": u.user_id,
                    "display_name": u.display_name,
                    "session_count": u.session_count,
                    "aliases": u.aliases,
                    "skills": u.skills,
                    "last_seen": u.last_seen,
                }
                for u in users
            ],
        }
