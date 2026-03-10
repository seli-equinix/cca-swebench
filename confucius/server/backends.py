# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Shared backend connections for CCA extensions.

Provides a singleton client layer for Qdrant, Memgraph, Redis, and the
embedding server.  Initialised once in app.py lifespan, then injected
into extensions via build_extensions_for_route().

All service URLs come from config.toml [services] section.
Env vars (QDRANT_URL, EMBEDDING_URL, etc.) override when set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from ..core.config import get_services_config

logger = logging.getLogger(__name__)

EMBEDDING_DIMS: int = 4096
EMBEDDING_BATCH_SIZE: int = 10


class BackendClients:
    """Shared backend connections for CCA code intelligence extensions.

    Follows the NoteObserver pattern: constructor takes URLs, initialize()
    connects.  Graceful degradation — if a backend is unreachable, its
    property returns None and extensions skip that functionality.
    """

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        embedding_url: str | None = None,
        memgraph_host: str | None = None,
        memgraph_port: int | None = None,
    ) -> None:
        svc = get_services_config()
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL") or svc.qdrant_url
        embedding_url = embedding_url or os.getenv("EMBEDDING_URL") or svc.embedding_url
        memgraph_host = memgraph_host or os.getenv("MEMGRAPH_HOST") or svc.memgraph_host
        memgraph_port = memgraph_port or int(os.getenv("MEMGRAPH_PORT", "0") or 0) or svc.memgraph_port
        self._qdrant_url = qdrant_url
        self._embedding_url = embedding_url.rstrip("/")
        self._memgraph_host = memgraph_host
        self._memgraph_port = memgraph_port

        # Backends (set during initialize())
        self._qdrant: Any = None  # qdrant_client.AsyncQdrantClient
        self._memgraph: Any = None  # neo4j.AsyncDriver
        self._redis: Any = None  # redis.asyncio.Redis (injected)
        self._http_client: Any = None  # httpx.AsyncClient (for embeddings)
        self._embedding_model: Optional[str] = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, redis: Any = None) -> None:
        """Connect to all backends.  Safe to call multiple times."""
        if self._initialized:
            return

        import httpx

        self._http_client = httpx.AsyncClient(timeout=120.0)
        self._redis = redis

        # ---- Qdrant ---------------------------------------------------
        try:
            from qdrant_client import AsyncQdrantClient

            self._qdrant = AsyncQdrantClient(url=self._qdrant_url)
            # Quick check: list collections
            await self._qdrant.get_collections()
            logger.info("BackendClients: Qdrant connected (%s)", self._qdrant_url)
        except Exception as e:
            logger.warning("BackendClients: Qdrant unavailable: %s", e)
            self._qdrant = None

        # ---- Memgraph (neo4j async driver) ----------------------------
        try:
            from neo4j import AsyncGraphDatabase

            uri = f"bolt://{self._memgraph_host}:{self._memgraph_port}"
            self._memgraph = AsyncGraphDatabase.driver(
                uri,
                auth=None,  # Memgraph has no auth configured
                max_connection_pool_size=10,
                connection_acquisition_timeout=30,
            )
            # Verify connectivity
            async with self._memgraph.session() as session:
                await session.run("RETURN 1")
            logger.info("BackendClients: Memgraph connected (%s)", uri)
        except Exception as e:
            logger.warning("BackendClients: Memgraph unavailable: %s", e)
            self._memgraph = None

        # ---- Discover embedding model ---------------------------------
        try:
            resp = await self._http_client.get(
                f"{self._embedding_url}/v1/models"
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                self._embedding_model = models[0].get("id")
                logger.info(
                    "BackendClients: embedding model: %s", self._embedding_model
                )
        except Exception as e:
            logger.warning("BackendClients: embedding discovery failed: %s", e)

        self._initialized = True

    async def close(self) -> None:
        """Clean shutdown of all connections."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        if self._memgraph is not None:
            await self._memgraph.close()
            self._memgraph = None

        if self._qdrant is not None:
            await self._qdrant.close()
            self._qdrant = None
        # Redis is shared with UserSessionManager — don't close it here

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qdrant(self) -> Any:
        """Async Qdrant client (or None if unavailable)."""
        return self._qdrant

    @property
    def memgraph(self) -> Any:
        """Neo4j async driver for Memgraph (or None if unavailable)."""
        return self._memgraph

    @property
    def redis(self) -> Any:
        """Redis async client (shared with UserSessionManager, or None)."""
        return self._redis

    @property
    def embedding_model(self) -> Optional[str]:
        """Discovered embedding model name."""
        return self._embedding_model

    @property
    def available(self) -> bool:
        """True if at least Qdrant and embedding are available."""
        return self._qdrant is not None and self._embedding_model is not None

    # ------------------------------------------------------------------
    # Health check + auto-reconnect
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, str]:
        """Ping each backend. Reconnect if down. Returns status dict."""
        status: dict[str, str] = {}

        # ---- Qdrant ----
        if self._qdrant is not None:
            try:
                await self._qdrant.get_collections()
                status["qdrant"] = "ok"
            except Exception:
                logger.warning(
                    "BackendClients: Qdrant connection lost — reconnecting"
                )
                try:
                    await self._qdrant.close()
                except Exception:
                    pass
                self._qdrant = None
                status["qdrant"] = "down"

        if self._qdrant is None and self._qdrant_url:
            try:
                from qdrant_client import AsyncQdrantClient

                client = AsyncQdrantClient(url=self._qdrant_url)
                await client.get_collections()
                self._qdrant = client
                status["qdrant"] = "reconnected"
                logger.info("BackendClients: Qdrant reconnected")
            except Exception:
                status.setdefault("qdrant", "down")

        # ---- Memgraph ----
        if self._memgraph is not None:
            try:
                async with self._memgraph.session() as session:
                    await session.run("RETURN 1")
                status["memgraph"] = "ok"
            except Exception:
                logger.warning(
                    "BackendClients: Memgraph connection lost — reconnecting"
                )
                try:
                    await self._memgraph.close()
                except Exception:
                    pass
                self._memgraph = None
                status["memgraph"] = "down"

        if self._memgraph is None and self._memgraph_host:
            try:
                from neo4j import AsyncGraphDatabase

                uri = f"bolt://{self._memgraph_host}:{self._memgraph_port}"
                driver = AsyncGraphDatabase.driver(
                    uri,
                    auth=None,
                    max_connection_pool_size=10,
                    connection_acquisition_timeout=30,
                )
                async with driver.session() as session:
                    await session.run("RETURN 1")
                self._memgraph = driver
                status["memgraph"] = "reconnected"
                logger.info("BackendClients: Memgraph reconnected")
            except Exception:
                status.setdefault("memgraph", "down")

        # ---- Embedding ----
        if self._http_client is not None:
            try:
                resp = await self._http_client.get(
                    f"{self._embedding_url}/v1/models", timeout=10.0
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    self._embedding_model = models[0].get("id")
                status["embedding"] = "ok"
            except Exception:
                self._embedding_model = None
                status["embedding"] = "down"

        return status

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via the embedding server.

        Processes in batches of EMBEDDING_BATCH_SIZE.  Returns a list of
        4096-dim float vectors, one per input text.

        Raises RuntimeError if the embedding server is unavailable.
        """
        if self._http_client is None or self._embedding_model is None:
            raise RuntimeError(
                "Embedding server not available — "
                "BackendClients not initialized or embedding model not discovered"
            )

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            resp = await self._http_client.post(
                f"{self._embedding_url}/v1/embeddings",
                json={
                    "model": self._embedding_model,
                    "input": batch,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order
            items = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in items)

        return all_embeddings
