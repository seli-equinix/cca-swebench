# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Concurrent Confucius session management for the HTTP server.

Each HTTP session maps to one Confucius instance with:
- Its own HttpIOInterface (fresh per request)
- Its own memory (persisted across requests in the same session)
- A per-session lock (prevents concurrent requests from conflicting)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ..lib.confucius import Confucius
from .io_adapter import HttpIOInterface

logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    """A single session in the pool."""

    cf: Confucius
    last_used: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_count: int = 0


class SessionPool:
    """Manages concurrent Confucius instances for HTTP sessions.

    Thread-safe via asyncio.Lock. Each session has its own per-session
    lock to serialize requests (the orchestrator is not concurrent-safe).
    """

    def __init__(
        self,
        max_sessions: int = 50,
        session_ttl: int = 3600,
        workspace: str = "/workspace",
    ) -> None:
        self._sessions: dict[str, SessionEntry] = {}
        self._pool_lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl
        self._workspace = workspace

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    async def acquire(self, session_id: str, io: HttpIOInterface) -> SessionEntry:
        """Get or create a session, inject fresh IO, and acquire its lock.

        The caller MUST call release() when done (use try/finally).
        """
        async with self._pool_lock:
            if session_id not in self._sessions:
                # Evict expired sessions if at capacity
                if len(self._sessions) >= self._max_sessions:
                    await self._evict_oldest()

                # Create new Confucius instance
                cf = Confucius(
                    session=session_id,
                    io=io,
                    user="http",
                )
                # Try to load persisted session state
                try:
                    await cf.load()
                    logger.info(f"Loaded existing session state for {session_id[:16]}")
                except Exception:
                    logger.debug(f"No existing state for session {session_id[:16]}, starting fresh")

                entry = SessionEntry(cf=cf)
                self._sessions[session_id] = entry
                logger.info(
                    f"Created new session {session_id[:16]} "
                    f"(pool size: {len(self._sessions)})"
                )
            else:
                entry = self._sessions[session_id]

        # Acquire per-session lock BEFORE mutating entry state.
        # This prevents two concurrent requests for the same session
        # from overwriting each other's IO adapter.
        await entry.lock.acquire()
        entry.cf.io = io
        entry.last_used = time.time()
        entry.request_count += 1
        return entry

    async def release(self, session_id: str) -> None:
        """Save session state and release the per-session lock."""
        async with self._pool_lock:
            entry = self._sessions.get(session_id)

        if entry is None:
            return

        try:
            # Persist session state (memory, storage, artifacts)
            await entry.cf.save(raise_exception=False)
        except Exception as e:
            logger.warning(f"Failed to save session {session_id[:16]}: {e}")
        finally:
            if entry.lock.locked():
                entry.lock.release()

    async def cleanup_expired(self) -> int:
        """Remove sessions that haven't been used within TTL.

        Returns the number of sessions evicted.
        """
        now = time.time()
        expired: list[str] = []

        async with self._pool_lock:
            for sid, entry in self._sessions.items():
                if (now - entry.last_used) > self._session_ttl:
                    expired.append(sid)

            for sid in expired:
                entry = self._sessions.pop(sid)
                try:
                    await entry.cf.save(raise_exception=False)
                except Exception as e:
                    logger.warning(f"Failed to save expired session {sid[:16]}: {e}")

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    async def _evict_oldest(self) -> None:
        """Evict the oldest session to make room for a new one."""
        if not self._sessions:
            return

        oldest_sid = min(self._sessions, key=lambda s: self._sessions[s].last_used)
        entry = self._sessions.pop(oldest_sid)

        try:
            await entry.cf.save(raise_exception=False)
        except Exception as e:
            logger.warning(f"Failed to save evicted session {oldest_sid[:16]}: {e}")

        logger.info(f"Evicted oldest session {oldest_sid[:16]} to make room")

    async def get_session_info(self) -> list[dict]:
        """Get info about all active sessions (for /health or debugging)."""
        now = time.time()
        info = []
        async with self._pool_lock:
            for sid, entry in self._sessions.items():
                info.append({
                    "session_id": sid[:16] + "...",
                    "user": entry.cf.user,
                    "request_count": entry.request_count,
                    "age_seconds": int(now - entry.last_used),
                    "locked": entry.lock.locked(),
                })
        return info
