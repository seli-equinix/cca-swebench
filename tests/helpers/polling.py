"""Polling helpers for async CCA subsystems.

NoteObserver is fire-and-forget async: 8-14s locally, 15-20s in CI
(semaphore contention + Docker network).  Fixed ``time.sleep()`` calls
in tests are flaky — polling avoids false negatives.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.cca_client import CCAClient


def wait_for_notes(
    cca: CCAClient,
    query: str,
    user_id: str | None = None,
    max_wait: int = 45,
    interval: int = 3,
) -> list[dict]:
    """Poll ``GET /v1/notes/search`` until results appear or timeout.

    Args:
        cca: CCAClient instance.
        query: Search query for notes.
        user_id: Optional user filter.
        max_wait: Total seconds to wait before giving up.
        interval: Seconds between polls.

    Returns:
        List of note dicts, or empty list on timeout.
    """
    for _ in range(max_wait // interval):
        time.sleep(interval)
        notes = cca.search_notes(query, user_id=user_id)
        if notes:
            return notes
    return []
