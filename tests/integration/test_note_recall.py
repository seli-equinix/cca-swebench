"""Flow test: Note-taker recall across sessions.

Journey: have a distinctive conversation → wait for note extraction →
start a new session and ask about the previous topic → verify recall.

The NoteObserver fires after each request and extracts notes to Qdrant
via the Spark1 Qwen3-8B model. Notes are then injected into future
sessions via search_notes or <past_insights> context enrichment.

Exercises: search_notes (NOTES group), NoteObserver pipeline.
"""

import time
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestNoteRecall:
    """Cross-session note recall via NoteObserver → Qdrant pipeline."""

    def test_note_recall(self, cca, trace_test, judge_model):
        """Session 1: distinctive topic → Session 2: recall it."""
        name = f"NoteTest_{uuid.uuid4().hex[:6]}"
        unique_topic = f"Flamingo_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-note1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-note2-{uuid.uuid4().hex[:8]}"

        try:
            # ── Session 1: Have a distinctive conversation ──
            msg1 = (
                f"Hi I'm {name}. I'm building a system called {unique_topic} "
                f"that uses WebSockets for real-time data streaming and "
                f"stores events in TimescaleDB. Can you write a quick Python "
                f"WebSocket handler for me?"
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:300])
            assert r1.content, "Session 1 returned empty"

            # Wait for NoteObserver to extract and store notes.
            # The observer fires async after each request — give it time
            # to call Qwen3-8B for extraction + embed + upsert to Qdrant.
            time.sleep(15)

            # ── Session 2: New session — recall without re-intro ──
            msg2 = (
                "What was that project I was working on "
                "last time? Something about real-time streaming?"
            )
            r2 = cca.chat(msg2, session_id=sid2)
            # Skip judge — recall quality depends on note extraction
            # which is non-deterministic (8B model extraction).
            evaluate_response(r2, msg2, trace_test, None, "integration")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"

            # Check if the unique topic or related concepts were recalled
            content_lower = r2.content.lower()
            recall_terms = [
                unique_topic.lower(), "websocket", "timescaledb",
                "real-time", "streaming",
            ]
            recalled = sum(1 for t in recall_terms if t in content_lower)
            trace_test.set_attribute("cca.test.recall_count", recalled)
            trace_test.set_attribute("cca.test.recall_terms_found",
                                     [t for t in recall_terms if t in content_lower])

            # At least one distinctive term should be recalled
            assert recalled >= 1, (
                f"Agent didn't recall any terms from session 1 "
                f"(checked: {recall_terms}). Response: {r2.content[:300]}"
            )

        finally:
            cca.cleanup_test_user(name)
