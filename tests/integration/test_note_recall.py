"""Flow test: Note-taker recall across sessions + code validation.

Journey: have a distinctive conversation → wait for note extraction →
start a new session and ask about the previous topic → verify recall →
return to original session and validate the code runs.

The NoteObserver fires after each request and extracts notes to Qdrant
via the Spark1 Qwen3-8B model. Notes are then injected into future
sessions via search_notes or <past_insights> context enrichment.

Exercises: search_notes (NOTES group), NoteObserver pipeline,
code validation (bash tool execution).
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
        tracker = cca.tracker()
        name = f"NoteTest_{uuid.uuid4().hex[:6]}"
        unique_topic = f"Flamingo_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-note1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-note2-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        tracker.track_session(sid1)
        tracker.track_session(sid2)

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

            # Capture user_id for cross-session tracking
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created in session 1"
            user_id = user["user_id"]

            # Verify notes were actually stored before proceeding
            notes = cca.search_notes("websocket timescaledb", user_id=user_id)
            trace_test.set_attribute("cca.test.notes_found", len(notes))
            if notes:
                trace_test.set_attribute(
                    "cca.test.note_contents",
                    [n.get("content", "")[:100] for n in notes[:3]],
                )

            # ── Session 2: New session — recall without re-intro ──
            # Question deliberately avoids session-1 keywords so that
            # any mention in the response proves genuine recall via notes.
            msg2 = (
                "Hey, what was that project I was working on "
                "last time? I can't remember the details."
            )
            r2 = cca.chat(msg2, session_id=sid2, user_id=user_id)
            # Skip judge — recall quality depends on note extraction
            # which is non-deterministic (8B model extraction).
            evaluate_response(r2, msg2, trace_test, None, "integration")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"

            # Check if distinctive terms from session 1 were recalled.
            # These terms do NOT appear in the session 2 question, so
            # the only way they show up is through genuine note recall.
            content_lower = r2.content.lower()
            recall_terms = [
                unique_topic.lower(),  # Flamingo_xxxxx
                "websocket",
                "timescaledb",
            ]
            recalled = sum(1 for t in recall_terms if t in content_lower)
            trace_test.set_attribute("cca.test.recall_count", recalled)
            trace_test.set_attribute("cca.test.recall_terms_found",
                                     [t for t in recall_terms if t in content_lower])

            # At least one distinctive term should be recalled
            assert recalled >= 1, (
                f"Agent didn't recall any terms from session 1 "
                f"(checked: {recall_terms}). "
                f"Notes stored: {len(notes)}. "
                f"Response: {r2.content[:300]}"
            )

            # ── Turn 3: Validate the code actually runs ──
            # Back in session 1 where the agent created files.
            # Don't specify filename (we don't know it) — the agent
            # has session context and knows what it wrote.
            msg3 = (
                "Can you run the WebSocket handler code you wrote "
                "and show me the output? Just a quick syntax or import "
                "check is fine if it's a server."
            )
            r3 = cca.chat(msg3, session_id=sid1, user_id=user_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't use tools to run code (iters={iters3})"
            )

            # Verify the agent actually executed something — not just
            # described it. Check for execution indicators in the response.
            content3_lower = r3.content.lower()
            execution_indicators = [
                "python", "import", "output", "error", "traceback",
                "syntax", "exit", "listening", "running", "success",
                "no errors", "py_compile", "check",
            ]
            executed = any(
                ind in content3_lower for ind in execution_indicators
            )
            trace_test.set_attribute("cca.test.code_executed", executed)
            assert executed, (
                f"Agent didn't appear to execute the code. "
                f"Response: {r3.content[:300]}"
            )

        finally:
            tracker.cleanup()
