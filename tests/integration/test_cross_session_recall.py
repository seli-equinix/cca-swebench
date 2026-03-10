"""Flow test: Cross-session recall — notes and plans persist across sessions.

Journey: store project facts → create a plan → new session recalls both.
A developer who told CCA about their project setup yesterday, discussed
a plan, and comes back today expecting CCA to remember everything.

Exercises: note extraction, plan persistence, cross-session memory recall,
context enrichment via <past_insights>.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestCrossSessionRecall:
    """Cross-session memory: notes and plans persist and are recalled."""

    def test_cross_session_recall(self, cca, trace_test, judge_model):
        """4-turn flow across 3 sessions: store facts → plan → recall both.

        Absorbs: test_note_recall, test_plan_recall_across_sessions.

        Session 1 (2 turns): Introduce and share project details.
        Session 2 (1 turn): Ask for a plan related to the project.
        Session 3 (1 turn): New session — recall facts from S1 + plan from S2.
        """
        tracker = cca.tracker()
        name = f"Recall_{uuid.uuid4().hex[:6]}"
        unique_tag = uuid.uuid4().hex[:8]
        sid1 = f"test-recall1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-recall2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-recall3-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        for s in (sid1, sid2, sid3):
            tracker.track_session(s)

        try:
            # ── Session 1, Turn 1: Introduce and share project facts ──
            msg1 = (
                f"Hi I'm {name}. I'm working on Project Helios ({unique_tag}). "
                f"The stack is PostgreSQL 16 with pgvector for embeddings, "
                f"FastAPI backend with SQLAlchemy ORM, and React frontend. "
                f"We deploy on AWS EKS with ArgoCD for GitOps. "
                f"Can you write a quick health check endpoint in FastAPI?"
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1t1_response", r1.content[:300])
            assert r1.content, "Session 1 Turn 1 returned empty"
            assert r1.user_identified, "User should be identified"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"
            user_id = user["user_id"]

            # ── Session 1, Turn 2: Add more distinctive details ──
            msg2 = (
                "One more thing — our team uses a custom caching layer called "
                f"HeliosCache ({unique_tag}) that sits between the API and "
                "PostgreSQL. It's Redis-backed with a 5-minute TTL. "
                "Can you show me a Redis PING check in Python?"
            )
            r2 = cca.chat(msg2, session_id=sid1, user_id=user_id)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1t2_response", r2.content[:300])
            assert r2.content, "Session 1 Turn 2 returned empty"

            # ── Session 2: Ask for a plan (new session, same user) ──
            msg3 = (
                "I need a plan to add real-time notifications to Project Helios. "
                "Users should get instant alerts when their dashboards update. "
                "What's the architecture and implementation steps?"
            )
            r3 = cca.chat(msg3, session_id=sid2, user_id=user_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s2_response", r3.content[:500])
            assert r3.content, "Session 2 returned empty"
            assert len(r3.content) > 200, (
                f"Plan response too short ({len(r3.content)} chars)"
            )

            content3 = r3.content.lower()
            has_plan = any(w in content3 for w in [
                "notification", "websocket", "real-time", "realtime",
                "alert", "push", "event", "sse",
            ])
            trace_test.set_attribute("cca.test.s2_has_plan", has_plan)
            assert has_plan, (
                f"Response doesn't discuss notifications: {r3.content[:300]}"
            )

            # ── Session 3: New session — recall facts + plan ──
            msg4 = (
                "What do you remember about my project setup? "
                "And what was the notification plan we discussed?"
            )
            r4 = cca.chat(msg4, session_id=sid3, user_id=user_id)
            evaluate_response(r4, msg4, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r4.content[:500])
            assert r4.content, "Session 3 returned empty"

            content4 = r4.content.lower()

            # Should recall project facts from Session 1
            fact_keywords = [
                "postgresql", "pgvector", "fastapi", "sqlalchemy",
                "eks", "argocd", "helioscache", "helios",
            ]
            recalled_facts = [k for k in fact_keywords if k in content4]
            trace_test.set_attribute(
                "cca.test.s3_recalled_facts", str(recalled_facts),
            )
            assert len(recalled_facts) >= 2, (
                f"Agent recalled too few project facts ({recalled_facts}). "
                f"Response: {r4.content[:400]}"
            )

            # Should recall the notification plan from Session 2
            plan_keywords = [
                "notification", "websocket", "real-time", "realtime",
                "alert", "push", "dashboard",
            ]
            recalled_plan = [k for k in plan_keywords if k in content4]
            trace_test.set_attribute(
                "cca.test.s3_recalled_plan", str(recalled_plan),
            )
            assert len(recalled_plan) >= 1, (
                f"Agent didn't recall notification plan ({recalled_plan}). "
                f"Response: {r4.content[:400]}"
            )

        finally:
            tracker.cleanup()
