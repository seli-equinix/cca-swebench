"""Flow test: A returning user across multiple sessions.

Journey: create user → return in new session → verify recall →
update facts (job change) → verify overwrite → verify no duplicates.

Replaces 9 individual tests: identify_returning_user, no_duplicate_on_return,
infer_known_user, fact_recalled_in_conversation, fact_persists_across_sessions,
fact_overwrite, context_shows_stored_facts, context_enriches_responses,
preference_recalled_next_session.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.slow]


class TestReturningUserMemory:
    """Returning user: recall facts, update facts, context enrichment."""

    def test_returning_user_memory(self, cca, trace_test, judge_model):
        """Full memory flow: store → recall → overwrite → verify."""
        name = f"Memory_{uuid.uuid4().hex[:6]}"
        old_company = f"OldCorp_{uuid.uuid4().hex[:4]}"
        new_company = f"NewCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-mem1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-mem2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-mem3-{uuid.uuid4().hex[:8]}"
        sid4 = f"test-mem4-{uuid.uuid4().hex[:8]}"

        try:
            # ── Session 1: First visit — introduce with distinctive info ──
            msg1 = (
                f"Hi I'm {name}. I'm a backend engineer at {old_company}, "
                f"my main language is Rust, and I work on the search team. "
                f"Write a Python function to parse a URL."
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:300])
            assert r1.content, "Session 1 returned empty"
            assert r1.user_identified, "User should be identified"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"

            # ── Session 2: Return — system should track without re-intro ──
            msg2 = (
                "Can you remind me what company I work at? Also write me "
                "a health check function in Python."
            )
            r2 = cca.chat(msg2, session_id=sid2)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"

            # Should recall the company
            content_lower = r2.content.lower()
            recalled = old_company.lower() in content_lower
            trace_test.set_attribute("cca.test.company_recalled", recalled)
            assert recalled, (
                f"Agent didn't recall '{old_company}': {r2.content[:300]}"
            )

            # No duplicate user created
            users_data = cca.list_users()
            matches = [
                u for u in users_data.get("users", [])
                if u.get("display_name", "").lower() == name.lower()
            ]
            trace_test.set_attribute("cca.test.user_count", len(matches))
            assert len(matches) == 1, (
                f"Expected 1 user named '{name}', found {len(matches)}"
            )

            # Session count should reflect multiple sessions
            assert matches[0]["session_count"] >= 2, (
                f"Expected session_count >= 2, got {matches[0]['session_count']}"
            )

            # ── Session 3: Job change — fact overwrite (no re-intro) ──
            msg3 = (
                f"I switched jobs — I now work at "
                f"{new_company}. Write me a one-liner to check disk usage."
            )
            r3 = cca.chat(msg3, session_id=sid3)
            evaluate_response(r3, msg3, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Session 3 returned empty"

            # ── Session 4: Verify overwrite persisted (no re-intro) ──
            msg4 = (
                "Where do I work now? "
                "Also write a quick Python timestamp function."
            )
            r4 = cca.chat(msg4, session_id=sid4)
            evaluate_response(r4, msg4, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s4_response", r4.content[:300])
            assert r4.content, "Session 4 returned empty"

            content_lower = r4.content.lower()
            has_new = new_company.lower() in content_lower
            trace_test.set_attribute("cca.test.has_new_company", has_new)
            assert has_new, (
                f"Agent didn't mention new company "
                f"'{new_company}': {r4.content[:300]}"
            )

        finally:
            cca.cleanup_test_user(name)
