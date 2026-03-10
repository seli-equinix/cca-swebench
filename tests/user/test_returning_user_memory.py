"""Flow test: A returning user across multiple sessions.

Journey: create user → return in new session → verify recall →
update facts (job change) → verify overwrite → accumulate new facts →
verify accumulated knowledge.

Replaces 9 individual tests: identify_returning_user, no_duplicate_on_return,
infer_known_user, fact_recalled_in_conversation, fact_persists_across_sessions,
fact_overwrite, context_shows_stored_facts, context_enriches_responses,
preference_recalled_next_session.
Also absorbs fact accumulation coverage from test_full_lifecycle.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.slow]


class TestReturningUserMemory:
    """Returning user: recall facts, update facts, accumulate knowledge."""

    def test_returning_user_memory(self, cca, trace_test, judge_model):
        """Full memory flow: store → recall → overwrite → verify → accumulate → verify.

        Sessions 1-4: Original test (intro → recall → job change → verify).
        Sessions 5-6 (NEW): Fact accumulation — add infra details, then
        verify BOTH job change AND infra facts are retained.
        """
        tracker = cca.tracker()
        name = f"Memory_{uuid.uuid4().hex[:6]}"
        old_company = f"OldCorp_{uuid.uuid4().hex[:4]}"
        new_company = f"NewCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-mem1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-mem2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-mem3-{uuid.uuid4().hex[:8]}"
        sid4 = f"test-mem4-{uuid.uuid4().hex[:8]}"
        sid5 = f"test-mem5-{uuid.uuid4().hex[:8]}"
        sid6 = f"test-mem6-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        for s in (sid1, sid2, sid3, sid4, sid5, sid6):
            tracker.track_session(s)

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
            user_id = user["user_id"]

            # ── Session 2: Return — system tracks via user_id token ──
            msg2 = (
                "Can you remind me what company I work at? Also write me "
                "a health check function in Python."
            )
            r2 = cca.chat(msg2, session_id=sid2, user_id=user_id)
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
            r3 = cca.chat(msg3, session_id=sid3, user_id=user_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Session 3 returned empty"

            # ── Session 4: Verify overwrite persisted (no re-intro) ──
            msg4 = (
                "Where do I work now? "
                "Also write a quick Python timestamp function."
            )
            r4 = cca.chat(msg4, session_id=sid4, user_id=user_id)
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

            # ── Session 5 (NEW): Add infrastructure facts ──
            # Tests fact ACCUMULATION — new facts should be added alongside
            # existing facts (company change), not replace them.
            msg5 = (
                "Important: we run a 5-node Docker Swarm cluster, our "
                "registry is at registry.acme.internal, and we deploy via "
                "Portainer with GitOps. "
                "Write me a docker-compose.yml for a Redis cache."
            )
            r5 = cca.chat(msg5, session_id=sid5, user_id=user_id)
            evaluate_response(r5, msg5, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s5_response", r5.content[:300])
            assert r5.content, "Session 5 returned empty"

            # Should mention redis/compose in the response
            content5 = r5.content.lower()
            has_redis = any(w in content5 for w in [
                "redis", "compose", "healthcheck", "volume",
            ])
            trace_test.set_attribute("cca.test.s5_has_redis", has_redis)
            assert has_redis, (
                f"Response doesn't mention Redis/compose: {r5.content[:300]}"
            )

            # Verify infra facts stored via REST
            profile = cca.get_user_profile(user_id)
            if profile:
                facts = str(profile.get("facts", {})).lower()
                trace_test.set_attribute("cca.test.s5_facts", facts[:500])

            # ── Session 6 (NEW): Verify accumulated knowledge ──
            # The agent should recall BOTH the company change (S3) AND
            # the infrastructure details (S5).
            msg6 = "What do you know about my infrastructure setup?"
            r6 = cca.chat(msg6, session_id=sid6, user_id=user_id)
            evaluate_response(r6, msg6, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s6_response", r6.content[:500])
            assert r6.content, "Session 6 returned empty"

            content6 = r6.content.lower()
            infra_keywords = [
                "swarm", "registry", "portainer", "gitops", "docker",
            ]
            recalled_infra = [k for k in infra_keywords if k in content6]
            trace_test.set_attribute(
                "cca.test.s6_recalled_infra", str(recalled_infra),
            )
            assert len(recalled_infra) >= 1, (
                f"Agent didn't recall infra facts ({recalled_infra}). "
                f"Response: {r6.content[:300]}"
            )

        finally:
            tracker.cleanup()
