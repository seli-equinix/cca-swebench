"""End-to-end integration test: full user lifecycle.

Tests the complete journey a real person would have with CCA:
create → enrich across sessions → web search → REST verify → delete.

The server auto-creates users when it detects a name introduction.
Tests validate state via the /users REST API for ground-truth.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.user]


class TestFullUserLifecycle:
    """Full lifecycle: new user → enrich profile → recall across sessions → web search → verify → delete.

    This test walks through the real journey a person would have with CCA:
    they arrive, introduce themselves, do some work across multiple sessions,
    and we verify the system accumulated everything correctly via REST API.
    """

    @pytest.mark.slow
    def test_full_lifecycle(self, cca, trace_test, judge_model):
        """Complete user lifecycle across 3 sessions + REST verification."""
        name = f"Lifecycle_{uuid.uuid4().hex[:6]}"
        company = "AcmeSystems"
        sid1 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-life-{uuid.uuid4().hex[:8]}"

        try:
            # ── Session 1: New user arrives, introduces themselves, codes ──
            msg1 = (
                f"Hi I'm {name}, I'm a DevOps engineer at {company}. "
                f"I mainly work with Docker and Kubernetes. "
                f"Write me a Python function to check if a port is open."
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:300])
            assert r1.content, "Session 1 returned empty"
            assert r1.user_identified, "User should be identified after introduction"

            # Verify user was created via REST API
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found after session 1"
            user_id = user["user_id"]
            trace_test.set_attribute("cca.test.user_id", user_id)

            # ── Session 2: Returns, past context should enrich response ──
            msg2 = (
                f"Hey it's {name} again from {company}. "
                f"I need help with a container health check script — "
                f"write a bash one-liner that checks if a Docker container is running."
            )
            r2 = cca.chat(msg2, session_id=sid2)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"
            assert r2.user_identified, "Returning user should be identified"

            # Session 2 response should reference their infrastructure context
            s2_lower = r2.content.lower()
            assert any(w in s2_lower for w in ["docker", "container", "bash"]), (
                f"Session 2 response doesn't address Docker/container request: "
                f"{r2.content[:200]}"
            )

            # ── Session 3: Different route — web search ──
            msg3 = (
                f"Hi, I'm {name}. Can you look up the latest Docker Compose "
                f"release notes and tell me what's new? Give me a link."
            )
            r3 = cca.chat(msg3, session_id=sid3)
            # Skip LLM judge for session 3 — this tests route diversity in
            # the lifecycle, not search quality.  Dedicated websearch tests
            # (test_basic_search, test_search_tech_topic) cover that.
            evaluate_response(r3, msg3, trace_test, None, "integration")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Session 3 returned empty"

            # Web search should have used tools and returned URLs
            iters = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.s3_tool_iterations", iters)
            assert "http" in r3.content.lower() or "docker" in r3.content.lower(), (
                f"Session 3 (web search) didn't return Docker info or URLs: "
                f"{r3.content[:200]}"
            )

            # ── REST API verification: profile accumulated correctly ──
            profile = cca.get_user_profile(user_id)
            assert profile is not None, "Full profile not returned from REST API"
            trace_test.set_attribute(
                "cca.test.profile_facts", str(profile.get("facts", {})),
            )
            trace_test.set_attribute(
                "cca.test.profile_sessions",
                len(profile.get("session_history", [])),
            )

            # Profile should have facts from sessions
            facts = profile.get("facts", {})
            facts_str = str(facts).lower()
            has_company = company.lower() in facts_str
            has_infra = any(
                w in facts_str
                for w in ["docker", "kubernetes", "devops", "container"]
            )
            trace_test.set_attribute("cca.test.has_company", has_company)
            trace_test.set_attribute("cca.test.has_infra_facts", has_infra)

            # At minimum, the company fact should be stored
            assert has_company or has_infra, (
                f"Profile has no facts about {company} or infrastructure work. "
                f"Facts: {facts}"
            )

            # Should have session history from multiple sessions
            sessions = profile.get("session_history", [])
            assert len(sessions) >= 2, (
                f"Profile should show multiple sessions, got {len(sessions)}: "
                f"{sessions}"
            )

            # ── Cleanup: delete user and verify gone ──
            cca.cleanup_test_user(name)

            user_after = cca.find_user_by_name(name)
            trace_test.set_attribute("cca.test.user_deleted", user_after is None)
            assert user_after is None, (
                f"User '{name}' still exists after deletion"
            )

        finally:
            cca.cleanup_test_user(name)
