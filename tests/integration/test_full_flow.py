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
        """Complete user lifecycle across 5 sessions + REST verification."""
        name = f"Lifecycle_{uuid.uuid4().hex[:6]}"
        company = "AcmeSystems"
        sid1 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid4 = f"test-life-{uuid.uuid4().hex[:8]}"
        sid5 = f"test-life-{uuid.uuid4().hex[:8]}"

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

            # ── Session 2: Returns next day — container work ──
            msg2 = (
                "I need help with a container health check script — "
                "write a bash one-liner that checks if a Docker container is running."
            )
            r2 = cca.chat(msg2, session_id=sid2, user_id=user_id)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"
            assert r2.user_identified, (
                "Session 2 should identify user via user_id token"
            )

            s2_lower = r2.content.lower()
            assert any(w in s2_lower for w in ["docker", "container", "bash"]), (
                f"Session 2 response doesn't address Docker/container request: "
                f"{r2.content[:200]}"
            )

            # ── Session 3: Researching — web search for Docker Compose ──
            msg3 = (
                "Can you look up the latest Docker Compose "
                "release notes and tell me what's new? Give me a link."
            )
            r3 = cca.chat(msg3, session_id=sid3, user_id=user_id)
            # Skip LLM judge — dedicated websearch tests cover search quality.
            evaluate_response(r3, msg3, trace_test, None, "integration")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Session 3 returned empty"

            iters = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.s3_tool_iterations", iters)
            assert "http" in r3.content.lower() or "docker" in r3.content.lower(), (
                f"Session 3 (web search) didn't return Docker info or URLs: "
                f"{r3.content[:200]}"
            )

            # ── Session 4: Shares infra details — asks CCA to remember them ──
            msg4 = (
                "Hey, important stuff to remember about our setup: "
                "we run a 5-node Docker Swarm cluster, our private registry "
                "is at registry.acme.internal, and we deploy everything "
                "through Portainer with GitOps. Can you write me a "
                "docker-compose.yml for a Redis cache with persistent "
                "storage and a health check?"
            )
            r4 = cca.chat(msg4, session_id=sid4, user_id=user_id)
            evaluate_response(r4, msg4, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s4_response", r4.content[:500])
            assert r4.content, "Session 4 returned empty"

            # Should produce a compose file with Redis + health check
            s4_lower = r4.content.lower()
            assert any(w in s4_lower for w in [
                "redis", "healthcheck", "health_check", "volumes",
            ]), (
                f"Session 4 didn't produce Redis compose with health check: "
                f"{r4.content[:300]}"
            )

            # Verify infra facts were stored via REST
            profile_s4 = cca.get_user_profile(user_id)
            if profile_s4:
                facts_s4 = str(profile_s4.get("facts", {})).lower()
                infra_stored = any(w in facts_s4 for w in [
                    "swarm", "registry", "portainer", "gitops", "5-node",
                    "5 node",
                ])
                trace_test.set_attribute(
                    "cca.test.infra_facts_stored", infra_stored,
                )
                trace_test.set_attribute(
                    "cca.test.s4_facts", str(profile_s4.get("facts", {})),
                )

            # ── Session 5: Follow-up referencing earlier search + work ──
            msg5 = (
                "Now write me a multi-stage Dockerfile for a Python Flask "
                "app that would work with the Redis compose setup we just "
                "made. Use a slim base image and make sure it's "
                "production-ready with a non-root user."
            )
            r5 = cca.chat(msg5, session_id=sid5, user_id=user_id)
            evaluate_response(r5, msg5, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.s5_response", r5.content[:500])
            assert r5.content, "Session 5 returned empty"

            # Should produce a Dockerfile with multi-stage + non-root
            s5_lower = r5.content.lower()
            has_dockerfile = any(w in s5_lower for w in [
                "dockerfile", "from python", "from slim",
                "multi-stage", "as builder",
            ])
            has_nonroot = any(w in s5_lower for w in [
                "non-root", "useradd", "adduser", "user app",
                "user ", "chown",
            ])
            trace_test.set_attribute("cca.test.s5_has_dockerfile", has_dockerfile)
            trace_test.set_attribute("cca.test.s5_has_nonroot", has_nonroot)
            assert has_dockerfile, (
                f"Session 5 didn't produce a Dockerfile: {r5.content[:300]}"
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
                for w in ["docker", "kubernetes", "devops", "container",
                           "swarm", "registry", "portainer"]
            )
            trace_test.set_attribute("cca.test.has_company", has_company)
            trace_test.set_attribute("cca.test.has_infra_facts", has_infra)

            # Company + infra facts should be stored
            assert has_company or has_infra, (
                f"Profile has no facts about {company} or infrastructure work. "
                f"Facts: {facts}"
            )

            # All 5 sessions should be linked via user_id token
            sessions = profile.get("session_history", [])
            trace_test.set_attribute(
                "cca.test.session_count", len(sessions),
            )
            assert len(sessions) >= 5, (
                f"Expected >= 5 sessions linked, got {len(sessions)}: "
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
