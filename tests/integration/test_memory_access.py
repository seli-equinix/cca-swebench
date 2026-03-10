"""Flow test: Memory accessibility across all routes.

Validates that users can access their data through the chat interface:
- PLANNER: complete plan in response, no "Memory updated" narration
- CODER: execution results in response, memory accessible via REST
- INFRA: infrastructure results in response, memory accessible via REST
- Cross-session: plan created in session 1 → recalled in session 2

The chat response IS the deliverable for Continue.dev users who can't
call REST APIs or know session IDs. Memory files are a secondary access
path for programmatic use.

Exercises: PLANNER prompt, NoteObserver memory_files, REST memory endpoint,
cross-session recall via Qdrant notes.
"""

import re
import time
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestMemoryAccess:
    """Memory accessibility: chat-first UX + REST fallback."""

    def test_planner_anonymous_clean_response(self, cca, trace_test, judge_model):
        """Anonymous PLANNER — complete plan in response, no tool narration."""
        tracker = cca.tracker()
        sid = f"test-plan-anon-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = (
                "I need to design a CI/CD pipeline for a Python monorepo with "
                "3 services. It should handle testing, Docker image builds, and "
                "deployment to Kubernetes. What's the architecture and steps?"
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            assert r.content, "Returned empty"
            assert len(r.content) > 200, f"Plan too short ({len(r.content)} chars)"

            content_lower = r.content.lower()

            # Plan must be on-topic
            assert any(t in content_lower for t in [
                "pipeline", "ci/cd", "ci cd", "continuous", "build",
                "deploy", "stage", "workflow", "automation",
            ]), f"Not about CI/CD: {r.content[:300]}"

            # Plan must have structure
            has_structure = (
                bool(re.search(r"\d+[\.\\):]\s", r.content))
                or r.content.count("\n- ") >= 2
                or r.content.count("##") >= 1
            )
            assert has_structure, f"No structure: {r.content[:400]}"

            # CRITICAL: No "Memory updated" narration
            assert "memory updated" not in content_lower, (
                f"Response contains 'Memory updated': {r.content[:300]}"
            )

            # Check metadata
            trace_test.set_attribute("cca.test.route", r.metadata.get("route", ""))
            trace_test.set_attribute("cca.test.memory_files",
                                     str(r.metadata.get("memory_files")))

        finally:
            tracker.cleanup()

    def test_planner_identified_user_complete_plan(self, cca, trace_test, judge_model):
        """Identified user PLANNER — plan in response, memory accessible."""
        tracker = cca.tracker()
        name = f"Planner_{uuid.uuid4().hex[:6]}"
        sid = f"test-plan-user-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        tracker.track_session(sid)

        try:
            # Turn 1: Identify
            r1 = cca.chat(
                f"Hi, I'm {name}. I'm a DevOps engineer working with Kubernetes.",
                session_id=sid,
            )
            assert r1.content, "Turn 1 returned empty"
            user_id = r1.metadata.get("user_id")

            # Turn 2: Ask for a plan
            msg = (
                "Design a monitoring stack for a production Kubernetes cluster "
                "with 20 microservices. I need metrics, logs, and traces. "
                "Give me the complete architecture and implementation plan."
            )
            r2 = cca.chat(msg, session_id=sid, user_id=user_id)
            evaluate_response(r2, msg, trace_test, judge_model, "integration")

            assert r2.content, "Turn 2 returned empty"
            assert len(r2.content) > 200, f"Plan too short ({len(r2.content)} chars)"

            content_lower = r2.content.lower()

            # Plan on-topic
            assert any(t in content_lower for t in [
                "monitor", "metric", "alert", "prometheus", "grafana",
                "log", "trace", "observ", "dashboard",
            ]), f"Not about monitoring: {r2.content[:300]}"

            # Plan has structure
            has_structure = (
                bool(re.search(r"\d+[\.\\):]\s", r2.content))
                or r2.content.count("\n- ") >= 2
                or r2.content.count("##") >= 1
            )
            assert has_structure, f"No structure: {r2.content[:400]}"

            # No narration
            assert "memory updated" not in content_lower, (
                f"'Memory updated' in response: {r2.content[:300]}"
            )

            # User was identified
            assert r2.metadata.get("user_identified"), "User not identified"

            # If memory was created, verify it's accessible via REST
            memory_files = r2.metadata.get("memory_files")
            trace_test.set_attribute("cca.test.memory_files", str(memory_files))
            if memory_files:
                import httpx
                resp = httpx.get(
                    f"{cca.base_url}/v1/sessions/{sid}/memory",
                    timeout=10,
                )
                assert resp.status_code == 200, (
                    f"Memory endpoint returned {resp.status_code}"
                )
                data = resp.json()
                assert data["file_count"] > 0

        finally:
            tracker.cleanup()

    def test_coder_identified_user_memory_access(self, cca, trace_test, judge_model):
        """Identified user CODER — response has results, memory accessible."""
        tracker = cca.tracker()
        name = f"Coder_{uuid.uuid4().hex[:6]}"
        sid = f"test-coder-mem-{uuid.uuid4().hex[:8]}"
        prefix = f"mem_{uuid.uuid4().hex[:6]}"
        tracker.track_user(name)
        tracker.track_session(sid)
        tracker.track_workspace_prefix(prefix)

        try:
            # Turn 1: Identify + request code
            msg1 = (
                f"Hi I'm {name}. Create a Python module at "
                f"/workspace/{prefix}_utils.py with two functions:\n"
                f"1. fibonacci(n) - returns the nth Fibonacci number\n"
                f"2. is_prime(n) - returns True if n is prime\n"
                f"Then run both with n=10 and show the output."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            assert r1.content, "Turn 1 returned empty"

            # Agent should have used tools
            iters = r1.metadata.get("tool_iterations", 0)
            assert iters >= 1, f"No tools used (iters={iters})"

            # Response should contain execution results
            content_lower = r1.content.lower()
            has_fibonacci = "55" in r1.content or "fibonacci" in content_lower
            has_prime = "prime" in content_lower
            assert has_fibonacci or has_prime, (
                f"Response lacks execution results: {r1.content[:300]}"
            )

            # No narration
            assert "memory updated" not in content_lower, (
                f"'Memory updated' in response: {r1.content[:300]}"
            )

            # Check memory metadata
            memory_files = r1.metadata.get("memory_files")
            trace_test.set_attribute("cca.test.memory_files", str(memory_files))

            # If memory was created, verify REST access
            if memory_files:
                import httpx
                resp = httpx.get(
                    f"{cca.base_url}/v1/sessions/{sid}/memory",
                    timeout=10,
                )
                assert resp.status_code == 200

            # User identified
            assert r1.metadata.get("user_identified"), "User not identified"

        finally:
            tracker.cleanup()

    def test_infra_identified_user_memory_access(self, cca, trace_test, judge_model):
        """Identified user INFRA — response has results, memory accessible."""
        tracker = cca.tracker()
        name = f"Infra_{uuid.uuid4().hex[:6]}"
        sid = f"test-infra-mem-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        tracker.track_session(sid)

        try:
            # Identify + ask infra question
            msg = (
                f"Hi I'm {name}, a sysadmin. Check the health of the Redis "
                f"service and tell me its status. Also list all running "
                f"Docker containers on this node."
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            assert r.content, "Returned empty"

            # Agent should have used tools (bash for docker/redis checks)
            iters = r.metadata.get("tool_iterations", 0)
            assert iters >= 1, f"No tools used (iters={iters})"

            # Response should contain infra results
            content_lower = r.content.lower()
            has_redis = any(t in content_lower for t in [
                "redis", "pong", "6379", "connected",
            ])
            has_docker = any(t in content_lower for t in [
                "docker", "container", "running", "status",
            ])
            assert has_redis or has_docker, (
                f"Response lacks infra results: {r.content[:300]}"
            )

            # No narration
            assert "memory updated" not in content_lower, (
                f"'Memory updated' in response: {r.content[:300]}"
            )

            # Check memory metadata
            memory_files = r.metadata.get("memory_files")
            trace_test.set_attribute("cca.test.memory_files", str(memory_files))

            # If memory was created, verify REST access
            if memory_files:
                import httpx
                resp = httpx.get(
                    f"{cca.base_url}/v1/sessions/{sid}/memory",
                    timeout=10,
                )
                assert resp.status_code == 200

        finally:
            tracker.cleanup()

    @pytest.mark.slow
    def test_plan_recall_across_sessions(self, cca, trace_test, judge_model):
        """Session 1: create a distinctive plan -> Session 2: recall it."""
        tracker = cca.tracker()
        name = f"Recall_{uuid.uuid4().hex[:6]}"
        unique_topic = f"Pegasus_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-recall1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-recall2-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        tracker.track_session(sid1)
        tracker.track_session(sid2)

        try:
            # -- Session 1: Create a distinctive plan --
            msg1 = (
                f"Hi I'm {name}. I'm designing a system called {unique_topic} -- "
                f"a real-time event processing pipeline using Apache Kafka for "
                f"ingestion, Apache Flink for stream processing, and ClickHouse "
                f"for analytical storage. Give me the complete architecture with "
                f"at least 4 components and their interactions."
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            assert r1.content, "Session 1 returned empty"
            assert len(r1.content) > 200, f"Plan too short ({len(r1.content)} chars)"

            # Plan should mention distinctive terms
            content1_lower = r1.content.lower()
            assert any(t in content1_lower for t in [
                "kafka", "flink", "clickhouse", "event", "stream",
            ]), f"Plan doesn't mention key technologies: {r1.content[:300]}"

            # Get user_id for cross-session tracking
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"
            user_id = user["user_id"]

            # Wait for NoteObserver to extract and store notes
            time.sleep(15)

            # Verify notes were stored
            notes = cca.search_notes(
                f"kafka flink clickhouse {unique_topic}",
                user_id=user_id,
            )
            trace_test.set_attribute("cca.test.notes_found", len(notes))

            # -- Session 2: New session -- ask about the plan --
            # Question avoids session-1 keywords -- any mention proves
            # genuine recall via notes.
            msg2 = (
                "Hey, what was that architecture I was working on last time? "
                "I can't remember the components we discussed."
            )
            r2 = cca.chat(msg2, session_id=sid2, user_id=user_id)
            evaluate_response(r2, msg2, trace_test, None, "integration")

            assert r2.content, "Session 2 returned empty"

            # Check recall of distinctive terms
            content2_lower = r2.content.lower()
            recall_terms = [
                unique_topic.lower(),  # Pegasus_xxxxx
                "kafka",
                "flink",
                "clickhouse",
            ]
            recalled = sum(1 for t in recall_terms if t in content2_lower)
            trace_test.set_attribute("cca.test.recall_count", recalled)
            trace_test.set_attribute("cca.test.recall_terms",
                                     [t for t in recall_terms if t in content2_lower])

            # At least one distinctive term recalled
            assert recalled >= 1, (
                f"Agent didn't recall session 1 terms "
                f"(checked: {recall_terms}). Notes: {len(notes)}. "
                f"Response: {r2.content[:300]}"
            )

        finally:
            tracker.cleanup()
