"""Flow test: Router edge cases — direct answers, planner route, complexity, cross-route identity.

Tests routing decisions that don't have dedicated test coverage:
- Quick knowledge → detailed planning (route escalation)
- Identified user across PLANNER → CODER → INFRA routes
- Complex multi-file coding task (expert extensions: code review + test gen)

Exercises: Expert router classification, PLANNER route, complexity detection,
cross-route memory persistence, CodeReviewerExtension, TestGeneratorExtension.
"""

import re
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestRoutingEdgeCases:
    """Expert router: edge cases and underexercised routes."""

    def test_planning_flow(self, cca, trace_test, judge_model):
        """2-turn flow: quick factual answer → detailed planning request.

        Absorbs: test_direct_answer, test_planner_route,
        test_planner_anonymous_clean_response.

        Journey: a developer starts with a quick knowledge question,
        then asks for a project plan — testing route escalation from
        direct answer → planner.
        """
        tracker = cca.tracker()
        sid = f"test-plan-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Quick factual question (direct answer route) ──
            msg1 = "What does the acronym REST stand for in web development?"
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"

            content1 = r1.content.lower()
            has_answer = any(w in content1 for w in [
                "representational", "state", "transfer",
            ])
            trace_test.set_attribute("cca.test.t1_has_answer", has_answer)
            assert has_answer, (
                f"Response doesn't explain REST: {r1.content[:300]}"
            )

            # ── Turn 2: Detailed planning request (planner route) ──
            msg2 = (
                "I need to design a CI/CD pipeline for a Python microservices "
                "project with 5 services. The pipeline should handle testing, "
                "building Docker images, and deploying to Kubernetes. "
                "What would be the high-level architecture and steps?"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"
            assert len(r2.content) > 200, (
                f"Planning response too short ({len(r2.content)} chars)"
            )

            content2 = r2.content.lower()
            is_on_topic = any(t in content2 for t in [
                "pipeline", "ci/cd", "ci cd", "continuous",
                "build", "deploy", "stage", "workflow", "automation",
            ])
            trace_test.set_attribute("cca.test.t2_on_topic", is_on_topic)
            assert is_on_topic, (
                f"Response is not about CI/CD: {r2.content[:300]}"
            )

            has_structure = (
                bool(re.search(r"\d+[\.\):]", r2.content))
                or r2.content.count("\n- ") >= 2
                or r2.content.count("##") >= 1
                or r2.content.count("**") >= 2
                or any(s in content2 for s in ["step", "phase", "stage"])
            )
            trace_test.set_attribute("cca.test.t2_has_structure", has_structure)
            assert has_structure, (
                f"Response lacks structured breakdown: {r2.content[:400]}"
            )

            # No tool narration in response
            assert "memory updated" not in content2, (
                f"Response contains 'Memory updated': {r2.content[:300]}"
            )

        finally:
            tracker.cleanup()

    def test_identified_user_across_routes(self, cca, trace_test, judge_model):
        """4-turn flow: intro → plan → code → infra check.

        Absorbs: test_planner_identified_user_complete_plan,
        test_coder_identified_user_memory_access,
        test_infra_identified_user_memory_access.

        Journey: an identified power user does planning, coding, and infra
        checks in one session. Each route should produce actual results
        (not narration) and memory should be REST-accessible.
        """
        tracker = cca.tracker()
        name = f"RouteUser_{uuid.uuid4().hex[:6]}"
        sid = f"test-xroute-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        tracker.track_session(sid)

        try:
            # ── Turn 1: Introduce yourself ──
            msg1 = (
                f"Hi I'm {name}. I'm a DevOps engineer working with "
                f"Kubernetes and microservices."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"
            assert r1.user_identified, "User should be identified after intro"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"
            user_id = user["user_id"]

            # ── Turn 2: Planning request (PLANNER route) ──
            msg2 = (
                "Design a monitoring stack for a production K8s cluster with "
                "20 microservices. I need metrics, logs, and traces. "
                "What's the architecture?"
            )
            r2 = cca.chat(msg2, session_id=sid, user_id=user_id)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"
            assert len(r2.content) > 200, (
                f"Planning response too short ({len(r2.content)} chars)"
            )

            content2 = r2.content.lower()
            is_on_topic = any(t in content2 for t in [
                "monitor", "metric", "alert", "prometheus", "grafana",
                "log", "trace", "observ",
            ])
            trace_test.set_attribute("cca.test.t2_on_topic", is_on_topic)
            assert is_on_topic, (
                f"Response is not about monitoring: {r2.content[:300]}"
            )
            assert "memory updated" not in content2, (
                f"Turn 2 contains tool narration: {r2.content[:300]}"
            )

            # ── Turn 3: Code task (CODER route) ──
            msg3 = (
                "Now write a Python module with fibonacci(n) and is_prime(n). "
                "Run both with n=10 and show me the output."
            )
            r3 = cca.chat(msg3, session_id=sid, user_id=user_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't use tools for coding (iters={iters3})"
            )

            content3 = r3.content.lower()
            has_fib = "55" in r3.content or "fibonacci" in content3
            trace_test.set_attribute("cca.test.t3_has_fib", has_fib)
            assert has_fib, (
                f"Response doesn't show fibonacci results: {r3.content[:300]}"
            )
            assert "memory updated" not in content3, (
                f"Turn 3 contains tool narration: {r3.content[:300]}"
            )

            # ── Turn 4: Infra check (INFRA route) ──
            msg4 = (
                "Check the health of Redis and list all running Docker "
                "containers on this node."
            )
            r4 = cca.chat(msg4, session_id=sid, user_id=user_id)
            evaluate_response(r4, msg4, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
            assert r4.content, "Turn 4 returned empty"

            iters4 = r4.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t4_iters", iters4)
            assert iters4 >= 1, (
                f"Agent didn't use tools for infra check (iters={iters4})"
            )

            content4 = r4.content.lower()
            has_infra = any(w in content4 for w in [
                "redis", "pong", "6379", "docker", "container", "running",
            ])
            trace_test.set_attribute("cca.test.t4_has_infra", has_infra)
            assert has_infra, (
                f"Response doesn't mention infra results: {r4.content[:300]}"
            )
            assert "memory updated" not in content4, (
                f"Turn 4 contains tool narration: {r4.content[:300]}"
            )

        finally:
            tracker.cleanup()

    def test_complex_multi_file_project(self, cca, trace_test, judge_model):
        """Complex multi-file project — triggers expert extensions + code execution.

        Journey: ask for a multi-file Python project with separate modules →
        agent creates 3+ files (triggers TestGeneratorExtension) with 3+
        edit operations (triggers CodeReviewerExtension) → then run the
        code and verify output.

        Expert extensions require estimated_steps >= 8 (is_complex).
        CodeReviewerExtension fires after 3+ file edits (create/str_replace/insert).
        TestGeneratorExtension fires after any file creation.

        Expert output appears as <code_review> and <test_suggestions> XML tags
        injected into the conversation — we soft-check for these.
        """
        tracker = cca.tracker()
        sid = f"test-complex-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)
        prefix = f"calc_{uuid.uuid4().hex[:6]}"
        tracker.track_workspace_prefix(prefix)

        try:
            # ── Turn 1: Create a multi-file Python project ──
            # Request is deliberately complex — multiple files, clear structure,
            # enough work for the router to estimate 8+ steps.
            msg1 = (
                f"Build me a Python calculator project in /workspace with "
                f"these separate files:\n"
                f"1. /workspace/{prefix}_ops.py — a module with functions: "
                f"add(a, b), subtract(a, b), multiply(a, b), divide(a, b). "
                f"divide should raise ValueError on division by zero.\n"
                f"2. /workspace/{prefix}_formatter.py — a module with a function "
                f"format_result(operation, a, b, result) that returns a string "
                f"like 'add(3, 5) = 8'.\n"
                f"3. /workspace/{prefix}_main.py — imports from both modules, "
                f"runs all 4 operations on 10 and 3, formats each result, "
                f"and prints them. Also tests divide by zero handling.\n"
                f"Make sure all files work together."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            steps = r1.metadata.get("estimated_steps", 0)
            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            trace_test.set_attribute("cca.test.estimated_steps", steps)
            trace_test.set_attribute("cca.test.route", route)
            assert iters1 >= 1, (
                f"Agent didn't use tools to create files (iters={iters1})"
            )

            # Verify files were created via REST
            files = cca.list_workspace_files()
            file_list = files.get("files", [])
            file_names = [
                f.get("name", "") if isinstance(f, dict) else str(f)
                for f in file_list
            ]
            created_files = [n for n in file_names if prefix in n]
            trace_test.set_attribute("cca.test.files_created", len(created_files))
            trace_test.set_attribute("cca.test.file_names", str(created_files[:10]))
            assert len(created_files) >= 2, (
                f"Expected 3 files with prefix '{prefix}', found "
                f"{len(created_files)}: {created_files}"
            )

            # Check for expert extension output (soft assertion — not guaranteed)
            has_code_review = "<code_review>" in r1.content
            has_test_suggestions = "<test_suggestions>" in r1.content
            trace_test.set_attribute("cca.test.has_code_review", has_code_review)
            trace_test.set_attribute("cca.test.has_test_suggestions", has_test_suggestions)
            trace_test.set_attribute(
                "cca.test.experts_would_fire",
                f"steps={steps} >= 8 required, iters={iters1}, "
                f"review={'yes' if has_code_review else 'no'}, "
                f"test_gen={'yes' if has_test_suggestions else 'no'}",
            )

            # ── Turn 2: Run the code and verify output ──
            msg2 = (
                f"Now run /workspace/{prefix}_main.py with python3 and "
                f"show me the complete output. Does everything work correctly?"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't run the code (iters={iters2})"
            )

            # --- Execution evidence: does the response prove the code ran? ---
            content_lower = r2.content.lower()

            # Tier 1: Specific computed values (strongest proof)
            # 10+3=13, 10*3=30, 10/3≈3.33 — only from execution
            # Note: "7" (10-3) excluded — too common in unrelated contexts
            has_raw_13 = "13" in r2.content
            has_raw_30 = "30" in r2.content
            has_raw_333 = "3.33" in r2.content or "3.3" in r2.content
            raw_values = sum([has_raw_13, has_raw_30, has_raw_333])

            # Tier 2: Operation names (agent knows what was computed)
            has_add = "add" in content_lower
            has_subtract = "subtract" in content_lower
            has_multiply = "multiply" in content_lower
            has_divide = "divide" in content_lower
            op_names = sum([has_add, has_subtract, has_multiply, has_divide])

            # Tier 3: Blanket confirmation with specifics
            has_all_work = any(phrase in content_lower for phrase in [
                "all four operations", "all 4 operations",
                "all operations work", "everything works",
                "working correctly", "all tests pass",
            ])

            has_zero_check = any(x in content_lower for x in [
                "zero", "error", "valueerror", "exception",
                "divide by zero", "division by zero",
            ])

            # Accept: concrete values OR (named ops + confirmation) OR mix
            execution_proven = (
                raw_values >= 2
                or (op_names >= 3 and has_all_work)
                or (raw_values >= 1 and op_names >= 2)
            )

            trace_test.set_attribute("cca.test.raw_values", raw_values)
            trace_test.set_attribute("cca.test.op_names", op_names)
            trace_test.set_attribute("cca.test.has_all_work", has_all_work)
            trace_test.set_attribute("cca.test.has_zero_check", has_zero_check)
            trace_test.set_attribute("cca.test.execution_proven", execution_proven)

            assert execution_proven, (
                f"Response doesn't demonstrate actual execution results "
                f"(raw_values={raw_values}, op_names={op_names}, "
                f"blanket_ok={has_all_work}): {r2.content[:500]}"
            )

        finally:
            tracker.cleanup()
