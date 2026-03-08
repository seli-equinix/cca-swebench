"""Flow test: Router edge cases — direct answer, planner route, complexity.

Tests routing decisions that don't have dedicated test coverage:
- Simple questions that need no tools (answer_directly)
- Multi-step planning questions (PLANNER route)
- Complex multi-file coding task (expert extensions: code review + test gen)

Exercises: Expert router classification, PLANNER route, complexity detection,
CodeReviewerExtension, TestGeneratorExtension, write→run→verify.
"""

import re
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestRoutingEdgeCases:
    """Expert router: edge cases and underexercised routes."""

    def test_direct_answer(self, cca, trace_test, judge_model):
        """Simple factual question — should answer without tools."""
        tracker = cca.tracker()
        sid = f"test-direct-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = "What does the acronym REST stand for in web development?"
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.response", r.content[:300])
            assert r.content, "Returned empty"

            # Should contain the answer
            content_lower = r.content.lower()
            has_answer = any(w in content_lower for w in [
                "representational", "state", "transfer",
            ])
            trace_test.set_attribute("cca.test.has_answer", has_answer)
            assert has_answer, (
                f"Response doesn't explain REST: {r.content[:300]}"
            )

            # Should be relatively fast (no heavy tool calling)
            route = r.metadata.get("route", "")
            iters = r.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.route", route)
            trace_test.set_attribute("cca.test.iters", iters)

        finally:
            tracker.cleanup()

    def test_planner_route(self, cca, trace_test, judge_model):
        """Multi-step planning question — should engage planning logic."""
        tracker = cca.tracker()
        sid = f"test-planner-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = (
                "I need to design a CI/CD pipeline for a Python microservices "
                "project with 5 services. The pipeline should handle testing, "
                "building Docker images, and deploying to Kubernetes. "
                "What would be the high-level architecture and steps?"
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.response", r.content[:500])
            assert r.content, "Returned empty"
            assert len(r.content) > 200, (
                f"Planning response too short ({len(r.content)} chars) — "
                f"expected detailed breakdown"
            )

            route = r.metadata.get("route", "")
            trace_test.set_attribute("cca.test.route", route)

            # --- Brokenness check 1: Is this about CI/CD at all? ---
            content_lower = r.content.lower()
            is_on_topic = any(t in content_lower for t in [
                "pipeline", "ci/cd", "ci cd", "continuous",
                "build", "deploy", "stage", "workflow", "automation",
            ])
            trace_test.set_attribute("cca.test.is_on_topic", is_on_topic)
            assert is_on_topic, (
                f"Response is not about CI/CD: {r.content[:300]}"
            )

            # --- Brokenness check 2: Has structure? (user asked for "steps") ---
            has_structure = (
                bool(re.search(r"\d+[\.\):]", r.content))
                or r.content.count("\n- ") >= 2
                or r.content.count("##") >= 1
                or r.content.count("**") >= 2
                or any(s in content_lower for s in ["step", "phase", "stage"])
            )
            trace_test.set_attribute("cca.test.has_structure", has_structure)
            assert has_structure, (
                f"Response lacks structured breakdown — user asked for "
                f"'architecture and steps': {r.content[:400]}"
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
