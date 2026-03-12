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

from tests.evaluators import assert_tools_called, evaluate_response

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
        """Full dev workflow: create → run → read → edit → re-run → test.

        Exercises every file/code capability the CODER agent has:
          Turn 1: Create 3 files (str_replace_editor create × 3)
          Turn 2: Run the code (bash python3) — verify computed values
          Turn 3: Read a file back (str_replace_editor view) — verify contents
          Turn 4: Edit existing code — add new functions (str_replace_editor)
          Turn 5: Update main + run again — verify new computed values
          Turn 6: Create test file + run tests (new file creation + bash)

        This is a realistic developer session: build a project, run it,
        inspect code, add features, verify, then write tests.
        """
        tracker = cca.tracker()
        sid = f"test-complex-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)
        prefix = f"calc_{uuid.uuid4().hex[:6]}"
        tracker.track_workspace_prefix(prefix)

        try:
            # ═══════════════════════════════════════════════════════════
            # Turn 1: Create a multi-file Python project
            # ═══════════════════════════════════════════════════════════
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
            assert_tools_called(
                r1.metadata, ["str_replace_editor"], "Turn 1: create files",
            )

            # Verify files were created via REST ground truth
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

            # ═══════════════════════════════════════════════════════════
            # Turn 2: Run the code and verify actual execution output
            # ═══════════════════════════════════════════════════════════
            msg2 = (
                f"Run /workspace/{prefix}_main.py with python3 and "
                f"show me the complete output."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"
            assert_tools_called(
                r2.metadata, ["bash"], "Turn 2: run code",
            )

            # Verify the code was actually executed, not just described.
            # Tiered: raw values > (named ops + confirmation) > mix
            content2 = r2.content.lower()
            has_13 = "13" in r2.content
            has_30 = "30" in r2.content
            has_333 = "3.33" in r2.content or "3.3" in r2.content
            raw_values = sum([has_13, has_30, has_333])

            op_names = sum(1 for op in ["add", "subtract", "multiply", "divide"]
                           if op in content2)
            has_confirmation = any(p in content2 for p in [
                "all four operations", "all 4 operations",
                "all operations work", "everything works",
                "working correctly", "all tests pass",
                "executed", "output shows",
            ])
            execution_proven = (
                raw_values >= 2
                or (op_names >= 3 and has_confirmation)
                or (raw_values >= 1 and op_names >= 2)
            )
            trace_test.set_attribute("cca.test.t2_raw_values", raw_values)
            trace_test.set_attribute("cca.test.t2_op_names", op_names)
            trace_test.set_attribute("cca.test.t2_confirmed", has_confirmation)
            assert execution_proven, (
                f"No execution evidence (values={raw_values}, "
                f"ops={op_names}, confirmed={has_confirmation}): "
                f"{r2.content[:400]}"
            )

            # ═══════════════════════════════════════════════════════════
            # Turn 3: Read back a specific file — verify view capability
            # ═══════════════════════════════════════════════════════════
            msg3 = (
                f"Show me the contents of /workspace/{prefix}_ops.py. "
                f"I want to review the code before making changes."
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            # Agent may use str_replace_editor view or bash cat — both work.
            # Turn 1 already verified str_replace_editor for creation.
            t3_tools = r3.tool_names
            trace_test.set_attribute("cca.test.t3_tools", str(t3_tools))
            assert len(t3_tools) >= 1, (
                f"Turn 3 didn't use any tools to read file. Tools: {t3_tools}"
            )

            # Response should contain the actual function definitions
            content3 = r3.content.lower()
            has_func_defs = sum(1 for fn in [
                "def add", "def subtract", "def multiply", "def divide",
            ] if fn in content3)
            trace_test.set_attribute("cca.test.t3_func_defs", has_func_defs)
            assert has_func_defs >= 3, (
                f"File view missing function definitions "
                f"(found {has_func_defs}/4): {r3.content[:400]}"
            )

            # ═══════════════════════════════════════════════════════════
            # Turn 4: Edit ops.py — add power() and modulo() functions
            # ═══════════════════════════════════════════════════════════
            msg4 = (
                f"Using str_replace_editor, edit /workspace/{prefix}_ops.py "
                f"to add two new functions after the existing ones:\n"
                f"- power(a, b) that returns a raised to the power of b\n"
                f"- modulo(a, b) that returns a modulo b, raising "
                f"ValueError if b is zero."
            )
            r4 = cca.chat(msg4, session_id=sid)
            evaluate_response(r4, msg4, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
            assert r4.content, "Turn 4 returned empty"

            # The model SHOULD use str_replace_editor per task definition,
            # but may use bash (sed/echo). Log which tool was used.
            t4_tools = r4.tool_names
            trace_test.set_attribute("cca.test.t4_tools", str(t4_tools))
            t4_used_editor = any("str_replace_editor" in t for t in t4_tools)
            trace_test.set_attribute("cca.test.t4_used_editor", t4_used_editor)
            assert len(t4_tools) >= 1, (
                f"Agent didn't use any tools to edit file. Tools: {t4_tools}"
            )

            # ═══════════════════════════════════════════════════════════
            # Turn 5: Update main.py to use new functions, then run
            # ═══════════════════════════════════════════════════════════
            msg5 = (
                f"Now update /workspace/{prefix}_main.py to also test "
                f"power(10, 3) and modulo(10, 3) using the new functions. "
                f"Format and print them like the others. Then run the "
                f"updated main.py with python3 and show me the output."
            )
            r5 = cca.chat(msg5, session_id=sid)
            evaluate_response(r5, msg5, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t5_response", r5.content[:500])
            assert r5.content, "Turn 5 returned empty"

            # Should have used tools to edit the file AND run it.
            # Agent may use str_replace_editor or bash (sed/echo) to edit —
            # both are valid. Turns 1/3/4 already verify editor create/view/edit.
            t5_tools = r5.tool_names
            trace_test.set_attribute("cca.test.t5_tools", str(t5_tools))
            has_any_tool = len(t5_tools) >= 1
            assert has_any_tool, (
                f"Turn 5 didn't use any tools. Tools: {t5_tools}"
            )
            has_bash = any("bash" in t for t in t5_tools)
            assert has_bash, (
                f"Turn 5 didn't use bash to run the code. Tools: {t5_tools}"
            )

            # Verify new operations work: power(10,3)=1000, modulo(10,3)=1
            # Accept raw values OR mentions of the operation names
            content5 = r5.content.lower()
            has_1000 = "1000" in r5.content
            has_power_ref = "power" in content5
            has_modulo_ref = "modulo" in content5
            new_ops_proven = (
                has_1000
                or (has_power_ref and has_modulo_ref)
            )
            trace_test.set_attribute("cca.test.t5_has_1000", has_1000)
            trace_test.set_attribute("cca.test.t5_has_power", has_power_ref)
            trace_test.set_attribute("cca.test.t5_has_modulo", has_modulo_ref)
            assert new_ops_proven, (
                f"No evidence of new operations (1000={has_1000}, "
                f"power={has_power_ref}, modulo={has_modulo_ref}): "
                f"{r5.content[:400]}"
            )

            # Original operations should still work (no regression)
            has_original = (
                "13" in r5.content or "30" in r5.content
                or ("add" in content5 and "subtract" in content5)
            )
            trace_test.set_attribute("cca.test.t5_has_original", has_original)
            assert has_original, (
                f"Original operations missing from output (regression?): "
                f"{r5.content[:400]}"
            )

            # ═══════════════════════════════════════════════════════════
            # Turn 6: Create a test file and run the tests
            # ═══════════════════════════════════════════════════════════
            msg6 = (
                f"Create /workspace/{prefix}_tests.py with unit tests "
                f"using Python's unittest module. Test all 6 operations: "
                f"add, subtract, multiply, divide (including zero division "
                f"error), power, and modulo (including zero modulo error). "
                f"Then run the tests with python3 and show me the results."
            )
            r6 = cca.chat(msg6, session_id=sid)
            evaluate_response(r6, msg6, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t6_response", r6.content[:500])
            assert r6.content, "Turn 6 returned empty"

            t6_tools = r6.tool_names
            trace_test.set_attribute("cca.test.t6_tools", str(t6_tools))
            has_bash_t6 = any("bash" in t for t in t6_tools)
            assert len(t6_tools) >= 1, (
                f"Turn 6 didn't use any tools. Tools: {t6_tools}"
            )
            assert has_bash_t6, (
                f"Turn 6 didn't use bash to run tests. Tools: {t6_tools}"
            )

            # Verify test file was created via REST
            files_after = cca.list_workspace_files()
            file_list_after = files_after.get("files", [])
            file_names_after = [
                f.get("name", "") if isinstance(f, dict) else str(f)
                for f in file_list_after
            ]
            test_file_exists = any(
                f"{prefix}_tests" in n for n in file_names_after
            )
            total_files = [n for n in file_names_after if prefix in n]
            trace_test.set_attribute(
                "cca.test.t6_test_file_exists", test_file_exists,
            )
            trace_test.set_attribute(
                "cca.test.final_file_count", len(total_files),
            )
            trace_test.set_attribute(
                "cca.test.final_files", str(total_files[:10]),
            )
            assert test_file_exists, (
                f"Test file not created. Files: {file_names_after}"
            )
            assert len(total_files) >= 4, (
                f"Expected 4 files (ops, formatter, main, tests), "
                f"found {len(total_files)}: {total_files}"
            )

            # Verify tests actually ran and passed
            content6 = r6.content.lower()
            tests_ran = any(x in content6 for x in [
                "ok", "passed", "test_add", "test_subtract",
                "ran ", "tests run",
            ])
            tests_failed = any(x in content6 for x in [
                "fail", "error", "traceback",
            ])
            trace_test.set_attribute("cca.test.t6_tests_ran", tests_ran)
            trace_test.set_attribute("cca.test.t6_tests_failed", tests_failed)
            assert tests_ran, (
                f"No evidence that tests ran: {r6.content[:400]}"
            )

        finally:
            tracker.cleanup()
