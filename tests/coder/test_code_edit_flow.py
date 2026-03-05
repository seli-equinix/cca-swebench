"""Flow test: Write code, edit it, then run it to verify it works.

Journey: ask agent to create a Python file → edit it to add a function →
run the file with python3 to verify output → confirm both functions work.

This is the core developer workflow: write → edit → run → verify.

Exercises: str_replace_editor (create, str_replace), bash_tool (python3),
CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeEditFlow:
    """CODER route: write code, edit it, run it, verify output."""

    def test_code_edit_flow(self, cca, trace_test, judge_model):
        """Full dev workflow: create → edit → run → verify output."""
        filename = f"test_edit_{uuid.uuid4().hex[:6]}.py"
        sid = f"test-edit-{uuid.uuid4().hex[:8]}"

        try:
            # ── Turn 1: Create a Python file with a function ──
            msg1 = (
                f"Create a Python file called {filename} in /workspace "
                f"with a function called greet(name) that returns "
                f"'Hello, {{name}}!'. Add a main block that prints "
                f"greet('World')."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"

            iters = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters)
            assert iters >= 1, (
                f"Agent didn't use tools to create file (iters={iters})"
            )

            # Verify file exists via REST
            files = cca.list_workspace_files()
            file_list = files.get("files", [])
            file_names = [
                f.get("name", "") if isinstance(f, dict) else str(f)
                for f in file_list
            ]
            has_file = any(filename in name for name in file_names)
            trace_test.set_attribute("cca.test.file_created", has_file)
            assert has_file, (
                f"File '{filename}' not found in workspace. "
                f"Files: {file_names[:10]}"
            )

            # ── Turn 2: Edit the file — add a second function ──
            msg2 = (
                f"Add a function called farewell(name) to {filename} "
                f"that returns 'Goodbye, {{name}}!'. Also update the "
                f"main block to print farewell('World') after greet."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use tools to edit file (iters={iters2})"
            )

            # ── Turn 3: Run the file and verify output ──
            msg3 = (
                f"Now run /workspace/{filename} with python3 and "
                f"show me the output. Did both functions work?"
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't run the file (iters={iters3})"
            )

            # The output should contain both greeting and farewell
            content_lower = r3.content.lower()
            has_hello = "hello" in content_lower
            has_goodbye = "goodbye" in content_lower
            trace_test.set_attribute("cca.test.has_hello", has_hello)
            trace_test.set_attribute("cca.test.has_goodbye", has_goodbye)
            assert has_hello, (
                f"Output doesn't contain 'Hello': {r3.content[:300]}"
            )
            assert has_goodbye, (
                f"Output doesn't contain 'Goodbye': {r3.content[:300]}"
            )

        finally:
            cca.clean_workspace_files(prefix=filename.replace(".py", ""))
