"""Flow test: Route tool isolation — verify routes respect boundaries.

Journey 1: Search-routed question that asks to create a file →
should NOT use str_replace_editor (SEARCH has no FILE tools).

Journey 2: User-routed question that asks to run a command →
should NOT use bash_executor (USER has no SHELL tools).

Exercises: route classification, tool group boundaries.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestToolIsolation:
    """Route tool boundaries: verify routes only use allowed tool groups."""

    def test_search_route_no_file_tools(self, cca, trace_test, judge_model):
        """Search-routed request asking to save to file → should not use editor.

        The SEARCH route only has: WEB, USER_MEMORY, NOTES, DOCUMENT.
        It should NOT have access to FILE (str_replace_editor) or
        SHELL (bash_executor).
        """
        tracker = cca.tracker()
        sid = f"test-isolation-search-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # This should route to SEARCH (web search request)
            # but also asks to save to a file (which SEARCH can't do)
            msg1 = (
                "Search the web for Python best practices in 2026 and "
                "save the top 5 results to /workspace/best_practices.txt"
            )
            r1 = cca.chat(msg1, session_id=sid)
            # Task asks to save to file, but SEARCH route blocks file tools.
            # Incomplete task completion is the expected/correct outcome.
            evaluate_response(
                r1, msg1, trace_test, judge_model, "websearch",
                expected_incomplete=True,
            )

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_route", route)

            # Check which tools were actually called
            tool_names = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names))

            # If it routed to SEARCH, it should NOT have file tools
            if route.upper() == "SEARCH":
                used_file_tools = any(
                    t in name for name in tool_names
                    for t in ["str_replace_editor", "bash_executor"]
                )
                trace_test.set_attribute(
                    "cca.test.t1_used_file_tools", used_file_tools,
                )
                assert not used_file_tools, (
                    f"SEARCH route used file/shell tools: {tool_names}. "
                    f"These should be restricted to CODER/INFRA routes."
                )

            # If it routed to CODER instead, that's also acceptable
            # (the request is ambiguous — "search" + "save to file")
            if route.upper() in ("CODER", "INFRASTRUCTURE"):
                trace_test.set_attribute(
                    "cca.test.t1_note",
                    f"Routed to {route} (has file tools) — acceptable",
                )

        finally:
            tracker.cleanup()

    def test_user_route_no_bash(self, cca, trace_test, judge_model):
        """User-routed question that asks to run a command → should not use bash.

        The USER route only has: USER, NOTES.
        It should NOT have access to SHELL (bash_executor) or
        FILE (str_replace_editor).
        """
        tracker = cca.tracker()
        sid = f"test-isolation-user-{uuid.uuid4().hex[:8]}"
        unique_id = uuid.uuid4().hex[:6]
        user_name = f"RouteUser_{unique_id}"
        tracker.track_session(sid)
        tracker.track_user(user_name)

        try:
            # Frame as a user introduction that also asks to run a command.
            # The introduction should trigger USER route, which shouldn't
            # have bash access.
            msg1 = (
                f"Hi I'm {user_name}, I'm a software engineer. "
                f"Can you run 'ls /workspace' for me?"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_route", route)

            tool_names = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names))

            # If it routed to USER, bash should not be available
            if route.upper() == "USER":
                used_bash = any("bash" in name for name in tool_names)
                trace_test.set_attribute("cca.test.t1_used_bash", used_bash)
                assert not used_bash, (
                    f"USER route used bash tools: {tool_names}. "
                    f"Bash should only be on CODER/INFRA routes."
                )

            # Track what happened for observability
            trace_test.set_attribute(
                "cca.test.t1_user_identified", r1.user_identified,
            )

        finally:
            tracker.cleanup()
