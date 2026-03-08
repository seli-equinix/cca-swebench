"""Flow test: Codebase search via the CODER route.

Journey: ask about code in the indexed workspace → verify agent uses
search_codebase or search_knowledge tools → verify relevant results.

The CODE_SEARCH tool group (search_codebase, search_knowledge) is on the
CODER route, NOT the SEARCH route. Messages must be phrased as coding
tasks ("I'm working on X, find the code for Y") rather than search-y
phrasing ("Search for X") to avoid SEARCH route classification.

Exercises: search_codebase, search_knowledge (CODE_SEARCH group), CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder]


class TestCodebaseSearch:
    """CODER route: search indexed codebase for functions and patterns."""

    def test_codebase_search(self, cca, trace_test, judge_model):
        """Search the indexed codebase and verify relevant results."""
        tracker = cca.tracker()
        sid = f"test-codesearch-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Ask about code structure (coding task framing) ──
            # The MCP server codebase is indexed — ask about something
            # that definitely exists. Frame as a coding task to route
            # to CODER (which has CODE_SEARCH tools), not SEARCH.
            msg1 = (
                "I need to modify the health check endpoint in our project. "
                "Look through the codebase and find which files implement "
                "health check functions. Show me the relevant code locations."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            # Should have used search tools
            iters = r1.metadata.get("tool_iterations", 0)
            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_iters", iters)
            trace_test.set_attribute("cca.test.t1_route", route)
            assert iters >= 1, (
                f"Agent didn't use search tools (route={route}, iters={iters}). "
                f"Response: {r1.content[:200]}"
            )

            # Response should mention file paths or function names
            content_lower = r1.content.lower()
            has_code_refs = any(w in content_lower for w in [
                ".py", "def ", "health", "endpoint", "function",
            ])
            trace_test.set_attribute("cca.test.has_code_refs", has_code_refs)
            assert has_code_refs, (
                f"Response doesn't reference code: {r1.content[:300]}"
            )

            # ── Turn 2: Ask about a specific technology in the codebase ──
            msg2 = (
                "I also need to understand our Qdrant integration. "
                "Find the files in the codebase that use Qdrant and tell me "
                "what collections they create or reference."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use search tools for Qdrant query (iters={iters2})"
            )

            # Should mention Qdrant-related content
            content_lower = r2.content.lower()
            has_qdrant = "qdrant" in content_lower
            trace_test.set_attribute("cca.test.has_qdrant", has_qdrant)
            assert has_qdrant, (
                f"Response doesn't mention Qdrant: {r2.content[:300]}"
            )

        finally:
            tracker.cleanup()
