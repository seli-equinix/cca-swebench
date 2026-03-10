"""Flow test: Code intelligence — call graph and dependency analysis.

Journey: query function callers/callees → find orphan functions →
analyze file dependencies. All via the CODER route using GRAPH tools.

Exercises: query_call_graph, find_orphan_functions, analyze_dependencies
(GRAPH group), CODER route.

Requires: Memgraph (see config.toml [services]) with indexed codebase.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeIntelligence:
    """CODER route: call graph queries, orphan detection, dependency analysis."""

    def test_call_graph_query(self, cca, trace_test, judge_model):
        """Ask about function call relationships in the codebase."""
        tracker = cca.tracker()
        sid = f"test-graph-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = (
                "Using the code graph, what functions call 'build_user_context'? "
                "Show me the callers and which files they're in."
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.response", r.content[:500])
            assert r.content, "Returned empty"

            iters = r.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.iters", iters)
            assert iters >= 1, (
                f"Agent didn't use graph tools (iters={iters}). "
                f"Response: {r.content[:200]}"
            )

            # Should mention function names or file paths
            content_lower = r.content.lower()
            has_graph_data = any(w in content_lower for w in [
                "caller", "calls", "call_graph", "build_user_context",
                ".py", "function",
            ])
            trace_test.set_attribute("cca.test.has_graph_data", has_graph_data)
            assert has_graph_data, (
                f"Response doesn't contain graph data: {r.content[:300]}"
            )

        finally:
            tracker.cleanup()

    def test_orphan_detection(self, cca, trace_test, judge_model):
        """Ask the agent to find unused/orphan functions."""
        tracker = cca.tracker()
        sid = f"test-orphan-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = (
                "Can you check the code graph for orphan functions — "
                "functions that are defined but never called by anything? "
                "Just show me the top 5 if there are many."
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.response", r.content[:500])
            assert r.content, "Returned empty"

            iters = r.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.iters", iters)
            assert iters >= 1, (
                f"Agent didn't use orphan detection tools (iters={iters})"
            )

            # Should list function names
            content_lower = r.content.lower()
            has_functions = any(w in content_lower for w in [
                "def ", "function", "orphan", "unused", "no caller",
                "never called", ".py",
            ])
            trace_test.set_attribute("cca.test.has_functions", has_functions)
            assert has_functions, (
                f"Response doesn't list orphan functions: {r.content[:300]}"
            )

        finally:
            tracker.cleanup()

    def test_dependency_analysis(self, cca, trace_test, judge_model):
        """Ask about dependencies and impact of changing a file."""
        tracker = cca.tracker()
        sid = f"test-deps-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg = (
                "Analyze the dependencies of the user_context.py file. "
                "What other files depend on it, and what would be impacted "
                "if I changed its main functions?"
            )
            r = cca.chat(msg, session_id=sid)
            evaluate_response(r, msg, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.response", r.content[:500])
            assert r.content, "Returned empty"

            iters = r.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.iters", iters)
            assert iters >= 1, (
                f"Agent didn't use dependency tools (iters={iters})"
            )

            # Should mention files or dependencies
            content_lower = r.content.lower()
            has_deps = any(w in content_lower for w in [
                "depend", "import", "impact", "user_context",
                ".py", "module", "coupling",
            ])
            trace_test.set_attribute("cca.test.has_deps", has_deps)
            assert has_deps, (
                f"Response doesn't discuss dependencies: {r.content[:300]}"
            )

        finally:
            tracker.cleanup()
