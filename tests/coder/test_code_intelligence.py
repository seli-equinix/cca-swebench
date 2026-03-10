"""Flow test: Code intelligence — call graph, orphan detection, dependency analysis.

Journey: query function callers → find orphan functions →
analyze file dependencies. A developer exploring a codebase
before making changes.

Exercises: query_call_graph, find_orphan_functions, analyze_dependencies
(GRAPH group), CODER route.

Requires: Memgraph (see config.toml [services]) with indexed codebase.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeIntelligence:
    """CODER route: multi-turn code graph exploration."""

    def test_code_intelligence_flow(self, cca, trace_test, judge_model):
        """3-turn flow: call graph → orphans → dependencies."""
        tracker = cca.tracker()
        sid = f"test-graph-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Query the call graph ──
            # Developer wants to understand who calls a function before modifying it
            msg1 = (
                "Using the code graph, what functions call 'build_user_context'? "
                "Show me the callers and which files they're in."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            assert iters1 >= 1, (
                f"Agent didn't use graph tools (iters={iters1}). "
                f"Response: {r1.content[:200]}"
            )

            # Should mention function names or file paths
            content1 = r1.content.lower()
            has_graph_data = any(w in content1 for w in [
                "caller", "calls", "call_graph", "build_user_context",
                ".py", "function",
            ])
            trace_test.set_attribute("cca.test.t1_has_graph_data", has_graph_data)
            assert has_graph_data, (
                f"Response doesn't contain graph data: {r1.content[:300]}"
            )

            # ── Turn 2: Find orphan functions ──
            # Natural follow-up — now looking for dead code
            msg2 = (
                "Can you check the code graph for orphan functions — "
                "functions that are defined but never called by anything? "
                "Just show me the top 5 if there are many."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use orphan detection tools (iters={iters2})"
            )

            # Should list function names
            content2 = r2.content.lower()
            has_functions = any(w in content2 for w in [
                "def ", "function", "orphan", "unused", "no caller",
                "never called", ".py",
            ])
            trace_test.set_attribute("cca.test.t2_has_functions", has_functions)
            assert has_functions, (
                f"Response doesn't list orphan functions: {r2.content[:300]}"
            )

            # ── Turn 3: Analyze dependencies ──
            # Developer wants to know impact before changing a file
            msg3 = (
                "Analyze the dependencies of the user_context.py file. "
                "What other files depend on it, and what would be impacted "
                "if I changed its main functions?"
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't use dependency tools (iters={iters3})"
            )

            # Should mention files or dependencies
            content3 = r3.content.lower()
            has_deps = any(w in content3 for w in [
                "depend", "import", "impact", "user_context",
                ".py", "module", "coupling",
            ])
            trace_test.set_attribute("cca.test.t3_has_deps", has_deps)
            assert has_deps, (
                f"Response doesn't discuss dependencies: {r3.content[:300]}"
            )

        finally:
            tracker.cleanup()
