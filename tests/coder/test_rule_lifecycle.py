"""Flow test: Rules lifecycle — create, list, request, delete.

Journey: create a coding behavior rule → list+verify it → delete it.
All turns must have explicit coding context so the router selects CODER
(which has the RULES tools), not USER or DIRECT.

Exercises: create_rule, list_rules, request_rule, delete_rule
(RULES group), CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestRuleLifecycle:
    """CODER route: full CRUD lifecycle for behavior rules."""

    def test_rule_lifecycle(self, cca, trace_test, judge_model):
        """Create → list+verify → delete a coding rule."""
        sid = f"test-rule-{uuid.uuid4().hex[:8]}"
        rule_name = f"test-rule-{uuid.uuid4().hex[:6]}"

        # ── Turn 1: Create a coding rule ──
        # Frame as a coding workflow so the router selects CODER.
        msg1 = (
            f"I'm setting up coding standards for our Python project. "
            f"Create a new coding rule called '{rule_name}' with type "
            f"'manual' that says: 'Always use type hints in Python "
            f"function signatures. Include return type annotations.' "
            f"Set the description to 'Python type hint enforcement'."
        )
        r1 = cca.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
        route1 = r1.metadata.get("route", "")
        trace_test.set_attribute("cca.test.t1_route", route1)
        assert r1.content, "Turn 1 returned empty"

        iters = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters)
        assert iters >= 1, (
            f"Agent didn't use tools to create rule "
            f"(route={route1}, iters={iters})"
        )

        # ── Turn 2: List all coding rules, verify ours, then delete it ──
        # Combine list + delete to reduce routing risk on follow-ups.
        msg2 = (
            f"List all the coding rules in the system to confirm "
            f"'{rule_name}' was created, then delete it — I was just "
            f"testing the rules setup. Show me the rule details before "
            f"you remove it."
        )
        r2 = cca.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        route2 = r2.metadata.get("route", "")
        trace_test.set_attribute("cca.test.t2_route", route2)
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use rule tools "
            f"(route={route2}, iters={iters2})"
        )

        # Should have listed the rule and confirmed deletion
        content_lower = r2.content.lower()
        has_rule = rule_name in content_lower or "type hint" in content_lower
        has_delete = any(w in content_lower for w in [
            "deleted", "removed", "delete",
        ])
        trace_test.set_attribute("cca.test.rule_found", has_rule)
        trace_test.set_attribute("cca.test.delete_confirmed", has_delete)
        assert has_rule, (
            f"Rule '{rule_name}' not found: {r2.content[:300]}"
        )
