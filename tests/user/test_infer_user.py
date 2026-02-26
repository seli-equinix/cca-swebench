"""Tests for the infer_user tool.

Validates semantic user matching — the agent's ability to identify
a user from contextual clues without an explicit name introduction.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user]


class TestInferUser:
    """infer_user tool — semantic matching via embeddings."""

    def test_infer_known_user(self, cca, trace_test, judge_model):
        """Agent should recognize a known user from a returning intro."""
        name = f"InferKnown_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-inf1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-inf2-{uuid.uuid4().hex[:8]}"

        # Session 1: create user with distinctive info
        cca.chat(
            f"Hi I'm {name}. I work on the vLLM server. "
            f"Write a Python one-liner to read a JSON file.",
            session_id=sid1,
        )

        # Session 2: return with name + task
        message = (
            f"Hey it's {name} again, the vLLM person. "
            f"Help me write a function to parse YAML."
        )
        result = cca.chat(message, session_id=sid2)

        evaluate_response(result, message, trace_test, judge_model, "user")

        tool_iters = result.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute("cca.test.tool_iterations", tool_iters)
        assert result.content, "Agent returned empty response"

        # Coding task should have used tools (bash, file edit, etc.)
        assert tool_iters >= 1, (
            f"Coding task should trigger tool use but got "
            f"tool_iterations={tool_iters}. "
            f"Response: {result.content[:200]}"
        )

        # Response should not contain raw tool_call XML
        assert "<tool_call>" not in result.content, (
            "Response contains raw <tool_call> XML — tools leaked into "
            f"text instead of being executed. Response: {result.content[:300]}"
        )

        # User should exist (not duplicated)
        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_found", user is not None)
        assert user is not None, f"User '{name}' not found"

        cca.cleanup_test_user(name)

    def test_infer_no_match(self, cca, trace_test, judge_model):
        """Generic message with no user clues should not auto-identify."""
        session_id = f"test-noinf-{uuid.uuid4().hex[:8]}"
        message = "What is the capital of France?"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        # Should not claim identification
        trace_test.set_attribute("cca.test.user_identified", result.user_identified)
        assert not result.user_identified, \
            "Anonymous question should not trigger identification"
