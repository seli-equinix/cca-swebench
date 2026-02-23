"""Tests for the update_user_preference tool.

Pairs preference setting with a coding task to ensure the agent loop runs.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.timeout(300)]


class TestUserPreference:
    """update_user_preference tool — adjust response style per user."""

    def test_set_preference(self, cca, trace_test, judge_model):
        """Agent should store a user preference during a coding task."""
        name = f"PrefUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-pref-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}. I prefer concise responses. "
            f"Write a Python function to check if a number is prime."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        assert user is not None, f"User '{name}' not found via /users API"

        cca.cleanup_test_user(name)

    @pytest.mark.timeout(600)
    def test_preference_recalled_next_session(self, cca, trace_test, judge_model):
        """Preferences should persist and be available in the next session."""
        name = f"PrefRecall_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-prec1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-prec2-{uuid.uuid4().hex[:8]}"

        # Session 1: set a preference
        cca.chat(
            f"Hi I'm {name}. I always want code examples in Python "
            f"with type hints. Can you write a function that "
            f"reverses a string?",
            session_id=sid1,
            timeout=240,
        )

        # Session 2: come back, ask for code
        message = (
            f"Hey {name} again. Write me a function to find "
            f"duplicates in a list."
        )
        result = cca.chat(message, session_id=sid2, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        # Agent should produce code (validates it's still functional)
        content_lower = result.content.lower()
        has_code = "def " in content_lower or "```" in result.content
        trace_test.set_attribute("cca.test.has_code", has_code)
        assert has_code, \
            f"Expected code in response: {result.content[:300]}"

        cca.cleanup_test_user(name)

    def test_preference_acknowledged(self, cca, trace_test, judge_model):
        """Agent should respond with code when given a coding preference."""
        name = f"PrefAck_{uuid.uuid4().hex[:6]}"
        session_id = f"test-ack-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi, I'm {name}. I prefer Python code with type hints. "
            f"Write a function to calculate factorial."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        assert len(result.content) > 50, "Response too short for a code task"

        cca.cleanup_test_user(name)
