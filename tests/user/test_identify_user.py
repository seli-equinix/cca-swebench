"""Tests for user identification via the CCA agent.

The server auto-creates user profiles when it detects a name in the
first message (via smart_identify → identify_user). This happens
BEFORE the expert router, so users are created even for DIRECT answers.

Tests pair introductions with coding tasks to also exercise the agent
loop. Validates via the /users REST API for ground-truth.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.timeout(300)]


class TestIdentifyUser:
    """identify_user tool — session-to-user linking."""

    def test_identify_new_user(self, cca, trace_test, judge_model):
        """A new name + coding task should create a user profile."""
        name = f"NewUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-new-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi, I'm {name}. Help me write a Python one-liner "
            f"that prints 'hello world'."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_created", user is not None)
        assert user is not None, f"User '{name}' not found via /users API"

        cca.cleanup_test_user(name)

    def test_identify_returning_user(self, cca, trace_test, judge_model):
        """Same name in a new session should find the existing profile."""
        name = f"Return_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-ret1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-ret2-{uuid.uuid4().hex[:8]}"

        cca.chat(
            f"Hi, I'm {name}. Write a one-liner to reverse a string.",
            session_id=sid1,
        )

        message = f"Hey, it's {name} again. Now write a one-liner to sort a list."
        result = cca.chat(message, session_id=sid2)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        assert user is not None, f"User '{name}' not found"
        trace_test.set_attribute("cca.test.session_count", user["session_count"])
        assert user["session_count"] >= 2, \
            f"Expected session_count >= 2, got {user['session_count']}"

        cca.cleanup_test_user(name)

    def test_identify_with_greeting(self, cca, trace_test, judge_model):
        """Natural greeting + task should trigger identification."""
        name = f"Greeter_{uuid.uuid4().hex[:6]}"
        session_id = f"test-greet-{uuid.uuid4().hex[:8]}"
        message = (
            f"Good morning! My name is {name}. Can you help me write "
            f"a Python function that checks if a number is even?"
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        assert user is not None, f"User '{name}' not created from greeting"

        cca.cleanup_test_user(name)

    @pytest.mark.timeout(600)
    def test_no_duplicate_on_return(self, cca, trace_test, judge_model):
        """Coming back in a new session shouldn't create a duplicate profile."""
        name = f"NoDup_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-dup1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-dup2-{uuid.uuid4().hex[:8]}"

        # Session 1: first visit
        cca.chat(
            f"Hi, I'm {name}. Can you help me write a quick "
            f"Python script to read a CSV file?",
            session_id=sid1,
            timeout=240,
        )

        # Session 2: come back
        message = (
            f"Hey it's {name} again. I need help parsing JSON this time."
        )
        result = cca.chat(message, session_id=sid2, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        # Verify: exactly ONE user with this name
        users_data = cca.list_users()
        matches = [
            u for u in users_data.get("users", [])
            if u.get("display_name", "").lower() == name.lower()
        ]
        trace_test.set_attribute("cca.test.match_count", len(matches))
        assert len(matches) == 1, \
            f"Expected 1 user named '{name}', found {len(matches)}"

        # Verify: session_count >= 2
        user = matches[0]
        trace_test.set_attribute("cca.test.session_count", user["session_count"])
        assert user["session_count"] >= 2, \
            f"Expected session_count >= 2, got {user['session_count']}"

        cca.cleanup_test_user(name)

    def test_identify_updates_metadata(self, cca, trace_test, judge_model):
        """After identification, context_metadata should show identified."""
        name = f"MetaUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-meta-{uuid.uuid4().hex[:8]}"
        message = f"Hi, I'm {name}. Help me debug this: print('hello'"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:300])
        trace_test.set_attribute("cca.test.user_identified", result.user_identified)

        assert result.content, "Agent returned empty response"
        user = cca.find_user_by_name(name)
        assert result.user_identified or user is not None, \
            "User not identified via metadata or REST API"

        cca.cleanup_test_user(name)
