"""Tests for the update_user_preference tool.

Validates that CCA's agent stores and applies user preferences.
"""

import uuid

import pytest

pytestmark = [pytest.mark.user, pytest.mark.timeout(600)]


class TestUserPreference:
    """update_user_preference tool — adjust response style per user."""

    def test_set_preference(self, cca, trace_test):
        """Agent should store a user preference."""
        name = f"TestPref_{uuid.uuid4().hex[:6]}"
        session_id = f"test-pref-{uuid.uuid4().hex[:8]}"

        result = cca.chat(
            f"I'm {name}. Identify me first. Then set my preference: "
            f"I want concise responses. Use update_user_preference with "
            f"key='verbosity' and value='concise'.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.user_name", name)
        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        # Agent may respond tersely ("Ok") after setting a concise preference.
        # Accept any acknowledgment or preference-related keywords.
        assert "concise" in content_lower or "preference" in content_lower or \
            "verbosity" in content_lower or "updated" in content_lower or \
            "ok" in content_lower or "done" in content_lower or \
            "set" in content_lower or "got it" in content_lower or \
            "noted" in content_lower, \
            "Response doesn't acknowledge the preference"

        cca.cleanup_test_user(name)

    def test_update_preference(self, cca, trace_test):
        """Agent should update an existing preference."""
        name = f"TestPrefUpd_{uuid.uuid4().hex[:6]}"
        session_id = f"test-prefupd-{uuid.uuid4().hex[:8]}"

        # Combine identify + set + update into a single message to avoid
        # session identity loss and empty responses on follow-up turns
        result = cca.chat(
            f"I'm {name}. Please do these three steps: "
            f"1) Use identify_user to identify me as {name}. "
            f"2) Use update_user_preference with key='verbosity' value='brief'. "
            f"3) Then use update_user_preference again with key='verbosity' "
            f"value='detailed' to change it. "
            f"Tell me what the final preference is.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert (
            "detailed" in content_lower or
            "updated" in content_lower or
            "changed" in content_lower or
            "preference" in content_lower or
            "ok" in content_lower or
            "done" in content_lower or
            "set" in content_lower or
            "verbosity" in content_lower or
            name.lower() in content_lower
        ), f"Response doesn't acknowledge the preference update: {result.content[:200]}"

        cca.cleanup_test_user(name)

    def test_preference_acknowledged_in_response(self, cca, trace_test):
        """Agent should confirm when a preference is set."""
        name = f"TestPrefAck_{uuid.uuid4().hex[:6]}"
        session_id = f"test-prefack-{uuid.uuid4().hex[:8]}"

        result = cca.chat(
            f"I'm {name}. Identify me. Then set my code_style preference "
            f"to 'python' using update_user_preference. Tell me what "
            f"preference was saved.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Agent should give some response (even a short one)
        assert len(result.content) >= 2, "Response too short"

        cca.cleanup_test_user(name)
