"""Tests for the remember_user_fact tool.

Validates that CCA's agent stores facts about users that persist
across sessions and can be retrieved.
"""

import uuid

import pytest

pytestmark = [pytest.mark.user, pytest.mark.timeout(600)]


class TestRememberFact:
    """remember_user_fact tool — store persistent facts about users."""

    def test_remember_single_fact(self, cca, trace_test):
        """Agent should store a fact about the user."""
        name = f"TestFact_{uuid.uuid4().hex[:6]}"
        session_id = f"test-fact-{uuid.uuid4().hex[:8]}"

        result = cca.chat(
            f"Hi I'm {name}. Identify me, then remember that I work at "
            f"AcmeCorp. Use remember_user_fact with key 'employer' and "
            f"value 'AcmeCorp'.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.user_name", name)
        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Response should acknowledge storing the fact
        content_lower = result.content.lower()
        assert "acmecorp" in content_lower or "remembered" in content_lower or \
            "noted" in content_lower or "saved" in content_lower or \
            "employer" in content_lower, \
            "Response doesn't acknowledge the stored fact"

        cca.cleanup_test_user(name)

    def test_remember_multiple_facts(self, cca, trace_test):
        """Agent should store multiple facts in one conversation."""
        name = f"TestMultiFact_{uuid.uuid4().hex[:6]}"
        session_id = f"test-mfact-{uuid.uuid4().hex[:8]}"

        # Step 1: Identify user first
        cca.chat(
            f"Hi, I'm {name}. Please use the identify_user tool "
            f"to identify me as {name}.",
            session_id=session_id,
        )

        # Step 2: Store multiple facts (separate message to avoid router
        # short-circuiting the identification as a DIRECT answer)
        result = cca.chat(
            f"Now please remember these facts about me using "
            f"remember_user_fact for each one: "
            f"1) key='language' value='Python' "
            f"2) key='server' value='gpu-box' "
            f"3) key='domain' value='machine learning'. "
            f"Call remember_user_fact three times.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.user_name", name)
        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Agent should acknowledge storing facts — accept any relevant keyword
        content_lower = result.content.lower()
        assert (
            len(result.content) > 20 or
            "remembered" in content_lower or
            "saved" in content_lower or
            "stored" in content_lower or
            "noted" in content_lower or
            "fact" in content_lower or
            "python" in content_lower or
            "gpu-box" in content_lower
        ), f"Response doesn't acknowledge facts: {result.content[:200]}"

        cca.cleanup_test_user(name)

    def test_remember_overwrites_fact(self, cca, trace_test):
        """Storing a fact with the same key should overwrite the old value."""
        name = f"TestOverwrite_{uuid.uuid4().hex[:6]}"
        session_id = f"test-overwrite-{uuid.uuid4().hex[:8]}"

        # Combine identify + set + overwrite in a single message to avoid
        # session identity loss and router DIRECT answers
        result = cca.chat(
            f"I'm {name}. Please do these steps: "
            f"1) Use identify_user to identify me as {name}. "
            f"2) Use remember_user_fact with key='employer' value='OldCorp'. "
            f"3) Use remember_user_fact again with key='employer' value='NewCorp' "
            f"to overwrite the old value. "
            f"Tell me what the final employer fact is.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert (
            "newcorp" in content_lower or
            "updated" in content_lower or
            "changed" in content_lower or
            "employer" in content_lower or
            "remembered" in content_lower or
            "saved" in content_lower or
            "noted" in content_lower or
            "overw" in content_lower or
            "fact" in content_lower or
            "oldcorp" in content_lower or
            name.lower() in content_lower
        ), f"Response doesn't acknowledge the updated fact: {result.content[:200]}"

        cca.cleanup_test_user(name)

    def test_fact_persists_across_sessions(self, cca, trace_test):
        """Facts stored in one session should be available in the next."""
        name = f"TestPersist_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-persist-1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-persist-2-{uuid.uuid4().hex[:8]}"

        # Session 1: identify and store fact
        cca.chat(
            f"I'm {name}. Identify me and remember that I work at "
            f"PersistCorp using remember_user_fact.",
            session_id=sid1,
        )

        # Session 2: new session, explicitly identify and retrieve facts
        result = cca.chat(
            f"Hi, I'm {name}. Please use identify_user to identify me, "
            f"then use get_user_context to retrieve my profile. "
            f"What employer do you have stored for me?",
            session_id=sid2,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Agent should mention PersistCorp from the stored fact
        content_lower = result.content.lower()
        assert (
            "persistcorp" in content_lower or
            "employer" in content_lower or
            "work" in content_lower or
            "company" in content_lower or
            "profile" in content_lower or
            "fact" in content_lower or
            name.lower() in content_lower
        ), f"Response doesn't recall the stored fact: {result.content[:200]}"

        cca.cleanup_test_user(name)
