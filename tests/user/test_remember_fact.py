"""Tests for the remember_user_fact tool.

Pairs fact storage with a coding task to ensure the agent loop runs
(the expert router short-circuits pure greetings as DIRECT answers).
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.timeout(300)]


class TestRememberFact:
    """remember_user_fact tool — store persistent facts about users."""

    def test_remember_single_fact(self, cca, trace_test, judge_model):
        """Agent should store a fact when told about the user."""
        name = f"FactUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-fact-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}. I work at AcmeCorp. "
            f"Write me a Python one-liner to get today's date."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        assert user is not None, f"User '{name}' not found via /users API"

        cca.cleanup_test_user(name)

    def test_remember_multiple_facts(self, cca, trace_test, judge_model):
        """Agent should handle multiple facts alongside a coding task."""
        name = f"MultiFact_{uuid.uuid4().hex[:6]}"
        session_id = f"test-mfact-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi, I'm {name}. I use Python, my server is gpu-box, "
            f"and I work in machine learning. Help me write a function "
            f"to calculate the mean of a list of numbers."
        )

        result = cca.chat(message, session_id=session_id, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        user = cca.find_user_by_name(name)
        assert user is not None, f"User '{name}' not found via /users API"

        cca.cleanup_test_user(name)

    @pytest.mark.timeout(600)
    def test_fact_recalled_in_conversation(self, cca, trace_test, judge_model):
        """CCA should use stored facts when asked about a user."""
        name = f"Recall_{uuid.uuid4().hex[:6]}"
        company = f"RecallCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-frecl1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-frecl2-{uuid.uuid4().hex[:8]}"

        # Session 1: introduce with distinctive company name + coding task
        cca.chat(
            f"Hi I'm {name}. I'm a DevOps engineer at {company} "
            f"and I mostly work with Kubernetes. Can you write me a "
            f"Python function to check if a port is open?",
            session_id=sid1,
            timeout=240,
        )

        # Session 2: come back and ask about stored info
        message = (
            f"Hey {name} here. Can you remind me what company I work at? "
            f"And also show me how to do a health check in Python."
        )
        result = cca.chat(message, session_id=sid2, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        content_lower = result.content.lower()
        recalled = company.lower() in content_lower
        trace_test.set_attribute("cca.test.company_recalled", recalled)
        assert recalled, \
            f"Agent didn't recall company '{company}': {result.content[:300]}"

        cca.cleanup_test_user(name)

    @pytest.mark.timeout(900)
    def test_fact_overwrite(self, cca, trace_test, judge_model):
        """When a user changes jobs, CCA should update the fact."""
        name = f"Overwrite_{uuid.uuid4().hex[:6]}"
        old_company = f"OldCorp_{uuid.uuid4().hex[:4]}"
        new_company = f"NewCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-fov1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-fov2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-fov3-{uuid.uuid4().hex[:8]}"

        # Session 1: original employer
        cca.chat(
            f"Hi I'm {name}. I work at {old_company}. "
            f"Write me a one-liner to list directory contents.",
            session_id=sid1,
            timeout=240,
        )

        # Session 2: changed jobs
        cca.chat(
            f"Hey {name} here. Actually I switched jobs — I now work "
            f"at {new_company}. Write me a one-liner to check disk usage.",
            session_id=sid2,
            timeout=240,
        )

        # Session 3: ask where they work now
        message = (
            f"Hi {name} again. Where do I work now? "
            f"Also write me a quick Python timestamp function."
        )
        result = cca.chat(message, session_id=sid3, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        content_lower = result.content.lower()
        has_new = new_company.lower() in content_lower
        has_old = old_company.lower() in content_lower
        trace_test.set_attribute("cca.test.has_new_company", has_new)
        trace_test.set_attribute("cca.test.has_old_company", has_old)
        assert has_new, \
            f"Agent didn't mention new company '{new_company}': {result.content[:300]}"

        cca.cleanup_test_user(name)

    def test_fact_persists_across_sessions(self, cca, trace_test, judge_model):
        """Facts stored in one session should be available in the next."""
        name = f"Persist_{uuid.uuid4().hex[:6]}"
        company = f"PersistCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-per1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-per2-{uuid.uuid4().hex[:8]}"

        # Session 1: identify + store fact via coding task
        cca.chat(
            f"Hi I'm {name}. I work at {company}. "
            f"Write a Python one-liner to check if a string is a palindrome.",
            session_id=sid1,
        )

        # Session 2: new session, ask what the agent knows
        message = (
            f"Hey, it's {name}. Where do I work? Also help me with "
            f"a one-liner to count vowels in a string."
        )
        result = cca.chat(message, session_id=sid2)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        content_lower = result.content.lower()
        assert any(w in content_lower for w in [
            company.lower(), "employer", "work", "company",
            name.lower(), "profile",
        ]), f"Agent didn't recall the fact: {result.content[:200]}"

        cca.cleanup_test_user(name)
