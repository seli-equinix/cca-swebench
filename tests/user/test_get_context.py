"""Tests for session context and user identification state.

Validates that user identification persists across messages in the
same session, and that anonymous sessions stay anonymous.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.timeout(300)]


class TestGetContext:
    """Session context — identification state across messages."""

    def test_context_identified_user(self, cca, trace_test, judge_model):
        """After identification, the session should remain identified."""
        name = f"CtxUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-ctx-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}. I work at ContextCorp. "
            f"Write a Python one-liner to get the current timestamp."
        )

        r1 = cca.chat(message, session_id=session_id)

        evaluate_response(r1, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.r1_response", r1.content[:300])

        # User should be auto-created
        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_created", user is not None)
        assert user is not None, f"User '{name}' not created after first message"

        # Second message: same session — should still be identified
        msg2 = "Now write a Python function to calculate fibonacci numbers."
        r2 = cca.chat(msg2, session_id=session_id)

        evaluate_response(r2, msg2, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.r2_response", r2.content[:300])
        assert r2.content, "Second message returned empty response"

        # Session should show user as identified in metadata
        trace_test.set_attribute("cca.test.user_identified", r2.user_identified)
        assert r2.user_identified, \
            "Session should remain identified across messages"

        cca.cleanup_test_user(name)

    def test_context_anonymous(self, cca, trace_test, judge_model):
        """Anonymous session should not show identification."""
        session_id = f"test-anon-{uuid.uuid4().hex[:8]}"
        message = "What is the capital of France?"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        # Should not be identified
        trace_test.set_attribute("cca.test.user_identified", result.user_identified)
        assert not result.user_identified, \
            "Anonymous session should not be identified"

    @pytest.mark.timeout(600)
    def test_context_shows_stored_facts(self, cca, trace_test, judge_model):
        """After storing facts, asking about them should surface the info."""
        name = f"CtxFacts_{uuid.uuid4().hex[:6]}"
        company = f"CtxCorp_{uuid.uuid4().hex[:4]}"
        session_id = f"test-ctxf-{uuid.uuid4().hex[:8]}"

        # First message: introduce with lots of context
        cca.chat(
            f"Hi I'm {name}. I'm a backend engineer at {company}, "
            f"I work on the search team, and my main language is "
            f"Rust. Help me write a Python function to parse a URL.",
            session_id=session_id,
            timeout=240,
        )

        # Same session: ask what the agent knows
        message = "What do you know about me? Give me a summary."
        result = cca.chat(message, session_id=session_id, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        content_lower = result.content.lower()
        matches = sum(1 for term in [
            company.lower(), "backend", "search", "rust",
        ] if term in content_lower)
        trace_test.set_attribute("cca.test.context_terms_found", matches)
        assert matches >= 2, \
            f"Expected at least 2 stored context terms, found {matches}: {result.content[:300]}"

        cca.cleanup_test_user(name)

    @pytest.mark.timeout(600)
    def test_context_enriches_responses(self, cca, trace_test, judge_model):
        """CCA should use stored context to give more relevant answers."""
        name = f"CtxEnrich_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-ctxe1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-ctxe2-{uuid.uuid4().hex[:8]}"

        # Session 1: tell CCA about your infrastructure
        cca.chat(
            f"Hi I'm {name}. I work on Docker Swarm and manage "
            f"a cluster with 5 nodes. My main tool is Portainer. "
            f"Write me a Python function to check container health.",
            session_id=sid1,
            timeout=240,
        )

        # Session 2: ask a vague question — should be enriched by context
        message = (
            f"Hey {name}. I'm having issues with my cluster — "
            f"what should I check first?"
        )
        result = cca.chat(message, session_id=sid2, timeout=240)

        evaluate_response(result, message, trace_test, judge_model, "user")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"

        content_lower = result.content.lower()
        context_used = any(term in content_lower for term in [
            "docker", "swarm", "portainer", "node", "cluster",
            "container", "service",
        ])
        trace_test.set_attribute("cca.test.context_used", context_used)
        assert context_used, \
            f"Response doesn't reference stored infrastructure context: {result.content[:300]}"

        cca.cleanup_test_user(name)
