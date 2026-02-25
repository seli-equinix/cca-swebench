"""Tests for web search capability.

Validates that CCA actually searches the web when asked questions that
require current information. Tests use natural human language — the way
a real person would ask, not robotic tool-name prompts.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.websearch, pytest.mark.timeout(600)]


def _assert_tools_used(result, min_iterations=1):
    """Assert the agent actually called tools (not just answered from memory)."""
    meta = result.metadata
    tool_iters = meta.get("tool_iterations", 0)
    assert tool_iters >= min_iterations, (
        f"Agent answered from memory without using tools "
        f"(tool_iterations={tool_iters}, expected >={min_iterations}). "
        f"Response: {result.content[:200]}"
    )


def _has_real_urls(text):
    """Check if response contains real URLs (http:// or https://)."""
    return "http://" in text or "https://" in text


class TestWebSearchBasic:
    """Basic web search — agent should search and return real results."""

    def test_basic_search(self, cca, trace_test, judge_model):
        """Ask about something that requires current web info."""
        session_id = f"test-search-{uuid.uuid4().hex[:8]}"
        message = "What are the main new features in Python 3.13? Show me the sources."

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content)
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)
        assert _has_real_urls(result.content), (
            "Response has no real URLs — agent likely answered from memory. "
            f"Response: {result.content[:300]}"
        )

    def test_search_no_results(self, cca, trace_test, judge_model):
        """Ask about something completely made up — should still try searching."""
        session_id = f"test-search-empty-{uuid.uuid4().hex[:8]}"
        nonsense = f"xyzzy_{uuid.uuid4().hex[:12]}_nonexistent"
        message = f"Can you find any information about '{nonsense}' online?"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.query", nonsense)
        trace_test.set_attribute("cca.test.response", result.content)
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        content_lower = result.content.lower()
        handled = (
            "no result" in content_lower
            or "couldn't find" in content_lower
            or "not find" in content_lower
            or "no match" in content_lower
            or "0 result" in content_lower
            or "nothing" in content_lower
            or "unable" in content_lower
            or nonsense.lower() in content_lower
        )
        assert handled, (
            f"Agent didn't acknowledge empty results: {result.content[:200]}"
        )


class TestWebSearchAdvanced:
    """More complex search scenarios a real user would ask."""

    def test_search_tech_topic(self, cca, trace_test, judge_model):
        """Ask about a specific technology — natural question."""
        session_id = f"test-search-tech-{uuid.uuid4().hex[:8]}"
        message = "What's the latest version of vLLM and what are its key features?"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content)
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        content_lower = result.content.lower()
        assert "vllm" in content_lower, "Response doesn't mention vLLM"

    def test_search_recent_news(self, cca, trace_test, judge_model):
        """Ask for recent news — should search and find current results."""
        session_id = f"test-search-recent-{uuid.uuid4().hex[:8]}"
        message = "What's the biggest AI news this week? Give me links."

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content)
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)
        assert _has_real_urls(result.content), (
            "Response has no real URLs from web search"
        )

    @pytest.mark.slow
    def test_comparison_search(self, cca, trace_test, judge_model):
        """Ask to compare two things — should search for both."""
        session_id = f"test-search-compare-{uuid.uuid4().hex[:8]}"
        message = (
            "I'm trying to decide between vLLM and TGI for serving LLMs. "
            "What are the pros and cons of each? Please look it up."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content)
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result, min_iterations=1)

        content_lower = result.content.lower()
        has_vllm = "vllm" in content_lower
        has_tgi = (
            "tgi" in content_lower
            or "text generation inference" in content_lower
        )
        trace_test.set_attribute("cca.test.mentions_vllm", has_vllm)
        trace_test.set_attribute("cca.test.mentions_tgi", has_tgi)

        assert has_vllm or has_tgi, (
            "Response doesn't mention either vLLM or TGI"
        )
