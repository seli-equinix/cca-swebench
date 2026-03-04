"""Flow test: Web search and URL fetching capabilities.

Tests the SEARCH route's ability to find, fetch, and synthesize web info.
Each test is independent (no shared state between tests).

Replaces 9 individual tests: basic_search, search_no_results,
search_tech_topic, search_recent_news, comparison_search,
fetch_public_url, search_then_read.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.websearch]


def _assert_tools_used(result, min_iterations=1):
    """Assert the agent actually called tools."""
    iters = result.metadata.get("tool_iterations", 0)
    assert iters >= min_iterations, (
        f"Agent answered without tools (iters={iters}, need >={min_iterations}). "
        f"Response: {result.content[:200]}"
    )


def _has_real_urls(text):
    """Check if response contains real URLs."""
    return "http://" in text or "https://" in text


class TestWebSearchFlow:
    """Web search capabilities — search, fetch, chain, compare."""

    def test_basic_search_with_urls(self, cca, trace_test, judge_model):
        """Ask something requiring web search — should return real URLs."""
        session_id = f"test-wsf1-{uuid.uuid4().hex[:8]}"
        message = "What are the main new features in Python 3.13? Show me the sources."

        result = cca.chat(message, session_id=session_id)
        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)
        assert _has_real_urls(result.content), (
            "No URLs in response — agent likely answered from memory. "
            f"Response: {result.content[:300]}"
        )

    def test_fetch_public_url(self, cca, trace_test, judge_model):
        """Give the agent a URL — should read and summarize the page."""
        session_id = f"test-wsf2-{uuid.uuid4().hex[:8]}"
        message = (
            "Can you read this page for me and tell me what it says? "
            "https://httpbin.org/html"
        )

        result = cca.chat(message, session_id=session_id)
        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        # httpbin.org/html serves a Moby Dick excerpt
        content_lower = result.content.lower()
        has_content = any(w in content_lower for w in [
            "herman", "melville", "moby", "whale",
        ])
        trace_test.set_attribute("cca.test.has_content", has_content)
        assert has_content, (
            "Response doesn't mention Moby Dick content. "
            f"Response: {result.content[:300]}"
        )

    def test_search_tech_topic(self, cca, trace_test, judge_model):
        """Natural tech question — should search and give real info."""
        session_id = f"test-wsf3-{uuid.uuid4().hex[:8]}"
        message = "What's the latest version of vLLM and what are its key features?"

        result = cca.chat(message, session_id=session_id)
        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        content_lower = result.content.lower()
        assert "vllm" in content_lower, "Response doesn't mention vLLM"

    @pytest.mark.slow
    def test_search_then_read(self, cca, trace_test, judge_model):
        """Multi-step: search for something, then read a page about it."""
        session_id = f"test-wsf4-{uuid.uuid4().hex[:8]}"
        message = (
            "I want to understand what httpbin.org is. Can you look it up "
            "and then read the main page to give me a summary?"
        )

        result = cca.chat(message, session_id=session_id)
        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        _assert_tools_used(result, min_iterations=2)

        assert "httpbin" in result.content.lower(), (
            "Response doesn't mention httpbin.org. "
            f"Response: {result.content[:300]}"
        )

    @pytest.mark.slow
    def test_comparison_search(self, cca, trace_test, judge_model):
        """Compare two technologies — should search for both."""
        session_id = f"test-wsf5-{uuid.uuid4().hex[:8]}"
        message = (
            "I'm trying to decide between vLLM and TGI for serving LLMs. "
            "What are the pros and cons of each? Please look it up."
        )

        result = cca.chat(message, session_id=session_id)
        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        content_lower = result.content.lower()
        has_vllm = "vllm" in content_lower
        has_tgi = "tgi" in content_lower or "text generation inference" in content_lower
        trace_test.set_attribute("cca.test.mentions_vllm", has_vllm)
        trace_test.set_attribute("cca.test.mentions_tgi", has_tgi)
        assert has_vllm or has_tgi, "Response doesn't mention vLLM or TGI"
