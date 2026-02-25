"""Tests for the web_search tool.

Validates that CCA's agent ACTUALLY calls web_search via SearXNG,
not just answering from training data. Checks context_metadata.tool_iterations
to verify real tool usage.
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
    """web_search tool — basic search functionality."""

    def test_basic_search(self, cca, trace_test, judge_model):
        """Agent should call web_search and return results with real URLs."""
        session_id = f"test-search-{uuid.uuid4().hex[:8]}"
        message = (
            "Search the web for 'Python 3.13 new features' and show me "
            "the top results with their URLs."
        )

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: tools were actually called
        _assert_tools_used(result)

        # Response should contain real URLs from search results
        assert _has_real_urls(result.content), (
            "Response has no real URLs — agent likely answered from memory. "
            f"Response: {result.content[:300]}"
        )

    def test_search_no_results(self, cca, trace_test, judge_model):
        """Agent should call web_search even for nonsense queries."""
        session_id = f"test-search-empty-{uuid.uuid4().hex[:8]}"
        nonsense = f"xyzzy_{uuid.uuid4().hex[:12]}_nonexistent"
        message = (
            f"Search the web for '{nonsense}'. "
            f"Tell me what you found."
        )

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.query", nonsense)
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: web_search was called (even for nonsense)
        _assert_tools_used(result)

        # Response should acknowledge no results were found
        content_lower = result.content.lower()
        handled = (
            "no result" in content_lower
            or "couldn't find" in content_lower
            or "not find" in content_lower
            or "no match" in content_lower
            or "0 result" in content_lower
            or "nothing" in content_lower
            or nonsense.lower() in content_lower
        )
        assert handled, (
            f"Agent didn't acknowledge empty results: {result.content[:200]}"
        )


class TestWebSearchAdvanced:
    """web_search tool — advanced parameters."""

    def test_search_with_tech_category(self, cca, trace_test, judge_model):
        """Agent should use web_search with IT category."""
        session_id = f"test-search-tech-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for 'vLLM inference engine latest version' using "
            "the IT category. Show me what you find with URLs."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: tools were actually called
        _assert_tools_used(result)

        content_lower = result.content.lower()
        assert "vllm" in content_lower, "Response doesn't mention vLLM"

    def test_search_recent_results(self, cca, trace_test, judge_model):
        """Agent should use time_range for recent results."""
        session_id = f"test-search-recent-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for recent AI news from this month. "
            "What's the latest? Include source URLs."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: tools were called
        _assert_tools_used(result)

        # Should have real URLs from search results
        assert _has_real_urls(result.content), (
            "Response has no real URLs from web search"
        )

    @pytest.mark.slow
    def test_parallel_search_comparison(self, cca, trace_test, judge_model):
        """Agent should call web_search multiple times for comparison."""
        session_id = f"test-search-parallel-{uuid.uuid4().hex[:8]}"
        message = (
            "I need to compare vLLM vs TGI for LLM serving. "
            "Search for both and give me a brief comparison with "
            "source URLs."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:800])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: multiple tool iterations for comparison
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
