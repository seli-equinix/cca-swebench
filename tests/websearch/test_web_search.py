"""Tests for the web_search tool.

Validates that CCA's agent can search the web via SearXNG
with various parameters (categories, time_range, engines).
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.websearch, pytest.mark.timeout(600)]


class TestWebSearchBasic:
    """web_search tool — basic search functionality."""

    def test_basic_search(self, cca, trace_test, judge_model):
        """Agent should perform a web search and return results."""
        session_id = f"test-search-{uuid.uuid4().hex[:8]}"
        message = (
            "Search the web for 'Python 3.12 new features' using the "
            "web_search tool. Show me the top results with their URLs."
        )

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert "python" in content_lower, \
            "Response doesn't mention Python"
        has_results = (
            "http" in result.content or
            "result" in content_lower or
            ".org" in result.content or
            ".com" in result.content or
            ".io" in result.content
        )
        trace_test.set_attribute("cca.test.has_results", has_results)
        assert has_results, "Response doesn't contain any search results or URLs"

    def test_search_no_results(self, cca, trace_test, judge_model):
        """Agent should handle empty search results gracefully."""
        session_id = f"test-search-empty-{uuid.uuid4().hex[:8]}"
        nonsense = f"xyzzy_{uuid.uuid4().hex[:12]}_nonexistent"
        message = (
            f"Search the web for '{nonsense}' using web_search. "
            f"Tell me what you found."
        )

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.query", nonsense)
        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        handled = (
            "no result" in content_lower or
            "couldn't find" in content_lower or
            "not find" in content_lower or
            "no match" in content_lower or
            "0 result" in content_lower or
            "nothing" in content_lower or
            len(result.content) > 20
        )
        assert handled, "Agent didn't handle empty results gracefully"


class TestWebSearchAdvanced:
    """web_search tool — advanced parameters."""

    def test_search_with_tech_category(self, cca, trace_test, judge_model):
        """Agent should use IT/tech category for programming topics."""
        session_id = f"test-search-tech-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for 'vLLM inference engine' using web_search. "
            "Use the IT category for better results. Show me what you find."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert "vllm" in content_lower, \
            "Response doesn't mention vLLM"

    def test_search_recent_results(self, cca, trace_test, judge_model):
        """Agent should use time_range for recent results."""
        session_id = f"test-search-recent-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for recent AI news from this month using web_search "
            "with time_range='month'. What's the latest?"
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert "ai" in content_lower or "artificial" in content_lower or \
            "model" in content_lower or "llm" in content_lower, \
            "Response doesn't contain AI-related content"

    @pytest.mark.slow
    def test_parallel_search_comparison(self, cca, trace_test, judge_model):
        """Agent should call multiple searches for a comparison task."""
        session_id = f"test-search-parallel-{uuid.uuid4().hex[:8]}"
        message = (
            "I need to compare vLLM vs TGI for LLM serving. "
            "Search for both using web_search (you can make multiple "
            "parallel calls) and give me a brief comparison."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:800])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        has_vllm = "vllm" in content_lower
        has_tgi = "tgi" in content_lower or "text generation inference" in content_lower
        trace_test.set_attribute("cca.test.mentions_vllm", has_vllm)
        trace_test.set_attribute("cca.test.mentions_tgi", has_tgi)

        assert has_vllm or has_tgi, \
            "Response doesn't mention either vLLM or TGI"
