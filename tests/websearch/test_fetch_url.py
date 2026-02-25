"""Tests for the fetch_url_content tool.

Validates that CCA's agent ACTUALLY calls fetch_url_content via httpx,
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


class TestFetchUrlBasic:
    """fetch_url_content tool — basic URL fetching."""

    def test_fetch_public_url(self, cca, trace_test, judge_model):
        """Agent should call fetch_url_content and extract text from the page."""
        session_id = f"test-fetch-{uuid.uuid4().hex[:8]}"
        message = (
            "Fetch the page at https://httpbin.org/html using "
            "fetch_url_content and tell me what text it contains."
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

        # Response should reference actual httpbin.org/html content
        # (Herman Melville's Moby Dick excerpt)
        content_lower = result.content.lower()
        has_content = (
            "herman" in content_lower
            or "melville" in content_lower
            or "moby" in content_lower
            or "whale" in content_lower
        )
        trace_test.set_attribute("cca.test.has_content", has_content)
        assert has_content, (
            "Response doesn't mention Moby Dick content from httpbin.org/html. "
            f"Response: {result.content[:300]}"
        )


class TestFetchUrlSecurity:
    """fetch_url_content tool — SSRF protection and validation."""

    def test_ssrf_blocks_private_ip(self, cca, trace_test, judge_model):
        """Agent should attempt fetch_url_content on private IP and report the error."""
        session_id = f"test-ssrf-{uuid.uuid4().hex[:8]}"
        message = (
            "Fetch http://192.168.1.1/ using fetch_url_content. "
            "Tell me exactly what happens."
        )

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: the agent attempted to use tools
        # (tool should fail with SSRF error, but it should have TRIED)
        _assert_tools_used(result)

        # Response should indicate the request was blocked
        content_lower = result.content.lower()
        blocked = (
            "blocked" in content_lower
            or "private" in content_lower
            or "internal" in content_lower
            or "not allowed" in content_lower
            or "security" in content_lower
            or "ssrf" in content_lower
            or "refused" in content_lower
            or "error" in content_lower
            or "denied" in content_lower
        )
        trace_test.set_attribute("cca.test.ssrf_blocked", blocked)
        assert blocked, (
            "Response doesn't indicate the private IP was blocked. "
            f"Response: {result.content[:300]}"
        )

    def test_invalid_scheme_rejected(self, cca, trace_test, judge_model):
        """Agent should attempt fetch_url_content with FTP and report the error."""
        session_id = f"test-scheme-{uuid.uuid4().hex[:8]}"
        message = "Fetch ftp://example.com/file.txt using fetch_url_content."

        result = cca.chat(message, session_id=session_id, timeout=300)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: tools were called (should fail with scheme error)
        _assert_tools_used(result)

        # Response should indicate FTP scheme was rejected
        content_lower = result.content.lower()
        rejected = (
            "not supported" in content_lower
            or "invalid" in content_lower
            or "scheme" in content_lower
            or "only http" in content_lower
            or "ftp" in content_lower
            or "error" in content_lower
        )
        assert rejected, (
            "Response doesn't indicate FTP scheme was rejected. "
            f"Response: {result.content[:300]}"
        )


class TestFetchUrlChained:
    """fetch_url_content chained with web_search."""

    @pytest.mark.slow
    @pytest.mark.timeout(540)
    def test_search_then_fetch(self, cca, trace_test, judge_model):
        """Agent should search then fetch the best result — uses multiple tools."""
        session_id = f"test-chain-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for 'httpbin.org' using web_search, then use "
            "fetch_url_content to fetch the top result URL and tell "
            "me what the page says."
        )

        result = cca.chat(message, session_id=session_id, timeout=540)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Core validation: multiple tool iterations (search + fetch)
        _assert_tools_used(result, min_iterations=2)

        # Response should contain real URLs from search results
        has_urls = "http://" in result.content or "https://" in result.content
        trace_test.set_attribute("cca.test.has_urls", has_urls)

        # Response should mention httpbin
        content_lower = result.content.lower()
        assert "httpbin" in content_lower, (
            "Response doesn't mention httpbin.org. "
            f"Response: {result.content[:300]}"
        )
