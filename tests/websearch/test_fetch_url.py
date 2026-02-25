"""Tests for URL fetching capability.

Validates that CCA can fetch and read web pages when asked. Tests use
natural human language — the way a real person would ask.
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
    """Basic URL fetching — agent should read pages when given a link."""

    def test_fetch_public_url(self, cca, trace_test, judge_model):
        """Give the agent a URL and ask what's on the page."""
        session_id = f"test-fetch-{uuid.uuid4().hex[:8]}"
        message = (
            "Can you read this page for me and tell me what it says? "
            "https://httpbin.org/html"
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

        # httpbin.org/html serves a Moby Dick excerpt
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
    """Security — agent should handle bad URLs gracefully."""

    def test_ssrf_blocks_private_ip(self, cca, trace_test, judge_model):
        """Ask the agent to read a private IP — should be blocked."""
        session_id = f"test-ssrf-{uuid.uuid4().hex[:8]}"
        message = "Hey, can you check what's running at http://192.168.1.1/ for me?"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result)

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
            or "can't access" in content_lower
            or "unable" in content_lower
        )
        trace_test.set_attribute("cca.test.ssrf_blocked", blocked)
        assert blocked, (
            "Response doesn't indicate the private IP was blocked. "
            f"Response: {result.content[:300]}"
        )

    def test_invalid_scheme_rejected(self, cca, trace_test, judge_model):
        """Ask the agent to fetch an FTP link — should explain it can't."""
        session_id = f"test-scheme-{uuid.uuid4().hex[:8]}"
        message = "Can you download this file for me? ftp://example.com/file.txt"

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"

        # Agent might use tools or might know FTP isn't supported
        # — either way, it should explain the limitation
        content_lower = result.content.lower()
        rejected = (
            "not supported" in content_lower
            or "invalid" in content_lower
            or "scheme" in content_lower
            or "only http" in content_lower
            or "ftp" in content_lower
            or "error" in content_lower
            or "can't" in content_lower
            or "unable" in content_lower
            or "doesn't support" in content_lower
        )
        assert rejected, (
            "Response doesn't indicate FTP scheme was rejected. "
            f"Response: {result.content[:300]}"
        )


class TestFetchUrlChained:
    """Search then fetch — multi-step real-world usage."""

    @pytest.mark.slow
    @pytest.mark.timeout(540)
    def test_search_then_read(self, cca, trace_test, judge_model):
        """Ask about something that requires searching then reading a page."""
        session_id = f"test-chain-{uuid.uuid4().hex[:8]}"
        message = (
            "I want to understand what httpbin.org is. Can you look it up "
            "and then read the main page to give me a summary?"
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")
        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute(
            "cca.test.tool_iterations",
            result.metadata.get("tool_iterations", 0),
        )

        assert result.content, "Agent returned empty response"
        _assert_tools_used(result, min_iterations=2)

        content_lower = result.content.lower()
        assert "httpbin" in content_lower, (
            "Response doesn't mention httpbin.org. "
            f"Response: {result.content[:300]}"
        )
