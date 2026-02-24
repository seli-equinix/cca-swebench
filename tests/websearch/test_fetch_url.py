"""Tests for the fetch_url_content tool.

Validates URL fetching with text extraction, SSRF protection,
and scheme validation.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.websearch, pytest.mark.timeout(600)]


class TestFetchUrlBasic:
    """fetch_url_content tool — basic URL fetching."""

    def test_fetch_public_url(self, cca, trace_test, judge_model):
        """Agent should fetch a public URL and extract text content."""
        session_id = f"test-fetch-{uuid.uuid4().hex[:8]}"
        message = (
            "Fetch the page at https://httpbin.org/html using "
            "fetch_url_content and tell me what text it contains."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        has_content = (
            "herman" in content_lower or
            "melville" in content_lower or
            "moby" in content_lower or
            "whale" in content_lower or
            "httpbin" in content_lower or
            "html" in content_lower or
            "text" in content_lower
        )
        trace_test.set_attribute("cca.test.has_content", has_content)
        assert has_content, "Response doesn't mention fetched page content"


class TestFetchUrlSecurity:
    """fetch_url_content tool — SSRF protection and validation."""

    def test_ssrf_blocks_private_ip(self, cca, trace_test, judge_model):
        """Agent should refuse to fetch private/internal IP addresses."""
        session_id = f"test-ssrf-{uuid.uuid4().hex[:8]}"
        message = (
            "Fetch http://192.168.1.1/ using fetch_url_content. "
            "Tell me exactly what happens."
        )

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        blocked = (
            "blocked" in content_lower or
            "private" in content_lower or
            "internal" in content_lower or
            "not allowed" in content_lower or
            "security" in content_lower or
            "ssrf" in content_lower or
            "refused" in content_lower or
            "error" in content_lower
        )
        trace_test.set_attribute("cca.test.ssrf_blocked", blocked)
        assert blocked, \
            "Response doesn't indicate the private IP was blocked"

    def test_invalid_scheme_rejected(self, cca, trace_test, judge_model):
        """Agent should reject non-http/https schemes."""
        session_id = f"test-scheme-{uuid.uuid4().hex[:8]}"
        message = "Fetch ftp://example.com/file.txt using fetch_url_content."

        result = cca.chat(message, session_id=session_id)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        rejected = (
            "http" in content_lower or
            "not supported" in content_lower or
            "only" in content_lower or
            "invalid" in content_lower or
            "scheme" in content_lower or
            "ftp" in content_lower
        )
        assert rejected, "Response doesn't indicate FTP scheme was rejected"


class TestFetchUrlChained:
    """fetch_url_content chained with web_search."""

    @pytest.mark.slow
    @pytest.mark.timeout(360)
    def test_search_then_fetch(self, cca, trace_test, judge_model):
        """Agent should search then fetch the best result."""
        session_id = f"test-chain-{uuid.uuid4().hex[:8]}"
        message = (
            "Search for 'httpbin.org' using web_search, then use "
            "fetch_url_content to fetch the top result URL and tell "
            "me what the page says."
        )

        result = cca.chat(message, session_id=session_id, timeout=360)

        evaluate_response(result, message, trace_test, judge_model, "websearch")

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        assert len(result.content) > 50, \
            "Response too short for a search + fetch chain"
