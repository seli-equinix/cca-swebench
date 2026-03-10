"""Flow test: Web search and URL fetching capabilities.

Tests the SEARCH route's ability to find, fetch, and synthesize web info
through a realistic multi-turn research session.

Exercises: web_search, fetch_url_content, multi-source synthesis,
search-then-read chaining.
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
    """Web search: research, fetch URLs, compare, and discover — like a real person would."""

    @pytest.mark.slow
    def test_web_search_flow(self, cca, trace_test, judge_model):
        """4-turn research session: search → fetch URL → compare → discover.

        Absorbs: test_research_session, test_search_and_read.

        Simulates a developer researching LLM serving frameworks:
        1. Ask about vLLM — expects web search with real URLs
        2. Paste a URL to read — expects page content extraction
        3. Compare vLLM vs TGI — expects multi-source synthesis
        4. Discover a new tool — expects search + read chaining
        """
        tracker = cca.tracker()
        session_id = f"test-wsf-{uuid.uuid4().hex[:8]}"
        tracker.track_session(session_id)

        try:
            # ── Turn 1: Research a tech topic (search + URLs) ──
            msg1 = (
                "I'm looking into LLM serving frameworks. "
                "What's vLLM and what version is it at now? "
                "Show me where you found the info."
            )
            r1 = cca.chat(msg1, session_id=session_id)
            evaluate_response(r1, msg1, trace_test, judge_model, "websearch")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"
            _assert_tools_used(r1)
            assert "vllm" in r1.content.lower(), (
                "Turn 1 doesn't mention vLLM. "
                f"Response: {r1.content[:300]}"
            )
            assert _has_real_urls(r1.content), (
                "Turn 1 has no URLs — agent likely answered from memory. "
                f"Response: {r1.content[:300]}"
            )

            # ── Turn 2: Read a specific page (URL fetching) ──
            msg2 = (
                "Oh by the way, can you read this page and tell me what it's about? "
                "https://httpbin.org/html"
            )
            r2 = cca.chat(msg2, session_id=session_id)
            # Skip LLM judge: httpbin.org/html serves a Moby Dick excerpt, but
            # the judge doesn't know this and rates correct content as "hallucination".
            evaluate_response(r2, msg2, trace_test, None, "websearch")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"
            _assert_tools_used(r2)

            # httpbin.org/html serves a Moby Dick excerpt
            content_lower = r2.content.lower()
            has_content = any(w in content_lower for w in [
                "herman", "melville", "moby", "whale",
            ])
            trace_test.set_attribute("cca.test.t2_has_moby_dick", has_content)
            assert has_content, (
                "Turn 2 doesn't mention Moby Dick content from httpbin.org/html. "
                f"Response: {r2.content[:300]}"
            )

            # ── Turn 3: Compare technologies (builds on turn 1) ──
            msg3 = (
                "Going back to my original question about LLM serving — "
                "how does vLLM compare to TGI? What are the trade-offs?"
            )
            r3 = cca.chat(msg3, session_id=session_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "websearch")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"
            _assert_tools_used(r3)

            content_lower = r3.content.lower()
            has_vllm = "vllm" in content_lower
            has_tgi = "tgi" in content_lower or "text generation inference" in content_lower
            trace_test.set_attribute("cca.test.t3_mentions_vllm", has_vllm)
            trace_test.set_attribute("cca.test.t3_mentions_tgi", has_tgi)
            assert has_vllm or has_tgi, (
                "Turn 3 doesn't mention vLLM or TGI. "
                f"Response: {r3.content[:300]}"
            )

            # ── Turn 4: Discover something new (search + read chaining) ──
            # Tests the "I don't know what this is, go figure it out" pattern.
            # Agent should search, find a page, fetch it, and summarize.
            msg4 = (
                "I keep hearing about httpbin.org but I've never used it. "
                "Can you find out what it is and read the site for me?"
            )
            r4 = cca.chat(msg4, session_id=session_id)
            evaluate_response(r4, msg4, trace_test, judge_model, "websearch")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
            assert r4.content, "Turn 4 returned empty"
            _assert_tools_used(r4, min_iterations=2)

            content_lower = r4.content.lower()
            assert "httpbin" in content_lower, (
                "Response doesn't mention httpbin. "
                f"Response: {r4.content[:300]}"
            )
            # Should have actually read something — not just parroted the name
            has_substance = any(w in content_lower for w in [
                "http", "api", "test", "request", "endpoint", "response",
            ])
            trace_test.set_attribute("cca.test.t4_has_substance", has_substance)
            assert has_substance, (
                "Response mentions httpbin but has no substance about what it does. "
                f"Response: {r4.content[:300]}"
            )

        finally:
            tracker.cleanup()
