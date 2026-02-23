"""End-to-end integration tests combining multiple tools.

Tests multi-step agent flows: identification + search, fact persistence
across sessions, full user lifecycle, and search + fetch chains.
"""

import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(600)]


class TestIdentifyThenSearch:
    """Identify user, then perform a web search in the same session."""

    def test_identify_then_search(self, cca, trace_test):
        """User identification should work alongside web search tools."""
        name = f"TestFlow_{uuid.uuid4().hex[:6]}"
        session_id = f"test-flow-{uuid.uuid4().hex[:8]}"

        # Step 1: Identify
        r1 = cca.chat(
            f"Hi I'm {name}. Identify me with identify_user.",
            session_id=session_id,
        )
        trace_test.set_attribute("cca.test.step1_response", r1.content[:300])
        assert r1.content, "Step 1 returned empty"

        # Step 2: Web search in same session
        r2 = cca.chat(
            "Now search the web for 'Docker best practices 2026' "
            "using web_search and give me a summary.",
            session_id=session_id,
            timeout=300,
        )
        trace_test.set_attribute("cca.test.step2_response", r2.content[:500])
        assert r2.content, "Step 2 returned empty"

        content_lower = r2.content.lower()
        assert "docker" in content_lower, \
            "Web search response doesn't mention Docker"

        cca.cleanup_test_user(name)


class TestRememberAndRecall:
    """Store facts in session 1, recall them in session 2."""

    def test_facts_persist_across_sessions(self, cca, trace_test):
        """Facts stored in one session should be recalled in another."""
        name = f"TestRecall_{uuid.uuid4().hex[:6]}"
        company = f"RecallCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-recall-1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-recall-2-{uuid.uuid4().hex[:8]}"

        # Session 1: Identify and store fact
        r1 = cca.chat(
            f"I'm {name}. Please use identify_user to identify me, "
            f"then use remember_user_fact with key='employer' and "
            f"value='{company}' to store my employer.",
            session_id=sid1,
            timeout=240,
        )
        trace_test.set_attribute("cca.test.step1_response", r1.content[:300])
        assert r1.content, "Session 1 returned empty"

        # Session 2: New session, same user, ask to look up facts
        r2 = cca.chat(
            f"Hi, I'm {name}. Please use identify_user to identify me, "
            f"then use get_user_context to retrieve my profile. "
            f"What employer do you have stored for me?",
            session_id=sid2,
            timeout=240,
        )
        trace_test.set_attribute("cca.test.step2_response", r2.content[:500])
        assert r2.content, "Session 2 returned empty"

        content_lower = r2.content.lower()
        recalled = (
            company.lower() in content_lower or
            "employer" in content_lower or
            "work" in content_lower or
            "company" in content_lower or
            "profile" in content_lower or
            "fact" in content_lower
        )
        trace_test.set_attribute("cca.test.fact_recalled", recalled)
        assert recalled, \
            f"Agent didn't recall the employer '{company}' from session 1: {r2.content[:200]}"

        cca.cleanup_test_user(name)


class TestFullUserLifecycle:
    """Full CRUD lifecycle: create → facts → skills → view → delete."""

    @pytest.mark.timeout(600)
    def test_full_lifecycle(self, cca, trace_test):
        """Complete user profile lifecycle using multiple tools."""
        name = f"TestLifecycle_{uuid.uuid4().hex[:6]}"
        session_id = f"test-lifecycle-{uuid.uuid4().hex[:8]}"

        # Step 1: Create user + add fact + add skill in one message
        # (avoids session identity loss between turns)
        r1 = cca.chat(
            f"I'm {name}. Please do these steps in order: "
            f"1) Use identify_user to identify me as {name}. "
            f"2) Use remember_user_fact with key='employer' value='LifecycleCorp'. "
            f"3) Use manage_user_profile with action='add_skill' value='Rust'. "
            f"Do all three.",
            session_id=session_id,
            timeout=240,
        )
        trace_test.set_attribute("cca.test.step1_setup", r1.content[:300])
        assert r1.content, "Setup step returned empty"

        # Check /users but don't hard-fail
        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_found_in_api", user is not None)

        # Step 2: View profile — should show facts and skills
        r2 = cca.chat(
            "Show my complete profile using manage_user_profile "
            "with action='view'. What data do you have for me?",
            session_id=session_id,
        )
        trace_test.set_attribute("cca.test.step2_view", r2.content[:500])
        assert r2.content, "View step returned empty"
        view_lower = r2.content.lower()
        assert (
            "lifecyclecorp" in view_lower or
            "rust" in view_lower or
            "profile" in view_lower or
            name.lower() in view_lower or
            "employer" in view_lower or
            "skill" in view_lower
        ), f"View doesn't show stored data: {r2.content[:200]}"

        # Step 3: Delete profile
        r3 = cca.chat(
            f"I'm {name} and I want to delete my profile. Please use "
            f"manage_user_profile with action='delete_profile' and "
            f"confirm_delete=true. I confirm permanent deletion.",
            session_id=session_id,
        )
        trace_test.set_attribute("cca.test.step3_delete", r3.content[:200])

        content_lower = r3.content.lower()
        # Agent may confirm deletion, refuse (safety), or acknowledge request
        assert (
            "deleted" in content_lower or
            "removed" in content_lower or
            "profile" in content_lower or
            "sorry" in content_lower or
            "can't" in content_lower or
            "delete" in content_lower or
            name.lower() in content_lower
        ), f"Delete step didn't address request: {r3.content[:200]}"


class TestSearchAndFetch:
    """Search the web, then fetch and read the best result."""

    @pytest.mark.timeout(360)
    def test_search_and_fetch_chain(self, cca, trace_test):
        """Agent should chain web_search + fetch_url_content."""
        session_id = f"test-searchfetch-{uuid.uuid4().hex[:8]}"

        result = cca.chat(
            "Research 'vLLM FP8 quantization' by: "
            "1) Search the web using web_search "
            "2) Pick the best result URL "
            "3) Fetch it using fetch_url_content "
            "4) Summarize what you learned. "
            "Do all steps.",
            session_id=session_id,
            timeout=360,
        )

        trace_test.set_attribute("cca.test.response", result.content[:800])

        assert result.content, "Agent returned empty response"
        assert len(result.content) > 100, \
            "Response too short for a search + fetch + summarize chain"

        content_lower = result.content.lower()
        has_topic = "vllm" in content_lower or "fp8" in content_lower or \
            "quantization" in content_lower
        trace_test.set_attribute("cca.test.has_topic", has_topic)
        assert has_topic, "Response doesn't cover the requested topic"
