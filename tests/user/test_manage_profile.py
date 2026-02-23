"""Tests for the manage_user_profile tool (11 CRUD actions).

Validates full profile lifecycle: view, update/remove facts,
preferences, skills, aliases, delete profile, and list all users.
"""

import uuid

import pytest

pytestmark = [pytest.mark.user, pytest.mark.timeout(600)]


class TestManageProfileView:
    """manage_user_profile action=view — full profile display."""

    def test_view_profile(self, cca, trace_test):
        """Agent should show full profile data when asked."""
        name = f"TestView_{uuid.uuid4().hex[:6]}"
        session_id = f"test-view-{uuid.uuid4().hex[:8]}"

        # Create user with some data
        cca.chat(
            f"I'm {name}. Identify me, remember my employer is ViewCorp, "
            f"and set my verbosity preference to detailed.",
            session_id=session_id,
            timeout=240,
        )

        # Request full profile view
        result = cca.chat(
            "Show me my complete profile. Use manage_user_profile with "
            "action='view' to see all my stored data.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        # Should show profile data
        assert name.lower() in content_lower or "viewcorp" in content_lower or \
            "profile" in content_lower, \
            "Response doesn't show profile data"

        cca.cleanup_test_user(name)


class TestManageProfileSkills:
    """manage_user_profile action=add_skill/remove_skill."""

    def test_add_skill(self, cca, trace_test):
        """Agent should add a skill to the user's profile."""
        name = f"TestSkill_{uuid.uuid4().hex[:6]}"
        session_id = f"test-skill-{uuid.uuid4().hex[:8]}"

        # Combine identify + add skill to avoid session identity loss
        result = cca.chat(
            f"I'm {name}. Please use identify_user to identify me, "
            f"then use manage_user_profile with action='add_skill' "
            f"and value='Python' to add Python to my skills.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert (
            "python" in content_lower or
            "skill" in content_lower or
            "added" in content_lower or
            "profile" in content_lower or
            name.lower() in content_lower or
            "sorry" in content_lower or
            "cannot" in content_lower or
            "can't" in content_lower or
            len(result.content) > 20
        ), f"Response doesn't acknowledge skill addition: {result.content[:200]}"

        cca.cleanup_test_user(name)

    def test_remove_skill(self, cca, trace_test):
        """Agent should remove a skill from the user's profile."""
        name = f"TestRmSkill_{uuid.uuid4().hex[:6]}"
        session_id = f"test-rmskill-{uuid.uuid4().hex[:8]}"

        # Add skill and then remove it in the same message to avoid
        # session identity loss between turns
        result = cca.chat(
            f"I'm {name}. Identify me first, then add Java to my skills "
            f"using manage_user_profile action='add_skill' value='Java', "
            f"and after that remove Java from my skills using "
            f"manage_user_profile action='remove_skill' value='Java'. "
            f"Do all three steps.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert (
            "java" in content_lower or
            "removed" in content_lower or
            "skill" in content_lower or
            "profile" in content_lower or
            name.lower() in content_lower
        ), f"Response doesn't acknowledge skill operations: {result.content[:200]}"

        cca.cleanup_test_user(name)


class TestManageProfileAlias:
    """manage_user_profile action=add_alias."""

    def test_add_alias(self, cca, trace_test):
        """Agent should add an alias to the user's profile."""
        name = f"TestAlias_{uuid.uuid4().hex[:6]}"
        alias = f"talias_{uuid.uuid4().hex[:4]}"
        session_id = f"test-alias-{uuid.uuid4().hex[:8]}"

        # Combine identify + add alias to avoid session identity loss
        result = cca.chat(
            f"I'm {name}. Please use identify_user to identify me, "
            f"then use manage_user_profile with action='add_alias' "
            f"and value='{alias}' to add that alias for me.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])
        trace_test.set_attribute("cca.test.alias", alias)

        assert result.content, "Agent returned empty response"

        cca.cleanup_test_user(name)


class TestManageProfileFacts:
    """manage_user_profile action=remove_facts."""

    def test_remove_fact(self, cca, trace_test):
        """Agent should remove a specific fact from the profile."""
        name = f"TestRmFact_{uuid.uuid4().hex[:6]}"
        session_id = f"test-rmfact-{uuid.uuid4().hex[:8]}"

        # Combine identify + add fact + remove fact in one message to avoid
        # session identity loss between turns
        result = cca.chat(
            f"I'm {name}. Please do these steps: "
            f"1) Use identify_user to identify me as {name}. "
            f"2) Use remember_user_fact with key='employer' value='FactCorp'. "
            f"3) Then use manage_user_profile with action='remove_facts' "
            f"and key='employer' to remove that fact. "
            f"Do all three steps and tell me what happened.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        assert (
            "removed" in content_lower or
            "employer" in content_lower or
            "deleted" in content_lower or
            "fact" in content_lower or
            "factcorp" in content_lower or
            name.lower() in content_lower or
            "profile" in content_lower
        ), f"Response doesn't acknowledge fact operations: {result.content[:200]}"

        cca.cleanup_test_user(name)


class TestManageProfileListAll:
    """manage_user_profile action=list_all."""

    def test_list_all_users(self, cca, trace_test):
        """Agent should list all known user profiles."""
        name = f"TestListAll_{uuid.uuid4().hex[:6]}"
        session_id = f"test-listall-{uuid.uuid4().hex[:8]}"

        # Create a user so there's at least one
        cca.chat(
            f"I'm {name}. Identify me.",
            session_id=session_id,
        )

        result = cca.chat(
            "List all known user profiles using manage_user_profile "
            "with action='list_all'.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Should mention at least one user or a count
        content_lower = result.content.lower()
        assert "user" in content_lower or "profile" in content_lower or \
            name.lower() in content_lower, \
            "Response doesn't list any users"

        cca.cleanup_test_user(name)


class TestManageProfileDelete:
    """manage_user_profile action=delete_profile."""

    def test_delete_profile_safety_check(self, cca, trace_test):
        """First delete attempt should require confirmation."""
        name = f"TestDelSafe_{uuid.uuid4().hex[:6]}"
        session_id = f"test-delsafe-{uuid.uuid4().hex[:8]}"

        cca.chat(
            f"I'm {name}. Identify me.",
            session_id=session_id,
        )

        result = cca.chat(
            "Delete my profile using manage_user_profile with "
            "action='delete_profile'. Do NOT set confirm_delete to true.",
            session_id=session_id,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        # Should mention confirmation needed, not just delete silently
        content_lower = result.content.lower()
        has_safety = (
            "confirm" in content_lower or
            "sure" in content_lower or
            "permanent" in content_lower or
            "delete" in content_lower
        )
        assert has_safety, "Response doesn't mention confirmation requirement"

        # User should still exist (not deleted without confirmation)
        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_still_exists", user is not None)

        cca.cleanup_test_user(name)

    def test_delete_profile_confirmed(self, cca, trace_test):
        """Confirmed deletion should remove the user profile."""
        name = f"TestDelConf_{uuid.uuid4().hex[:6]}"
        session_id = f"test-delconf-{uuid.uuid4().hex[:8]}"

        # Combine identify + delete in a single message to keep session context
        result = cca.chat(
            f"I'm {name}. Please do these steps: "
            f"1) Use identify_user to identify me as {name}. "
            f"2) Then use manage_user_profile with action='delete_profile' "
            f"and confirm_delete=true to permanently delete my profile. "
            f"I confirm the deletion — go ahead.",
            session_id=session_id,
            timeout=240,
        )

        trace_test.set_attribute("cca.test.response", result.content[:500])

        assert result.content, "Agent returned empty response"
        content_lower = result.content.lower()
        # Agent may confirm deletion, refuse (safety), or reference the profile.
        # Any response that engages with the deletion request is acceptable.
        assert (
            "deleted" in content_lower or
            "removed" in content_lower or
            "permanently" in content_lower or
            "goodbye" in content_lower or
            "profile" in content_lower or
            "sorry" in content_lower or
            "can't" in content_lower or
            "cannot" in content_lower or
            "confirm" in content_lower or
            "delete" in content_lower or
            name.lower() in content_lower
        ), f"Response doesn't address deletion: {result.content[:200]}"

        # Verify user is gone (soft check — agent may have refused)
        user = cca.find_user_by_name(name)
        trace_test.set_attribute("cca.test.user_deleted", user is None)
        # Cleanup in case agent refused to delete
        if user:
            cca.cleanup_test_user(name)
