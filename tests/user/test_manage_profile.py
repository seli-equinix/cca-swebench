"""Tests for the manage_user_profile tool.

Pairs profile operations with coding tasks to ensure the agent loop runs.
Uses REST API for ground truth validation.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user]


class TestManageProfileView:
    """manage_user_profile action=view — full profile display."""

    def test_view_profile(self, cca, trace_test, judge_model):
        """Agent should show profile data when asked."""
        name = f"ViewUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-view-{uuid.uuid4().hex[:8]}"
        try:
            # Create user with a coding task
            cca.chat(
                f"Hi I'm {name}. I work at ViewCorp. "
                f"Write a Python one-liner to generate a random number.",
                session_id=session_id,
            )

            # Ask for profile view (same session, user already identified)
            message = (
                "What do you know about me? Show my profile. "
                "Also, how do I generate a random string in Python?"
            )
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            content_lower = result.content.lower()
            assert any(w in content_lower for w in [
                name.lower(), "viewcorp", "profile",
            ]), f"Profile view missing user data: {result.content[:200]}"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileSkills:
    """manage_user_profile action=add_skill."""

    def test_add_skill(self, cca, trace_test, judge_model):
        """Agent should add a skill to the user's profile."""
        name = f"SkillUser_{uuid.uuid4().hex[:6]}"
        session_id = f"test-skill-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}. I know Python and Docker. "
            f"Write a Dockerfile for a simple Python Flask app."
        )
        try:
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found via /users API"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileAlias:
    """manage_user_profile action=add_alias."""

    def test_add_alias(self, cca, trace_test, judge_model):
        """Agent should handle alias introduction alongside a task."""
        name = f"AliasUser_{uuid.uuid4().hex[:6]}"
        alias = f"al_{uuid.uuid4().hex[:4]}"
        session_id = f"test-alias-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}, also known as {alias}. "
            f"Write a Python function to merge two dictionaries."
        )
        try:
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileSkillVerify:
    """Verify skills are actually stored on the profile via REST API."""

    def test_skill_appears_on_profile(self, cca, trace_test, judge_model):
        """Skills mentioned in conversation should appear on the profile."""
        name = f"SkillCheck_{uuid.uuid4().hex[:6]}"
        session_id = f"test-skchk-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}. I'm experienced with Python, Docker, "
            f"and Terraform. Help me write a Dockerfile for a "
            f"Python FastAPI app."
        )
        try:
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"

            skills_lower = [s.lower() for s in user.get("skills", [])]
            trace_test.set_attribute("cca.test.skills", str(skills_lower))
            has_any_skill = any(
                s in skills_lower
                for s in ["python", "docker", "terraform"]
            )
            trace_test.set_attribute("cca.test.has_expected_skill", has_any_skill)
            assert has_any_skill, \
                f"Expected at least one skill on profile, got: {skills_lower}"
        finally:
            cca.cleanup_test_user(name)

    def test_remove_skill(self, cca, trace_test, judge_model):
        """Asking to remove a skill should actually remove it from the profile."""
        name = f"RmSkill_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-rmsk1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-rmsk2-{uuid.uuid4().hex[:8]}"
        try:
            # Session 1: create user with skills
            cca.chat(
                f"Hi I'm {name}. I know Python and Java. "
                f"Write me a Python hello world.",
                session_id=sid1,
            )

            # Session 2: ask to remove Java
            message = (
                f"Hey {name}, I don't really use Java anymore — can you "
                f"remove it from my profile? Also write me a one-liner "
                f"to get the current time."
            )
            result = cca.chat(message, session_id=sid2)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found"

            skills_lower = [s.lower() for s in user.get("skills", [])]
            trace_test.set_attribute("cca.test.skills_after", str(skills_lower))
            java_gone = "java" not in skills_lower
            trace_test.set_attribute("cca.test.java_removed", java_gone)
            assert java_gone, \
                f"Java still in skills after removal: {skills_lower}"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileAliasVerify:
    """Verify aliases are stored and removable via REST API."""

    def test_alias_stored_on_profile(self, cca, trace_test, judge_model):
        """Introducing with an alias should store it on the profile."""
        name = f"AliasChk_{uuid.uuid4().hex[:6]}"
        alias = f"nick_{uuid.uuid4().hex[:4]}"
        session_id = f"test-alchk-{uuid.uuid4().hex[:8]}"
        message = (
            f"Hi I'm {name}, but my friends call me {alias}. "
            f"Can you write a Python function to generate a "
            f"random password?"
        )
        try:
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"

            aliases_lower = [a.lower() for a in user.get("aliases", [])]
            trace_test.set_attribute("cca.test.aliases", str(aliases_lower))
            has_alias = alias.lower() in aliases_lower
            trace_test.set_attribute("cca.test.alias_stored", has_alias)
            assert has_alias, \
                f"Alias '{alias}' not found in profile aliases: {aliases_lower}"
        finally:
            cca.cleanup_test_user(name)

    def test_remove_alias(self, cca, trace_test, judge_model):
        """Asking to remove an alias should actually remove it."""
        name = f"RmAlias_{uuid.uuid4().hex[:6]}"
        alias = f"old_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-rmal1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-rmal2-{uuid.uuid4().hex[:8]}"
        try:
            # Session 1: create with alias
            cca.chat(
                f"Hi I'm {name}, also known as {alias}. "
                f"Write me a one-liner to read environment variables.",
                session_id=sid1,
            )

            # Session 2: ask to remove alias
            message = (
                f"Hey {name}. Please remove the alias {alias} from my "
                f"profile — I don't go by that anymore. Also write a "
                f"one-liner to get the hostname."
            )
            result = cca.chat(message, session_id=sid2)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found"

            aliases_lower = [a.lower() for a in user.get("aliases", [])]
            trace_test.set_attribute("cca.test.aliases_after", str(aliases_lower))
            alias_gone = alias.lower() not in aliases_lower
            trace_test.set_attribute("cca.test.alias_removed", alias_gone)
            assert alias_gone, \
                f"Alias '{alias}' still in profile after removal: {aliases_lower}"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileRemove:
    """Tests for removing facts and preferences from user profiles."""

    def test_remove_fact(self, cca, trace_test, judge_model):
        """After asking to forget a fact, CCA should no longer recall it."""
        name = f"RmFact_{uuid.uuid4().hex[:6]}"
        company = f"ForgetCorp_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-rmf1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-rmf2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-rmf3-{uuid.uuid4().hex[:8]}"
        try:
            # Session 1: store a fact
            cca.chat(
                f"Hi I'm {name}. I work at {company}. "
                f"Write me a one-liner to get the current timestamp.",
                session_id=sid1,
            )

            # Session 2: ask to forget it
            cca.chat(
                f"Hey {name}, I'd rather you forget where I work — "
                f"please remove that from my profile. "
                f"Show me how to clear a list in Python.",
                session_id=sid2,
            )

            # Session 3: ask about it
            message = (
                f"Hi {name}. Do you know where I work? "
                f"Also write a one-liner to reverse a list."
            )
            result = cca.chat(message, session_id=sid3)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            # Ground truth: check REST API — company should not be in facts
            user = cca.find_user_by_name(name)
            if user is not None:
                facts_lower = " ".join(
                    str(f) for f in user.get("facts", [])
                ).lower()
                fact_gone = company.lower() not in facts_lower
                trace_test.set_attribute("cca.test.company_forgotten", fact_gone)
                assert fact_gone, \
                    f"Company '{company}' still in user facts: {user.get('facts', [])}"
        finally:
            cca.cleanup_test_user(name)

    def test_remove_preference(self, cca, trace_test, judge_model):
        """Removing a preference should be acknowledged."""
        name = f"RmPref_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-rmp1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-rmp2-{uuid.uuid4().hex[:8]}"
        try:
            # Session 1: set a preference
            cca.chat(
                f"Hi I'm {name}. I prefer very detailed explanations "
                f"with examples. Write me a function to sort a "
                f"dictionary by value.",
                session_id=sid1,
            )

            # Session 2: remove it
            message = (
                f"Hey {name}. Forget about the detailed explanations "
                f"preference — remove it please. Write a function to "
                f"flatten a nested list."
            )
            result = cca.chat(message, session_id=sid2)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"
        finally:
            cca.cleanup_test_user(name)


class TestManageProfileViewAndDelete:
    """View profile data and full delete lifecycle."""

    def test_view_profile_shows_data(self, cca, trace_test, judge_model):
        """Asking for profile should show previously stored information."""
        name = f"ViewData_{uuid.uuid4().hex[:6]}"
        company = f"ViewCorp_{uuid.uuid4().hex[:4]}"
        session_id = f"test-vpd-{uuid.uuid4().hex[:8]}"
        try:
            # First message: introduce with lots of info
            cca.chat(
                f"Hi I'm {name}. I work at {company}, I know Python "
                f"and Go, and I prefer concise code. Write me a "
                f"function to validate an email address.",
                session_id=session_id,
            )

            # Same session: ask to see profile
            message = "Can you show me everything you know about me?"
            result = cca.chat(message, session_id=session_id)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            content_lower = result.content.lower()
            matches = sum(1 for term in [
                name.lower(), company.lower(), "python", "go", "concise",
            ] if term in content_lower)
            trace_test.set_attribute("cca.test.profile_terms_found", matches)
            assert matches >= 2, \
                f"Expected at least 2 profile terms in response, found {matches}: {result.content[:300]}"
        finally:
            cca.cleanup_test_user(name)

    def test_delete_and_verify_gone(self, cca, trace_test, judge_model):
        """After deletion, user should not exist via REST API."""
        name = f"DelGone_{uuid.uuid4().hex[:6]}"
        sid1 = f"test-dlg1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-dlg2-{uuid.uuid4().hex[:8]}"
        try:
            # Create user
            cca.chat(
                f"Hi I'm {name}. I work on infrastructure. "
                f"Write me a Python one-liner to ping a host.",
                session_id=sid1,
            )

            user = cca.find_user_by_name(name)
            trace_test.set_attribute("cca.test.user_exists_before", user is not None)
            assert user is not None, f"User '{name}' not created"

            # Request deletion
            message = (
                f"Hi {name} here. I want to delete my profile completely. "
                f"Yes I confirm the deletion. Also show me how to "
                f"delete a file in Python."
            )
            result = cca.chat(message, session_id=sid2)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            # Verify user is gone
            user_after = cca.find_user_by_name(name)
            trace_test.set_attribute("cca.test.user_gone", user_after is None)
            assert user_after is None, \
                f"User '{name}' still exists after deletion request"
        finally:
            # Safety cleanup if agent refused deletion
            cca.cleanup_test_user(name)


class TestManageProfileListAll:
    """REST API /users endpoint."""

    def test_list_all_users(self, cca, trace_test):
        """REST API should list user profiles."""
        users_data = cca.list_users()
        trace_test.set_attribute("cca.test.user_count", users_data.get("count", 0))
        assert users_data.get("count", 0) > 0, "No users found via /users API"


class TestManageProfileDelete:
    """manage_user_profile action=delete_profile."""

    def test_delete_profile(self, cca, trace_test, judge_model):
        """Agent should delete a user profile when confirmed."""
        name = f"DelUser_{uuid.uuid4().hex[:6]}"
        sid_create = f"test-delc-{uuid.uuid4().hex[:8]}"
        sid_delete = f"test-deld-{uuid.uuid4().hex[:8]}"
        try:
            # Create user via coding task
            cca.chat(
                f"Hi I'm {name}. Write a Python one-liner to flatten a list.",
                session_id=sid_create,
            )

            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"

            # Request deletion in a new session
            message = (
                f"Hi I'm {name}. Please delete my profile completely. "
                f"I confirm permanent deletion. Also show me how to "
                f"delete a file in Python."
            )
            result = cca.chat(message, session_id=sid_delete)

            evaluate_response(result, message, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.response", result.content[:500])
            assert result.content, "Agent returned empty response"

            content_lower = result.content.lower()
            assert any(w in content_lower for w in [
                "deleted", "removed", "confirm", "delete", "profile",
            ]), f"Response doesn't address deletion: {result.content[:200]}"

            user_after = cca.find_user_by_name(name)
            trace_test.set_attribute("cca.test.user_deleted", user_after is None)
            assert user_after is None, (
                f"Profile for '{name}' still exists in Qdrant after deletion request"
            )
        finally:
            # Cleanup if agent refused deletion
            cca.cleanup_test_user(name)
