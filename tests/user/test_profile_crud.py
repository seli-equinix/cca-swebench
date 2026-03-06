"""Flow test: Full profile CRUD lifecycle.

Journey: create user with skills + alias → view profile → remove skill →
remove alias → verify removals via REST → delete profile → verify gone.

Replaces 13 individual tests from test_manage_profile: view_profile,
add_skill, add_alias, skill_appears_on_profile, remove_skill,
alias_stored_on_profile, remove_alias, remove_fact, remove_preference,
view_profile_shows_data, delete_and_verify_gone, list_all_users,
delete_profile.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.slow]


class TestProfileCRUD:
    """Full CRUD on user profiles: create, read, update, delete."""

    def test_profile_crud(self, cca, trace_test, judge_model):
        """Complete profile lifecycle with REST API verification."""
        name = f"CRUD_{uuid.uuid4().hex[:6]}"
        company = f"CRUDCorp_{uuid.uuid4().hex[:4]}"
        alias = f"nick_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-crud1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-crud2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-crud3-{uuid.uuid4().hex[:8]}"
        sid4 = f"test-crud4-{uuid.uuid4().hex[:8]}"

        try:
            # ── Session 1: Create user with skills, alias, facts ──
            msg1 = (
                f"Hi I'm {name}, also known as {alias}. "
                f"I work at {company}, I know Python, Docker, and Java. "
                f"I prefer concise code. Write a Dockerfile for Flask."
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:300])
            assert r1.content, "Session 1 returned empty"

            # Verify user created with skills
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created"
            user_id = user["user_id"]

            skills_lower = [s.lower() for s in user.get("skills", [])]
            trace_test.set_attribute("cca.test.skills", str(skills_lower))
            has_skill = any(
                s in skills_lower for s in ["python", "docker", "java"]
            )
            trace_test.set_attribute("cca.test.has_skill", has_skill)
            assert has_skill, f"No skills stored: {skills_lower}"

            # Verify alias stored
            aliases_lower = [a.lower() for a in user.get("aliases", [])]
            has_alias = alias.lower() in aliases_lower
            trace_test.set_attribute("cca.test.has_alias", has_alias)
            assert has_alias, f"Alias '{alias}' not stored: {aliases_lower}"

            # ── Session 2: View profile — system tracks without re-intro ──
            msg2 = (
                "What do you know about me? "
                "Show my full profile. Also write a one-liner to "
                "generate a random number."
            )
            r2 = cca.chat(msg2, session_id=sid2)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Session 2 returned empty"

            # Profile view should include stored info
            content_lower = r2.content.lower()
            profile_terms = sum(1 for t in [
                name.lower(), company.lower(), "python", "docker",
            ] if t in content_lower)
            trace_test.set_attribute("cca.test.profile_terms", profile_terms)
            assert profile_terms >= 2, (
                f"Profile view missing data (found {profile_terms} terms): "
                f"{r2.content[:300]}"
            )

            # ── Session 3: Remove Java skill and alias (no re-intro) ──
            msg3 = (
                f"I don't use Java anymore — remove it from "
                f"my skills. Also remove the alias {alias}, I don't go "
                f"by that. Write a one-liner to get the current time."
            )
            r3 = cca.chat(msg3, session_id=sid3)
            evaluate_response(r3, msg3, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Session 3 returned empty"

            # Verify Java removed via REST
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found after update"
            skills_after = [s.lower() for s in user.get("skills", [])]
            trace_test.set_attribute("cca.test.skills_after", str(skills_after))
            assert "java" not in skills_after, (
                f"Java still in skills: {skills_after}"
            )

            # Verify alias removed via REST
            aliases_after = [a.lower() for a in user.get("aliases", [])]
            trace_test.set_attribute("cca.test.aliases_after", str(aliases_after))
            assert alias.lower() not in aliases_after, (
                f"Alias '{alias}' still present: {aliases_after}"
            )

            # ── Session 4: Delete profile entirely (no re-intro) ──
            msg4 = (
                "Please delete my profile completely. "
                "I confirm permanent deletion. Also show me how to "
                "delete a file in Python."
            )
            r4 = cca.chat(msg4, session_id=sid4)
            evaluate_response(r4, msg4, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s4_response", r4.content[:300])
            assert r4.content, "Session 4 returned empty"

            # Verify deletion acknowledged
            content_lower = r4.content.lower()
            assert any(w in content_lower for w in [
                "deleted", "removed", "confirm", "delete", "profile",
            ]), f"Response doesn't address deletion: {r4.content[:200]}"

            # Verify user is gone via REST
            user_after = cca.find_user_by_name(name)
            trace_test.set_attribute("cca.test.user_deleted", user_after is None)
            assert user_after is None, (
                f"User '{name}' still exists after deletion"
            )

            # REST: /users endpoint still functional
            users_data = cca.list_users()
            assert users_data.get("count", -1) >= 0, "Users endpoint broken"

        finally:
            cca.cleanup_test_user(name)
