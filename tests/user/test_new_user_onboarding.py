"""Flow test: A brand-new person arrives at CCA for the first time.

Journey: introduce with name + company + skills → verify profile created
via REST → same-session context persistence → preference setting.

Replaces 8 individual tests: identify_new_user, identify_with_greeting,
remember_single_fact, remember_multiple_facts, context_identified_user,
set_preference, preference_acknowledged, identify_updates_metadata.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.slow]


class TestNewUserOnboarding:
    """A new person's first interaction with CCA."""

    def test_new_user_onboarding(self, cca, trace_test, judge_model):
        """Full onboarding: intro → facts stored → context persists → preference set."""
        name = f"Onboard_{uuid.uuid4().hex[:6]}"
        company = "OnboardCorp"
        session_id = f"test-onb-{uuid.uuid4().hex[:8]}"

        try:
            # ── Turn 1: Arrive, introduce with name + company + skills ──
            msg1 = (
                f"Hi, I'm {name}. I'm a DevOps engineer at {company}, "
                f"and I mainly work with Python, Docker, and Terraform. "
                f"Write me a Python function to check if a port is open."
            )
            r1 = cca.chat(msg1, session_id=session_id)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"
            assert r1.user_identified, "User should be identified after introduction"

            # REST: user should exist
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not found via REST API"
            user_id = user["user_id"]
            trace_test.set_attribute("cca.test.user_id", user_id)

            # REST: check profile for facts/skills
            profile = cca.get_user_profile(user_id)
            if profile:
                facts_str = str(profile.get("facts", {})).lower()
                skills_lower = [s.lower() for s in profile.get("skills", [])]
                has_company = company.lower() in facts_str
                has_skill = any(
                    s in skills_lower for s in ["python", "docker", "terraform"]
                )
                trace_test.set_attribute("cca.test.has_company", has_company)
                trace_test.set_attribute("cca.test.has_skill", has_skill)
                # At minimum, the company or a skill should be stored
                assert has_company or has_skill, (
                    f"Profile missing company or skills. "
                    f"Facts: {profile.get('facts', {})} Skills: {skills_lower}"
                )

            # ── Turn 2: Same session — context should persist ──
            msg2 = (
                "What do you know about me? Also write a one-liner to "
                "check disk usage."
            )
            r2 = cca.chat(msg2, session_id=session_id)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:300])
            assert r2.content, "Turn 2 returned empty"
            assert r2.user_identified, "Should remain identified in same session"

            # Response should reference stored context
            content_lower = r2.content.lower()
            context_terms = sum(1 for t in [
                company.lower(), "devops", "python", "docker", "terraform",
            ] if t in content_lower)
            trace_test.set_attribute("cca.test.context_terms", context_terms)
            assert context_terms >= 2, (
                f"Context not surfaced (found {context_terms} terms): "
                f"{r2.content[:300]}"
            )

            # ── Turn 3: Set a preference ──
            msg3 = (
                "I prefer concise code with type hints. "
                "Write a function to calculate factorial."
            )
            r3 = cca.chat(msg3, session_id=session_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:300])
            assert r3.content, "Turn 3 returned empty"
            assert len(r3.content) > 50, "Response too short for a code task"

        finally:
            cca.cleanup_test_user(name)
