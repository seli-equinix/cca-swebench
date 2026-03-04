"""Flow test: A brand-new person arrives at CCA for the first time.

Journey: introduce with name + company + skills → verify profile created
via REST → same-session context persistence → preference setting →
new session verifies notes + preference recall.

Replaces 8 individual tests: identify_new_user, identify_with_greeting,
remember_single_fact, remember_multiple_facts, context_identified_user,
set_preference, preference_acknowledged, identify_updates_metadata.
"""

import re
import time
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

            # ── Notes check: NoteObserver should have extracted the preference ──
            # NoteObserver is async fire-and-forget — give it time to process
            time.sleep(10)
            notes = cca.search_notes("type hints concise code", user_id=user_id)
            trace_test.set_attribute("cca.test.notes_count", len(notes))
            trace_test.set_attribute(
                "cca.test.notes_preview",
                str(notes[:2])[:500] if notes else "no notes found",
            )
            # Note should exist about the preference
            assert len(notes) > 0, (
                "No notes found about 'type hints' preference — "
                "NoteObserver may not be extracting preferences"
            )

            # ── Turn 4: New session — verify preference was remembered ──
            sid2 = f"test-onb-{uuid.uuid4().hex[:8]}"
            msg4 = (
                f"Hey it's {name} again. Write me a function that "
                f"reverses a linked list."
            )
            r4 = cca.chat(msg4, session_id=sid2)
            evaluate_response(r4, msg4, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s4_response", r4.content[:500])
            assert r4.content, "Turn 4 returned empty"
            assert r4.user_identified, "Returning user should be identified"

            # Code should have ACTUAL type annotations in function signatures,
            # not just type names in docstrings or comments.
            # Look for `def func(param: Type)` or `-> ReturnType` patterns.
            content = r4.content
            has_signature_hints = bool(re.search(
                r'def \w+\(.*?:\s*\w+', content, re.DOTALL
            ))
            has_return_hint = "->" in content
            has_type_hints = has_signature_hints or has_return_hint
            trace_test.set_attribute("cca.test.s4_has_signature_hints", has_signature_hints)
            trace_test.set_attribute("cca.test.s4_has_return_hint", has_return_hint)
            assert has_type_hints, (
                "Turn 4 code doesn't have type hints in function signatures — "
                "preference not applied. "
                f"Response: {content[:400]}"
            )

        finally:
            cca.cleanup_test_user(name)
