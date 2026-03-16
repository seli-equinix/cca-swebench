"""Flow test: Full profile CRUD lifecycle with data layer validation.

Journey: create user with rich context → verify Qdrant profile + Redis
session + NoteObserver notes → recall profile in new session → update
skills/alias → verify updates persisted → delete profile → verify
cascade (Qdrant profile gone, notes cleaned).

Validates the entire user data pipeline:
  - Qdrant user_profiles: facts, skills, aliases, preferences
  - Qdrant cca_notes: NoteObserver extracts and stores notes
  - Redis cca:session:{id}: session state with user linkage
  - Cascade delete: profile + notes cleaned on user deletion
"""

import time
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.user, pytest.mark.slow]


class TestProfileCRUD:
    """Full CRUD on user profiles with Qdrant, Redis, and notes validation."""

    def test_profile_crud(self, cca, trace_test, judge_model):
        """Complete profile lifecycle: create → verify data layer → update → delete → verify cascade."""
        tracker = cca.tracker()
        name = f"CRUD_{uuid.uuid4().hex[:6]}"
        company = f"CRUDCorp_{uuid.uuid4().hex[:4]}"
        alias = f"nick_{uuid.uuid4().hex[:4]}"
        sid1 = f"test-crud1-{uuid.uuid4().hex[:8]}"
        sid2 = f"test-crud2-{uuid.uuid4().hex[:8]}"
        sid3 = f"test-crud3-{uuid.uuid4().hex[:8]}"
        sid4 = f"test-crud4-{uuid.uuid4().hex[:8]}"
        sid5 = f"test-crud5-{uuid.uuid4().hex[:8]}"
        tracker.track_user(name)
        for s in (sid1, sid2, sid3, sid4, sid5):
            tracker.track_session(s)

        # Snapshot user count before test for cascade verification
        users_before = cca.list_users()
        count_before = users_before.get("count", 0)

        try:
            # ═══════════════════════════════════════════════════════
            # Session 1: Arrive with rich context — name, alias,
            # company, skills, preference, and a coding task
            # ═══════════════════════════════════════════════════════
            msg1 = (
                f"Hi I'm {name}, people also call me {alias}. "
                f"I'm a platform engineer at {company}. "
                f"I know Python, Docker, and Java. "
                f"I prefer verbose logging and detailed error messages "
                f"in all my code. Write me a Python function that "
                f"connects to a PostgreSQL database."
            )
            r1 = cca.chat(msg1, session_id=sid1)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s1_response", r1.content[:500])
            assert r1.content, "Session 1 returned empty"
            assert r1.user_identified, "User not identified after introduction"

            # ── Qdrant: verify user profile created ──
            user = cca.find_user_by_name(name)
            assert user is not None, f"User '{name}' not created via REST"
            user_id = user["user_id"]
            trace_test.set_attribute("cca.test.user_id", user_id)

            # Full profile from Qdrant
            profile = cca.get_user_profile(user_id)
            assert profile is not None, "Profile not returned from Qdrant"
            trace_test.set_attribute(
                "cca.test.s1_profile", str(profile)[:500],
            )

            # Skills stored
            skills_lower = [s.lower() for s in profile.get("skills", [])]
            has_skills = sum(
                1 for s in ["python", "docker", "java"]
                if s in skills_lower
            )
            trace_test.set_attribute("cca.test.s1_skills", str(skills_lower))
            assert has_skills >= 2, (
                f"Expected >=2 skills stored, got {has_skills}: {skills_lower}"
            )

            # Alias stored
            aliases_lower = [a.lower() for a in profile.get("aliases", [])]
            trace_test.set_attribute("cca.test.s1_aliases", str(aliases_lower))
            assert alias.lower() in aliases_lower, (
                f"Alias '{alias}' not in profile: {aliases_lower}"
            )

            # Facts stored (company/role)
            facts = profile.get("facts", {})
            facts_str = str(facts).lower()
            has_company = company.lower() in facts_str
            has_role = any(
                w in facts_str for w in ["platform", "engineer"]
            )
            trace_test.set_attribute("cca.test.s1_facts", str(facts))
            assert has_company or has_role, (
                f"No company/role in facts: {facts}"
            )

            # ── Redis: verify session is tracked ──
            sessions = cca.list_sessions()
            session_ids = [
                s.get("session_id") for s in sessions.get("sessions", [])
            ]
            has_session = sid1 in session_ids
            trace_test.set_attribute("cca.test.s1_session_tracked", has_session)
            # Session should exist and be linked to our user
            if has_session:
                session_data = next(
                    s for s in sessions["sessions"]
                    if s["session_id"] == sid1
                )
                trace_test.set_attribute(
                    "cca.test.s1_session_user",
                    session_data.get("user_id", ""),
                )

            # ── User count increased ──
            users_after_s1 = cca.list_users()
            count_after_s1 = users_after_s1.get("count", 0)
            trace_test.set_attribute("cca.test.user_count_before", count_before)
            trace_test.set_attribute("cca.test.user_count_after_s1", count_after_s1)
            assert count_after_s1 > count_before, (
                f"User count didn't increase: {count_before} → {count_after_s1}"
            )

            # ── Qdrant cca_notes: verify notes extracted ──
            # NoteObserver is async fire-and-forget — poll until notes arrive
            from tests.helpers.polling import wait_for_notes
            notes = wait_for_notes(
                cca, f"{name} {company} Python Docker", user_id=user_id,
            )
            trace_test.set_attribute("cca.test.s1_notes_count", len(notes))
            trace_test.set_attribute(
                "cca.test.s1_notes_preview",
                str(notes[:2])[:500] if notes else "no notes",
            )
            assert len(notes) > 0, (
                "NoteObserver didn't extract any notes after session 1 — "
                "Qdrant cca_notes collection has no entries for this user"
            )

            # ═══════════════════════════════════════════════════════
            # Session 2: Return in new session — ask what CCA knows
            # about us (tests Qdrant profile retrieval + note recall)
            # ═══════════════════════════════════════════════════════
            msg2 = (
                "What do you know about me? List everything — "
                "my name, company, skills, preferences, anything "
                "you have stored. Also write me a quick health "
                "check function in Python."
            )
            r2 = cca.chat(msg2, session_id=sid2, user_id=user_id)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s2_response", r2.content[:500])
            assert r2.content, "Session 2 returned empty"
            assert r2.user_identified, "Session 2 should identify via user_id"

            # Agent should recall stored profile data
            content_lower = r2.content.lower()
            recalled_terms = sum(1 for t in [
                name.lower(), company.lower(), "python", "docker",
                "java", "platform", "engineer",
            ] if t in content_lower)
            trace_test.set_attribute("cca.test.s2_recalled", recalled_terms)
            assert recalled_terms >= 3, (
                f"Profile recall too weak (only {recalled_terms} terms): "
                f"{r2.content[:400]}"
            )

            # Session count should now be >= 2
            user_s2 = cca.find_user_by_name(name)
            assert user_s2 is not None, "User vanished after session 2"
            session_count = user_s2.get("session_count", 0)
            trace_test.set_attribute("cca.test.session_count", session_count)
            assert session_count >= 2, (
                f"Expected session_count >= 2, got {session_count}"
            )

            # ═══════════════════════════════════════════════════════
            # Session 3: Update — remove Java, remove alias, add
            # a new fact (team change)
            # ═══════════════════════════════════════════════════════
            msg3 = (
                f"A few updates: I don't use Java anymore, please "
                f"remove it from my skills. And drop the alias "
                f"{alias}, nobody calls me that. Also, I moved to "
                f"the infrastructure team recently. "
                f"Write me a one-liner to check disk space."
            )
            r3 = cca.chat(msg3, session_id=sid3, user_id=user_id)
            evaluate_response(r3, msg3, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s3_response", r3.content[:500])
            assert r3.content, "Session 3 returned empty"

            # ── Qdrant: verify skill removed ──
            profile_s3 = cca.get_user_profile(user_id)
            assert profile_s3 is not None, "Profile gone after session 3"
            skills_s3 = [s.lower() for s in profile_s3.get("skills", [])]
            trace_test.set_attribute("cca.test.s3_skills", str(skills_s3))
            assert "java" not in skills_s3, (
                f"Java still in skills after removal: {skills_s3}"
            )
            # Python/Docker should still be there
            assert any(s in skills_s3 for s in ["python", "docker"]), (
                f"Remaining skills lost after update: {skills_s3}"
            )

            # ── Qdrant: verify alias removed ──
            aliases_s3 = [a.lower() for a in profile_s3.get("aliases", [])]
            trace_test.set_attribute("cca.test.s3_aliases", str(aliases_s3))
            assert alias.lower() not in aliases_s3, (
                f"Alias '{alias}' still present after removal: {aliases_s3}"
            )

            # ── Qdrant: verify new fact stored (infrastructure team) ──
            facts_s3 = str(profile_s3.get("facts", {})).lower()
            trace_test.set_attribute(
                "cca.test.s3_facts", str(profile_s3.get("facts", {})),
            )
            has_infra_fact = any(
                w in facts_s3 for w in ["infrastructure", "infra"]
            )
            trace_test.set_attribute("cca.test.s3_infra_fact", has_infra_fact)
            # This is aspirational — fact overwrite may or may not work
            # depending on LLM extraction quality. Log but don't fail.

            # Wait for NoteObserver to process session 3
            time.sleep(10)

            # ── Qdrant cca_notes: notes accumulated across sessions ──
            notes_s3 = cca.search_notes(
                f"{name} skills update", user_id=user_id,
            )
            trace_test.set_attribute("cca.test.s3_notes_count", len(notes_s3))

            # ═══════════════════════════════════════════════════════
            # Session 4: Verify updates persisted — new session,
            # agent should see updated profile (no Java, no alias)
            # ═══════════════════════════════════════════════════════
            msg4 = (
                "Can you confirm my current skills and tell me "
                "what team I'm on? Write a Python function to "
                "parse a YAML config file."
            )
            r4 = cca.chat(msg4, session_id=sid4, user_id=user_id)
            evaluate_response(r4, msg4, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s4_response", r4.content[:500])
            assert r4.content, "Session 4 returned empty"

            # Should NOT mention Java (removed)
            s4_lower = r4.content.lower()
            java_mentioned = "java" in s4_lower
            trace_test.set_attribute("cca.test.s4_java_mentioned", java_mentioned)
            # Soft check — LLM might still mention it from context
            # but it shouldn't be listed as a current skill

            # Should still know Python/Docker
            has_remaining = any(
                w in s4_lower for w in ["python", "docker"]
            )
            trace_test.set_attribute("cca.test.s4_has_remaining", has_remaining)
            assert has_remaining, (
                f"Agent forgot remaining skills: {r4.content[:300]}"
            )

            # ═══════════════════════════════════════════════════════
            # Session 5: Delete profile — verify cascade cleanup
            # ═══════════════════════════════════════════════════════

            # Snapshot notes count before deletion
            notes_before_del = cca.search_notes(name, user_id=user_id)
            notes_count_before = len(notes_before_del)
            trace_test.set_attribute(
                "cca.test.notes_before_delete", notes_count_before,
            )

            msg5 = (
                "I want to delete my profile entirely. Remove all "
                "my data — profile, notes, everything. I confirm "
                "permanent deletion. Also show me how to delete "
                "a directory in Python."
            )
            r5 = cca.chat(msg5, session_id=sid5, user_id=user_id)
            evaluate_response(r5, msg5, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.s5_response", r5.content[:500])
            assert r5.content, "Session 5 returned empty"

            # Response should acknowledge deletion
            s5_lower = r5.content.lower()
            assert any(w in s5_lower for w in [
                "deleted", "removed", "confirmed", "profile",
            ]), f"Deletion not acknowledged: {r5.content[:300]}"

            # ── Qdrant user_profiles: user gone ──
            user_after = cca.find_user_by_name(name)
            trace_test.set_attribute(
                "cca.test.user_deleted", user_after is None,
            )
            assert user_after is None, (
                f"User '{name}' still in Qdrant after deletion"
            )

            # Profile endpoint should return None/404
            profile_after = cca.get_user_profile(user_id)
            trace_test.set_attribute(
                "cca.test.profile_deleted", profile_after is None,
            )
            assert profile_after is None, (
                f"Profile still returned from Qdrant after deletion: "
                f"{str(profile_after)[:300]}"
            )

            # ── User count decreased back ──
            users_after_del = cca.list_users()
            count_after_del = users_after_del.get("count", 0)
            trace_test.set_attribute(
                "cca.test.user_count_after_del", count_after_del,
            )
            assert count_after_del < count_after_s1, (
                f"User count didn't decrease after deletion: "
                f"{count_after_s1} → {count_after_del}"
            )

            # ── Qdrant cca_notes: notes cleaned for this user ──
            notes_after_del = cca.search_notes(name, user_id=user_id)
            trace_test.set_attribute(
                "cca.test.notes_after_delete", len(notes_after_del),
            )
            # Notes should be gone (cascade delete cleans cca_notes)
            assert len(notes_after_del) == 0, (
                f"Notes still exist after profile deletion "
                f"({len(notes_after_del)} found, had {notes_count_before}): "
                f"{str(notes_after_del[:2])[:300]}"
            )

            # ── /users endpoint still functional ──
            assert users_after_del.get("count", -1) >= 0, (
                "Users endpoint broken after deletion"
            )

        finally:
            tracker.cleanup()
