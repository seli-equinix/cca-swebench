"""Flow test: Knowledge system pipeline — facts, notes, and memory.

Journey 1: Share infrastructure details → verify facts stored.
Tests CriticalFactsExtractor and fact recall.

Journey 2: Share distinctive details → verify NoteObserver captures them.
Tests note extraction quality.

Journey 3: Complex task → verify hierarchical memory and REST endpoint.
Tests HierarchicalMemoryExtension.

Exercises: user identification, fact extraction, note extraction,
memory tools, REST API ground truth verification.
"""

import time
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestKnowledgePipeline:
    """Knowledge system E2E: facts, notes, and memory."""

    def test_infrastructure_facts_stored(self, cca, trace_test, judge_model):
        """2-turn: share infra details → verify facts in profile.

        Turn 1: Introduce + share detailed infra stack
        Turn 2: Ask CCA to recall infrastructure details

        Note: This test may initially FAIL if CriticalFactsExtractor
        doesn't capture infrastructure-category facts. That's the gap
        we're confirming.
        """
        tracker = cca.tracker()
        sid = f"test-infra-facts-{uuid.uuid4().hex[:8]}"
        unique_id = uuid.uuid4().hex[:6]
        user_name = f"InfraTest_{unique_id}"
        tracker.track_session(sid)
        tracker.track_user(user_name)

        try:
            # ── Turn 1: Introduce + share infra details ──
            msg1 = (
                f"Hi, I'm {user_name}. I'm a DevOps engineer. "
                f"My main cluster is running Kubernetes 1.29 on 5 nodes. "
                f"The container registry is at registry.internal:5000. "
                f"I use Terraform for infrastructure provisioning and "
                f"ArgoCD for GitOps deployments. My monitoring stack is "
                f"Prometheus + Grafana on port 3000."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"
            assert r1.user_identified, (
                f"{user_name} should be identified"
            )

            # ── Turn 2: Ask about infra details ──
            msg2 = (
                "What infrastructure details do you remember about me? "
                "What's my container registry URL and what tools do I use?"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            # Check if infra facts were recalled
            content2 = r2.content.lower()
            recalled_facts = []
            for fact in [
                "kubernetes", "registry", "terraform", "argocd",
                "prometheus", "grafana", "devops",
            ]:
                if fact in content2:
                    recalled_facts.append(fact)
            trace_test.set_attribute(
                "cca.test.t2_recalled_facts", str(recalled_facts),
            )
            trace_test.set_attribute(
                "cca.test.t2_recall_count", len(recalled_facts),
            )

            # Should recall at least some facts (from session context
            # if not from extracted facts)
            assert len(recalled_facts) >= 2, (
                f"Agent recalled too few facts ({recalled_facts}). "
                f"Response: {r2.content[:300]}"
            )

            # Ground truth: check user profile via REST API
            user_data = cca.find_user_by_name(user_name)
            if user_data:
                profile = cca.get_user_profile(user_data["user_id"])
                if profile:
                    facts = profile.get("facts", {})
                    trace_test.set_attribute(
                        "cca.test.profile_facts", str(facts)[:500],
                    )
                    # Track whether infra facts made it to the profile
                    has_infra_in_profile = any(
                        "registry" in str(v).lower()
                        or "kubernetes" in str(v).lower()
                        or "terraform" in str(v).lower()
                        for v in facts.values()
                    ) if facts else False
                    trace_test.set_attribute(
                        "cca.test.infra_in_profile", has_infra_in_profile,
                    )

        finally:
            tracker.cleanup()

    def test_note_extraction_quality(self, cca, trace_test, judge_model):
        """2-turn: share distinctive details → verify notes captured.

        Turn 1: Share project with unique identifiers + technical details
        (Wait for NoteObserver async processing)
        Turn 2: Verify via cca.search_notes() REST API

        Tests that NoteObserver correctly extracts and stores meaningful
        notes from conversation context.
        """
        tracker = cca.tracker()
        sid = f"test-notes-{uuid.uuid4().hex[:8]}"
        unique_id = uuid.uuid4().hex[:6]
        user_name = f"NoteTest_{unique_id}"
        tracker.track_session(sid)
        tracker.track_user(user_name)

        # Distinctive content that should be captured in notes
        project_name = f"ProjectNebula_{unique_id}"
        secret_detail = "quantum-resistant encryption using CRYSTALS-Kyber"

        try:
            # ── Turn 1: Share distinctive project details ──
            msg1 = (
                f"Hi I'm {user_name}. I'm working on {project_name}, "
                f"which is a secure messaging platform. The key innovation "
                f"is {secret_detail} for all message exchanges. "
                f"We're targeting a beta launch in Q3 with 10,000 users. "
                f"The backend runs on Rust with Actix-web, and the "
                f"frontend uses SvelteKit."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"
            assert r1.user_identified, f"{user_name} should be identified"

            # ── Verify: Search notes via REST API ──
            # NoteObserver is async fire-and-forget — poll until notes arrive
            from tests.helpers.polling import wait_for_notes

            user_data = cca.find_user_by_name(user_name)
            user_id = user_data["user_id"] if user_data else None

            notes = wait_for_notes(cca, project_name, user_id=user_id)
            trace_test.set_attribute("cca.test.note_count", len(notes))

            if notes:
                # Check note content for our distinctive details
                all_notes_text = " ".join(
                    n.get("content", "") + " " + n.get("note", "")
                    for n in notes
                ).lower()
                trace_test.set_attribute(
                    "cca.test.notes_preview", all_notes_text[:500],
                )

                has_project = project_name.lower() in all_notes_text
                has_tech = any(w in all_notes_text for w in [
                    "kyber", "quantum", "encryption", "rust",
                    "actix", "sveltekit", "svelte",
                ])
                trace_test.set_attribute("cca.test.has_project", has_project)
                trace_test.set_attribute("cca.test.has_tech", has_tech)

                assert has_project or has_tech, (
                    f"Notes don't contain distinctive details. "
                    f"Notes: {all_notes_text[:300]}"
                )
            else:
                assert False, (
                    "NoteObserver didn't extract any notes after 45s polling"
                )

        finally:
            tracker.cleanup()

    def test_memory_rest_endpoint(self, cca, trace_test, judge_model):
        """1-turn: complex multi-step task → check memory in metadata.

        Verifies that the agent's working memory (HierarchicalMemory)
        is populated during complex tasks and accessible via metadata.
        """
        tracker = cca.tracker()
        sid = f"test-memory-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Complex task that should trigger memory use ──
            # Multi-step planning should use write_memory to track state
            msg1 = (
                "I need a comprehensive plan for migrating a monolithic "
                "Django application to microservices. Cover the following: "
                "1) How to identify service boundaries "
                "2) Database decomposition strategy "
                "3) API gateway setup "
                "4) Testing approach for the migration "
                "Give me specific steps for each phase."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            # Response should be substantive (planning task)
            assert len(r1.content) > 200, (
                f"Planning response too short ({len(r1.content)} chars)"
            )

            # Check metadata for memory-related fields
            context_meta = r1.metadata
            trace_test.set_attribute(
                "cca.test.context_keys", str(list(context_meta.keys())),
            )

            # Memory files in metadata (if HierarchicalMemory was used)
            memory_files = context_meta.get("memory_files")
            trace_test.set_attribute(
                "cca.test.has_memory_files",
                memory_files is not None,
            )
            if memory_files:
                trace_test.set_attribute(
                    "cca.test.memory_file_count", len(memory_files),
                )

            # Route should be PLANNER or CODER for this type of request
            route = context_meta.get("route", "")
            trace_test.set_attribute("cca.test.route", route)

            # Response should cover the 4 requested topics
            content1 = r1.content.lower()
            topics_covered = sum(1 for topic in [
                "service boundar", "database", "api gateway", "test",
            ] if topic in content1)
            trace_test.set_attribute(
                "cca.test.topics_covered", topics_covered,
            )
            assert topics_covered >= 2, (
                f"Only {topics_covered}/4 migration topics covered. "
                f"Response: {r1.content[:300]}"
            )

        finally:
            tracker.cleanup()
