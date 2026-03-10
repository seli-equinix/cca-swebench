"""Flow test: Document upload, search, list, and promotion workflow.

Journey: paste text for agent to store → search for it → list and
promote to project knowledge. A developer storing reference material
while working on a project.

Exercises: upload_document, search_documents, list_session_docs,
promote_doc_to_knowledge (DOCUMENT group), CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestDocumentWorkflow:
    """CODER route: upload, search, list, and promote documents."""

    def test_document_workflow(self, cca, trace_test, judge_model):
        """3-turn flow: upload notes → search for content → list & promote."""
        tracker = cca.tracker()
        sid = f"test-doc-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            unique_id = uuid.uuid4().hex[:8]
            doc_content = (
                f"Project Zephyr ({unique_id}) Architecture Notes:\n"
                f"The Zephyr system uses a three-tier cache: L1 in-process, "
                f"L2 Redis, L3 PostgreSQL. The ingestion pipeline processes "
                f"events at 50K/sec via Kafka partitions. The anomaly detector "
                f"uses an isolation forest model retrained nightly."
            )

            # ── Turn 1: Upload architecture notes ──
            msg1 = (
                f"Please store these architecture notes for me, I'll need "
                f"to reference them later:\n\n{doc_content}"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            assert iters1 >= 1, (
                f"Agent didn't use document tools (iters={iters1}). "
                f"Response: {r1.content[:200]}"
            )

            # ── Turn 2: Search for specific content ──
            msg2 = (
                "Search my documents for the anomaly detection approach "
                "in Project Zephyr. What model does it use?"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
            assert r2.content, "Turn 2 returned empty"

            # Should recall the isolation forest detail
            content2 = r2.content.lower()
            has_recall = any(w in content2 for w in [
                "isolation forest", "anomaly", "zephyr", unique_id,
            ])
            trace_test.set_attribute("cca.test.t2_has_recall", has_recall)
            assert has_recall, (
                f"Agent didn't recall document content: {r2.content[:300]}"
            )

            # ── Turn 3: List documents and promote ──
            # Framed as workspace management — developer wants to keep
            # these notes for future sessions.
            msg3 = (
                "List the documents in this session. Is the Zephyr "
                "architecture doc still available? If so, promote it "
                "to permanent project knowledge so I can find it next time."
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't use document tools (iters={iters3})"
            )

            # Should mention the document
            content3 = r3.content.lower()
            has_doc_ref = any(w in content3 for w in [
                "zephyr", unique_id, "architecture", "promoted",
                "knowledge", "permanent",
            ])
            trace_test.set_attribute("cca.test.t3_has_doc_ref", has_doc_ref)
            assert has_doc_ref, (
                f"Response doesn't reference the document: {r3.content[:300]}"
            )

        finally:
            tracker.cleanup()
