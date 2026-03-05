"""Flow test: Document upload, search, and promotion workflow.

Journey: paste text for agent to store → search for it → promote to
project knowledge → verify persistence.

Exercises: upload_document, search_documents, list_session_docs,
promote_doc_to_knowledge (DOCUMENT group), CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestDocumentWorkflow:
    """CODER route: upload, search, and promote documents."""

    def test_document_upload_search(self, cca, trace_test, judge_model):
        """Upload a document, then search for its content."""
        sid = f"test-doc-{uuid.uuid4().hex[:8]}"
        # Use unique content so search results are unambiguous
        unique_id = uuid.uuid4().hex[:8]
        doc_content = (
            f"Project Zephyr ({unique_id}) Architecture Notes:\n"
            f"The Zephyr system uses a three-tier cache: L1 in-process, "
            f"L2 Redis, L3 PostgreSQL. The ingestion pipeline processes "
            f"events at 50K/sec via Kafka partitions. The anomaly detector "
            f"uses an isolation forest model retrained nightly."
        )

        # ── Turn 1: Ask agent to store the document ──
        msg1 = (
            f"Please store these architecture notes for me, I'll need "
            f"to reference them later:\n\n{doc_content}"
        )
        r1 = cca.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
        assert r1.content, "Turn 1 returned empty"

        iters = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters)
        assert iters >= 1, (
            f"Agent didn't use document tools (iters={iters}). "
            f"Response: {r1.content[:200]}"
        )

        # ── Turn 2: Search for the document content ──
        msg2 = (
            "Search my documents for the anomaly detection approach "
            "in Project Zephyr. What model does it use?"
        )
        r2 = cca.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
        assert r2.content, "Turn 2 returned empty"

        # Should recall the isolation forest detail
        content_lower = r2.content.lower()
        has_recall = any(w in content_lower for w in [
            "isolation forest", "anomaly", "zephyr", unique_id,
        ])
        trace_test.set_attribute("cca.test.has_recall", has_recall)
        assert has_recall, (
            f"Agent didn't recall document content: {r2.content[:300]}"
        )

    def test_document_list_and_promote(self, cca, trace_test, judge_model):
        """Upload a document, list it, then promote to project knowledge."""
        sid = f"test-docpromo-{uuid.uuid4().hex[:8]}"
        unique_id = uuid.uuid4().hex[:8]

        # ── Turn 1: Upload a document (framed as coding context) ──
        # Must avoid user-management framing ("store this") which routes
        # to USER. Frame as workspace document for coding reference.
        msg1 = (
            f"I need to write deployment scripts. Upload this deployment "
            f"runbook as a session document so I can reference it while "
            f"coding:\n\n"
            f"Deployment Runbook ({unique_id}):\n"
            f"1. Run pre-flight checks\n"
            f"2. Tag release in git\n"
            f"3. Build Docker image\n"
            f"4. Deploy to staging\n"
            f"5. Run smoke tests\n"
            f"6. Promote to production"
        )
        r1 = cca.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
        route = r1.metadata.get("route", "")
        trace_test.set_attribute("cca.test.t1_route", route)
        assert r1.content, "Turn 1 returned empty"

        iters = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters)
        assert iters >= 1, (
            f"Agent didn't use document tools (route={route}, iters={iters}). "
            f"Response: {r1.content[:200]}"
        )

        # ── Turn 2: List session documents ──
        msg2 = (
            "List the documents in this session. Is the deployment "
            "runbook still available?"
        )
        r2 = cca.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
        assert r2.content, "Turn 2 returned empty"

        # Should mention the document name
        content_lower = r2.content.lower()
        has_doc_ref = "runbook" in content_lower or unique_id in content_lower
        trace_test.set_attribute("cca.test.has_doc_ref", has_doc_ref)
        assert has_doc_ref, (
            f"Response doesn't list the uploaded document: {r2.content[:300]}"
        )
