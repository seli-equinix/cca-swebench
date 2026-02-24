"""Phoenix evaluators for the CCA test suite.

Two types of evaluators:
  1. Code evaluators (instant, deterministic) — always run
  2. LLM judge evaluators (via direct vLLM call) — run when judge_model available

The LLM judge talks DIRECTLY to vLLM on Spark2:8000/v1 (raw OpenAI-compatible
API). It does NOT go through CCA. CCA is the system under test; the judge is
an independent assessor.

Results are posted to Phoenix as **span annotations** (not just span attributes)
so they appear in the Annotations tab of the Phoenix UI.

Usage in tests:
    evals = evaluate_response(result, message, trace_test, judge_model, "user")
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .cca_client import ChatResult

log = logging.getLogger(__name__)

# Score constants
SCORE_PASS = 1.0
SCORE_PARTIAL = 0.5
SCORE_FAIL = 0.0

# Lazy singleton for Phoenix client
_phoenix_client = None


def _get_phoenix_client():
    """Get or create a singleton Phoenix client."""
    global _phoenix_client
    if _phoenix_client is None:
        try:
            from phoenix.client import Client
            _phoenix_client = Client(base_url="http://192.168.4.204:6006")
        except Exception as e:
            log.warning("Could not create Phoenix client: %s", e)
    return _phoenix_client


def _get_span_id(span) -> Optional[str]:
    """Extract the OTLP hex span ID from an OpenTelemetry span.

    Phoenix needs the hex format (e.g. '7c2304adbf38783d'), not the
    base64 internal ID.
    """
    try:
        ctx = span.get_span_context()
        if ctx and ctx.span_id:
            return format(ctx.span_id, "016x")
    except Exception:
        pass
    return None


def _post_annotation(
    span_id: str,
    name: str,
    annotator_kind: str,
    label: str,
    score: float,
    explanation: str = "",
) -> None:
    """Post a single annotation to Phoenix via the client API.

    Best-effort: logs warnings on failure, never raises.
    """
    client = _get_phoenix_client()
    if client is None:
        return

    try:
        client.spans.add_span_annotation(
            span_id=span_id,
            annotation_name=name,
            annotator_kind=annotator_kind,
            label=label,
            score=score,
            explanation=explanation[:500] if explanation else "",
            sync=True,
        )
    except Exception as e:
        log.warning("Failed to post annotation '%s': %s", name, e)


def post_deferred_annotations(span) -> None:
    """Post all annotations that were collected during the test.

    Called by the trace_test fixture in conftest.py AFTER the span is
    closed and force_flush() has delivered it to Phoenix.
    """
    pending = getattr(span, "_pending_annotations", [])
    if not pending:
        return

    for anno in pending:
        _post_annotation(
            span_id=anno["span_id"],
            name=anno["name"],
            annotator_kind=anno["annotator_kind"],
            label=anno["label"],
            score=anno["score"],
            explanation=anno.get("explanation", ""),
        )

    log.info("Posted %d annotations for span %s", len(pending), pending[0]["span_id"])


# ==================== Code Evaluators (no LLM) ====================


def eval_not_empty(result: ChatResult) -> Dict[str, Any]:
    """Score 1.0 if response has content, 0.0 if empty."""
    content = result.content.strip()
    passed = len(content) > 0
    return {
        "name": "response_not_empty",
        "annotator_kind": "CODE",
        "score": SCORE_PASS if passed else SCORE_FAIL,
        "label": "pass" if passed else "fail",
        "explanation": f"Response length: {len(content)} chars",
    }


def eval_no_error(result: ChatResult) -> Dict[str, Any]:
    """Score 1.0 if no HTTP/API error in response."""
    has_error = "error" in result.raw and result.raw["error"]
    return {
        "name": "no_error",
        "annotator_kind": "CODE",
        "score": SCORE_FAIL if has_error else SCORE_PASS,
        "label": "fail" if has_error else "pass",
        "explanation": str(result.raw.get("error", ""))[:200] if has_error else "Clean response",
    }


def eval_latency(result: ChatResult) -> Dict[str, Any]:
    """Score: <30s = 1.0, <60s = 0.5, >60s = 0.0."""
    ms = result.elapsed_ms
    if ms < 30000:
        score, label = SCORE_PASS, "fast"
    elif ms < 60000:
        score, label = SCORE_PARTIAL, "moderate"
    else:
        score, label = SCORE_FAIL, "slow"
    return {
        "name": "latency",
        "annotator_kind": "CODE",
        "score": score,
        "label": label,
        "explanation": f"{ms:.0f}ms",
    }


def eval_code_present(result: ChatResult) -> Optional[Dict[str, Any]]:
    """Score 1.0 if code blocks/patterns detected. Returns None if N/A."""
    content = result.content
    has_code_block = "```" in content
    has_def = re.search(r"\bdef\s+\w+", content) is not None
    has_code_pattern = re.search(
        r"(import |from |class |if __name__|print\(|return )", content
    ) is not None

    passed = has_code_block or has_def or has_code_pattern
    return {
        "name": "code_present",
        "annotator_kind": "CODE",
        "score": SCORE_PASS if passed else SCORE_FAIL,
        "label": "pass" if passed else "fail",
        "explanation": f"code_block={has_code_block}, def={has_def}, pattern={has_code_pattern}",
    }


def eval_user_identified(result: ChatResult) -> Optional[Dict[str, Any]]:
    """Score 1.0 if user_identified flag is True in response metadata."""
    identified = result.user_identified
    return {
        "name": "user_identified",
        "annotator_kind": "CODE",
        "score": SCORE_PASS if identified else SCORE_FAIL,
        "label": "pass" if identified else "fail",
        "explanation": f"user_name={result.user_name or 'none'}",
    }


# ==================== LLM Judge Evaluators (direct vLLM) ====================

# Prompt templates for llm_classify. Variables in {curly_braces} are
# replaced by column names from the DataFrame passed to llm_classify.

RESPONSE_QUALITY_TEMPLATE = """You are evaluating the quality of an AI coding assistant's response.

User's message: {message}

Assistant's response: {response}

Rate the quality of this response:
- good: The response is helpful, accurate, and addresses the user's request
- adequate: The response partially addresses the request but has issues
- poor: The response is unhelpful, incorrect, or doesn't address the request"""

TASK_COMPLETION_TEMPLATE = """You are evaluating whether an AI coding assistant completed the user's task.

User's message: {message}

Assistant's response: {response}

Did the assistant complete what was asked?
- completed: The task was fully accomplished
- partial: The task was partially done or the answer is incomplete
- failed: The task was not accomplished or the response is off-topic"""

RESPONSE_QUALITY_RAILS = ["good", "adequate", "poor"]
TASK_COMPLETION_RAILS = ["completed", "partial", "failed"]

RAIL_SCORES = {
    "good": SCORE_PASS,
    "adequate": SCORE_PARTIAL,
    "poor": SCORE_FAIL,
    "completed": SCORE_PASS,
    "partial": SCORE_PARTIAL,
    "failed": SCORE_FAIL,
}


def _run_llm_classify(
    judge_model,
    message: str,
    response: str,
    template: str,
    rails: list,
    eval_name: str,
) -> Dict[str, Any]:
    """Run llm_classify with a single example. Returns eval dict."""
    from phoenix.evals import llm_classify

    df = pd.DataFrame([{"message": message[:2000], "response": response[:2000]}])

    try:
        result_df = llm_classify(
            data=df,
            model=judge_model,
            template=template,
            rails=rails,
            provide_explanation=True,
            verbose=False,
        )
        label = result_df["label"].iloc[0]
        explanation = result_df.get("explanation", pd.Series([""])).iloc[0] or ""

        if label is None or (isinstance(label, float) and pd.isna(label)):
            label = "error"
            score = SCORE_FAIL
            explanation = "Judge returned no label"
        else:
            score = RAIL_SCORES.get(label, SCORE_FAIL)

    except Exception as e:
        label = "error"
        score = SCORE_FAIL
        explanation = f"Judge error: {e}"

    return {
        "name": eval_name,
        "annotator_kind": "LLM",
        "score": score,
        "label": str(label),
        "explanation": str(explanation)[:500],
    }


def eval_response_quality(
    result: ChatResult, message: str, judge_model
) -> Dict[str, Any]:
    """LLM judges: Is this a helpful, accurate, complete response?"""
    return _run_llm_classify(
        judge_model,
        message=message,
        response=result.content,
        template=RESPONSE_QUALITY_TEMPLATE,
        rails=RESPONSE_QUALITY_RAILS,
        eval_name="response_quality",
    )


def eval_task_completion(
    result: ChatResult, message: str, judge_model
) -> Dict[str, Any]:
    """LLM judges: Did the agent accomplish the stated task?"""
    return _run_llm_classify(
        judge_model,
        message=message,
        response=result.content,
        template=TASK_COMPLETION_TEMPLATE,
        rails=TASK_COMPLETION_RAILS,
        eval_name="task_completion",
    )


# ==================== Main Entry Point ====================


def evaluate_response(
    result: ChatResult,
    message: str,
    trace_span,
    judge_model=None,
    category: str = "user",
) -> Dict[str, Dict[str, Any]]:
    """Run all applicable evaluators and post results as Phoenix annotations.

    Each eval result is:
      1. Set as a span attribute (for quick filtering in Phoenix)
      2. Posted as a Phoenix span annotation (appears in Annotations tab)

    Args:
        result: ChatResult from cca.chat()
        message: The original user message sent to CCA
        trace_span: The trace_test span fixture (OTel span)
        judge_model: OpenAIModel for LLM judge (None = skip LLM evals)
        category: Test category ("user", "websearch", "integration")

    Returns:
        Dict of {eval_name: {name, score, label, explanation}}
    """
    evals: Dict[str, Dict[str, Any]] = {}

    # --- Code evaluators (always run) ---
    for evaluator in [eval_not_empty, eval_no_error, eval_latency]:
        ev = evaluator(result)
        evals[ev["name"]] = ev

    # Code presence — only for tests that involve code
    if category in ("user", "integration"):
        ev = eval_code_present(result)
        if ev is not None:
            evals[ev["name"]] = ev

    # User identification — only for user tests
    if category == "user":
        ev = eval_user_identified(result)
        if ev is not None:
            evals[ev["name"]] = ev

    # --- LLM judge evaluators (only if judge available) ---
    if judge_model is not None and result.content:
        for llm_eval in [eval_response_quality, eval_task_completion]:
            ev = llm_eval(result, message, judge_model)
            evals[ev["name"]] = ev

    # --- Log to span attributes (for filtering) ---
    for ev in evals.values():
        name = ev["name"]
        trace_span.set_attribute(f"cca.eval.{name}.score", ev["score"])
        trace_span.set_attribute(f"cca.eval.{name}.label", ev["label"])

    # --- Defer annotations until span is closed + flushed ---
    # The trace_test fixture in conftest.py calls post_deferred_annotations()
    # AFTER the span closes and force_flush() completes. This avoids 404
    # errors from Phoenix when the span hasn't arrived yet.
    span_id = _get_span_id(trace_span)
    if span_id and hasattr(trace_span, "_pending_annotations"):
        for ev in evals.values():
            trace_span._pending_annotations.append({
                "span_id": span_id,
                "name": ev["name"],
                "annotator_kind": ev.get("annotator_kind", "CODE"),
                "label": ev["label"],
                "score": ev["score"],
                "explanation": ev.get("explanation", ""),
            })
    elif span_id:
        # Fallback: post immediately if no deferred queue (shouldn't happen)
        for ev in evals.values():
            _post_annotation(
                span_id=span_id,
                name=ev["name"],
                annotator_kind=ev.get("annotator_kind", "CODE"),
                label=ev["label"],
                score=ev["score"],
                explanation=ev.get("explanation", ""),
            )

    return evals
