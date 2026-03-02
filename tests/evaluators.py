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
    """Score: <120s = 1.0, <300s = 0.5, >300s = 0.0.

    Thresholds tuned for local LLMs on DGX Spark hardware. Each CCA
    request involves routing (Functionary) + multi-iteration agent loop
    (Qwen3-80B) + tool execution, so 60-120s is normal.
    """
    ms = result.elapsed_ms
    if ms < 120_000:
        score, label = SCORE_PASS, "fast"
    elif ms < 300_000:
        score, label = SCORE_PARTIAL, "moderate"
    else:
        score, label = SCORE_FAIL, "slow"
    return {
        "name": "latency",
        "annotator_kind": "CODE",
        "score": score,
        "label": label,
        "explanation": f"{ms/1000:.1f}s",
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


def eval_iteration_efficiency(result: ChatResult) -> Optional[Dict[str, Any]]:
    """Score based on iteration count vs estimated complexity.

    Simple tasks (estimated_steps <= 3) with >5 iterations = fail.
    All tasks with >15 iterations = fail.
    Advisory only — posted to Phoenix but doesn't gate pass/fail.
    """
    iters = result.metadata.get("tool_iterations", 0)
    steps = result.metadata.get("estimated_steps", 10)

    if steps <= 3 and iters > 5:
        score, label = SCORE_FAIL, "over-iterated"
    elif iters > 15:
        score, label = SCORE_FAIL, "excessive"
    elif iters > steps * 2:
        score, label = SCORE_PARTIAL, "high"
    else:
        score, label = SCORE_PASS, "efficient"

    return {
        "name": "iteration_efficiency",
        "annotator_kind": "CODE",
        "score": score,
        "label": label,
        "explanation": f"iterations={iters}, estimated_steps={steps}",
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

RESPONSE_QUALITY_TEMPLATE = """Rate this AI agent response. The agent (CCA) is a tool-augmented assistant with REAL capabilities beyond text generation:
- It HAS tools to create, delete, and modify user profiles (names, skills, aliases, facts)
- It CAN store and recall personal facts about users across sessions via persistent storage
- It CAN search the web and fetch URLs via integrated search tools
- It CAN execute code, manage files, and run shell commands

When the agent says it stored a fact, deleted a profile, added a skill, or recalled user info, it actually performed these actions via tool calls. Do NOT penalize these as fabricated or impossible.

Answer with ONLY the label on the first line, then a one-sentence explanation.

Labels: good | adequate | poor

Task category: {category}
User asked: {message}

Response: {response}

good = helpful, accurate, addresses the request appropriately for the task category
adequate = partially addresses but has gaps or minor issues
poor = unhelpful, incorrect, or completely off-topic

Label:"""

TASK_COMPLETION_TEMPLATE = """Did this AI agent complete the user's task? The agent (CCA) has real tools for: coding, user profile management (create/delete/modify profiles, skills, aliases), persistent fact storage/recall across sessions, web search, and URL fetching. When it says it performed an action, it actually did via tool calls.

Answer with ONLY the label on the first line, then a one-sentence explanation.

Labels: completed | partial | failed

Task category: {category}
User asked: {message}

Response: {response}

completed = task fully accomplished for the given category
partial = task partially done or incomplete
failed = task not accomplished or completely off-topic

Label:"""

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


def _extract_rail_from_thinking(content: str, rails: list) -> Optional[str]:
    """Extract a rail label from thinking-model output.

    Qwen3 thinking models embed reasoning in the content field (with or
    without <think> tags). This parser handles both cases:
      1. Strip <think>...</think> blocks, then look for the rail.
      2. Look for bold rail words like **good** (common in analysis).
      3. Look for a rail word on its own line.
      4. Fallback: last rail word mentioned (thinking comes first,
         answer comes last).
    """
    # 1. Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # 2. Check cleaned content for a rail on its own line
    for line in cleaned.split("\n"):
        word = line.strip().lower().rstrip(".:!,")
        if word in [r.lower() for r in rails]:
            return word

    # 3. Check for bold rail word like **good**
    for rail in rails:
        if f"**{rail}**" in cleaned.lower():
            return rail

    # 4. Fallback: last rail word in the full content (answer comes after thinking)
    last_match = None
    content_lower = content.lower()
    for rail in rails:
        idx = content_lower.rfind(rail.lower())
        if idx >= 0:
            if last_match is None or idx > last_match[1]:
                last_match = (rail, idx)
    if last_match:
        return last_match[0]

    return None


def _run_llm_classify(
    judge_model,
    message: str,
    response: str,
    template: str,
    rails: list,
    eval_name: str,
    category: str = "general",
) -> Dict[str, Any]:
    """Run LLM judge via direct API call with thinking-model support.

    Uses httpx to call vLLM directly instead of phoenix llm_classify,
    which can't handle Qwen3's thinking output (produces NOT_PARSABLE
    for ~50% of responses). Our _extract_rail_from_thinking parser
    handles <think> tags and embedded reasoning robustly.
    """
    import httpx

    # Map category codes to human-readable descriptions for the judge
    category_labels = {
        "user": "User profile management + coding task",
        "websearch": "Web search and information retrieval",
        "integration": "End-to-end integration (user lifecycle + coding)",
    }
    category_desc = category_labels.get(category, category)

    prompt = template.format(
        message=message[:2000],
        response=response[:2000],
        category=category_desc,
    )

    try:
        api_resp = httpx.post(
            f"{judge_model['base_url']}/chat/completions",
            json={
                "model": judge_model["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2048,  # Qwen3 thinking model needs room to think + label
            },
            timeout=120,
        )
        api_resp.raise_for_status()
        data = api_resp.json()

        content = data["choices"][0]["message"]["content"] or ""

        # Try reasoning_content first (vLLM may separate it)
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")

        # Extract rail label from content (handles thinking output)
        label = _extract_rail_from_thinking(content, rails)

        if label is None:
            label = "error"
            score = SCORE_FAIL
            explanation = f"Could not extract rail from: {content[:300]}"
        else:
            score = RAIL_SCORES.get(label, SCORE_FAIL)
            # Use cleaned content as explanation
            explanation = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()

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
    result: ChatResult, message: str, judge_model, category: str = "general",
) -> Dict[str, Any]:
    """LLM judges: Is this a helpful, accurate, complete response?"""
    return _run_llm_classify(
        judge_model,
        message=message,
        response=result.content,
        template=RESPONSE_QUALITY_TEMPLATE,
        rails=RESPONSE_QUALITY_RAILS,
        eval_name="response_quality",
        category=category,
    )


def eval_task_completion(
    result: ChatResult, message: str, judge_model, category: str = "general",
) -> Dict[str, Any]:
    """LLM judges: Did the agent accomplish the stated task?"""
    return _run_llm_classify(
        judge_model,
        message=message,
        response=result.content,
        template=TASK_COMPLETION_TEMPLATE,
        rails=TASK_COMPLETION_RAILS,
        eval_name="task_completion",
        category=category,
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

    # Iteration efficiency — always run (advisory, not gating)
    ev = eval_iteration_efficiency(result)
    if ev is not None:
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
            ev = llm_eval(result, message, judge_model, category=category)
            evals[ev["name"]] = ev

    # --- Set OpenInference I/O so Phoenix shows input/output columns ---
    trace_span.set_attribute("input.value", message)
    trace_span.set_attribute("output.value", result.content)

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

    # --- Gate on code evaluator failures ---
    # CODE evaluators are deterministic and catch real problems.
    # LLM judge results remain advisory (posted to Phoenix but don't gate).
    for ev in evals.values():
        if ev["annotator_kind"] != "CODE" or ev["score"] != SCORE_FAIL:
            continue
        # Latency is informational — slow != wrong
        if ev["name"] == "latency":
            continue
        # Code presence is advisory — not all "user" tests ask for code.
        # Individual tests assert code presence when they expect it.
        # Posted to Phoenix for visibility but doesn't gate pass/fail.
        if ev["name"] == "code_present":
            continue
        # User identification is advisory — anonymous sessions, deleted
        # users, and cross-session scenarios don't expect identification.
        # Individual tests assert identification when they expect it.
        if ev["name"] == "user_identified":
            continue
        # Iteration efficiency is advisory — posted to Phoenix for
        # visibility but doesn't gate pass/fail.
        if ev["name"] == "iteration_efficiency":
            continue
        raise AssertionError(
            f"Code evaluator '{ev['name']}' FAILED: "
            f"label={ev['label']}, explanation={ev.get('explanation', '')}"
        )

    return evals
