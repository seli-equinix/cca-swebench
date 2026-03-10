"""Phoenix evaluators for the CCA test suite.

Two types of evaluators:
  1. Code evaluators (instant, deterministic) — always run inline during tests
  2. LLM judge evaluators (via direct vLLM call) — deferred to after all tests

Every evaluate_response() call is queued for a Phoenix Dataset + Experiment.
After all tests complete, the deferred runner:
  - Creates a Phoenix Dataset (input/output/metadata for each call)
  - Creates a Phoenix Experiment with code eval scores (always)
  - Runs LLM judge evaluators if --with-judge was used (deferred, no vLLM contention)
  - Posts LLM annotations to spans (Traces tab)

Usage in tests:
    evaluate_response(result, message, trace_test, judge_model, "user")
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from .cca_client import ChatResult

log = logging.getLogger(__name__)

# Score constants
SCORE_PASS = 1.0
SCORE_PARTIAL = 0.5
SCORE_FAIL = 0.0

# Evaluation queue — ALL evaluate_response() calls append here.
# Drained after all tests to create Phoenix Dataset + Experiment.
_EVAL_QUEUE: list[dict] = []
_TURN_COUNTERS: dict[str, int] = {}

PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")

# Lazy singleton for Phoenix client
_phoenix_client = None


def _get_phoenix_client():
    """Get or create a singleton Phoenix client."""
    global _phoenix_client
    if _phoenix_client is None:
        try:
            from phoenix.client import Client
            _phoenix_client = Client(base_url=PHOENIX_URL)
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


def _get_trace_id(span) -> Optional[str]:
    """Extract the OTLP hex trace ID from an OpenTelemetry span.

    Returns 32-char hex string (128-bit trace ID) for Phoenix experiment
    run linking.
    """
    try:
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
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


def eval_tool_errors(result: ChatResult) -> Optional[Dict[str, Any]]:
    """Score 0.0 if any tool execution errors occurred during the response.

    Captures SSE comment labels that indicate tool failures (validation
    errors, command failures, parsing errors). Surfaces these prominently
    so they don't get lost in test output.
    """
    errors = getattr(result, "tool_errors", [])
    if not errors:
        return {
            "name": "tool_errors",
            "annotator_kind": "CODE",
            "score": SCORE_PASS,
            "label": "clean",
            "explanation": "No tool errors",
        }
    # Truncate to first 5 errors for readability
    error_summary = "; ".join(errors[:5])
    if len(errors) > 5:
        error_summary += f" (+{len(errors) - 5} more)"
    return {
        "name": "tool_errors",
        "annotator_kind": "CODE",
        "score": SCORE_FAIL,
        "label": f"{len(errors)}_errors",
        "explanation": error_summary,
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

Agent execution metadata (ground truth from server):
- Route: {route} (the processing pipeline used)
- Tool iterations: {tool_iterations} (0 = direct answer, 1+ = agent called tools)
- User identified: {user_identified} (True = agent recognized and/or created user profile)

Use this metadata to inform your rating. If tool_iterations > 0, the agent actively worked on the task. If user_identified is True, the introduction was processed.

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

Agent execution metadata (ground truth from server):
- Route: {route} (the processing pipeline used)
- Tool iterations: {tool_iterations} (0 = direct answer, 1+ = agent called tools)
- User identified: {user_identified} (True = agent recognized and/or created user profile)

Use this metadata to inform your rating. If tool_iterations > 0, the agent actively worked on the task.

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
    metadata: Optional[Dict[str, Any]] = None,
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
        "coder": "Code generation, editing, search, and execution",
        "websearch": "Web search and information retrieval",
        "integration": "End-to-end integration (user lifecycle + coding)",
    }
    category_desc = category_labels.get(category, category)

    format_kwargs = {
        "message": message,
        "response": response,
        "category": category_desc,
        "route": "",
        "tool_iterations": 0,
        "user_identified": False,
    }
    if metadata:
        format_kwargs.update(metadata)
    prompt = template.format(**format_kwargs)

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
        metadata={
            "user_identified": getattr(result, "user_identified", False),
            "tool_iterations": getattr(result, "metadata", {}).get("tool_iterations", 0),
            "route": getattr(result, "metadata", {}).get("route", ""),
        },
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
        metadata={
            "user_identified": getattr(result, "user_identified", False),
            "tool_iterations": getattr(result, "metadata", {}).get("tool_iterations", 0),
            "route": getattr(result, "metadata", {}).get("route", ""),
        },
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

    # Tool errors — always run (advisory, surfaces SSE error labels)
    ev = eval_tool_errors(result)
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

    # --- Surface tool errors prominently in test output ---
    tool_errors = getattr(result, "tool_errors", [])
    if tool_errors:
        log.warning(
            "TOOL ERRORS detected (%d): %s",
            len(tool_errors),
            "; ".join(tool_errors[:5]),
        )

    # --- Set OpenInference I/O so Phoenix shows input/output columns ---
    trace_span.set_attribute("input.value", message)
    trace_span.set_attribute("output.value", result.content)
    if tool_errors:
        trace_span.set_attribute("cca.eval.tool_errors", "; ".join(tool_errors[:10]))

    # --- Log to span attributes (for filtering) ---
    for ev in evals.values():
        name = ev["name"]
        trace_span.set_attribute(f"cca.eval.{name}.score", ev["score"])
        trace_span.set_attribute(f"cca.eval.{name}.label", ev["label"])

    # --- Defer code eval annotations until span is closed + flushed ---
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

    # --- Queue for Phoenix Dataset + Experiment (ALWAYS) ---
    # Every evaluate_response() call creates a dataset example.
    # LLM judge runs deferred after all tests (only if judge_model provided).
    test_name = getattr(trace_span, "name", "unknown")
    turn = _TURN_COUNTERS.get(test_name, 0) + 1
    _TURN_COUNTERS[test_name] = turn

    _EVAL_QUEUE.append({
        "message": message,
        "response": result.content,
        "category": category,
        "test_name": test_name,
        "turn": turn,
        "span_id": span_id,
        "trace_id": _get_trace_id(trace_span),
        "run_judge": judge_model is not None,
        "judge_model": judge_model,
        "code_evals": {k: dict(v) for k, v in evals.items()},
        "user_identified": result.user_identified,
        "tool_iterations": result.metadata.get("tool_iterations", 0),
        "route": result.metadata.get("route", ""),
    })

    # --- Gate on code evaluator failures ---
    # CODE evaluators are deterministic and catch real problems.
    for ev in evals.values():
        if ev["annotator_kind"] != "CODE" or ev["score"] != SCORE_FAIL:
            continue
        # Latency is informational — slow != wrong
        if ev["name"] == "latency":
            continue
        # Code presence is advisory — not all "user" tests ask for code.
        # Individual tests assert code presence when they expect it.
        if ev["name"] == "code_present":
            continue
        # User identification is advisory — anonymous sessions, deleted
        # users, and cross-session scenarios don't expect identification.
        if ev["name"] == "user_identified":
            continue
        # Iteration efficiency is advisory.
        if ev["name"] == "iteration_efficiency":
            continue
        # Tool errors are advisory — surfaced for visibility, not gating.
        if ev["name"] == "tool_errors":
            continue
        raise AssertionError(
            f"Code evaluator '{ev['name']}' FAILED: "
            f"label={ev['label']}, explanation={ev.get('explanation', '')}"
        )

    # LLM judge gating is deferred — runs after all tests via
    # run_deferred_experiment() in conftest.py's deferred_experiment fixture.

    return evals


# ==================== Deferred Experiment Runner ====================


def run_deferred_experiment(run_judge: bool = False) -> list[dict]:
    """Create Phoenix Dataset + Experiment from ALL test results.

    Always creates dataset + experiment with code evaluator scores.
    If run_judge=True, also runs LLM judge on eligible items.

    Args:
        run_judge: Whether to run LLM judge evaluators (--with-judge flag)

    Returns:
        List of LLM judge result dicts (empty if run_judge=False)
    """
    if not _EVAL_QUEUE:
        return []

    import httpx
    from datetime import datetime, timezone

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    total = len(_EVAL_QUEUE)
    judge_eligible = sum(1 for item in _EVAL_QUEUE if item["run_judge"])

    # --- 1. Create Phoenix Dataset (ALL items) ---
    client = _get_phoenix_client()
    if client is None:
        log.warning("Phoenix unavailable — running judge-only fallback")
        return _run_judge_only() if run_judge else []

    try:
        dataset = client.datasets.create_dataset(
            name=f"cca-tests-{timestamp}",
            inputs=[
                {"message": item["message"], "category": item["category"]}
                for item in _EVAL_QUEUE
            ],
            outputs=[
                {"response": item["response"][:4000]}
                for item in _EVAL_QUEUE
            ],
            metadata=[
                {
                    "test_name": item["test_name"],
                    "turn": item["turn"],
                    "span_id": item.get("span_id", ""),
                    "run_judge": item["run_judge"],
                }
                for item in _EVAL_QUEUE
            ],
        )
        dataset_id = dataset.id
        print(f"  Dataset: cca-tests-{timestamp} ({total} examples)")
    except Exception as e:
        log.warning("Dataset creation failed: %s", e)
        return _run_judge_only() if run_judge else []

    # --- 2. Create Experiment ---
    try:
        http = httpx.Client(timeout=30)
        exp_resp = http.post(
            f"{PHOENIX_URL}/v1/datasets/{dataset_id}/experiments",
            json={
                "name": f"pytest-{timestamp}",
                "metadata": {
                    "source": "pytest",
                    "total_items": total,
                    "judge_eligible": judge_eligible,
                    "judge_enabled": run_judge,
                },
            },
        )
        exp_resp.raise_for_status()
        experiment_id = exp_resp.json()["data"]["id"]
        print(f"  Experiment: pytest-{timestamp}")
    except Exception as e:
        log.warning("Experiment creation failed: %s", e)
        http.close()
        return _run_judge_only() if run_judge else []

    # --- 3. Process ALL queue items ---
    judge_results = []
    examples = _get_dataset_examples(http, dataset_id)

    for i, item in enumerate(_EVAL_QUEUE):
        test_name = item["test_name"]
        turn = item["turn"]
        will_judge = run_judge and item["run_judge"]
        preview = item["message"][:50]

        status_prefix = "J" if will_judge else "C"
        print(
            f"  [{status_prefix}] [{i+1}/{total}] {test_name} t{turn}: "
            f"{preview}...",
            end="", flush=True,
        )

        now = datetime.now(timezone.utc).isoformat()
        example_id = examples[i]["id"] if i < len(examples) else None
        if not example_id:
            print(" skip", flush=True)
            continue

        # 3a. Submit run (pre-collected CCA output)
        try:
            run_payload = {
                "dataset_example_id": example_id,
                "output": {"response": item["response"][:2000]},
                "repetition_number": 1,
                "start_time": now,
                "end_time": now,
            }
            if item.get("trace_id"):
                run_payload["trace_id"] = item["trace_id"]
            run_resp = http.post(
                f"{PHOENIX_URL}/v1/experiments/{experiment_id}/runs",
                json=run_payload,
            )
            run_resp.raise_for_status()
            run_id = run_resp.json()["data"]["id"]
        except Exception as e:
            log.warning("Run submission failed: %s", e)
            print(" run_err", flush=True)
            continue

        # 3b. Submit code evaluator results (ALL items)
        for ev in item.get("code_evals", {}).values():
            _submit_evaluation(http, run_id, ev, annotator_kind="CODE")

        # 3c. Run LLM judge (only for eligible items with --with-judge)
        if will_judge:
            item_judge = []
            mock_result = type("R", (), {
                "content": item["response"],
                "user_identified": item.get("user_identified", False),
                "metadata": {
                    "tool_iterations": item.get("tool_iterations", 0),
                    "route": item.get("route", ""),
                },
            })()
            for llm_eval in [eval_response_quality, eval_task_completion]:
                ev = llm_eval(
                    mock_result,
                    item["message"],
                    item["judge_model"],
                    category=item["category"],
                )
                ev["test_name"] = test_name
                ev["turn"] = turn
                ev["span_id"] = item.get("span_id")
                item_judge.append(ev)
                judge_results.append(ev)

                # Submit to experiment
                _submit_evaluation(http, run_id, ev, annotator_kind="LLM")

            # Post LLM annotations to span (Traces tab)
            if item.get("span_id"):
                for ev in item_judge:
                    _post_annotation(
                        span_id=ev["span_id"],
                        name=ev["name"],
                        annotator_kind="LLM",
                        label=ev["label"],
                        score=ev["score"],
                        explanation=ev.get("explanation", ""),
                    )

            has_fail = any(
                ev["label"] in ("poor", "failed") for ev in item_judge
            )
            print(f" {'FAIL' if has_fail else 'ok'}", flush=True)
        else:
            print(" ok", flush=True)

    http.close()
    _EVAL_QUEUE.clear()
    _TURN_COUNTERS.clear()
    return judge_results


def _submit_evaluation(
    http, run_id: str, ev: dict, annotator_kind: str = "CODE",
):
    """Submit one evaluation to a Phoenix experiment run."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    try:
        http.post(
            f"{PHOENIX_URL}/v1/experiment_evaluations",
            json={
                "experiment_run_id": run_id,
                "name": ev["name"],
                "annotator_kind": annotator_kind,
                "result": {
                    "score": ev.get("score", 0),
                    "label": ev.get("label", ""),
                    "explanation": ev.get("explanation", "")[:500],
                },
                "start_time": now,
                "end_time": now,
            },
        )
    except Exception as e:
        log.warning("Eval submission failed (%s): %s", ev["name"], e)


def _get_dataset_examples(http, dataset_id: str) -> list:
    """Fetch dataset examples in creation order."""
    try:
        resp = http.get(
            f"{PHOENIX_URL}/v1/datasets/{dataset_id}/examples",
        )
        resp.raise_for_status()
        return resp.json()["data"]["examples"]
    except Exception as e:
        log.warning("Failed to fetch examples: %s", e)
        return []


def _run_judge_only() -> list[dict]:
    """Fallback: run LLM judge without Phoenix experiment."""
    results = []
    total = len(_EVAL_QUEUE)
    for i, item in enumerate(_EVAL_QUEUE):
        if not item["run_judge"]:
            continue
        test_name = item["test_name"]
        turn = item["turn"]
        print(
            f"  [{i+1}/{total}] {test_name} t{turn}...",
            end="", flush=True,
        )

        mock_result = type("R", (), {
            "content": item["response"],
            "user_identified": item.get("user_identified", False),
            "metadata": {
                "tool_iterations": item.get("tool_iterations", 0),
                "route": item.get("route", ""),
            },
        })()
        for llm_eval in [eval_response_quality, eval_task_completion]:
            ev = llm_eval(
                mock_result,
                item["message"],
                item["judge_model"],
                category=item["category"],
            )
            ev["test_name"] = test_name
            ev["turn"] = turn
            ev["span_id"] = item.get("span_id")
            results.append(ev)

        if item.get("span_id"):
            for ev in results[-2:]:
                _post_annotation(
                    span_id=ev["span_id"],
                    name=ev["name"],
                    annotator_kind="LLM",
                    label=ev["label"],
                    score=ev["score"],
                    explanation=ev.get("explanation", ""),
                )

        has_fail = any(
            ev["label"] in ("poor", "failed") for ev in results[-2:]
        )
        print(f" {'FAIL' if has_fail else 'ok'}", flush=True)

    _EVAL_QUEUE.clear()
    _TURN_COUNTERS.clear()
    return results
