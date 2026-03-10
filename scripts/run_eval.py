#!/usr/bin/env python3
"""CCA Agent Evaluation Runner — Phoenix Experiments.

Runs the cca-agent-eval dataset through the CCA Agent-as-a-Model endpoint
and records results as Phoenix experiments with automated evaluators.

Usage:
    # Run full evaluation
    python -m scripts.run_eval

    # Custom experiment name
    python -m scripts.run_eval --name "baseline-v1"

    # Filter by category
    python -m scripts.run_eval --category code_gen

    # Skip evaluators (just record outputs)
    python -m scripts.run_eval --no-eval

Environment:
    CCA_URL         CCA agent endpoint (default: http://localhost:8500)
    PHOENIX_URL     Phoenix server (default: http://localhost:6006)
    DATASET_NAME    Phoenix dataset name (default: cca-agent-eval)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

# ==================== Configuration ====================

DEFAULT_CCA_URL = "http://localhost:8500"
DEFAULT_PHOENIX_URL = "http://localhost:6006"
DEFAULT_DATASET = "cca-agent-eval"
CHAT_TIMEOUT = 180  # seconds per CCA call


# ==================== Phoenix Client ====================


class PhoenixClient:
    """Minimal Phoenix REST API client for experiments."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def close(self) -> None:
        self._client.close()

    def get_dataset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a dataset by name."""
        resp = self._client.get(f"{self.base_url}/v1/datasets")
        resp.raise_for_status()
        for ds in resp.json()["data"]:
            if ds["name"] == name:
                return ds
        return None

    def get_examples(
        self, dataset_id: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all examples from a dataset, optionally filtered by category."""
        resp = self._client.get(
            f"{self.base_url}/v1/datasets/{dataset_id}/examples"
        )
        resp.raise_for_status()
        examples = resp.json()["data"]["examples"]
        if category:
            examples = [
                e for e in examples
                if e.get("input", {}).get("category") == category
            ]
        return examples

    def create_experiment(
        self, dataset_id: str, name: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a new experiment on a dataset."""
        payload: Dict[str, Any] = {"name": name}
        if metadata:
            payload["metadata"] = metadata
        resp = self._client.post(
            f"{self.base_url}/v1/datasets/{dataset_id}/experiments",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def submit_run(
        self,
        experiment_id: str,
        example_id: str,
        output: Dict[str, Any],
        start_time: str,
        end_time: str,
        error: Optional[str] = None,
    ) -> str:
        """Submit an experiment run result."""
        payload: Dict[str, Any] = {
            "dataset_example_id": example_id,
            "output": output,
            "repetition_number": 1,
            "start_time": start_time,
            "end_time": end_time,
        }
        if error:
            payload["error"] = error
        resp = self._client.post(
            f"{self.base_url}/v1/experiments/{experiment_id}/runs",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["data"]["id"]

    def submit_evaluation(
        self,
        experiment_run_id: str,
        name: str,
        score: Optional[float] = None,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
    ) -> str:
        """Submit an evaluation for an experiment run."""
        now = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {}
        if score is not None:
            result["score"] = score
        if label:
            result["label"] = label
        if explanation:
            result["explanation"] = explanation

        payload = {
            "experiment_run_id": experiment_run_id,
            "name": name,
            "annotator_kind": "CODE",
            "result": result,
            "start_time": now,
            "end_time": now,
        }
        resp = self._client.post(
            f"{self.base_url}/v1/experiment_evaluations",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["data"]["id"]


# ==================== CCA Client ====================


def call_cca(
    base_url: str, message: str, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Send a message to the CCA agent and return the raw response."""
    session_id = session_id or f"eval-{uuid.uuid4().hex[:12]}"
    payload = {
        "model": "cca",
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "session_id": session_id,
    }
    with httpx.Client(timeout=CHAT_TIMEOUT) as client:
        t0 = time.time()
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"X-Session-Id": session_id},
        )
        elapsed_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        return {
            "error": resp.text[:500],
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
        }

    data = resp.json()
    choices = data.get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "") or ""

    return {
        "content": content,
        "model": data.get("model", ""),
        "finish_reason": choices[0].get("finish_reason", "") if choices else "",
        "usage": data.get("usage", {}),
        "elapsed_ms": elapsed_ms,
        "session_id": session_id,
    }


# ==================== Evaluators ====================


def eval_not_empty(output: Dict[str, Any], _example: Dict) -> Dict[str, Any]:
    """Check that the response is non-empty."""
    content = output.get("content", "")
    passed = len(content.strip()) > 0
    return {
        "name": "not_empty",
        "score": 1.0 if passed else 0.0,
        "label": "pass" if passed else "fail",
        "explanation": f"Response length: {len(content)} chars",
    }


def eval_no_error(output: Dict[str, Any], _example: Dict) -> Dict[str, Any]:
    """Check that no error occurred."""
    has_error = "error" in output
    return {
        "name": "no_error",
        "score": 0.0 if has_error else 1.0,
        "label": "fail" if has_error else "pass",
        "explanation": output.get("error", "No error")[:200] if has_error else "Clean response",
    }


def eval_latency(output: Dict[str, Any], _example: Dict) -> Dict[str, Any]:
    """Score response latency (under 30s = 1.0, under 60s = 0.5, else 0.0)."""
    ms = output.get("elapsed_ms", 999999)
    if ms < 30000:
        score, label = 1.0, "fast"
    elif ms < 60000:
        score, label = 0.5, "moderate"
    else:
        score, label = 0.0, "slow"
    return {
        "name": "latency",
        "score": score,
        "label": label,
        "explanation": f"{ms:.0f}ms",
    }


def eval_code_present(output: Dict[str, Any], example: Dict) -> Optional[Dict[str, Any]]:
    """For code_gen/code_refactor/code_debug: check if response contains code."""
    category = example.get("input", {}).get("category", "")
    if category not in ("code_gen", "code_refactor", "code_debug"):
        return None  # Skip for non-code categories

    content = output.get("content", "")
    # Check for code blocks or common code patterns
    has_code_block = "```" in content
    has_def = re.search(r'\bdef\s+\w+', content) is not None
    has_code_pattern = re.search(r'(import |from |class |if __name__|print\(|return )', content) is not None

    passed = has_code_block or has_def or has_code_pattern
    return {
        "name": "code_present",
        "score": 1.0 if passed else 0.0,
        "label": "pass" if passed else "fail",
        "explanation": f"code_block={has_code_block}, def={has_def}, pattern={has_code_pattern}",
    }


def eval_expected_behavior(output: Dict[str, Any], example: Dict) -> Dict[str, Any]:
    """Check if response aligns with expected behavior keywords."""
    content = output.get("content", "").lower()
    expected = example.get("output", {}).get("expected_behavior", "").lower()

    if not expected:
        return {
            "name": "expected_behavior",
            "score": 0.5,
            "label": "no_reference",
            "explanation": "No expected behavior defined",
        }

    # Extract key terms from expected behavior
    # Remove common words and check for keyword matches
    stop_words = {"the", "a", "an", "is", "in", "to", "and", "or", "with", "for", "of", "its", "should"}
    keywords = [
        w for w in re.findall(r'\b\w+\b', expected)
        if w not in stop_words and len(w) > 2
    ]

    if not keywords:
        return {
            "name": "expected_behavior",
            "score": 0.5,
            "label": "no_keywords",
            "explanation": "Could not extract keywords from expected behavior",
        }

    matches = sum(1 for kw in keywords if kw in content)
    ratio = matches / len(keywords)

    if ratio >= 0.5:
        score, label = 1.0, "pass"
    elif ratio >= 0.25:
        score, label = 0.5, "partial"
    else:
        score, label = 0.0, "fail"

    return {
        "name": "expected_behavior",
        "score": score,
        "label": label,
        "explanation": f"{matches}/{len(keywords)} keywords matched ({ratio:.0%})",
    }


ALL_EVALUATORS = [
    eval_not_empty,
    eval_no_error,
    eval_latency,
    eval_code_present,
    eval_expected_behavior,
]


# ==================== Main Runner ====================


def run_evaluation(
    cca_url: str = DEFAULT_CCA_URL,
    phoenix_url: str = DEFAULT_PHOENIX_URL,
    dataset_name: str = DEFAULT_DATASET,
    experiment_name: Optional[str] = None,
    category: Optional[str] = None,
    run_evals: bool = True,
) -> Dict[str, Any]:
    """Run CCA evaluation experiment."""

    phoenix = PhoenixClient(phoenix_url)
    try:
        # 1. Find dataset
        dataset = phoenix.get_dataset_by_name(dataset_name)
        if not dataset:
            print(f"ERROR: Dataset '{dataset_name}' not found in Phoenix")
            sys.exit(1)

        dataset_id = dataset["id"]
        print(f"Dataset: {dataset_name} (id={dataset_id}, {dataset['example_count']} examples)")

        # 2. Fetch examples
        examples = phoenix.get_examples(dataset_id, category=category)
        if not examples:
            print(f"ERROR: No examples found" + (f" for category '{category}'" if category else ""))
            sys.exit(1)

        print(f"Examples: {len(examples)}" + (f" (filtered: {category})" if category else ""))

        # 3. Create experiment
        exp_name = experiment_name or f"cca-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        exp = phoenix.create_experiment(
            dataset_id,
            name=exp_name,
            metadata={
                "cca_url": cca_url,
                "category_filter": category,
                "evaluators": [e.__name__ for e in ALL_EVALUATORS] if run_evals else [],
            },
        )
        experiment_id = exp["id"]
        print(f"Experiment: {exp_name} (id={experiment_id})")
        print(f"Phoenix project: {exp.get('project_name', 'unknown')}")
        print("=" * 60)

        # 4. Run each example through CCA
        results = {
            "experiment_id": experiment_id,
            "experiment_name": exp_name,
            "total": len(examples),
            "success": 0,
            "failed": 0,
            "eval_scores": {},
        }

        for i, example in enumerate(examples, 1):
            ex_input = example.get("input", {})
            message = ex_input.get("message", "")
            cat = ex_input.get("category", "unknown")
            example_id = example["id"]

            print(f"\n[{i}/{len(examples)}] {cat}: {message[:60]}...")

            start_time = datetime.now(timezone.utc).isoformat()
            try:
                output = call_cca(cca_url, message)
                end_time = datetime.now(timezone.utc).isoformat()

                error_msg = output.get("error")
                content = output.get("content", "")
                preview = content[:100].replace("\n", " ") if content else "(empty)"
                elapsed = output.get("elapsed_ms", 0)

                if error_msg:
                    print(f"  ERROR: {error_msg[:100]}")
                    results["failed"] += 1
                else:
                    print(f"  OK ({elapsed:.0f}ms, {len(content)} chars): {preview}")
                    results["success"] += 1

            except Exception as e:
                end_time = datetime.now(timezone.utc).isoformat()
                output = {"error": str(e), "content": "", "elapsed_ms": 0}
                error_msg = str(e)
                print(f"  EXCEPTION: {e}")
                results["failed"] += 1

            # 5. Submit run to Phoenix
            run_id = phoenix.submit_run(
                experiment_id=experiment_id,
                example_id=example_id,
                output=output,
                start_time=start_time,
                end_time=end_time,
                error=error_msg if error_msg else None,
            )

            # 6. Run evaluators
            if run_evals:
                for evaluator in ALL_EVALUATORS:
                    eval_result = evaluator(output, example)
                    if eval_result is None:
                        continue  # Evaluator skipped (not applicable)

                    eval_name = eval_result["name"]
                    phoenix.submit_evaluation(
                        experiment_run_id=run_id,
                        name=eval_name,
                        score=eval_result.get("score"),
                        label=eval_result.get("label"),
                        explanation=eval_result.get("explanation"),
                    )

                    # Track aggregate scores
                    if eval_name not in results["eval_scores"]:
                        results["eval_scores"][eval_name] = []
                    results["eval_scores"][eval_name].append(eval_result["score"])

        # 7. Summary
        print("\n" + "=" * 60)
        print(f"EXPERIMENT COMPLETE: {exp_name}")
        print(f"  Success: {results['success']}/{results['total']}")
        print(f"  Failed:  {results['failed']}/{results['total']}")

        if results["eval_scores"]:
            print("\n  Evaluator Scores:")
            for name, scores in sorted(results["eval_scores"].items()):
                avg = sum(scores) / len(scores) if scores else 0
                print(f"    {name}: {avg:.2f} avg ({len(scores)} runs)")

        print(f"\n  View in Phoenix: {phoenix_url}")
        return results

    finally:
        phoenix.close()


# ==================== CLI ====================


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Run CCA agent evaluation experiment"
    )
    parser.add_argument(
        "--name", default=None,
        help="Experiment name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--category", default=None,
        help="Filter examples by category (e.g., code_gen, reasoning)",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluators (just record raw outputs)",
    )
    parser.add_argument(
        "--cca-url", default=os.environ.get("CCA_URL", DEFAULT_CCA_URL),
        help=f"CCA endpoint URL (default: {DEFAULT_CCA_URL})",
    )
    parser.add_argument(
        "--phoenix-url", default=os.environ.get("PHOENIX_URL", DEFAULT_PHOENIX_URL),
        help=f"Phoenix URL (default: {DEFAULT_PHOENIX_URL})",
    )
    parser.add_argument(
        "--dataset", default=os.environ.get("DATASET_NAME", DEFAULT_DATASET),
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )

    args = parser.parse_args()

    run_evaluation(
        cca_url=args.cca_url,
        phoenix_url=args.phoenix_url,
        dataset_name=args.dataset,
        experiment_name=args.name,
        category=args.category,
        run_evals=not args.no_eval,
    )


if __name__ == "__main__":
    main()
