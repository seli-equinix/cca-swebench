"""CCA test suite — Phoenix-traced integration tests.

All traces go to a single Phoenix project (cca-http) so each request
shows as ONE unified trace with all steps: routing → agent → tool calls.
Tests are categorized via span attributes, not separate projects.

Every test creates spans visible in the Phoenix UI (set PHOENIX_URL env var).

Annotations are deferred until AFTER the span is closed and flushed to
Phoenix. This avoids 404 errors caused by posting annotations for spans
that haven't arrived at the server yet (BatchSpanProcessor buffers for
5 seconds by default).

Server load management:
  - Inter-test cooldown (CCA_TEST_COOLDOWN env var, default 3s) prevents
    overwhelming the CCA server and vLLM backend between tests.
  - LLM judge is opt-in (--with-judge flag or CCA_RUN_JUDGE=1 env var)
    since it doubles vLLM load by making 2 additional calls per test.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .cca_client import CCAClient, TIMEOUT_DIAGNOSTIC
from .evaluators import post_deferred_annotations

log = logging.getLogger(__name__)

# ==================== Configuration ====================

PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
CCA_BASE_URL = os.getenv("CCA_BASE_URL", "http://localhost:8500")

# Same Phoenix project as the CCA server — test + server spans unified
PROJECT_NAME = "cca-http"

# Inter-test cooldown (seconds) to prevent server overload.
# Each test triggers LLM inference on vLLM — without cooldown,
# sequential tests pile up requests faster than vLLM can drain them.
TEST_COOLDOWN = float(os.getenv("CCA_TEST_COOLDOWN", "3"))


# ==================== Markers ====================


def pytest_addoption(parser):
    parser.addoption(
        "--with-judge",
        action="store_true",
        default=False,
        help="Enable LLM judge evaluators (doubles vLLM load, off by default)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "user: User identification and profile tests")
    config.addinivalue_line("markers", "websearch: Web search and URL fetch tests")
    config.addinivalue_line("markers", "integration: Multi-tool integration tests")
    config.addinivalue_line("markers", "coder: CODER route tool tests (file, bash, search, graph, docs, rules)")
    config.addinivalue_line("markers", "slow: Tests that take more than 60 seconds")


# ==================== Phoenix / OpenTelemetry ====================


@pytest.fixture(scope="session")
def phoenix_provider():
    """Single TracerProvider for all tests — one Phoenix project.

    Also sets itself as the global TracerProvider so that W3C trace
    context propagation (traceparent injection) works automatically
    when the CCA test client makes HTTP requests.
    """
    resource = Resource.create({
        "service.name": PROJECT_NAME,
        "openinference.project.name": PROJECT_NAME,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    # Set as global so inject()/extract() use our test spans
    trace.set_tracer_provider(provider)
    yield provider
    provider.shutdown()


@pytest.fixture(scope="session")
def phoenix_tracer(phoenix_provider):
    """Single tracer for the entire test session."""
    tracer = phoenix_provider.get_tracer(PROJECT_NAME)
    return phoenix_provider, tracer


@pytest.fixture(autouse=True)
def trace_test(request, phoenix_tracer):
    """Wrap every test in a Phoenix span with test metadata.

    Creates a root span like 'websearch::test_basic_search' so tests
    are easy to find and filter in the Phoenix UI.

    Annotations are collected during the test via span._pending_annotations
    and posted AFTER the span closes + flushes — fixing the 404 race.
    """
    provider, tracer = phoenix_tracer

    test_path = request.node.nodeid
    if "/user/" in test_path:
        category = "user"
    elif "/websearch/" in test_path:
        category = "websearch"
    elif "/integration/" in test_path:
        category = "integration"
    elif "/coder/" in test_path:
        category = "coder"
    else:
        category = "other"

    test_name = request.node.name
    span_name = f"{category}::{test_name}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("cca.test.name", test_name)
        span.set_attribute("cca.test.category", category)
        span.set_attribute("cca.test.nodeid", test_path)

        # Evaluators append annotation dicts here during the test
        span._pending_annotations = []
        # CCA client stashes metrics here for per-test reporting
        span._test_metrics = {}

        yield span

        # Set I/O attributes from accumulated chat turns (done after yield so
        # ALL turns are collected before we write).
        # Single-turn: input.value / output.value as plain text (CHAIN span).
        # Multi-turn: llm.input_messages / llm.output_messages so Phoenix
        # renders each turn as its own independent message block (LLM span).
        turns = span._test_metrics.get("_turns", [])
        if turns:
            if len(turns) == 1:
                span.set_attribute("input.value", turns[0][0])
                span.set_attribute("output.value", turns[0][1])
            else:
                # Switch span kind to LLM so Phoenix uses the chat/conversation
                # renderer — each message becomes an independent block.
                span.set_attribute("openinference.span.kind", "LLM")
                idx = 0
                for i, (msg, resp) in enumerate(turns):
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.role", "user"
                    )
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.content",
                        f"[Turn {i + 1}] {msg}",
                    )
                    idx += 1
                    if i < len(turns) - 1:
                        # Intermediate assistant responses go into input context
                        span.set_attribute(
                            f"llm.input_messages.{idx}.message.role", "assistant"
                        )
                        span.set_attribute(
                            f"llm.input_messages.{idx}.message.content",
                            f"[Turn {i + 1}] {resp}",
                        )
                        idx += 1
                # Final assistant response is the output
                span.set_attribute(
                    "llm.output_messages.0.message.role", "assistant"
                )
                span.set_attribute(
                    "llm.output_messages.0.message.content",
                    f"[Turn {len(turns)}] {turns[-1][1]}",
                )

        if hasattr(request.node, "rep_call"):
            rep = request.node.rep_call
            span.set_attribute("cca.test.passed", rep.passed)
            span.set_attribute("cca.test.outcome", rep.outcome)
            if rep.passed:
                span.set_status(StatusCode.OK)
            else:
                span.set_status(StatusCode.ERROR, rep.longreprtext[:500] if hasattr(rep, "longreprtext") else "test failed")
        else:
            # No rep_call means setup failed or test was skipped
            span.set_status(StatusCode.ERROR, "no test result available")

    # Span is now CLOSED — flush to Phoenix, then post annotations.
    # force_flush() sends via gRPC; Phoenix needs a moment to persist
    # to Postgres before the span is queryable for annotations.
    provider.force_flush()
    time.sleep(1)
    post_deferred_annotations(span)

    # Per-test summary line for real-time monitoring
    metrics = getattr(span, "_test_metrics", {})
    if metrics:
        outcome = "?"
        if hasattr(request.node, "rep_call"):
            outcome = request.node.rep_call.outcome.upper()
        route = metrics.get("route", "?")
        steps = metrics.get("estimated_steps", "?")
        iters = metrics.get("tool_iterations", "?")
        elapsed_s = metrics.get("execution_time_ms", 0) / 1000
        flags = []
        if metrics.get("nudge_skipped"):
            flags.append("nudge_skipped")
        if metrics.get("circuit_breaker_fired"):
            flags.append("CB_FIRED")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"\n  >> {span_name}: {outcome} | "
            f"route={route} steps={steps} iters={iters} "
            f"{elapsed_s:.1f}s{flag_str}",
            flush=True,
        )

    # Inter-test cooldown: let vLLM drain its queue before the next test
    # starts a new LLM call. Without this, sequential tests pile up
    # requests and cause timeout cascades.
    if TEST_COOLDOWN > 0:
        log.debug("Cooldown %.1fs before next test", TEST_COOLDOWN)
        time.sleep(TEST_COOLDOWN)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result on node for trace_test fixture to read."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ==================== LLM Judge (direct vLLM, NOT CCA) ====================


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = "/models/Qwen3.5-35B-A3B-FP8"


@pytest.fixture(scope="session")
def judge_model(request):
    """Direct vLLM connection for LLM-as-judge evaluators.

    **Opt-in**: Returns None (disabled) unless --with-judge flag is passed
    or CCA_RUN_JUDGE=1 env var is set. This is because the LLM judge makes
    2 additional vLLM calls per test (response_quality + task_completion),
    doubling the load on the shared Spark2 vLLM server.

    Returns a dict with base_url and model (NOT a Phoenix OpenAIModel).
    The evaluators make direct httpx calls to vLLM with our own rail
    extraction that handles Qwen3 thinking-model output. Phoenix's
    llm_classify can't parse thinking tokens (produces NOT_PARSABLE
    for ~50% of responses).
    """
    # Check opt-in flag
    use_judge = (
        request.config.getoption("--with-judge", default=False)
        or os.getenv("CCA_RUN_JUDGE", "0") == "1"
    )
    if not use_judge:
        log.info("LLM judge disabled (use --with-judge or CCA_RUN_JUDGE=1 to enable)")
        return None

    try:
        import httpx

        resp = httpx.get(f"{VLLM_BASE_URL[:-3]}/health", timeout=5)
        if resp.status_code != 200:
            log.warning("vLLM not healthy at %s — judge disabled", VLLM_BASE_URL)
            return None
    except Exception:
        return None

    log.info("LLM judge enabled — 2 extra vLLM calls per test")
    return {
        "base_url": VLLM_BASE_URL,
        "model": VLLM_MODEL,
    }


# ==================== CCA Client ====================


@pytest.fixture(scope="session")
def cca(phoenix_tracer):
    """CCA AAAM client with Phoenix tracing.

    Session-scoped: one HTTP client for the entire test run.
    Uses streaming with idle timeout — no fixed total timeout.
    """
    _provider, tracer = phoenix_tracer
    client = CCAClient(base_url=CCA_BASE_URL, tracer=tracer)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def require_cca_healthy(cca):
    """Skip all tests if CCA AAAM server is unreachable."""
    health = cca.health()
    if health.get("status") != "healthy":
        pytest.skip(
            f"CCA AAAM server not healthy: {health.get('error', 'unknown')}"
        )


# ==================== Test Helpers ====================


# ==================== Session-Level Cleanup ====================


@pytest.fixture(scope="session", autouse=True)
def session_cleanup(cca):
    """Safety net: log leaked test resources after all tests (no deletions).

    Each test cleans up its own resources via TestResourceTracker.
    This fixture only warns about resources that leaked due to test crashes,
    so developers can fix their test cleanup.
    """
    yield  # All tests run here

    log.info("=== Session cleanup: checking for leaked test resources ===")

    try:
        users_data = cca.list_users()
        test_prefixes = (
            "Onboard_", "Memory_", "CRUD_", "Lifecycle_", "NoteTest_",
            "EditFlow_", "BashTest_", "TestUser_",
            "Planner_", "Coder_", "Infra_", "Recall_",
        )
        stale = [
            u.get("display_name", "")
            for u in users_data.get("users", [])
            if any(
                u.get("display_name", "").startswith(p)
                for p in test_prefixes
            )
        ]
        if stale:
            log.warning("LEAKED test users (not cleaned by tests): %s", stale)
    except Exception as e:
        log.warning("Session cleanup: failed to check users: %s", e)


# ==================== Deferred Experiment + LLM Judge ====================


@pytest.fixture(scope="session", autouse=True)
def deferred_experiment(judge_model, request):
    """Create Phoenix Dataset + Experiment after all tests.

    ALWAYS creates dataset/experiment (code eval scores for all 53 calls).
    If --with-judge: also runs deferred LLM judge on eligible items.
    """
    yield  # All tests run here

    from .evaluators import _EVAL_QUEUE, run_deferred_experiment

    if not _EVAL_QUEUE:
        return

    run_judge = judge_model is not None
    label = "LLM Judge + " if run_judge else ""
    print(f"\n{'=' * 60}")
    print(f"{label}Phoenix Experiment ({len(_EVAL_QUEUE)} responses)")
    print(f"{'=' * 60}")

    results = run_deferred_experiment(run_judge=run_judge)
    request.config._judge_results = results

    failures = [r for r in results if r["label"] in ("poor", "failed")]
    request.config._judge_failures = failures

    if run_judge and results:
        passed = len(results) - len(failures)
        print(f"Judge: {passed} passed, {len(failures)} failed")
    print(f"{'=' * 60}")


def pytest_sessionfinish(session, exitstatus):
    """Fail session if any LLM judge evaluations returned poor/failed."""
    failures = getattr(session.config, "_judge_failures", [])
    if failures and exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print LLM judge report with per-test/per-turn grouping."""
    results = getattr(config, "_judge_results", [])
    if not results:
        return

    failures = getattr(config, "_judge_failures", [])
    terminalreporter.write_sep("=", "LLM Judge Report")

    current_test = None
    for r in results:
        if r["test_name"] != current_test:
            current_test = r["test_name"]
            terminalreporter.write_line(f"\n  {current_test}:")

        is_fail = r["label"] in ("poor", "failed")
        marker = "FAIL" if is_fail else "PASS"
        turn = r.get("turn", "?")
        terminalreporter.write_line(
            f"    {marker}  [turn {turn}] {r['name']} = {r['label']}"
        )

    passed = len(results) - len(failures)
    terminalreporter.write_line(
        f"\n  {passed} passed, {len(failures)} failed "
        f"({len(results)} total)"
    )

    if failures:
        terminalreporter.write_sep("-", "Judge Failures")
        for f in failures:
            terminalreporter.write_line(
                f"  {f['test_name']} [turn {f.get('turn', '?')}] "
                f":: {f['name']}: {f['label']}"
            )
            terminalreporter.write_line(
                f"    {f.get('explanation', '')[:200]}"
            )
