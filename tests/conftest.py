"""CCA test suite — Phoenix-traced integration tests.

All traces go to a single Phoenix project (cca-http) so each request
shows as ONE unified trace with all steps: routing → agent → tool calls.
Tests are categorized via span attributes, not separate projects.

Every test creates spans visible in the Phoenix UI at
http://192.168.4.204:6006/.

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

PHOENIX_ENDPOINT = "http://192.168.4.204:4317"
CCA_BASE_URL = "http://192.168.4.205:8500"

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


VLLM_BASE_URL = "http://192.168.4.208:8000/v1"
VLLM_MODEL = "/models/Qwen3-Next-80B-A3B-Thinking-FP8"


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
    """Clean up stale test users and workspace files after all tests.

    Runs AFTER all tests complete. Sweeps any test users (display_name
    starts with "TestUser_" or common test prefixes) and workspace files
    that were left behind by failing tests.

    Individual tests still do their own cleanup in finally: blocks.
    This is the safety net for when tests crash mid-execution.
    """
    yield  # All tests run here

    log.info("=== Session cleanup: sweeping stale test data ===")

    # 1. Delete stale test users
    try:
        users_data = cca.list_users()
        test_prefixes = (
            "TestUser_", "SkillUser_", "AliasUser_", "AliasChk_",
            "Aliaschk_", "RmAlias_", "Rmalias_",
            "RmSkill_", "Rmskill_", "SkillCheck_", "Skillcheck_",
            "FactUser_", "Factuser_", "MultiFact_", "Multifact_",
            "PrefUser_", "Prefuser_", "PrefAck_", "Prefack_",
            "PrefRecall_", "Prefrecall_",
            "DelUser_", "Deluser_", "DelGone_", "Delgone_",
            "ViewUser_", "Viewuser_", "ViewData_", "Viewdata_",
            "UpdateUser_", "ListUser_",
            "NewUser_", "Newuser_", "NoDup_", "Nodup_",
            "Greeter_", "Return_", "MetaUser_", "Metauser_",
            "InferKnown_", "Inferknown_",
            "CtxUser_", "Ctxuser_", "CtxFacts_", "Ctxfacts_",
            "CtxEnrich_", "Ctxenrich_",
            "Lifecycle_", "Persist_", "Recall_",
            "Overwrite_", "RmFact_", "Rmfact_", "RmPref_", "Rmpref_",
        )
        for user in users_data.get("users", []):
            name = user.get("display_name", "")
            if any(name.startswith(p) for p in test_prefixes):
                user_id = user.get("user_id", "")
                if user_id:
                    try:
                        cca._client.delete(
                            f"{cca.base_url}/users/{user_id}",
                            timeout=TIMEOUT_DIAGNOSTIC,
                        )
                        log.info("Cleaned stale test user: %s (%s)", name, user_id)
                    except Exception as e:
                        log.warning("Failed to clean user %s: %s", name, e)
    except Exception as e:
        log.warning("Session cleanup: failed to list users: %s", e)

    # 2. Clean workspace files
    try:
        result = cca.clean_workspace_files()
        deleted = result.get("deleted_count", 0)
        if deleted:
            log.info("Cleaned %d workspace files: %s", deleted, result.get("deleted", []))
    except Exception as e:
        log.warning("Session cleanup: failed to clean workspace: %s", e)
