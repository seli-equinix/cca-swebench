"""CCA test suite — Phoenix-traced integration tests.

All traces go to a single Phoenix project (cca-tests) so each request
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
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .cca_client import CCAClient
from .evaluators import post_deferred_annotations

log = logging.getLogger(__name__)

# ==================== Configuration ====================

PHOENIX_ENDPOINT = "http://192.168.4.204:4317"
CCA_BASE_URL = "http://192.168.4.205:8500"

# Single Phoenix project for all tests — categorize via span attributes
PROJECT_NAME = "cca-tests"

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
    """Single TracerProvider for all tests — one Phoenix project."""
    resource = Resource.create({
        "service.name": PROJECT_NAME,
        "openinference.project.name": PROJECT_NAME,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
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

        yield span

        if hasattr(request.node, "rep_call"):
            rep = request.node.rep_call
            span.set_attribute("cca.test.passed", rep.passed)
            span.set_attribute("cca.test.outcome", rep.outcome)

    # Span is now CLOSED — flush to Phoenix, then post annotations.
    # force_flush() sends via gRPC; Phoenix needs a moment to persist
    # to Postgres before the span is queryable for annotations.
    provider.force_flush()
    time.sleep(1)
    post_deferred_annotations(span)

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

    When enabled, talks directly to vLLM on Spark2:8000/v1 (raw
    OpenAI-compatible API). Does NOT go through CCA. CCA is the system
    under test; the judge is an independent assessor.
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
        from phoenix.evals import OpenAIModel
    except ImportError:
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
    return OpenAIModel(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=1024,  # Qwen3 thinking model needs room for <think> + label
        request_timeout=120,  # Qwen3 thinking + CCA contention needs more than 30s
        timeout=300,  # Overall timeout for retries
    )


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


@pytest.fixture
def unique_name():
    """Generate a unique test user name to avoid cross-test interference."""
    return f"TestUser_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_session():
    """Generate a unique session ID."""
    return f"test-{uuid.uuid4().hex[:12]}"
