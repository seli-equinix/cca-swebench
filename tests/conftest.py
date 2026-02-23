"""CCA test suite — Phoenix-traced integration tests.

Routes traces to separate Phoenix projects:
  - cca-user-tests:   user identification, profiles, facts, preferences
  - cca-search-tests: web search, URL fetch, internet tools

Every test creates spans visible in the Phoenix UI at
http://192.168.4.204:6006/.
"""

from __future__ import annotations

import uuid

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .cca_client import CCAClient

# ==================== Configuration ====================

PHOENIX_ENDPOINT = "http://192.168.4.204:4317"
CCA_BASE_URL = "http://192.168.4.205:8500"

# Two separate Phoenix projects for different test domains
PROJECT_USER = "cca-user-tests"
PROJECT_SEARCH = "cca-search-tests"


# ==================== Markers ====================


def pytest_configure(config):
    config.addinivalue_line("markers", "user: User identification and profile tests")
    config.addinivalue_line("markers", "websearch: Web search and URL fetch tests")
    config.addinivalue_line("markers", "integration: Multi-tool integration tests")
    config.addinivalue_line("markers", "slow: Tests that take more than 60 seconds")


# ==================== Phoenix / OpenTelemetry ====================


def _make_tracer(project_name: str) -> tuple:
    """Create a TracerProvider + Tracer for a Phoenix project."""
    resource = Resource.create({
        "service.name": project_name,
        "openinference.project.name": project_name,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider.get_tracer(project_name)
    return provider, tracer


@pytest.fixture(scope="session")
def user_tracer():
    """Tracer that routes spans to the cca-user-tests project."""
    provider, tracer = _make_tracer(PROJECT_USER)
    yield tracer
    provider.shutdown()


@pytest.fixture(scope="session")
def search_tracer():
    """Tracer that routes spans to the cca-search-tests project."""
    provider, tracer = _make_tracer(PROJECT_SEARCH)
    yield tracer
    provider.shutdown()


@pytest.fixture(scope="session", autouse=True)
def default_tracer(user_tracer):
    """Set the global tracer provider to user_tracer as default.

    Individual test modules override via the phoenix_tracer fixture.
    """
    # Don't set_tracer_provider globally — we use explicit tracers
    yield user_tracer


@pytest.fixture
def phoenix_tracer(request, user_tracer, search_tracer):
    """Pick the correct tracer based on test location.

    Tests in tests/websearch/ → cca-search-tests project
    Everything else → cca-user-tests project
    """
    test_path = request.node.nodeid
    if "/websearch/" in test_path:
        return search_tracer
    return user_tracer


@pytest.fixture(autouse=True)
def trace_test(request, phoenix_tracer):
    """Wrap every test in a Phoenix span with test metadata.

    Creates a root span like 'user::test_identify_new_user' so tests
    are easy to find and filter in the Phoenix UI.
    """
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

    with phoenix_tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("cca.test.name", test_name)
        span.set_attribute("cca.test.category", category)
        span.set_attribute("cca.test.nodeid", test_path)

        yield span

        if hasattr(request.node, "rep_call"):
            rep = request.node.rep_call
            span.set_attribute("cca.test.passed", rep.passed)
            span.set_attribute("cca.test.outcome", rep.outcome)


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
def judge_model():
    """Direct vLLM connection for LLM-as-judge evaluators.

    Talks directly to vLLM on Spark2:8000/v1 (raw OpenAI-compatible API).
    Does NOT go through CCA. CCA is the system under test; this is the
    independent judge.
    """
    try:
        from phoenix.evals import OpenAIModel
    except ImportError:
        return None

    try:
        import httpx

        resp = httpx.get(f"{VLLM_BASE_URL[:-3]}/health", timeout=5)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    return OpenAIModel(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=1024,  # Qwen3 thinking model needs room for <think> + label
    )


# ==================== CCA Client ====================


@pytest.fixture(scope="session")
def cca(user_tracer):
    """CCA AAAM client with Phoenix tracing.

    Session-scoped: one HTTP client for the entire test run.
    Uses user_tracer by default; individual methods accept tracer override.
    """
    client = CCAClient(base_url=CCA_BASE_URL, tracer=user_tracer)
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
