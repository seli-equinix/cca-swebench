"""CCA AAAM test suite — Phoenix-traced integration tests.

Configures OpenTelemetry to export traces to Phoenix (192.168.4.204:4317)
under the project 'cca-aaam-tests'. Every test creates spans visible in
the Phoenix UI at http://192.168.4.204:6006/.
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
PROJECT_NAME = "cca-aaam-tests"
CCA_BASE_URL = "http://192.168.4.205:8500"


# ==================== Markers ====================


def pytest_configure(config):
    config.addinivalue_line("markers", "user: User identification and profile tests")
    config.addinivalue_line("markers", "websearch: Web search and URL fetch tests")
    config.addinivalue_line("markers", "integration: Multi-tool integration tests")
    config.addinivalue_line("markers", "slow: Tests that take more than 60 seconds")


# ==================== Phoenix / OpenTelemetry ====================


@pytest.fixture(scope="session", autouse=True)
def phoenix_tracer():
    """Initialize OpenTelemetry with Phoenix OTLP exporter.

    Creates the 'cca-aaam-tests' project in Phoenix automatically
    when the first trace arrives.
    """
    resource = Resource.create({
        "service.name": PROJECT_NAME,
        "openinference.project.name": PROJECT_NAME,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer("cca-aaam-tests")
    yield tracer

    # Flush remaining spans on shutdown
    provider.shutdown()


@pytest.fixture(autouse=True)
def trace_test(request, phoenix_tracer):
    """Wrap every test in a Phoenix span with test metadata.

    Creates a root span like 'user::test_identify_new_user' so tests
    are easy to find and filter in the Phoenix UI.
    """
    # Determine category from test path
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
        span.set_attribute("cca.test.name", test_name)
        span.set_attribute("cca.test.category", category)
        span.set_attribute("cca.test.nodeid", test_path)

        yield span

        # Mark test result on the span
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


# ==================== CCA Client ====================


@pytest.fixture(scope="session")
def cca(phoenix_tracer):
    """CCA AAAM client with Phoenix tracing.

    Session-scoped: one HTTP client for the entire test run.
    """
    client = CCAClient(base_url=CCA_BASE_URL, tracer=phoenix_tracer)
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
