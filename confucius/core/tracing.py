"""Centralized OpenTelemetry tracing for CCA → Phoenix.

Initializes a TracerProvider that exports spans to Phoenix via OTLP gRPC.
Traces are routed to a named project (not "default") using the
openinference.project.name resource attribute.

Configuration via environment variables:
    PHOENIX_COLLECTOR_ENDPOINT  (default: http://192.168.4.204:4317)
    PHOENIX_PROJECT_NAME        (default: cca-http)

Usage in server startup:
    from confucius.core.tracing import init_tracing, shutdown_tracing

    # In lifespan():
    init_tracing()
    ...
    shutdown_tracing()

Span helpers:
    from confucius.core.tracing import get_tracer, traced_span

    tracer = get_tracer()
    with traced_span(tracer, "cca.router", kind="LLM") as span:
        span.set_attribute("cca.router.model", "functionary-8b")
        ...
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

_provider: Optional[TracerProvider] = None

# OpenInference semantic convention keys (avoid hard dependency on the package)
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
TOOL_NAME = "tool.name"
TOOL_PARAMETERS = "tool.parameters"


def init_tracing() -> Optional[trace.Tracer]:
    """Initialize OpenTelemetry with Phoenix OTLP exporter.

    Returns a Tracer if setup succeeds, None if Phoenix is unreachable
    or tracing is disabled. Never raises — the server must not crash
    because of observability issues.
    """
    global _provider

    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://192.168.4.204:4317"
    )
    project_name = os.environ.get("PHOENIX_PROJECT_NAME", "cca-http")

    # Allow disabling tracing entirely
    if os.environ.get("PHOENIX_TRACING_DISABLED", "").lower() in ("1", "true"):
        logger.info("Phoenix tracing disabled via PHOENIX_TRACING_DISABLED")
        return None

    try:
        resource = Resource.create({
            "service.name": project_name,
            "openinference.project.name": project_name,
        })

        _provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        _provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)

        # Auto-instrument OpenAI client calls (captures all vLLM HTTP requests)
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument(tracer_provider=_provider)
            logger.info("OpenAI auto-instrumentation enabled (vLLM calls traced)")
        except Exception as e:
            logger.warning(f"OpenAI auto-instrumentation failed (non-fatal): {e}")

        tracer = trace.get_tracer(project_name)
        logger.info(
            f"Phoenix tracing initialized: project={project_name}, "
            f"endpoint={endpoint}"
        )
        return tracer

    except Exception as e:
        logger.warning(f"Phoenix tracing setup failed (non-fatal): {e}")
        _provider = None
        return None


def shutdown_tracing() -> None:
    """Flush remaining spans and shut down the tracer provider."""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
            logger.info("Phoenix tracing shut down (spans flushed)")
        except Exception as e:
            logger.warning(f"Phoenix tracing shutdown error: {e}")
        _provider = None


def get_tracer(name: str = "cca-http") -> trace.Tracer:
    """Get a tracer instance. Safe to call even if tracing is not initialized."""
    return trace.get_tracer(name)


def get_current_context() -> otel_context.Context:
    """Capture the current OTel context for propagation to async tasks.

    Usage:
        ctx = get_current_context()
        asyncio.create_task(_run_in_context(ctx, coro))

    async def _run_in_context(ctx, coro):
        token = otel_context.attach(ctx)
        try:
            await coro
        finally:
            otel_context.detach(token)
    """
    return otel_context.get_current()


def attach_context(ctx: otel_context.Context) -> object:
    """Attach a captured context (returns a token for detach)."""
    return otel_context.attach(ctx)


def detach_context(token: object) -> None:
    """Detach a previously attached context."""
    otel_context.detach(token)  # type: ignore[arg-type]
