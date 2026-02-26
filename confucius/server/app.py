# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""FastAPI application for the CCA Agent-as-a-Model Endpoint.

Accepts OpenAI-compatible /v1/chat/completions requests, runs the SAME
orchestrator and full agent loop as CLI mode, and returns responses
formatted as OpenAI chat completions — with user identification and
personalization from the ported MCP user-awareness system.

Both CLI and HTTP mode use the same agent path:
    invoke_analect(Entry, EntryInput(question=...))
      -> impl() -> build extensions -> AnthropicLLMOrchestrator -> full agent loop

The only differences in HTTP mode:
1. HttpRoutedEntry dynamically builds extensions per route via tool_groups.py
2. HttpIOInterface captures output instead of printing to terminal
3. User identification happens server-side (router extraction + regex fallback)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.config import CCAConfigError, get_llm_params, get_router_config
from opentelemetry.propagate import extract as otel_extract

from ..core.tracing import (
    init_tracing, shutdown_tracing, get_tracer,
    get_current_context, attach_context, detach_context,
    OPENINFERENCE_SPAN_KIND, INPUT_VALUE, OUTPUT_VALUE,
)
from ..core.entry.base import EntryInput
from ..lib.confucius import Confucius
from .expert_router import (
    ExpertType,
    RouteDecision,
    classify_request,
    close_client as close_router_client,
)
from .http_routed_entry import HttpRoutedEntry
from .tool_groups import get_max_iterations
from .io_adapter import HttpIOInterface
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelInfo,
    build_chunk,
    build_completion_response,
    generate_completion_id,
)
from .backends import BackendClients
from .note_observer import NoteObserver, format_notes_for_prompt
from .session_pool import SessionPool
from .streaming import sse_stream
from .user.critical_facts import CriticalFactsExtractor
from .user.session_manager import UserSessionManager
from .user.tools_extension import UserToolsExtension
from .user.user_context import (
    build_anonymous_context,
    build_uncertain_context,
    build_user_context,
)

logger = logging.getLogger(__name__)

# ==================== Model Name Resolution ====================


def _resolve_served_model_name() -> str:
    """Read the actual model name from CCA config (coder role).

    Returns the underlying LLM model name so clients like Open WebUI
    and Continue.dev see a recognizable model in their dropdown.
    Raises on failure — if config is broken, the server cannot function.
    """
    params = get_llm_params("coder")
    if not params.model:
        raise RuntimeError(
            "CCA config has no model name for 'coder' role. "
            "Check config.toml [providers.*.coder] section."
        )
    return params.model


# The model name advertised via /v1/models — resolved once at import time
SERVED_MODEL_NAME: str = _resolve_served_model_name()

# ==================== Globals (initialised in lifespan) ====================

session_pool: SessionPool
user_session_mgr: UserSessionManager
critical_facts_extractor: CriticalFactsExtractor
note_observer: Optional[NoteObserver] = None
backend_clients: Optional[BackendClients] = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application startup / shutdown lifecycle."""
    global session_pool, user_session_mgr, critical_facts_extractor, note_observer, backend_clients

    logger.info("CCA HTTP server starting up...")

    # Initialise Phoenix tracing (exports spans to Phoenix UI)
    init_tracing()

    # Initialise user session manager (Redis + Qdrant)
    user_session_mgr = UserSessionManager()
    await user_session_mgr.initialize()

    # Initialise critical facts extractor (shares Redis with session manager)
    critical_facts_extractor = CriticalFactsExtractor(
        redis_client=user_session_mgr._redis
    )

    # Initialise note observer (per-request note extraction → Qdrant)
    try:
        nt_params = get_llm_params("note_taker")
        base_url = (nt_params.additional_kwargs or {}).get(
            "base_url", "http://192.168.4.205:8400/v1"
        )
        note_observer = NoteObserver(
            llm_url=base_url,
            llm_model=nt_params.model or "",
            redis_client=user_session_mgr._redis,
        )
        await note_observer.initialize()
        logger.info("NoteObserver initialised (model=%s)", nt_params.model)
    except CCAConfigError:
        logger.info("NoteObserver disabled — 'note_taker' role not in config")
        note_observer = None
    except Exception as e:
        logger.warning("NoteObserver init failed (non-fatal): %s", e)
        note_observer = None

    # Initialise backend clients (Qdrant, Memgraph, Embedding — for code intelligence)
    try:
        backend_clients = BackendClients()
        await backend_clients.initialize(redis=user_session_mgr._redis)
        if backend_clients.available:
            logger.info("BackendClients initialised (Qdrant + Embedding available)")
        else:
            logger.warning("BackendClients: partial — some backends unavailable")
    except Exception as e:
        logger.warning("BackendClients init failed (non-fatal): %s", e)
        backend_clients = None

    # Initialise session pool (manages Confucius instances)
    session_pool = SessionPool(max_sessions=50, session_ttl=3600)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info(f"CCA HTTP server ready — serving as model: {SERVED_MODEL_NAME}")
    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    if backend_clients:
        await backend_clients.close()
    if note_observer:
        await note_observer.close()
    await close_router_client()
    await user_session_mgr.close()
    shutdown_tracing()
    logger.info("CCA HTTP server shut down")


async def _cleanup_loop() -> None:
    """Periodically clean up expired sessions."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            evicted = await session_pool.cleanup_expired()
            if evicted:
                logger.info(f"Cleaned up {evicted} expired sessions")
        except Exception as e:
            logger.warning(f"Session cleanup error: {e}")


# ==================== FastAPI App ====================

app = FastAPI(
    title="CCA Agent-as-a-Model Endpoint",
    description=(
        "OpenAI-compatible chat completions powered by CCA's full agent loop. "
        "File editing, command execution, planning, code review — all available."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Session ID Derivation ====================


def derive_session_id(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = None,
) -> str:
    """Derive a session ID from the request.

    Priority chain (ported from MCP server):
    1. request.session_id (CCA extension)
    2. X-Session-Id header
    3. request.user field -> "user-{user}"
    4. System prompt hash -> "sys-{hash}" (Continue.dev pattern)
    5. Conversation prefix hash -> "conv-{hash}" (last resort)
    """
    # 1. Explicit session_id in request body
    if request.session_id:
        return request.session_id

    # 2. X-Session-Id header
    if x_session_id:
        return x_session_id

    # 3. OpenAI standard user field
    if request.user:
        return f"user-{request.user}"

    # 4. System prompt hash (Continue.dev sends unique system prompts)
    for msg in request.messages:
        if msg.role == "system" and msg.content:
            h = hashlib.sha256(msg.content.encode()).hexdigest()[:16]
            return f"sys-{h}"

    # 5. Conversation prefix hash
    conv_text = "".join(
        (msg.content or "")[:100]
        for msg in request.messages[:3]
    )
    if conv_text:
        h = hashlib.sha256(conv_text.encode()).hexdigest()[:16]
        return f"conv-{h}"

    # Fallback: random
    import uuid
    return str(uuid.uuid4())


def extract_last_user_message(messages: List[Any]) -> str:
    """Extract the last user message from a message list."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


async def _run_note_observer_with_context(
    ctx: Any,
    observer: NoteObserver,
    messages: List[Any],
    session_id: str,
    user: Any,
) -> None:
    """Run note observer with propagated OTel span context.

    Without this, the background task would create orphaned spans
    disconnected from the parent request trace.
    """
    token = attach_context(ctx)
    try:
        await observer.process(
            messages=messages,
            session_id=session_id,
            user_id=user.user_id if user else None,
            user_name=user.display_name if user else None,
        )
    finally:
        detach_context(token)


# ==================== Routes ====================


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
    http_request: Request = None,  # type: ignore[assignment]
) -> Any:
    """OpenAI-compatible chat completions powered by CCA's full agent loop."""
    start_time = time.time()
    tracer = get_tracer()

    # Extract W3C trace context from incoming headers (traceparent).
    # If present, the cca.request span becomes a child of the caller's
    # trace — so test traces and server traces unify in Phoenix.
    incoming_ctx = None
    if http_request and http_request.headers:
        carrier = dict(http_request.headers)
        incoming_ctx = otel_extract(carrier)

    # Wrap the ENTIRE request in a parent span so router, agent, and
    # note observer all appear as children of a single trace.
    ctx_kwargs = {"context": incoming_ctx} if incoming_ctx else {}
    with tracer.start_as_current_span("cca.request", **ctx_kwargs) as request_span:
        request_span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")

        return await _handle_chat_completions(
            request, x_session_id, http_request,
            start_time, tracer, request_span,
        )


async def _handle_chat_completions(
    request: ChatCompletionRequest,
    x_session_id: Optional[str],
    http_request: Request,
    start_time: float,
    tracer: Any,
    request_span: Any,
) -> Any:
    """Inner handler for chat completions — runs inside cca.request span."""

    # 1. Derive session ID
    session_id = derive_session_id(request, x_session_id)
    logger.info(f"Request to session {session_id[:16]}...")
    request_span.set_attribute("cca.session_id", session_id)
    request_span.set_attribute(INPUT_VALUE, extract_last_user_message(request.messages)[:500])

    # 2. Detect client type from headers
    headers = dict(http_request.headers) if http_request else {}
    client_type = UserSessionManager.detect_client_type(headers)

    # 3. Get or create user session
    session = await user_session_mgr.get_or_create_session(session_id)
    session.message_count += 1
    if client_type:
        session.client_type = client_type

    # 4. Classify request via Functionary router (if enabled)
    #    Router runs FIRST so it can extract user info alongside routing.
    user_message = extract_last_user_message(request.messages)
    route: Optional[RouteDecision] = None
    router_config = get_router_config()

    if router_config.enabled and user_message:
        try:
            # Build recent context for follow-up awareness
            recent: List[Dict[str, str]] = []
            for msg in request.messages[-4:]:
                if msg.role in ("user", "assistant") and msg.content:
                    recent.append({"role": msg.role, "content": msg.content[:500]})

            route = await classify_request(user_message, router_config, recent)
            logger.info(
                f"Router: {route.expert.value} | {route.task_summary[:80]} "
                f"({route.classification_time_ms:.0f}ms)"
            )
            request_span.set_attribute("cca.route", route.expert.value)
            request_span.set_attribute("cca.route_summary", route.task_summary[:100])
            request_span.set_attribute("cca.estimated_steps", route.estimated_steps)
            request_span.set_attribute("cca.max_iterations", get_max_iterations(route))
            request_span.set_attribute("cca.experts_enabled", route.is_complex)

        except Exception as e:
            logger.warning(f"Router classification failed, falling back to coder: {e}")
            route = None

    # 5. User identification — router extraction first, regex fallback
    user = None
    id_result: Dict[str, Any] = {}
    id_source = ""

    if session.identified:
        user = await user_session_mgr.get_user_for_session(session)
        id_source = "session"
    elif user_message:
        # Try router-extracted name first (Functionary already parsed it)
        if route and route.detected_user_name:
            id_result = await user_session_mgr.identify_from_router(
                name=route.detected_user_name,
                facts=route.detected_user_facts,
                session=session,
                client_type=client_type,
            )
            logger.info(
                "Router ID: identified=%s, action=%s, name=%s, facts=%s",
                id_result.get("identified"), id_result.get("action"),
                route.detected_user_name, route.detected_user_facts,
            )
            if id_result.get("identified"):
                user = id_result.get("user")
                id_source = "router"

        # Fallback to regex extraction if router didn't extract or failed
        if not user and not session.identified:
            id_result = await user_session_mgr.smart_identify_on_first_message(
                user_message, session, client_type
            )
            logger.info(
                "Smart ID: identified=%s, action=%s, name=%s",
                id_result.get("identified"), id_result.get("action"),
                id_result.get("extracted_name", ""),
            )
            if id_result.get("identified"):
                user = id_result.get("user")
                id_source = "regex"
            elif id_result.get("action") in ("asking_new", "asking"):
                # Auto-identify when a name is detected — don't wait
                # for the LLM to call identify_user (unreliable).
                extracted_name = id_result.get("extracted_name", "")
                if extracted_name:
                    create_result = await user_session_mgr.identify_user(
                        extracted_name, session, client_type
                    )
                    if create_result.get("status") in (
                        "new_user", "welcome_back", "identified",
                    ):
                        user = await user_session_mgr.get_user_for_session(
                            session
                        )
                        id_source = "regex_auto"
                        logger.info(
                            "Auto ID: user=%s, status=%s",
                            extracted_name, create_result.get("status"),
                        )

    # 6. Build user context for system prompt injection
    user_context = ""
    if user:
        # Get critical facts for this session
        critical_facts_str = await critical_facts_extractor.format_facts_for_context(
            session_id
        )
        user_context = build_user_context(
            user, critical_facts_str, id_result or None
        )
        # Add router-identified hint so the 80B knows not to call identify tools
        if id_source == "router":
            user_context += (
                "\n[User identified by router — no identification tools needed. "
                "Use remember_user_fact if user mentions personal details during work.]"
            )
    elif id_result.get("action") == "asking":
        user_context = build_uncertain_context(
            id_result.get("potential_user", ""),
            id_result.get("confidence", 0.0),
        )
    else:
        user_context = build_anonymous_context()

    # 6b. Enrich context with relevant past notes from Qdrant
    if note_observer and user and user.user_id and user_message:
        try:
            relevant_notes = await note_observer.search_notes(
                query=user_message,
                user_id=user.user_id,
                n_results=3,
                min_score=0.2,
            )
            if relevant_notes:
                notes_ctx = format_notes_for_prompt(relevant_notes)
                user_context += f"\n\n{notes_ctx}"
                logger.debug(
                    "Injected %d past notes for user %s",
                    len(relevant_notes),
                    user.user_id,
                )
        except Exception as e:
            logger.debug("Note enrichment failed (non-fatal): %s", e)

    # 7. Create user tools extension
    user_ext = UserToolsExtension(
        session_mgr=user_session_mgr,
        session=session,
        critical_facts=critical_facts_extractor,
    )

    # 7b. Handle DIRECT and CLARIFY routes (no agent loop needed)
    if route:
        direct_content = None
        if route.is_direct_answer and route.direct_answer:
            direct_content = route.direct_answer
        elif route.is_clarification and route.clarification_question:
            direct_content = route.clarification_question

        if direct_content:
            execution_time = (time.time() - start_time) * 1000
            from .models import ContextMetadata

            metadata = ContextMetadata(
                user_identified=session.identified,
                user_name=user.display_name if user else None,
                execution_time_ms=execution_time,
                route=route.expert.value,
            )
            model_name = request.model or SERVED_MODEL_NAME

            if request.stream:
                # Streaming: wrap direct answer as SSE events
                completion_id = generate_completion_id()

                async def direct_sse():
                    # Role chunk
                    role_chunk = build_chunk(completion_id=completion_id, model=model_name, role="assistant")
                    yield f"data: {role_chunk.model_dump_json()}\n\n"
                    # Content chunk
                    content_chunk = build_chunk(completion_id=completion_id, model=model_name, content=direct_content)
                    yield f"data: {content_chunk.model_dump_json()}\n\n"
                    # Finish chunk
                    finish_chunk = build_chunk(completion_id=completion_id, model=model_name, finish_reason="stop")
                    yield f"data: {finish_chunk.model_dump_json()}\n\n"
                    # Context metadata
                    yield f"data: {json.dumps({'context_metadata': metadata.model_dump()})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    direct_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                return build_completion_response(
                    content=direct_content,
                    model=model_name,
                    metadata=metadata,
                )

    # 8. Select entry based on routing decision
    #    HttpRoutedEntry dynamically builds extensions from the route's tool groups.
    #    Each route only gets the tools it needs (USER=6, CODER=~13, INFRA=~13, etc.)
    if route:
        entry = HttpRoutedEntry(
            route=route,
            user_context=user_context,
            user_extension=user_ext,
            backend_clients=backend_clients,
            session_id=session_id,
            user_id=user.user_id if user else None,
        )
    else:
        # No router or router disabled — default coder route
        entry = HttpRoutedEntry(
            route=RouteDecision(expert=ExpertType.CODER, task_summary="Default (no router)"),
            user_context=user_context,
            user_extension=user_ext,
            backend_clients=backend_clients,
            session_id=session_id,
            user_id=user.user_id if user else None,
        )

    # 9. Validate user message (already extracted in step 7)
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
        )

    # 10. Create fresh IO for this request
    stream_queue = asyncio.Queue() if request.stream else None
    io = HttpIOInterface(stream_queue=stream_queue)

    # 11. Acquire session from pool → Confucius instance
    pool_entry = await session_pool.acquire(session_id, io)

    try:
        if request.stream:
            # Streaming mode: run agent as background task, stream output as SSE
            completion_id = generate_completion_id()

            async def run_agent() -> None:
                with tracer.start_as_current_span("cca.agent") as span:
                    span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
                    span.set_attribute(INPUT_VALUE, user_message[:500])
                    span.set_attribute("cca.session_id", session_id)
                    span.set_attribute("cca.mode", "streaming")
                    span.set_attribute("cca.message", user_message[:200])
                    if route:
                        span.set_attribute("cca.route", route.expert.value)
                        span.set_attribute("cca.route_summary", route.task_summary[:100])
                    try:
                        await pool_entry.cf.invoke_analect(
                            entry, EntryInput(question=user_message)
                        )
                        span.set_attribute("cca.status", "success")
                        span.set_attribute(
                            "cca.tool_iterations",
                            getattr(entry, "_tool_iterations", 0),
                        )

                        # Fire note observer in background with span context
                        if note_observer:
                            trajectory = pool_entry.cf.memory_manager.get_session_memory().messages
                            ctx = get_current_context()
                            asyncio.create_task(_run_note_observer_with_context(
                                ctx, note_observer, list(trajectory),
                                session_id, user,
                            ))
                    except Exception as e:
                        span.set_attribute("cca.status", "error")
                        span.set_attribute("cca.error", str(e)[:500])
                        raise
                    finally:
                        await io.signal_done()

            agent_task = asyncio.create_task(run_agent())

            def _build_stream_metadata():
                """Build context_metadata dict for the streaming finish event."""
                from .models import ContextMetadata
                return ContextMetadata(
                    tool_iterations=getattr(entry, "_tool_iterations", 0),
                    route=route.expert.value if route else None,
                    user_identified=session.identified,
                    user_name=user.display_name if user else None,
                    execution_time_ms=(time.time() - start_time) * 1000,
                ).model_dump()

            return StreamingResponse(
                sse_stream(io, agent_task, request, completion_id,
                           metadata_callback=_build_stream_metadata),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming mode: run agent, collect output, return complete response
            with tracer.start_as_current_span("cca.agent") as span:
                span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
                span.set_attribute(INPUT_VALUE, user_message[:500])
                span.set_attribute("cca.session_id", session_id)
                span.set_attribute("cca.mode", "non-streaming")
                span.set_attribute("cca.message", user_message[:200])
                if route:
                    span.set_attribute("cca.route", route.expert.value)
                    span.set_attribute("cca.route_summary", route.task_summary[:100])
                if user:
                    span.set_attribute("cca.user", user.display_name)

                try:
                    await pool_entry.cf.invoke_analect(
                        entry, EntryInput(question=user_message)
                    )
                except Exception as e:
                    span.set_attribute("cca.status", "error")
                    span.set_attribute("cca.error", str(e)[:500])
                    logger.error(f"Agent execution failed: {e}")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "message": f"Agent execution error: {e}",
                                "type": "server_error",
                            }
                        },
                    )

                # Fire note observer in background with span context
                if note_observer:
                    trajectory = pool_entry.cf.memory_manager.get_session_memory().messages
                    ctx = get_current_context()
                    asyncio.create_task(_run_note_observer_with_context(
                        ctx, note_observer, list(trajectory),
                        session_id, user,
                    ))

                # Collect response from IO buffer
                response_text = io.get_response_text()
                if not response_text:
                    response_text = io.get_all_text()

                thinking_text = io.get_thinking_text()

                # Process critical facts from this exchange
                await critical_facts_extractor.process_conversation(
                    session_id, user_message, response_text
                )

                # Build OpenAI-compatible response
                execution_time = (time.time() - start_time) * 1000
                from .models import ContextMetadata

                # Extract tool iteration count from the entry
                tool_iters = getattr(entry, "_tool_iterations", 0)
                route_name = route.expert.value if route else None

                metadata = ContextMetadata(
                    tool_iterations=tool_iters,
                    route=route_name,
                    user_identified=session.identified,
                    user_name=user.display_name if user else None,
                    execution_time_ms=execution_time,
                )

                span.set_attribute("cca.status", "success")
                span.set_attribute("cca.execution_time_ms", execution_time)
                span.set_attribute("cca.response_length", len(response_text))
                span.set_attribute("cca.tool_iterations", tool_iters)
                span.set_attribute(OUTPUT_VALUE, response_text[:500])

                # Estimated token counts (4 chars per token)
                est_completion_tokens = len(response_text) // 4
                span.set_attribute("llm.token_count.completion", est_completion_tokens)
                span.set_attribute("llm.token_count.total", est_completion_tokens)

                response = build_completion_response(
                    content=response_text,
                    model=request.model or SERVED_MODEL_NAME,
                    reasoning=thinking_text if thinking_text else None,
                    metadata=metadata,
                )

                return response

    finally:
        # Always release session (save state + unlock)
        await session_pool.release(session_id)

        # Update session activity
        await user_session_mgr.save_session(session)


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """List available models (OpenAI-compatible).

    Advertises the underlying LLM model name so clients like Open WebUI
    and Continue.dev show a recognizable model in their dropdown.
    Any model name is accepted in /v1/chat/completions — CCA is a single
    agent, not a model router.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "cca",
                "permission": [],
                "root": SERVED_MODEL_NAME,
                "parent": None,
            },
        ],
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        active_sessions=session_pool.active_count,
    )


@app.get("/users")
async def list_users() -> Dict[str, Any]:
    """List all known user profiles."""
    users = await user_session_mgr.get_all_users()
    return {
        "count": len(users),
        "users": [
            {
                "user_id": u.user_id,
                "display_name": u.display_name,
                "session_count": u.session_count,
                "aliases": u.aliases,
                "skills": u.skills,
                "last_seen": u.last_seen,
            }
            for u in users
        ],
    }


@app.delete("/users/{user_id}")
async def delete_user(user_id: str) -> Dict[str, Any]:
    """Delete a user profile by user_id (for testing/admin)."""
    result = await user_session_mgr.delete_user_profile(
        user_id, confirm_delete=True
    )
    return result


@app.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List active sessions (for debugging)."""
    info = await session_pool.get_session_info()
    return {
        "count": len(info),
        "sessions": info,
    }


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    """Diagnostic statistics."""
    sm_stats = await user_session_mgr.get_stats()
    return {
        "session_pool": {
            "active": session_pool.active_count,
        },
        "user_session_manager": sm_stats,
    }


@app.post("/route/test")
async def route_test(request: Request) -> Dict[str, Any]:
    """Test the expert router classification without running the agent loop.

    Usage:
        curl -X POST http://host:8500/route/test \\
            -d '{"message": "check docker swarm status"}'
    """
    body = await request.json()
    message = body.get("message", "")
    if not message:
        return {"error": "Missing 'message' field"}

    router_config = get_router_config()
    if not router_config.enabled:
        return {"error": "Router is disabled in config.toml", "enabled": False}

    try:
        route = await classify_request(message, router_config)
        return {
            "expert": route.expert.value,
            "task_summary": route.task_summary,
            "parameters": route.parameters,
            "direct_answer": route.direct_answer or None,
            "clarification_question": route.clarification_question or None,
            "classification_time_ms": round(route.classification_time_ms, 1),
            "context_header": route.to_context_header(),
            # Complexity-aware dynamic scaling
            "estimated_steps": route.estimated_steps,
            "computed_max_iterations": get_max_iterations(route),
            "experts_enabled": route.is_complex,
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== Note-Taker Endpoints ====================


@app.get("/v1/notes/stats")
async def notes_stats() -> Dict[str, Any]:
    """Statistics about the note-taker knowledge base."""
    if note_observer is None:
        return {"error": "NoteObserver not initialised", "enabled": False}
    return await note_observer.get_stats()


@app.get("/v1/notes/search")
async def notes_search(
    q: str = "",
    user_id: Optional[str] = None,
    n_results: int = 10,
) -> Dict[str, Any]:
    """Semantic search over extracted notes.

    Usage:
        curl "http://host:8500/v1/notes/search?q=vLLM+debugging&user_id=seli"
    """
    if note_observer is None:
        return {"error": "NoteObserver not initialised", "enabled": False}
    if not q:
        return {"error": "Missing 'q' query parameter"}

    results = await note_observer.search_notes(
        query=q,
        user_id=user_id,
        n_results=n_results,
        min_score=0.15,
    )
    return {
        "query": q,
        "user_id": user_id,
        "count": len(results),
        "notes": results,
    }


# ==================== Admin / Testing Endpoints ====================


@app.get("/workspace/files")
async def list_workspace_files() -> Dict[str, Any]:
    """List files in /workspace (for test cleanup verification)."""
    import os

    workspace = "/workspace"
    if not os.path.isdir(workspace):
        return {"error": "Workspace directory not found", "files": []}
    try:
        files = os.listdir(workspace)
        return {"count": len(files), "files": sorted(files)}
    except Exception as e:
        return {"error": str(e), "files": []}


@app.delete("/workspace/files")
async def clean_workspace_files(prefix: str = "") -> Dict[str, Any]:
    """Delete files from /workspace matching an optional prefix.

    Without prefix, deletes ALL files (use for test cleanup).
    With prefix, only deletes files whose name starts with the prefix.

    Only deletes regular files, not directories.
    """
    import os

    workspace = "/workspace"
    if not os.path.isdir(workspace):
        return {"error": "Workspace directory not found", "deleted": []}
    deleted = []
    errors = []
    try:
        for name in os.listdir(workspace):
            path = os.path.join(workspace, name)
            if not os.path.isfile(path):
                continue
            if prefix and not name.startswith(prefix):
                continue
            try:
                os.remove(path)
                deleted.append(name)
            except Exception as e:
                errors.append({"file": name, "error": str(e)})
    except Exception as e:
        return {"error": str(e), "deleted": deleted}
    return {"deleted_count": len(deleted), "deleted": deleted, "errors": errors}
