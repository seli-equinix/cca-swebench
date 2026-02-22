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
1. HttpCodeAssistEntry injects user context into the task definition
   and adds UserToolsExtension to the extensions list
2. HttpIOInterface captures output instead of printing to terminal
3. User identification happens before invocation via UserSessionManager
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.entry.base import EntryInput
from ..lib.confucius import Confucius
from .http_entry import HttpCodeAssistEntry
from .io_adapter import HttpIOInterface
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelInfo,
    build_completion_response,
    generate_completion_id,
)
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

# ==================== Globals (initialised in lifespan) ====================

session_pool: SessionPool
user_session_mgr: UserSessionManager
critical_facts_extractor: CriticalFactsExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application startup / shutdown lifecycle."""
    global session_pool, user_session_mgr, critical_facts_extractor

    logger.info("CCA HTTP server starting up...")

    # Initialise user session manager (Redis + Qdrant)
    user_session_mgr = UserSessionManager()
    await user_session_mgr.initialize()

    # Initialise critical facts extractor (shares Redis with session manager)
    critical_facts_extractor = CriticalFactsExtractor(
        redis_client=user_session_mgr._redis
    )

    # Initialise session pool (manages Confucius instances)
    session_pool = SessionPool(max_sessions=50, session_ttl=3600)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("CCA HTTP server ready")
    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await user_session_mgr.close()
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


# ==================== Routes ====================


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
    http_request: Request = None,  # type: ignore[assignment]
) -> Any:
    """OpenAI-compatible chat completions powered by CCA's full agent loop."""
    start_time = time.time()

    # 1. Derive session ID
    session_id = derive_session_id(request, x_session_id)
    logger.info(f"Request to session {session_id[:16]}...")

    # 2. Detect client type from headers
    headers = dict(http_request.headers) if http_request else {}
    client_type = UserSessionManager.detect_client_type(headers)

    # 3. Get or create user session
    session = await user_session_mgr.get_or_create_session(session_id)
    session.message_count += 1
    if client_type:
        session.client_type = client_type

    # 4. Smart auto-identification on first message of new sessions
    user = None
    id_result: Dict[str, Any] = {}
    if not session.identified:
        user_message = extract_last_user_message(request.messages)
        if user_message:
            id_result = await user_session_mgr.smart_identify_on_first_message(
                user_message, session, client_type
            )
            if id_result.get("identified"):
                user = id_result.get("user")
    else:
        user = await user_session_mgr.get_user_for_session(session)

    # 5. Build user context for system prompt injection
    user_context = ""
    if user:
        # Get critical facts for this session
        critical_facts_str = await critical_facts_extractor.format_facts_for_context(
            session_id
        )
        user_context = build_user_context(user, critical_facts_str, id_result or None)
    elif id_result.get("action") == "asking":
        user_context = build_uncertain_context(
            id_result.get("potential_user", ""),
            id_result.get("confidence", 0.0),
        )
    else:
        user_context = build_anonymous_context()

    # 6. Create user tools extension
    user_ext = UserToolsExtension(
        session_mgr=user_session_mgr,
        session=session,
    )

    # 7. Create HTTP entry with user context + tools
    entry = HttpCodeAssistEntry(
        user_context=user_context,
        user_extension=user_ext,
    )

    # 8. Extract user message
    user_message = extract_last_user_message(request.messages)
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
        )

    # 9. Create fresh IO for this request
    stream_queue = asyncio.Queue() if request.stream else None
    io = HttpIOInterface(stream_queue=stream_queue)

    # 10. Acquire session from pool → Confucius instance
    pool_entry = await session_pool.acquire(session_id, io)

    try:
        if request.stream:
            # Streaming mode: run agent as background task, stream output as SSE
            completion_id = generate_completion_id()

            async def run_agent() -> None:
                try:
                    await pool_entry.cf.invoke_analect(
                        entry, EntryInput(question=user_message)
                    )
                finally:
                    await io.signal_done()

            agent_task = asyncio.create_task(run_agent())

            return StreamingResponse(
                sse_stream(io, agent_task, request, completion_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming mode: run agent, collect output, return complete response
            try:
                await pool_entry.cf.invoke_analect(
                    entry, EntryInput(question=user_message)
                )
            except Exception as e:
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

            metadata = ContextMetadata(
                user_identified=session.identified,
                user_name=user.display_name if user else None,
                execution_time_ms=execution_time,
            )

            response = build_completion_response(
                content=response_text,
                model=request.model or "cca-agent",
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
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "cca-agent",
                "object": "model",
                "created": 1700000000,
                "owned_by": "cca",
                "permission": [],
                "root": "cca-agent",
                "parent": None,
            },
            {
                "id": "cca",
                "object": "model",
                "created": 1700000000,
                "owned_by": "cca",
                "permission": [],
                "root": "cca",
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
