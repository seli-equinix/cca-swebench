# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""SSE streaming with keepalive for the CCA HTTP server.

Uses an asyncio.Queue fed by HttpIOInterface to stream orchestrator
output as Server-Sent Events. Sends keepalive comments every 8 seconds
to prevent Continue.dev and other clients from timing out during long
tool execution or vLLM inference calls (20-30s).

Streaming architecture (v1):
- The orchestrator runs as a background asyncio.Task
- As it works, it pushes OutputChunks to the stream queue
- We format each chunk as an SSE event and yield it
- Thinking blocks → delta.reasoning_content (if include_reasoning)
- Assistant text → delta.content
- Progress/tool status → SSE comments (invisible to client)
- When the task completes, we send data: [DONE]

Note: In v1, the final response arrives as a complete block from io.ai(),
not token-by-token. Token-by-token streaming can be added in v2.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from .io_adapter import HttpIOInterface, OutputChunk
from .models import ChatCompletionRequest, build_chunk

logger = logging.getLogger(__name__)

# Keepalive interval in seconds — must be less than client timeout
# Continue.dev default timeout is ~30s, we use 8s for safety
KEEPALIVE_INTERVAL = 8.0


async def sse_stream(
    io: HttpIOInterface,
    agent_task: asyncio.Task,  # type: ignore[type-arg]
    request: ChatCompletionRequest,
    completion_id: str,
    metadata_callback: Optional[callable] = None,
) -> AsyncGenerator[str, None]:
    """Stream orchestrator output as SSE events.

    Reads from the HttpIOInterface's stream queue and formats each chunk
    as an SSE data event. Sends keepalive comments during periods of silence.

    Args:
        io: The HttpIOInterface with a stream_queue attached.
        agent_task: The asyncio.Task running the orchestrator.
        request: The original chat completion request.
        completion_id: Unique ID for this completion.

    Yields:
        SSE-formatted strings (data: {json}\\n\\n or : keepalive\\n\\n).
    """
    queue = io._stream_queue
    if queue is None:
        logger.error("sse_stream called without stream_queue on IO adapter")
        yield "data: [DONE]\n\n"
        return

    model = request.model or "cca-agent"
    include_reasoning = request.include_reasoning if request.include_reasoning is not None else True

    # Send initial role chunk
    initial_chunk = build_chunk(
        completion_id=completion_id,
        model=model,
        role="assistant",
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    try:
        while True:
            # Check if agent is done AND queue is empty
            if agent_task.done() and queue.empty():
                # Check for exceptions
                if agent_task.exception():
                    error_msg = str(agent_task.exception())
                    logger.error(f"Agent task failed: {error_msg}")
                    error_chunk = build_chunk(
                        completion_id=completion_id,
                        model=model,
                        content=f"\n\nError: {error_msg}",
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                break

            try:
                # Wait for next chunk with keepalive timeout
                chunk: Optional[OutputChunk] = await asyncio.wait_for(
                    queue.get(), timeout=KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # No output for 8 seconds — send keepalive comment
                # SSE comments (lines starting with :) keep the connection alive
                # but are invisible to the client
                yield ": keepalive\n\n"
                continue

            # None sentinel = agent is done
            if chunk is None:
                break

            # Format chunk based on type
            sse_data = _format_chunk(chunk, completion_id, model, include_reasoning)
            if sse_data:
                yield sse_data

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by client disconnect")
        agent_task.cancel()
        raise
    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        error_chunk = build_chunk(
            completion_id=completion_id,
            model=model,
            content=f"\n\nStream error: {e}",
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    # Send finish reason chunk
    finish_chunk = build_chunk(
        completion_id=completion_id,
        model=model,
        finish_reason="stop",
    )
    yield f"data: {finish_chunk.model_dump_json()}\n\n"

    # Send context_metadata event (if callback provided)
    # This lets streaming clients access tool_iterations, route, etc.
    if metadata_callback:
        try:
            metadata = metadata_callback()
            if metadata:
                yield f"data: {json.dumps({'context_metadata': metadata})}\n\n"
        except Exception as e:
            logger.warning(f"Failed to build streaming metadata: {e}")

    # Send terminal marker
    yield "data: [DONE]\n\n"


def _format_chunk(
    chunk: OutputChunk,
    completion_id: str,
    model: str,
    include_reasoning: bool,
) -> Optional[str]:
    """Format an OutputChunk as an SSE data event.

    Returns None for chunks that should not be sent to the client.
    """
    if chunk.chunk_type == "assistant":
        # Assistant text → delta.content
        sse_chunk = build_chunk(
            completion_id=completion_id,
            model=model,
            content=chunk.text,
        )
        return f"data: {sse_chunk.model_dump_json()}\n\n"

    elif chunk.chunk_type == "thinking":
        if include_reasoning:
            # Thinking → delta.reasoning_content
            sse_chunk = build_chunk(
                completion_id=completion_id,
                model=model,
                reasoning_content=chunk.text,
            )
            return f"data: {sse_chunk.model_dump_json()}\n\n"
        else:
            # Thinking suppressed — send as SSE comment for keepalive
            return ": thinking\n\n"

    elif chunk.chunk_type == "progress":
        # Progress updates → SSE comment (invisible to client, keeps connection alive)
        label = chunk.label or "working"
        return f": {label}\n\n"

    elif chunk.chunk_type == "error":
        # Errors → visible as content
        sse_chunk = build_chunk(
            completion_id=completion_id,
            model=model,
            content=f"\n\nError: {chunk.text}",
        )
        return f"data: {sse_chunk.model_dump_json()}\n\n"

    else:
        # Generic text — skip (debug/log output)
        return None
