# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""OpenAI-compatible Pydantic models for the CCA HTTP server.

Extracted and adapted from MCP server mcp_server.py (lines 3880-4040).
Provides full OpenAI Chat Completions API compatibility for clients
like Continue.dev, Open WebUI, LM Studio, Cursor, and Aider.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ==================== Request Models ====================


class FunctionDefinition(BaseModel):
    """OpenAI function definition for tool calling."""

    name: str = Field(..., description="The name of the function")
    description: Optional[str] = Field(
        None, description="Description of what the function does"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema for function parameters"
    )


class ToolDefinition(BaseModel):
    """OpenAI tool definition."""

    type: str = Field(
        "function", description="Type of tool (currently only 'function' supported)"
    )
    function: FunctionDefinition


class ToolCall(BaseModel):
    """Tool call in assistant message."""

    id: str = Field(..., description="Unique identifier for this tool call")
    type: str = Field("function", description="Type of tool call")
    function: Dict[str, str] = Field(..., description="Function name and arguments")


class ResponseFormat(BaseModel):
    """Response format specification."""

    type: str = Field("text", description="Response format: 'text' or 'json_object'")


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str = Field(..., description="Role: system, user, assistant, or tool")
    content: Optional[str] = Field(
        None, description="Message content (can be null for tool calls)"
    )
    name: Optional[str] = Field(
        None, description="Optional name for the message author"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        None, description="Tool calls made by assistant"
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of tool call this message responds to"
    )
    reasoning: Optional[str] = Field(
        None, description="Chain-of-thought reasoning from Thinking models"
    )


class ChatCompletionRequest(BaseModel):
    """Full OpenAI-compatible chat completion request."""

    model: Optional[str] = Field(default=None, description="Model to use (any name accepted)")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")

    # Generation parameters
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens (None = dynamic based on input)"
    )
    temperature: Optional[float] = Field(0.7, description="Sampling temperature (0-2)")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(False, description="Stream response via SSE")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")

    # Penalty parameters
    presence_penalty: Optional[float] = Field(
        0.0, description="Presence penalty (-2 to 2)"
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Frequency penalty (-2 to 2)"
    )

    # OpenAI standard fields
    n: Optional[int] = Field(1, description="Number of completions to generate")
    logprobs: Optional[bool] = Field(None, description="Return log probabilities")
    top_logprobs: Optional[int] = Field(
        None, description="Number of top logprobs to return (0-20)"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Token bias dictionary"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    user: Optional[str] = Field(None, description="User ID for tracking")

    # Tool/Function calling (Continue.dev uses this)
    tools: Optional[List[ToolDefinition]] = Field(
        None, description="Available tools for the model"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Tool selection: 'none', 'auto', or specific tool"
    )
    parallel_tool_calls: Optional[bool] = Field(
        True, description="Allow parallel tool calls"
    )

    # Response format (JSON mode)
    response_format: Optional[ResponseFormat] = Field(
        None, description="Response format specification"
    )

    # Reasoning/Thinking control
    enable_thinking: Optional[bool] = Field(
        None, description="Enable/disable thinking mode (None=model default)"
    )
    include_reasoning: Optional[bool] = Field(
        True, description="Include reasoning in response (default True)"
    )

    # CCA extension: session ID for session pooling
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation memory persistence"
    )


# ==================== Response Models ====================


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = Field(
        None, description="Log probabilities if requested"
    )


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCallInfo(BaseModel):
    """Information about a tool call made by the agent."""

    name: str = Field(..., description="Name of the tool called")
    success: bool = Field(True, description="Whether the tool executed successfully")
    iteration: int = Field(1, description="Which iteration of tool calling this was")


class ContextMetadata(BaseModel):
    """Metadata about the agent execution for this request."""

    tool_calls: Optional[List[ToolCallInfo]] = Field(
        None, description="Tools called by the agent during this request"
    )
    tool_iterations: int = Field(0, description="Number of tool calling iterations")
    route: Optional[str] = Field(
        None, description="Expert route used for this request"
    )
    user_identified: bool = Field(
        False, description="Whether the user was identified"
    )
    user_name: Optional[str] = Field(
        None, description="Identified user's display name"
    )
    execution_time_ms: float = Field(
        0.0, description="Total request execution time in milliseconds"
    )
    estimated_steps: int = Field(
        0, description="Router's estimated task complexity"
    )
    max_iterations: int = Field(
        0, description="Max iterations allowed for this request"
    )
    nudge_skipped: bool = Field(
        False, description="Whether the tool nudge was skipped"
    )
    circuit_breaker_fired: bool = Field(
        False, description="Whether error circuit breaker fired"
    )
    tools_escalated: bool = Field(
        False, description="Whether dynamic tool escalation was triggered"
    )
    escalated_groups: Optional[List[str]] = Field(
        None, description="Tool groups enabled via dynamic escalation"
    )


class ChatCompletionResponse(BaseModel):
    """Full OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = Field(
        None, description="Backend configuration fingerprint"
    )
    context_metadata: Optional[ContextMetadata] = Field(
        None, description="Metadata about agent execution"
    )


# ==================== Streaming Models ====================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk in a streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ==================== Helper Models ====================


class ModelInfo(BaseModel):
    """Model info for /v1/models endpoint."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    active_sessions: int = 0


# ==================== Utility Functions ====================


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def build_completion_response(
    content: str,
    model: str = "cca",
    finish_reason: str = "stop",
    reasoning: Optional[str] = None,
    metadata: Optional[ContextMetadata] = None,
) -> ChatCompletionResponse:
    """Build a complete ChatCompletionResponse from content."""
    message = ChatMessage(role="assistant", content=content)
    if reasoning:
        message.reasoning = reasoning

    # Rough token estimation (4 chars per token)
    prompt_tokens = 0  # Not tracked in this path
    completion_tokens = len(content) // 4

    return ChatCompletionResponse(
        id=generate_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        context_metadata=metadata,
    )


def build_chunk(
    completion_id: str,
    model: str,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ChatCompletionChunk:
    """Build a single SSE chunk."""
    delta = ChatCompletionChunkDelta(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
    )
    return ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )
