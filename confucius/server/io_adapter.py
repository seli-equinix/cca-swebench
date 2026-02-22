# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""HTTP IOInterface implementation for the CCA server.

The AnthropicLLMOrchestrator outputs through IOInterface as it works:
- io.ai(text)     → assistant text responses
- io.system(text, run_label="Thinking") → thinking/reasoning blocks
- io.system(text, run_label=...)        → tool progress updates
- io.error(text)  → error messages
- io.log(text)    → debug output

All methods delegate to print() via IOInterface defaults, except ai() and
system() which we override to tag chunks by type for proper SSE formatting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.io.base import IOInterface
from ..core import types as cf


@dataclass
class OutputChunk:
    """A tagged chunk of output from the orchestrator."""

    text: str
    chunk_type: str = "text"  # "assistant", "thinking", "progress", "error", "text"
    label: Optional[str] = None


class HttpIOInterface(IOInterface):
    """IOInterface that captures orchestrator output for HTTP responses.

    In non-streaming mode, output is buffered and read after the agent completes.
    In streaming mode, chunks are also pushed to an asyncio.Queue for real-time
    SSE forwarding.
    """

    def __init__(self, stream_queue: Optional[asyncio.Queue[OutputChunk | None]] = None) -> None:
        self._buffer: list[OutputChunk] = []
        self._stream_queue = stream_queue

    async def print(self, text: str, **kwargs: Any) -> None:
        """Default handler — captures untagged output."""
        chunk = OutputChunk(text=text, chunk_type="text")
        self._buffer.append(chunk)
        if self._stream_queue is not None:
            await self._stream_queue.put(chunk)

    async def ai(self, text: str, **kwargs: Any) -> None:
        """Capture assistant response text."""
        chunk = OutputChunk(text=text, chunk_type="assistant")
        self._buffer.append(chunk)
        if self._stream_queue is not None:
            await self._stream_queue.put(chunk)

    async def system(
        self,
        text: str,
        *,
        progress: int | None = None,
        run_status: cf.RunStatus | None = None,
        run_label: str | None = None,
        run_description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture system messages, tagging thinking blocks differently."""
        if run_label and "think" in run_label.lower():
            chunk_type = "thinking"
        elif run_status is not None:
            chunk_type = "progress"
        else:
            chunk_type = "text"

        chunk = OutputChunk(text=text, chunk_type=chunk_type, label=run_label)
        self._buffer.append(chunk)
        if self._stream_queue is not None:
            await self._stream_queue.put(chunk)

    async def error(self, text: str, **kwargs: Any) -> None:
        """Capture error messages."""
        chunk = OutputChunk(text=text, chunk_type="error")
        self._buffer.append(chunk)
        if self._stream_queue is not None:
            await self._stream_queue.put(chunk)

    async def log(self, text: str, **kwargs: Any) -> None:
        """Capture log messages (not forwarded to SSE, just buffered)."""
        chunk = OutputChunk(text=text, chunk_type="text")
        self._buffer.append(chunk)
        # Don't push logs to stream queue — they're debug info

    async def _get_input(self, prompt: str, placeholder: str | None = None) -> str:
        """HTTP mode does not support interactive input.

        Returns empty string so that confirm() defaults to False and
        choose_input() uses defaults, preventing the orchestrator from blocking.
        """
        return ""

    # ==================== Output Accessors ====================

    def get_response_text(self) -> str:
        """Get the final assistant response text (from io.ai() calls)."""
        parts = [c.text for c in self._buffer if c.chunk_type == "assistant"]
        return "\n".join(parts) if parts else ""

    def get_thinking_text(self) -> str:
        """Get thinking/reasoning text for response metadata."""
        parts = [c.text for c in self._buffer if c.chunk_type == "thinking"]
        return "\n".join(parts) if parts else ""

    def get_all_text(self) -> str:
        """Get all output text (fallback if no assistant-tagged chunks)."""
        return "\n".join(c.text for c in self._buffer if c.text)

    def get_buffer(self) -> list[OutputChunk]:
        """Get the raw output buffer."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear the output buffer for the next request."""
        self._buffer.clear()

    async def signal_done(self) -> None:
        """Signal the stream queue that the agent is done."""
        if self._stream_queue is not None:
            await self._stream_queue.put(None)
