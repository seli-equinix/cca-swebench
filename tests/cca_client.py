"""OTel-instrumented HTTP client for the CCA Agent-as-a-Model server.

Uses SSE streaming with idle timeout instead of fixed total timeouts.
This means tests won't fail just because a task takes longer than expected
— they only fail if the server stops sending data entirely.

Every method creates OpenTelemetry spans that are exported to Phoenix,
giving full visibility into test → HTTP request → response flow.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from opentelemetry import trace
from opentelemetry.propagate import inject as otel_inject

# Timeout defaults (seconds)
TIMEOUT_HEALTH = 10
TIMEOUT_CONNECT = 30
TIMEOUT_IDLE = 120       # No data for 120s → dead
TIMEOUT_DIAGNOSTIC = 15


class ChatResult:
    """Parsed result from a /v1/chat/completions call."""

    def __init__(self, raw: Dict[str, Any], elapsed_ms: float) -> None:
        self.raw = raw
        self.elapsed_ms = elapsed_ms
        self.id: str = raw.get("id", "")
        self.model: str = raw.get("model", "")

        choices = raw.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        self.content: str = msg.get("content", "") or ""
        self.role: str = msg.get("role", "")
        self.reasoning: Optional[str] = msg.get("reasoning")
        self.finish_reason: str = (
            choices[0].get("finish_reason", "") if choices else ""
        )

        self.usage: Dict[str, int] = raw.get("usage", {})
        self.metadata: Dict[str, Any] = raw.get("context_metadata", {}) or {}

    @property
    def user_identified(self) -> bool:
        return self.metadata.get("user_identified", False)

    @property
    def user_name(self) -> Optional[str]:
        return self.metadata.get("user_name")

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"ChatResult({self.elapsed_ms:.0f}ms, {len(self.content)} chars: {preview!r})"


class CCAClient:
    """HTTP client for CCA AAAM server with streaming + idle timeout.

    Uses SSE streaming to avoid fixed total timeouts. The idle_timeout
    resets every time data arrives (content, keepalive, progress comments).
    A task that takes 10 minutes but is actively working will succeed;
    a hung connection that goes silent for idle_timeout seconds will fail.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.4.205:8500",
        tracer: Optional[trace.Tracer] = None,
        idle_timeout: float = TIMEOUT_IDLE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tracer = tracer or trace.get_tracer("cca-tests")
        self.idle_timeout = idle_timeout
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=TIMEOUT_CONNECT,
                read=idle_timeout,
                write=30.0,
                pool=30.0,
            )
        )

    def close(self) -> None:
        self._client.close()

    # ==================== Core Methods ====================

    def health(self) -> Dict[str, Any]:
        """GET /health — check server status."""
        with self.tracer.start_as_current_span("cca.health") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            try:
                resp = self._client.get(
                    f"{self.base_url}/health", timeout=TIMEOUT_HEALTH
                )
                data = resp.json()
                span.set_attribute("cca.status", data.get("status", "unknown"))
                span.set_attribute("cca.active_sessions", data.get("active_sessions", 0))
                return data
            except Exception as e:
                span.set_attribute("cca.status", "error")
                span.set_attribute("cca.error", str(e))
                return {"status": "unreachable", "error": str(e)}

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        idle_timeout: Optional[float] = None,
        system: Optional[str] = None,
        # Legacy parameter — ignored, kept for backwards compatibility
        timeout: Optional[int] = None,
    ) -> ChatResult:
        """POST /v1/chat/completions — send a message to the CCA agent.

        Uses SSE streaming with idle timeout. The connection stays open as
        long as the server sends data (content, keepalives, progress).
        Only times out if no data arrives for idle_timeout seconds.

        Args:
            message: The user message to send.
            session_id: Optional session ID for multi-turn conversations.
            idle_timeout: Seconds of silence before timeout (default: 120s).
            system: Optional system message.
            timeout: DEPRECATED — ignored. Use idle_timeout instead.
        """
        session_id = session_id or f"test-{uuid.uuid4().hex[:12]}"
        read_timeout = idle_timeout or self.idle_timeout

        with self.tracer.start_as_current_span("cca.chat") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", message[:500])
            span.set_attribute("cca.session_id", session_id)
            span.set_attribute("cca.message", message[:200])
            span.set_attribute("cca.idle_timeout", read_timeout)

            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})

            payload = {
                "model": "cca",
                "messages": messages,
                "stream": True,
                "session_id": session_id,
            }

            t0 = time.time()
            try:
                result = self._stream_chat(
                    payload, session_id, read_timeout, span
                )
                elapsed_ms = (time.time() - t0) * 1000
                result.elapsed_ms = elapsed_ms

                span.set_attribute("cca.status", "success")
                span.set_attribute("cca.duration_ms", elapsed_ms)
                span.set_attribute("cca.response_length", len(result.content))
                span.set_attribute("cca.response_preview", result.content[:500])
                span.set_attribute("output.value", result.content[:500])
                span.set_attribute("cca.finish_reason", result.finish_reason)
                if result.metadata:
                    span.set_attribute(
                        "cca.tool_iterations",
                        result.metadata.get("tool_iterations", 0),
                    )
                    if result.metadata.get("route"):
                        span.set_attribute("cca.route", result.metadata["route"])
                if result.user_identified:
                    span.set_attribute("cca.user_identified", True)
                    span.set_attribute("cca.user_name", result.user_name or "")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                span.set_attribute("cca.status", "error")
                span.set_attribute("cca.error", str(e))
                span.set_attribute("cca.duration_ms", elapsed_ms)
                raise

    def _stream_chat(
        self,
        payload: Dict[str, Any],
        session_id: str,
        read_timeout: float,
        span: Any,
    ) -> ChatResult:
        """Execute streaming chat and accumulate response.

        Parses SSE events, accumulates content + reasoning, and extracts
        context_metadata from the final metadata event before [DONE].
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        completion_id = ""
        model = ""
        finish_reason = ""
        context_metadata: Dict[str, Any] = {}

        timeout = httpx.Timeout(
            connect=TIMEOUT_CONNECT,
            read=read_timeout,
            write=30.0,
            pool=30.0,
        )

        # Build headers with W3C trace context propagation
        # This injects traceparent so server spans become children
        # of the test trace in Phoenix (unified trace view).
        headers: Dict[str, str] = {"X-Session-Id": session_id}
        otel_inject(headers)

        with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode()
                return ChatResult(
                    {"error": body, "status_code": resp.status_code}, 0
                )

            for line in resp.iter_lines():
                if not line.strip():
                    continue

                # SSE comments (keepalive, progress) — skip but they
                # reset the read timeout, which is the whole point
                if line.startswith(":"):
                    continue

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # strip "data: " prefix

                # Terminal marker
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Context metadata event (sent before [DONE])
                if "context_metadata" in chunk and "choices" not in chunk:
                    context_metadata = chunk["context_metadata"]
                    continue

                completion_id = chunk.get("id", completion_id)
                model = chunk.get("model", model)

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])

        # Build a ChatResult that looks like a non-streaming response
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None

        raw = {
            "id": completion_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning": reasoning,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {},
            "context_metadata": context_metadata,
        }
        return ChatResult(raw, 0)

    # ==================== Diagnostic Endpoints ====================

    def list_users(self) -> Dict[str, Any]:
        """GET /users — list all known user profiles."""
        with self.tracer.start_as_current_span("cca.list_users") as span:
            resp = self._client.get(
                f"{self.base_url}/users", timeout=TIMEOUT_DIAGNOSTIC
            )
            data = resp.json()
            span.set_attribute("cca.user_count", data.get("count", 0))
            return data

    def list_sessions(self) -> Dict[str, Any]:
        """GET /sessions — list active sessions."""
        with self.tracer.start_as_current_span("cca.list_sessions") as span:
            resp = self._client.get(
                f"{self.base_url}/sessions", timeout=TIMEOUT_DIAGNOSTIC
            )
            data = resp.json()
            span.set_attribute("cca.session_count", data.get("count", 0))
            return data

    def get_stats(self) -> Dict[str, Any]:
        """GET /stats — diagnostic statistics."""
        with self.tracer.start_as_current_span("cca.get_stats"):
            resp = self._client.get(
                f"{self.base_url}/stats", timeout=TIMEOUT_DIAGNOSTIC
            )
            return resp.json()

    def list_models(self) -> Dict[str, Any]:
        """GET /v1/models — list available models."""
        with self.tracer.start_as_current_span("cca.list_models"):
            resp = self._client.get(
                f"{self.base_url}/v1/models", timeout=TIMEOUT_DIAGNOSTIC
            )
            return resp.json()

    # ==================== Helpers ====================

    def find_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Search /users for a user by display name (case-insensitive)."""
        with self.tracer.start_as_current_span("cca.find_user") as span:
            span.set_attribute("cca.search_name", name)
            data = self.list_users()
            for user in data.get("users", []):
                if user.get("display_name", "").lower() == name.lower():
                    span.set_attribute("cca.found", True)
                    return user
                # Also check aliases
                aliases = [a.lower() for a in user.get("aliases", [])]
                if name.lower() in aliases:
                    span.set_attribute("cca.found", True)
                    return user
            span.set_attribute("cca.found", False)
            return None

    def cleanup_test_user(self, name: str, session_id: Optional[str] = None) -> None:
        """Delete a test user profile via REST API.

        Uses DELETE /users/{user_id} directly — no LLM round-trip needed.
        Best-effort cleanup — failures are logged but don't raise.
        """
        with self.tracer.start_as_current_span("cca.cleanup_user") as span:
            span.set_attribute("cca.cleanup_target", name)
            try:
                user = self.find_user_by_name(name)
                if user is None:
                    span.set_attribute("cca.cleanup_status", "not_found")
                    return
                user_id = user["user_id"]
                resp = self._client.delete(
                    f"{self.base_url}/users/{user_id}",
                    timeout=TIMEOUT_DIAGNOSTIC,
                )
                span.set_attribute("cca.cleanup_status", "deleted")
                span.set_attribute("cca.cleanup_response", resp.text[:200])
            except Exception as e:
                span.set_attribute("cca.cleanup_status", f"failed: {e}")

    def list_workspace_files(self) -> Dict[str, Any]:
        """GET /workspace/files — list files in /workspace."""
        with self.tracer.start_as_current_span("cca.list_workspace"):
            resp = self._client.get(
                f"{self.base_url}/workspace/files", timeout=TIMEOUT_DIAGNOSTIC
            )
            return resp.json()

    def clean_workspace_files(self, prefix: str = "") -> Dict[str, Any]:
        """DELETE /workspace/files — remove files from /workspace.

        Without prefix, deletes ALL files. With prefix, only matching.
        """
        with self.tracer.start_as_current_span("cca.clean_workspace") as span:
            span.set_attribute("cca.cleanup_prefix", prefix)
            params = {"prefix": prefix} if prefix else {}
            resp = self._client.request(
                "DELETE",
                f"{self.base_url}/workspace/files",
                params=params,
                timeout=TIMEOUT_DIAGNOSTIC,
            )
            data = resp.json()
            span.set_attribute(
                "cca.cleanup_deleted", data.get("deleted_count", 0)
            )
            return data
