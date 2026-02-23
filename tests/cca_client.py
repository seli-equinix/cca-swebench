"""OTel-instrumented HTTP client for the CCA Agent-as-a-Model server.

Every method creates OpenTelemetry spans that are exported to Phoenix,
giving full visibility into test → HTTP request → response flow.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from opentelemetry import trace

# Timeout defaults (seconds)
TIMEOUT_HEALTH = 10
TIMEOUT_CHAT = 180
TIMEOUT_CHAT_SLOW = 360
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
    """HTTP client for CCA AAAM server with OpenTelemetry tracing."""

    def __init__(
        self,
        base_url: str = "http://192.168.4.205:8500",
        tracer: Optional[trace.Tracer] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tracer = tracer or trace.get_tracer("cca-aaam-tests")
        self._client = httpx.Client(timeout=TIMEOUT_CHAT)

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
        timeout: int = TIMEOUT_CHAT,
        system: Optional[str] = None,
    ) -> ChatResult:
        """POST /v1/chat/completions — send a message to the CCA agent.

        Creates a Phoenix-visible span with message, response, and timing.
        """
        session_id = session_id or f"test-{uuid.uuid4().hex[:12]}"

        with self.tracer.start_as_current_span("cca.chat") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", message[:500])
            span.set_attribute("cca.session_id", session_id)
            span.set_attribute("cca.message", message[:200])
            span.set_attribute("cca.timeout", timeout)

            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})

            payload = {
                "model": "cca",
                "messages": messages,
                "stream": False,
                "session_id": session_id,
            }

            t0 = time.time()
            try:
                resp = self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"X-Session-Id": session_id},
                    timeout=timeout,
                )
                elapsed_ms = (time.time() - t0) * 1000

                if resp.status_code != 200:
                    span.set_attribute("cca.status", "http_error")
                    span.set_attribute("cca.http_status", resp.status_code)
                    span.set_attribute("cca.error", resp.text[:500])
                    return ChatResult(
                        {"error": resp.text, "status_code": resp.status_code},
                        elapsed_ms,
                    )

                data = resp.json()
                result = ChatResult(data, elapsed_ms)

                span.set_attribute("cca.status", "success")
                span.set_attribute("cca.duration_ms", elapsed_ms)
                span.set_attribute("cca.response_length", len(result.content))
                span.set_attribute("cca.response_preview", result.content[:500])
                span.set_attribute("output.value", result.content[:500])
                span.set_attribute("cca.finish_reason", result.finish_reason)
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
