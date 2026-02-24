# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Expert Router — Functionary-based request classification.

Uses the small Functionary model (8B on node5) to classify incoming
user requests and route them to the appropriate CCA expert entry.

Called from app.py BEFORE building the entry. The classification result
determines which entry class and system prompt is used.

Architecture:
    User message → classify_request() → Functionary → RouteDecision
                                                        ├── "coder"     → HttpRoutedEntry (file+shell+memory)
                                                        ├── "infra"     → HttpRoutedEntry (shell+file+memory)
                                                        ├── "search"    → HttpRoutedEntry (web+file+memory)
                                                        ├── "planner"   → HttpRoutedEntry (memory only)
                                                        ├── "user"      → HttpRoutedEntry (user tools only)
                                                        ├── "direct"    → Functionary answers (no agent loop)
                                                        └── "clarify"   → Ask user for more info
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from ..core.config import RouterConfig

logger = logging.getLogger(__name__)


# ========================= Expert Types =========================


class ExpertType(str, Enum):
    """Expert categories for routing."""
    CODER = "coder"
    INFRASTRUCTURE = "infrastructure"
    SEARCH = "search"
    PLANNER = "planner"
    USER = "user"            # User identity, profiles, facts, preferences
    DIRECT = "direct"        # Simple Q&A — Functionary answers directly
    CLARIFY = "clarify"      # Need more info from user


# ========================= Route Decision =========================


@dataclass
class RouteDecision:
    """Result of the Functionary classification."""
    expert: ExpertType
    task_summary: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    direct_answer: str = ""             # Only set for ExpertType.DIRECT
    clarification_question: str = ""    # Only set for ExpertType.CLARIFY
    classification_time_ms: float = 0.0

    @property
    def is_direct_answer(self) -> bool:
        return self.expert == ExpertType.DIRECT

    @property
    def is_clarification(self) -> bool:
        return self.expert == ExpertType.CLARIFY

    def to_context_header(self) -> str:
        """Build a one-line routing context string for system prompt injection."""
        parts = [f"[Router: {self.expert.value}]"]
        if self.task_summary:
            parts.append(self.task_summary)
        # Add expert-specific hints
        p = self.parameters
        if p.get("files_mentioned"):
            parts.append(f"Files: {', '.join(p['files_mentioned'])}")
        if p.get("language"):
            parts.append(f"Lang: {p['language']}")
        if p.get("services_involved"):
            parts.append(f"Services: {', '.join(p['services_involved'])}")
        if p.get("nodes_involved"):
            parts.append(f"Nodes: {', '.join(p['nodes_involved'])}")
        if p.get("search_scope"):
            parts.append(f"Scope: {p['search_scope']}")
        if p.get("user_action"):
            parts.append(f"Action: {p['user_action']}")
        return " | ".join(parts)


# ========================= Routing Tools =========================

# These are the tool definitions sent to Functionary for classification.
# Each tool represents a routing decision the model can make.

ROUTING_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "route_to_coder",
            "description": (
                "Route to the coding expert. Use for: writing code, editing files, "
                "refactoring, debugging, implementing features, fixing bugs, "
                "creating files, modifying code, and Git operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {
                        "type": "string",
                        "description": "Brief summary of the coding task"
                    },
                    "files_mentioned": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths mentioned (if any)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Primary programming language"
                    },
                },
                "required": ["task_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_infrastructure",
            "description": (
                "Route to infrastructure/DevOps expert. Use for: Docker, "
                "Docker Compose, Docker Swarm, container management, "
                "server administration, networking, DNS, TLS/certificates, "
                "monitoring (Prometheus, Grafana), CI/CD, deployments, "
                "service health checks, log inspection, and system ops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {
                        "type": "string",
                        "description": "Infrastructure task description"
                    },
                    "services_involved": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Services involved (docker, traefik, redis, etc.)"
                    },
                    "nodes_involved": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Cluster nodes involved (node1-5, spark1, spark2)"
                    },
                },
                "required": ["task_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_search",
            "description": (
                "Route to search/research expert. Use for: searching codebases, "
                "finding files, looking up documentation, web research, "
                "and information retrieval that doesn't involve code changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {
                        "type": "string",
                        "description": "What to search for"
                    },
                    "search_scope": {
                        "type": "string",
                        "enum": ["codebase", "web", "docs", "all"],
                        "description": "Where to search"
                    },
                },
                "required": ["task_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_planner",
            "description": (
                "Route to planning/architecture expert. Use for: designing systems, "
                "creating implementation plans, breaking down complex tasks, "
                "evaluating approaches, and making architecture decisions. "
                "NOT for tasks that also need code execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {
                        "type": "string",
                        "description": "What needs to be planned"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "Planning scope"
                    },
                },
                "required": ["task_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_user",
            "description": (
                "Route to user management. Use for: user identification "
                "('I'm Sean', 'my name is...'), viewing/updating/deleting "
                "user profiles, storing personal facts ('remember I work at...'), "
                "updating preferences, managing skills and aliases. "
                "Use when the user's PRIMARY intent is about their identity "
                "or profile — NOT when they introduce themselves while "
                "asking a coding/infra question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {
                        "type": "string",
                        "description": "User management task description"
                    },
                    "user_action": {
                        "type": "string",
                        "enum": [
                            "identify", "view_profile", "update_profile",
                            "manage_facts", "manage_preferences", "delete_profile",
                        ],
                        "description": "The primary user management action"
                    },
                },
                "required": ["task_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_directly",
            "description": (
                "Answer the question directly without an expert. Use ONLY for: "
                "simple factual questions, greetings, concept explanations, "
                "quick calculations — anything that needs NO file access, "
                "NO code execution, and NO agent loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Direct answer to the user"
                    },
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": (
                "Ask the user for clarification. Use ONLY when the request "
                "is too ambiguous to route to any expert."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Clarifying question for the user"
                    },
                },
                "required": ["question"]
            }
        }
    },
]


# System prompt for the classifier
ROUTER_SYSTEM_PROMPT = """\
You are an expert request router for CCA (Confucius Code Agent), a full-stack \
coding and infrastructure management agent.

Analyze the user's request and call EXACTLY ONE routing function.

Routing rules:
- Code writing, editing, debugging, refactoring → route_to_coder
- Docker, Swarm, servers, networking, deployments, monitoring → route_to_infrastructure
- Searching codebases, docs, or the web (no code changes) → route_to_search
- Architecture design, planning, task breakdown (no code changes) → route_to_planner
- User identity, profiles, facts, preferences, skills/aliases → route_to_user
- Simple greetings, factual questions, concept explanations → answer_directly
- Ambiguous requests where you can't determine intent → request_clarification

Disambiguation:
- Coder vs infrastructure: Docker/containers/services/nodes → infrastructure; \
files/functions/classes/tests/bugs → coder.
- User vs coder: If user introduces themselves AND asks a coding question in the \
same message, the coding task is primary → coder. If the message is ONLY about \
their identity, profile, or stored data → user.
- "Delete my profile", "what do you know about me?", "remember I work at X" → user.
- "Hi I'm Sean, write me a Python script" → coder (coding is primary).

ALWAYS call exactly one function. Never respond with plain text."""


# ========================= Classifier =========================


# Module-level shared client (initialized lazily)
_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _client


async def close_client() -> None:
    """Close the shared HTTP client (call on shutdown)."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def classify_request(
    user_message: str,
    config: RouterConfig,
    recent_messages: Optional[List[Dict[str, str]]] = None,
) -> RouteDecision:
    """Classify a user request using Functionary.

    Args:
        user_message: The user's latest message
        config: Router configuration from config.toml
        recent_messages: Optional recent conversation context (last 2-3 messages)

    Returns:
        RouteDecision with the expert classification and extracted parameters.
        Falls back to config.fallback_entry on any error.
    """
    start = time.monotonic()
    client = await _get_client()

    # Build messages for classification
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
    ]

    # Add recent context if available (helps with follow-up routing)
    if recent_messages:
        for msg in recent_messages[-2:]:
            messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "functionary",
        "messages": messages,
        "tools": ROUTING_TOOLS,
        "tool_choice": "required",
        "temperature": config.temperature,
        "max_tokens": 512,
    }

    try:
        resp = await client.post(
            f"{config.url}/v1/chat/completions",
            json=payload,
            timeout=config.timeout_ms / 1000.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning(f"Router classification timed out ({elapsed:.0f}ms), using fallback")
        return _fallback(config, elapsed, "timeout")
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        logger.error(f"Router classification error: {e}")
        return _fallback(config, elapsed, str(e))

    elapsed_ms = (time.monotonic() - start) * 1000

    # Parse tool call from response
    choices = data.get("choices", [])
    if not choices:
        return _fallback(config, elapsed_ms, "empty choices")

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        # Model answered with text instead of tool call
        content = message.get("content", "")
        logger.warning(f"No tool call from router (text: {content[:80]}), using fallback")
        return _fallback(config, elapsed_ms, "no tool call")

    # Parse the first tool call
    tc = tool_calls[0]
    func_name = tc.get("function", {}).get("name", "")
    func_args_str = tc.get("function", {}).get("arguments", "{}")

    try:
        func_args = json.loads(func_args_str)
    except json.JSONDecodeError:
        func_args = {}

    # Map function name → RouteDecision
    decision = _parse_tool_call(func_name, func_args, elapsed_ms)

    logger.info(
        f"Route: {decision.expert.value} ({elapsed_ms:.0f}ms) "
        f"| {decision.task_summary[:60]}"
    )

    return decision


def _parse_tool_call(
    func_name: str,
    args: Dict[str, Any],
    elapsed_ms: float,
) -> RouteDecision:
    """Convert a Functionary tool call into a RouteDecision."""

    FUNC_TO_EXPERT = {
        "route_to_coder": ExpertType.CODER,
        "route_to_infrastructure": ExpertType.INFRASTRUCTURE,
        "route_to_search": ExpertType.SEARCH,
        "route_to_planner": ExpertType.PLANNER,
        "route_to_user": ExpertType.USER,
        "answer_directly": ExpertType.DIRECT,
        "request_clarification": ExpertType.CLARIFY,
    }

    expert = FUNC_TO_EXPERT.get(func_name, ExpertType.CODER)

    return RouteDecision(
        expert=expert,
        task_summary=args.get("task_summary", args.get("answer", args.get("question", ""))),
        parameters=args,
        direct_answer=args.get("answer", ""),
        clarification_question=args.get("question", ""),
        classification_time_ms=elapsed_ms,
    )


def _fallback(config: RouterConfig, elapsed_ms: float, reason: str) -> RouteDecision:
    """Build a fallback RouteDecision when classification fails."""
    expert_map = {
        "coder": ExpertType.CODER,
        "infrastructure": ExpertType.INFRASTRUCTURE,
        "search": ExpertType.SEARCH,
        "planner": ExpertType.PLANNER,
    }
    return RouteDecision(
        expert=expert_map.get(config.fallback_entry, ExpertType.CODER),
        task_summary=f"Fallback ({reason})",
        classification_time_ms=elapsed_ms,
    )
