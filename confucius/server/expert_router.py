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

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from ..core.config import RouterConfig, ToolRouterConfig
from opentelemetry.trace import StatusCode

from ..core.tracing import (
    get_tracer,
    OPENINFERENCE_SPAN_KIND,
    INPUT_VALUE,
    OUTPUT_VALUE,
    LLM_MODEL_NAME,
)

try:
    from openinference.instrumentation import (
        get_llm_input_message_attributes,
        get_llm_output_message_attributes,
        get_llm_invocation_parameter_attributes,
        get_llm_tool_attributes,
        Message,
        Tool,
        ToolCall,
        ToolCallFunction,
    )
    _HAS_OPENINFERENCE = True
except ImportError:
    _HAS_OPENINFERENCE = False

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
    # User extraction — populated by router alongside routing classification
    detected_user_name: str = ""
    detected_user_facts: List[str] = field(default_factory=list)
    greeting_detected: bool = False
    # Complexity estimation — router estimates distinct developer actions
    estimated_steps: int = 10  # default = moderate complexity

    @property
    def is_direct_answer(self) -> bool:
        return self.expert == ExpertType.DIRECT

    @property
    def is_clarification(self) -> bool:
        return self.expert == ExpertType.CLARIFY

    @property
    def is_complex(self) -> bool:
        return self.estimated_steps >= 8

    @property
    def is_simple(self) -> bool:
        return self.estimated_steps <= 3

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
        parts.append(f"Estimated steps: {self.estimated_steps}")
        return " | ".join(parts)


# ========================= Routing Tools =========================

# These are the tool definitions sent to Functionary for classification.
# Each tool represents a routing decision the model can make.

# User extraction properties — added to EVERY routing tool so the router
# can extract user info alongside its routing classification in one call.
_USER_EXTRACTION_PROPS: Dict[str, Any] = {
    "detected_user_name": {
        "type": "string",
        "description": (
            "User's name if they introduce themselves "
            "(e.g. 'I\\'m Sean', 'my name is Alex', 'this is John')"
        ),
    },
    "detected_user_facts": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Personal facts mentioned in the message "
            "(e.g. 'I work at Equinix', 'I prefer Python', 'my project uses Docker')"
        ),
    },
    "greeting_detected": {
        "type": "boolean",
        "description": "True if the message contains a greeting or introduction",
    },
}

# Step estimation property — added to agent-loop routing tools so the router
# can estimate task complexity alongside its routing classification.
_STEP_ESTIMATION_PROP: Dict[str, Any] = {
    "estimated_steps": {
        "type": "integer",
        "description": (
            "Estimate the number of distinct steps a developer would take "
            "to complete this task. Examples:\n"
            "- Answer a code question or write a one-liner: 1\n"
            "- Write a short function (< 20 lines): 2-3\n"
            "- Fix a typo: 2\n"
            "- Add error handling to one file: 5\n"
            "- Refactor a function and add tests: 12\n"
            "- Trace code across multiple files: 20-30\n"
            "- Build a new feature with multiple components: 40-60\n"
            "- Large-scale refactor or new application: 60-100\n"
            "Default to 10 if unsure."
        ),
    },
}

ROUTING_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "route_to_coder",
            "description": (
                "Route to the coding expert. Use for: writing code, editing files, "
                "refactoring, debugging, implementing features, fixing bugs, "
                "creating files, modifying code, and Git operations. "
                "This route also handles user facts — if the user introduces "
                "themselves or asks about their profile while requesting code, "
                "use this route."
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
                    **_USER_EXTRACTION_PROPS,
                    **_STEP_ESTIMATION_PROP,
                },
                "required": ["task_summary", "estimated_steps"]
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
                    **_USER_EXTRACTION_PROPS,
                    **_STEP_ESTIMATION_PROP,
                },
                "required": ["task_summary", "estimated_steps"]
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
                    **_USER_EXTRACTION_PROPS,
                    **_STEP_ESTIMATION_PROP,
                },
                "required": ["task_summary", "estimated_steps"]
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
                    **_USER_EXTRACTION_PROPS,
                    **_STEP_ESTIMATION_PROP,
                },
                "required": ["task_summary", "estimated_steps"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_user",
            "description": (
                "Route to user management. Use ONLY when the request is "
                "exclusively about user identity or profile management: "
                "identification ('I'm Sean'), viewing/deleting profiles, "
                "storing facts ('remember I work at...'), updating "
                "preferences, managing skills and aliases. "
                "If the user ALSO asks for code, infrastructure help, "
                "or any technical task alongside user management, route "
                "to coder or infrastructure instead — those routes have "
                "built-in user memory tools."
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
                    **_USER_EXTRACTION_PROPS,
                    **_STEP_ESTIMATION_PROP,
                },
                "required": ["task_summary", "estimated_steps"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_directly",
            "description": (
                "Answer the question directly without an expert. Use ONLY for: "
                "simple factual questions, concept explanations, "
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
                    **_USER_EXTRACTION_PROPS,
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
                    **_USER_EXTRACTION_PROPS,
                },
                "required": ["question"]
            }
        }
    },
]


# ========================= Tool Selection (Phase 2) =========================

# Tool definitions for Functionary-based tool selection during escalation.
# When the agent gets stuck (no tools available for the task), Functionary
# analyzes the agent's output and calls one or more of these to select
# which tool groups to enable.

TOOL_SELECTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "enable_file_editor",
            "description": (
                "Enable file viewing/editing (str_replace_editor). "
                "For creating, viewing, or modifying code files, configs, scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the agent needs file editing tools",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_shell",
            "description": (
                "Enable bash/command execution. "
                "For running commands, scripts, pip install, docker, git."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the agent needs shell execution",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_web_search",
            "description": (
                "Enable web search + URL fetching. "
                "For finding online docs, news, current information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the agent needs web search tools",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_memory",
            "description": (
                "Enable planning memory (write_memory, read_memory). "
                "For complex multi-step tasks needing a plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the agent needs memory/planning tools",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_code_search",
            "description": (
                "Enable codebase search (search_codebase, search_knowledge). "
                "For finding functions, classes, files in indexed repositories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the agent needs code search tools",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_additional_tools",
            "description": (
                "Current tools are sufficient. The agent can complete "
                "the task with what it has."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why no additional tools are needed",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

TOOL_SELECTOR_SYSTEM_PROMPT = """\
You are a tool selector for CCA (Confucius Code Agent). The main agent is \
stuck — it needs tools it doesn't currently have.

Analyze the agent's last output and determine which tool groups to enable.

Available tool groups:
- file_editor: str_replace_editor for creating/editing files (code, configs, scripts)
- shell: bash command execution (run commands, install packages, build, git)
- web_search: search the internet, read web pages, find documentation
- memory: write_memory/read_memory for planning and tracking complex tasks
- code_search: search_codebase/search_knowledge for finding code in the project

Rules:
- Enable ONLY the tools the agent actually needs — fewer is better (1-2 ideal)
- If the agent output code blocks (```) or mentions creating files → enable file_editor
- If the agent describes running commands or installing packages → enable shell
- If the agent wants to search online or find docs → enable web_search
- If the task needs planning (multiple steps) → enable memory
- If the agent needs to find existing code → enable code_search
- Call no_additional_tools if the agent's current tools are sufficient

The agent's current route: {current_route}
Currently available tools: {current_tools}\
"""

# Map Functionary tool-selection function names → ToolGroup values.
# Imported lazily in select_tools_for_escalation() to avoid circular import
# (tool_groups imports from expert_router).
_TOOL_FUNC_TO_GROUP_NAME = {
    "enable_file_editor": "file",
    "enable_shell": "shell",
    "enable_web_search": "web",
    "enable_memory": "memory",
    "enable_code_search": "code_search",
}


async def select_tools_for_escalation(
    agent_output: str,
    current_route: str,
    current_tools: list[str],
    config: ToolRouterConfig,
) -> list[str]:
    """Ask Functionary which tool groups the agent needs.

    Returns list of tool group names to enable (e.g., ["file", "shell"]).
    Returns empty list if Functionary says no additional tools needed.
    Falls back to ["file", "shell"] on error.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("cca.tool_selector") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(INPUT_VALUE, agent_output[:500])
        span.set_attribute(LLM_MODEL_NAME, "functionary-small-v3.2")
        span.set_attribute("cca.tool_selector.route", current_route)
        span.set_attribute("cca.tool_selector.current_tools", str(current_tools[:20]))

        start = time.monotonic()
        client = await _get_client()

        system_prompt = TOOL_SELECTOR_SYSTEM_PROMPT.format(
            current_route=current_route,
            current_tools=", ".join(current_tools),
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": agent_output[:2000]},
        ]

        payload = {
            "model": "functionary",
            "messages": messages,
            "tools": TOOL_SELECTION_TOOLS,
            "tool_choice": "required",
            "temperature": config.temperature,
            "max_tokens": 256,
        }

        # Record OpenInference attributes for Phoenix tracing
        if _HAS_OPENINFERENCE:
            try:
                oi_msgs = [Message(role=m["role"], content=m["content"]) for m in messages]
                for k, v in get_llm_input_message_attributes(oi_msgs).items():
                    span.set_attribute(k, v)
                oi_tools = [Tool(json_schema=t) for t in TOOL_SELECTION_TOOLS]
                for k, v in get_llm_tool_attributes(oi_tools).items():
                    span.set_attribute(k, v)
            except Exception as e:
                logger.warning("Failed to set tool_selector OpenInference attrs: %s", e)

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
            logger.warning(
                "Tool selector timed out (%.0fms), falling back to file+shell",
                elapsed,
            )
            span.set_attribute("cca.tool_selector.status", "timeout")
            span.set_status(StatusCode.ERROR, "timeout")
            return ["file", "shell"]
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool selector error: %s", e)
            span.set_attribute("cca.tool_selector.status", "error")
            span.set_attribute("cca.tool_selector.error", str(e)[:200])
            span.set_status(StatusCode.ERROR, str(e)[:200])
            return ["file", "shell"]

        elapsed_ms = (time.monotonic() - start) * 1000

        # Record response in OpenInference format
        if _HAS_OPENINFERENCE:
            try:
                resp_msg = data.get("choices", [{}])[0].get("message", {})
                oi_tool_calls = []
                for tc in resp_msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    oi_tool_calls.append(ToolCall(
                        function=ToolCallFunction(
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", "{}"),
                        )
                    ))
                oi_out = [Message(
                    role=resp_msg.get("role", "assistant"),
                    content=resp_msg.get("content"),
                    tool_calls=oi_tool_calls or None,
                )]
                for k, v in get_llm_output_message_attributes(oi_out).items():
                    span.set_attribute(k, v)
            except Exception as e:
                logger.warning("Failed to set tool_selector output attrs: %s", e)

        # Parse tool calls from response
        choices = data.get("choices", [])
        if not choices:
            span.set_attribute("cca.tool_selector.status", "empty_choices")
            return ["file", "shell"]

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            logger.warning("No tool call from tool selector, falling back")
            span.set_attribute("cca.tool_selector.status", "no_tool_call")
            return ["file", "shell"]

        # Extract requested tool groups from ALL tool calls
        requested_groups: list[str] = []
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            func_args_str = tc.get("function", {}).get("arguments", "{}")

            if func_name == "no_additional_tools":
                logger.info(
                    "Tool selector: no additional tools needed (%.0fms)",
                    elapsed_ms,
                )
                span.set_attribute("cca.tool_selector.status", "no_tools_needed")
                span.set_attribute("cca.tool_selector.elapsed_ms", elapsed_ms)
                span.set_status(StatusCode.OK)
                return []

            group_name = _TOOL_FUNC_TO_GROUP_NAME.get(func_name)
            if group_name and group_name not in requested_groups:
                try:
                    reason = json.loads(func_args_str).get("reason", "")
                except json.JSONDecodeError:
                    reason = ""
                logger.info(
                    "Tool selector: enable %s — %s", group_name, reason[:80],
                )
                requested_groups.append(group_name)

        span.set_attribute("cca.tool_selector.status", "success")
        span.set_attribute("cca.tool_selector.elapsed_ms", elapsed_ms)
        span.set_attribute("cca.tool_selector.groups", str(requested_groups))
        span.set_attribute(OUTPUT_VALUE, str(requested_groups))
        span.set_status(StatusCode.OK)

        # Token usage
        usage = data.get("usage", {})
        if usage:
            span.set_attribute("llm.token_count.prompt", usage.get("prompt_tokens", 0))
            span.set_attribute("llm.token_count.completion", usage.get("completion_tokens", 0))

        logger.info(
            "Tool selector: enable %s (%.0fms)",
            requested_groups, elapsed_ms,
        )
        return requested_groups


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
- Simple factual questions, concept explanations → answer_directly
- Ambiguous requests where you can't determine intent → request_clarification

CRITICAL — answer_directly restrictions:
- NEVER use answer_directly for requests to write, create, build, implement, or generate code.
- NEVER use answer_directly for requests that say "write a function", "write a script", \
"create a class", "implement X", "build a parser", "generate code", or similar.
- These are ALWAYS coding tasks → route_to_coder, even if they seem simple.
- answer_directly is ONLY for pure knowledge questions with no code output \
(e.g., "What is the capital of France?", "Explain what a mutex is").
- NEVER use answer_directly for questions about "latest", "recent", "current", \
"newest", "up-to-date", or "news". These require web search → route_to_search.
- Questions like "What is the latest version of X?" or "What are the latest \
AI news?" ALWAYS need web search because your training data may be outdated.

Disambiguation:
- Coder vs infrastructure: Docker/containers/services/nodes → infrastructure; \
files/functions/classes/tests/bugs → coder.
- User vs coder: If user asks about their profile or facts AND also requests \
code/technical work in the same message → coder (it has user memory tools). \
Route to user ONLY when the message is exclusively about identity, profile, \
or stored data with no other task.
- "Delete my profile", "what do you know about me?", "remember I work at X" → user.
- "Hi I'm Sean, write me a Python script" → coder (coding is primary).

User extraction (apply to ALL routing functions):
- If the user introduces themselves ("I'm Sean", "my name is Alex", "this is John"), \
set detected_user_name to the name.
- If they mention personal facts ("I work at Equinix", "I prefer Python", \
"my project uses Docker"), add each fact to detected_user_facts.
- Set greeting_detected=true for greetings or introductions.
- ALWAYS route based on the primary task — user extraction is secondary metadata. \
Example: "Hi I'm Sean, fix the Docker config" → route_to_infrastructure with \
detected_user_name="Sean" and greeting_detected=true.

Step estimation (apply to all agent routes — coder, infrastructure, search, planner, user):
- Estimate how many distinct actions (file reads, edits, commands, verifications) the task requires.
- Think about: how many files to read, how many edits to make, how many tests to run.
- Simple fixes: 2-5 steps. Standard tasks: 8-15 steps. Large multi-file work: 20-50. \
Full feature builds: 50-100. Default to 10 if unsure.

ALWAYS call exactly one function. Never respond with plain text."""


# ========================= Classifier =========================


# Module-level shared client (initialized lazily, guarded by lock)
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client."""
    global _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return _client


async def close_client() -> None:
    """Close the shared HTTP client (call on shutdown)."""
    global _client
    async with _client_lock:
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
    tracer = get_tracer()

    with tracer.start_as_current_span("cca.router") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(INPUT_VALUE, user_message[:500])
        span.set_attribute(LLM_MODEL_NAME, "functionary-small-v3.2")
        span.set_attribute("cca.router.url", config.url)

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

        # Record full LLM request in span using OpenInference format
        # (flattened dot-notation attributes that Phoenix renders natively)
        if _HAS_OPENINFERENCE:
            try:
                oi_msgs = [Message(role=m["role"], content=m["content"]) for m in messages]
                input_attrs = get_llm_input_message_attributes(oi_msgs)
                for k, v in input_attrs.items():
                    span.set_attribute(k, v)
                inv_attrs = get_llm_invocation_parameter_attributes({
                    "model": payload["model"],
                    "temperature": payload["temperature"],
                    "max_tokens": payload["max_tokens"],
                    "tool_choice": payload["tool_choice"],
                })
                for k, v in inv_attrs.items():
                    span.set_attribute(k, v)
                oi_tools = [Tool(json_schema=t) for t in ROUTING_TOOLS]
                tool_attrs = get_llm_tool_attributes(oi_tools)
                for k, v in tool_attrs.items():
                    span.set_attribute(k, v)
                logger.info(
                    "Router span enriched: %d input + %d param + %d tool attrs",
                    len(input_attrs), len(inv_attrs), len(tool_attrs),
                )
            except Exception as e:
                logger.warning("Failed to set OpenInference input attrs: %s", e)

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
            span.set_attribute("cca.router.status", "timeout")
            span.set_attribute("cca.router.elapsed_ms", elapsed)
            span.set_status(StatusCode.ERROR, "timeout")
            return _fallback(config, elapsed, "timeout")
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(f"Router classification error: {e}")
            span.set_attribute("cca.router.status", "error")
            span.set_attribute("cca.router.error", str(e)[:200])
            span.set_status(StatusCode.ERROR, str(e)[:200])
            return _fallback(config, elapsed, str(e))

        elapsed_ms = (time.monotonic() - start) * 1000

        # Record full LLM response in span using OpenInference format
        if _HAS_OPENINFERENCE:
            try:
                resp_msg = data.get("choices", [{}])[0].get("message", {})
                oi_tool_calls = []
                for tc in resp_msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    oi_tool_calls.append(ToolCall(
                        function=ToolCallFunction(
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", "{}"),
                        )
                    ))
                oi_out = [Message(
                    role=resp_msg.get("role", "assistant"),
                    content=resp_msg.get("content"),
                    tool_calls=oi_tool_calls or None,
                )]
                for k, v in get_llm_output_message_attributes(oi_out).items():
                    span.set_attribute(k, v)
            except Exception as e:
                logger.warning("Failed to set OpenInference output attrs: %s", e)

        # Parse tool call from response
        choices = data.get("choices", [])
        if not choices:
            span.set_attribute("cca.router.status", "empty_choices")
            return _fallback(config, elapsed_ms, "empty choices")

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            # Model answered with text instead of tool call
            content = message.get("content", "")
            logger.warning(f"No tool call from router (text: {content[:80]}), using fallback")
            span.set_attribute("cca.router.status", "no_tool_call")
            span.set_attribute(OUTPUT_VALUE, content[:200])
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

        # Record result in span
        span.set_attribute("cca.router.status", "success")
        span.set_attribute("cca.router.expert", decision.expert.value)
        span.set_attribute("cca.router.function", func_name)
        span.set_attribute("cca.router.elapsed_ms", elapsed_ms)
        span.set_attribute(OUTPUT_VALUE, decision.task_summary[:200])

        # Token usage if available
        usage = data.get("usage", {})
        if usage:
            span.set_attribute("llm.token_count.prompt", usage.get("prompt_tokens", 0))
            span.set_attribute("llm.token_count.completion", usage.get("completion_tokens", 0))

        # Complexity estimation tracing
        span.set_attribute("cca.router.estimated_steps", decision.estimated_steps)

        # User extraction tracing
        if decision.detected_user_name:
            span.set_attribute("cca.router.detected_user", decision.detected_user_name)
        if decision.detected_user_facts:
            span.set_attribute("cca.router.detected_facts", str(decision.detected_user_facts[:5]))
        if decision.greeting_detected:
            span.set_attribute("cca.router.greeting", True)

        # Post-classification guards: certain requests must NEVER be "direct"
        decision = _guard_coding_not_direct(decision, user_message)
        decision = _guard_search_not_direct(decision, user_message)
        decision = _guard_action_not_direct(decision, user_message)
        decision = _guard_copout_not_direct(decision, user_message)

        span.set_status(StatusCode.OK)

        user_info = f", user='{decision.detected_user_name}'" if decision.detected_user_name else ""
        logger.info(
            f"Route: {decision.expert.value} ({elapsed_ms:.0f}ms) "
            f"| {decision.task_summary[:60]}{user_info}"
        )

        return decision


# Patterns that indicate a coding task — NEVER answer directly
_CODING_PATTERNS = re.compile(
    r"\b("
    r"write\s+(a\s+|me\s+|the\s+)?(function|script|class|module|program|code|parser|test|method|decorator|handler|middleware|endpoint)"
    r"|create\s+(a\s+|me\s+|the\s+)?(function|script|class|module|program|file|parser|test|method)"
    r"|implement\s+(a\s+|the\s+)?"
    r"|build\s+(a\s+|me\s+|the\s+)?(function|script|class|module|parser|tool|api|server|client)"
    r"|generate\s+(a\s+|me\s+|the\s+)?(function|script|class|code)"
    r"|fix\s+(the\s+|this\s+|my\s+)?(bug|error|issue|code|function|test)"
    r"|refactor\s+"
    r"|debug\s+"
    r"|add\s+(a\s+|the\s+)?(function|method|endpoint|route|test|feature)"
    r"|make\s+(a\s+|me\s+)?(function|script|program)"
    r"|help\s+me\s+write"
    r"|write\s+a\s+python"
    r"|one-liner\s+to"
    r")\b",
    re.IGNORECASE,
)


def _guard_coding_not_direct(
    decision: RouteDecision, user_message: str
) -> RouteDecision:
    """Override direct→coder if the message contains coding task patterns.

    The router LLM sometimes misclassifies coding requests as direct
    answers. This guard catches those cases deterministically.
    """
    if decision.expert != ExpertType.DIRECT:
        return decision

    if _CODING_PATTERNS.search(user_message):
        logger.warning(
            "Router guard: overriding direct→coder for coding request: %s",
            user_message[:80],
        )
        decision.expert = ExpertType.CODER
        decision.direct_answer = ""
        if decision.estimated_steps < 3:
            decision.estimated_steps = 3
        return decision

    return decision


# Patterns that indicate a web search is needed — NEVER answer directly
_SEARCH_PATTERNS = re.compile(
    r"\b("
    r"latest\s+(version|release|update|news|development|announcement)"
    r"|recent\s+(news|update|development|release|change|announcement)"
    r"|current\s+(version|release|status|state)"
    r"|newest\s+(version|release|feature)"
    r"|up.to.date"
    r"|what.s\s+new\s+in"
    # "biggest/top/major/important AI news", "latest X news"
    r"|(latest|biggest|top|major|important)\s+(\w+\s+)?news"
    # "What's the latest X" (contraction-safe)
    r"|what.s\s+the\s+(latest|newest|current|most\s+recent)"
    # Explicit search/lookup language
    r"|search\s+(for|the\s+web|online)"
    r"|look\s+up"
    r"|find\s+(out|me|information)"
    r"|what\s+are\s+the\s+(latest|newest|recent)"
    r"|what\s+is\s+the\s+(latest|newest|current)"
    # Temporal signals — "this week", "give me links"
    r"|this\s+(week|month|year).s"
    r"|give\s+me\s+(links|sources|urls|references)"
    r"|show\s+me\s+(links|sources|urls|references)"
    r")\b",
    re.IGNORECASE,
)


def _guard_search_not_direct(
    decision: RouteDecision, user_message: str
) -> RouteDecision:
    """Override direct→search if the message needs current information.

    Questions about "latest", "recent", "current" versions/news
    require web search — the model's training data may be outdated.
    """
    if decision.expert != ExpertType.DIRECT:
        return decision

    if _SEARCH_PATTERNS.search(user_message):
        logger.warning(
            "Router guard: overriding direct→search for recency query: %s",
            user_message[:80],
        )
        decision.expert = ExpertType.SEARCH
        decision.direct_answer = ""
        if decision.estimated_steps < 3:
            decision.estimated_steps = 3
        return decision

    return decision


# Patterns that require shell execution, file ops, or network access
_ACTION_PATTERNS = re.compile(
    r"\b("
    r"download\s+(this|the|a|that|my)?\s*(file|page|image|document|url|content)"
    r"|upload\s+(this|the|a|that|my)?\s*(file|image|document)"
    r"|fetch\s+(this|the|a|that)?\s*(url|page|file|content)"
    r"|curl\s+"
    r"|wget\s+"
    r"|run\s+(this|the|a|that)?\s*(command|script|code|query)"
    r"|execute\s+(this|the|a|that)?\s*(command|script|code|query)"
    r"|install\s+(this|the|a)?\s*(package|module|library|dependency)"
    r"|connect\s+to"
    r"|ssh\s+to|ssh\s+into"
    r"|ping\s+"
    r"|transfer\s+"
    r")\b"
    # Messages containing non-http URLs (ftp://, file://, ssh://, scp://)
    r"|(?:ftp|file|ssh|scp)://\S+"
    # "download ... for me" phrasing
    r"|\bdownload\b.*\bfor\s+me\b",
    re.IGNORECASE,
)


def _guard_action_not_direct(
    decision: RouteDecision, user_message: str
) -> RouteDecision:
    """Override direct→coder if message requires execution or network access.

    Requests like "download this file", "run this command", or messages
    containing ftp:// URLs need shell tools — Functionary can't help.
    """
    if decision.expert != ExpertType.DIRECT:
        return decision

    if _ACTION_PATTERNS.search(user_message):
        logger.warning(
            "Router guard: overriding direct→coder for action request: %s",
            user_message[:80],
        )
        decision.expert = ExpertType.CODER
        decision.direct_answer = ""
        if decision.estimated_steps < 3:
            decision.estimated_steps = 3
        return decision

    return decision


# Patterns in Functionary's direct answer that indicate a cop-out refusal
_COPOUT_PATTERNS = re.compile(
    r"(i'?m sorry.{0,20}(can'?t|cannot|unable|don'?t have)"
    r"|i cannot\b"
    r"|i'?m unable\b"
    r"|i don'?t have (access|the ability|real-?time)"
    r"|beyond my (ability|capabilit)"
    r"|not (able|possible) (for me |to )"
    r"|as an ai.{0,20}(can'?t|cannot|unable)"
    r"|i lack the (ability|tools|access))",
    re.IGNORECASE,
)


def _guard_copout_not_direct(
    decision: RouteDecision, _user_message: str
) -> RouteDecision:
    """Override direct if the 'answer' is actually a refusal.

    When Functionary's direct answer says "I'm sorry, I can't..." or
    "I don't have access to real-time data", the 80B agent with tools
    can actually try.
    """
    if decision.expert != ExpertType.DIRECT:
        return decision
    if not decision.direct_answer:
        return decision

    if _COPOUT_PATTERNS.search(decision.direct_answer):
        logger.warning(
            "Router guard: direct answer is a cop-out, re-routing to coder: %s",
            decision.direct_answer[:80],
        )
        decision.expert = ExpertType.CODER
        decision.direct_answer = ""
        if decision.estimated_steps < 3:
            decision.estimated_steps = 3
        return decision

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

    # Parse estimated_steps with bounds (1-100), default 10
    estimated_steps_raw = args.get("estimated_steps", 10)
    try:
        estimated_steps = max(1, min(100, int(estimated_steps_raw)))
    except (ValueError, TypeError):
        estimated_steps = 10

    return RouteDecision(
        expert=expert,
        task_summary=args.get("task_summary", args.get("answer", args.get("question", ""))),
        parameters=args,
        direct_answer=args.get("answer", ""),
        clarification_question=args.get("question", ""),
        classification_time_ms=elapsed_ms,
        detected_user_name=args.get("detected_user_name", ""),
        detected_user_facts=args.get("detected_user_facts", []),
        greeting_detected=args.get("greeting_detected", False),
        estimated_steps=estimated_steps,
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
