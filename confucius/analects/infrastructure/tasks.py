# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""System prompt / task definition for the Infrastructure expert.

Builds the prompt dynamically from:
  1. infrastructure.toml  — cluster topology (nodes, containers, checks)
  2. config.toml [services] — service endpoints (Redis URL, Qdrant URL, etc.)
  3. CCA_SSH_PASSWORD env var — SSH credentials (never stored in files)

No hardcoded IPs, passwords, or infrastructure details in this file.
"""
from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Infrastructure TOML loader
# ---------------------------------------------------------------------------

_infra_config: dict[str, Any] | None = None


def _load_infra_config() -> dict[str, Any]:
    """Load infrastructure.toml from the same directory as config.toml."""
    global _infra_config
    if _infra_config is not None:
        return _infra_config

    # Look next to config.toml (CCA_CONFIG_PATH), or fall back to
    # CCA_INFRA_CONFIG_PATH if set explicitly.
    infra_path_env = os.environ.get("CCA_INFRA_CONFIG_PATH")
    if infra_path_env:
        infra_path = Path(infra_path_env)
    else:
        config_path = Path(os.environ.get(
            "CCA_CONFIG_PATH",
            str(Path.home() / ".confucius" / "config.toml"),
        ))
        infra_path = config_path.parent / "infrastructure.toml"

    if not infra_path.exists():
        logger.warning(
            f"infrastructure.toml not found at {infra_path} — "
            "INFRA expert will use a minimal prompt without topology details"
        )
        _infra_config = {}
        return _infra_config

    with open(infra_path, "rb") as f:
        _infra_config = tomllib.load(f)
    logger.info(f"Loaded infrastructure topology from {infra_path}")
    return _infra_config


def _extract_redis_password(redis_url: str) -> str:
    """Extract password from redis URL like redis://:password@host:port/db."""
    try:
        parsed = urlparse(redis_url)
        return parsed.password or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_swarm_table(infra: dict[str, Any]) -> str:
    """Build the Docker Swarm nodes markdown table."""
    nodes = infra.get("swarm_nodes", [])
    if not nodes:
        return ""
    lines = [
        "### Docker Swarm Nodes",
        "| Node | IP | Role | Hardware | GPU |",
        "|------|-----|------|----------|-----|",
    ]
    for n in nodes:
        gpu = n.get("gpu", "")
        gpu_col = gpu if gpu else "-"
        lines.append(
            f"| {n['name']} | {n['ip']} | {n.get('role', '')} "
            f"| {n.get('hardware', '')} | {gpu_col} |"
        )
    return "\n".join(lines)


def _build_standalone_table(infra: dict[str, Any]) -> str:
    """Build the standalone AI nodes markdown table."""
    nodes = infra.get("standalone_nodes", [])
    if not nodes:
        return ""
    lines = [
        "### Standalone AI Nodes (NOT in Swarm)",
        "| Node | IP | Purpose |",
        "|------|-----|---------|",
    ]
    for n in nodes:
        lines.append(f"| {n['name']} | {n['ip']} | {n.get('purpose', '')} |")
    return "\n".join(lines)


def _build_local_services_table(infra: dict[str, Any], redis_password: str) -> str:
    """Build the local services table with quick-check commands."""
    services = infra.get("local_services", [])
    if not services:
        return ""
    lines = [
        "## Local Services (this node)",
        "",
        "These services run as Docker containers on THIS machine. "
        "Check them directly without SSH:",
        "",
        "| Service | Container | Port | Quick Check |",
        "|---------|-----------|------|-------------|",
    ]
    for s in services:
        check = s.get("check", "")
        # Substitute password placeholder in check commands
        check = check.replace("$REDIS_PASSWORD", redis_password)
        lines.append(
            f"| {s['name']} | {s.get('container', '')} "
            f"| {s.get('port', '')} | `{check}` |"
        )
    lines.append("")
    lines.append("Container status: `docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'`")
    lines.append("Container logs: `docker logs --tail 50 <container-name>`")
    return "\n".join(lines)


def _build_remote_services_table(
    infra: dict[str, Any],
    memgraph_host: str,
    memgraph_port: int,
    vllm_host: str,
) -> str:
    """Build the remote services table."""
    services = infra.get("remote_services", [])
    if not services:
        return ""
    lines = [
        "### Remote Services",
        "| Service | Location | Quick Check |",
        "|---------|----------|-------------|",
    ]
    for s in services:
        check = s.get("check", "")
        # Substitute service endpoints from config
        check = check.replace("$MEMGRAPH_HOST", memgraph_host)
        check = check.replace("$MEMGRAPH_PORT", str(memgraph_port))
        check = check.replace("$VLLM_HOST", vllm_host)
        lines.append(f"| {s['name']} | {s.get('location', '')} | `{check}` |")
    return "\n".join(lines)


def _build_swarm_services_section(infra: dict[str, Any]) -> str:
    """Build the key swarm services description."""
    swarm_svc = infra.get("swarm_services", {})
    desc = swarm_svc.get("description", "").strip()
    if not desc:
        return ""
    return f"### Key Swarm Services\n{desc}"


def _build_ssh_section(infra: dict[str, Any], ssh_password: str) -> str:
    """Build the SSH access section."""
    ssh_cfg = infra.get("ssh", {})
    user = ssh_cfg.get("user", "user")
    if ssh_password:
        return (
            "### SSH Access to Standalone Nodes\n"
            "```bash\n"
            f"sshpass -p '{ssh_password}' ssh -o StrictHostKeyChecking=no "
            f"{user}@<IP> \"command\"\n"
            "```"
        )
    return (
        "### SSH Access to Standalone Nodes\n"
        "```bash\n"
        f"ssh -o StrictHostKeyChecking=no {user}@<IP> \"command\"\n"
        "```\n"
        "_Note: Set CCA_SSH_PASSWORD env var for password-based SSH via sshpass._"
    )


def get_infra_task_definition(current_time: str) -> str:
    """Build the infrastructure task definition dynamically.

    Reads topology from infrastructure.toml and service endpoints from
    config.toml [services]. Falls back to a minimal prompt if
    infrastructure.toml is not found.
    """
    # Load infrastructure topology
    infra = _load_infra_config()

    # Load service endpoints from config.toml
    try:
        from ...core.config import get_services_config
        svc = get_services_config()
    except Exception:
        # Fallback — config not loaded (tests, standalone usage)
        from types import SimpleNamespace
        svc = SimpleNamespace(
            redis_url="redis://localhost:6379/0",
            memgraph_host="localhost",
            memgraph_port=7687,
        )

    # Extract values
    redis_password = _extract_redis_password(svc.redis_url)
    ssh_password = os.environ.get("CCA_SSH_PASSWORD", "")
    memgraph_host = os.environ.get("MEMGRAPH_HOST") or svc.memgraph_host
    memgraph_port = int(os.environ.get("MEMGRAPH_PORT", "0") or 0) or svc.memgraph_port

    # Extract vLLM host from coder provider config (best-effort)
    vllm_host = "localhost"
    try:
        from ...core.config import get_llm_params
        coder_params = get_llm_params("coder")
        base_url = (coder_params.additional_kwargs or {}).get("base_url", "")
        if base_url:
            parsed = urlparse(base_url)
            vllm_host = parsed.hostname or "localhost"
    except Exception:
        pass

    # Build sections
    sections = [
        "# Infrastructure & DevOps Assistant",
        "",
        "You are an infrastructure and DevOps expert working with a Docker Swarm cluster",
        "and standalone AI inference nodes.",
        "",
        "Environment",
        f"- Current time: {current_time}",
        "- You have direct CLI access to the local node and can SSH to remote nodes.",
        "- You have `web_search` and `fetch_url_content` tools — use them for current docs, APIs, or any real-world information.",
        "",
        "## Cluster Overview",
        "",
    ]

    # Topology tables (from infrastructure.toml)
    swarm_table = _build_swarm_table(infra)
    if swarm_table:
        sections.append(swarm_table)
        sections.append("")

    standalone_table = _build_standalone_table(infra)
    if standalone_table:
        sections.append(standalone_table)
        sections.append("")

    swarm_services = _build_swarm_services_section(infra)
    if swarm_services:
        sections.append(swarm_services)
        sections.append("")

    # Local + remote services (from infrastructure.toml + config.toml)
    local_table = _build_local_services_table(infra, redis_password)
    if local_table:
        sections.append(local_table)
        sections.append("")

    remote_table = _build_remote_services_table(
        infra, memgraph_host, memgraph_port, vllm_host,
    )
    if remote_table:
        sections.append(remote_table)
        sections.append("")

    # Important notes
    sections.extend([
        "### Important",
        "- This container runs with host networking — `127.0.0.1` reaches host services directly.",
        "- `systemctl` is NOT available (no systemd in container). Use `docker ps`/`docker logs` instead.",
    ])

    # GPU check via SSH (if we have standalone nodes and SSH password)
    gpu_nodes = [n for n in infra.get("standalone_nodes", []) if n.get("purpose", "").lower().find("vllm") >= 0 or n.get("purpose", "").lower().find("gpu") >= 0]
    if gpu_nodes and ssh_password:
        ssh_user = infra.get("ssh", {}).get("user", "user")
        n = gpu_nodes[0]
        sections.append(
            f"- For GPU status on remote nodes: "
            f"`sshpass -p '{ssh_password}' ssh {ssh_user}@{n['ip']} nvidia-smi`"
        )
    sections.append("")

    # SSH section
    ssh_section = _build_ssh_section(infra, ssh_password)
    sections.append(ssh_section)
    sections.append("")

    # Static sections (goals, tools, rules, planning, past knowledge)
    sections.extend([
        "## Your Goals",
        "1. Understand the infrastructure request",
        "2. Plan the approach (which nodes, which services, what commands)",
        "3. Execute using Docker CLI, SSH, or system commands",
        "4. Verify the result (health checks, status commands)",
        "5. Report what was done",
        "",
        "## Available Tools",
        "",
        "| Tool | When to Use |",
        "|------|-------------|",
        "| `str_replace_editor` | Edit config files, docker-compose.yml, scripts — ALWAYS use this for file changes |",
        "| `bash` | Run commands (docker, ssh, curl, etc.) |",
        "| `web_search` / `fetch_url_content` | Research docs, APIs, current information |",
        "| `write_memory` / `read_memory` | Track multi-step plans, save progress |",
        "| `search_codebase` | Find code, configs, or patterns in the indexed repository |",
        "| `search_notes` | Check past session knowledge before re-investigating |",
        "",
        "## Rules",
        "- ALWAYS verify current state before making changes (docker ps, docker service ls, etc.)",
        "- Use `docker stack deploy` for Swarm services, `docker compose` for standalone nodes",
        "- NEVER put root_ca.crt in server cert chains (trust store only)",
        "- Overlay networks must have `attachable: true` for Swarm",
        "- GPU workloads go to nodes with nvidia runtime",
        "- Always `git push` before deploy (GitOps: Portainer polls GitHub)",
        "",
        "## Planning",
        "- For complex tasks involving multiple nodes or services, plan your approach before making changes. "
        "Use `write_memory` to create a todo/plan and track progress as you work.",
        "- Break down the task into steps: which nodes, which services, what commands, what verification.",
        "- Update your plan as you go — check off completed steps and document any issues.",
        "- For simple single-service checks or restarts, proceed directly.",
        "",
        "## Past Knowledge",
        "- If `<past_insights>` tags appear in your context, they contain verified knowledge from previous sessions.",
        "- Check past insights FIRST before running diagnostic commands. If you already know the answer from a previous session, use it directly.",
        "- You have a `search_notes` tool to search deeper into past session notes.",
        "- Reference past knowledge naturally: \"From our previous session, I know that...\" — don't re-discover what you already know.",
        "",
        "## Deliverables",
        "- Summary of what was changed and why",
        "- Verification output (health checks, status)",
        "- Any warnings or follow-up needed",
    ])

    return "\n".join(sections)
