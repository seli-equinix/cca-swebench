# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""System prompt / task definition for the Infrastructure expert.

Builds the prompt dynamically from:
  1. infrastructure.toml  — cluster topology (nodes, containers, checks)
  2. config.toml [services] — service endpoints (Redis URL, Qdrant URL, etc.)
  3. Per-node SSH credentials via env var references in infrastructure.toml

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


def _get_node_ssh_creds(node: dict[str, Any], infra: dict[str, Any]) -> tuple[str, str]:
    """Return (ssh_user, ssh_password) for a node, with global [ssh] fallback.

    Credential resolution order:
      1. Node-specific ``ssh_user`` / ``ssh_password_env`` fields
      2. Global ``[ssh].user`` / ``[ssh].password_env``
      3. Hardcoded defaults: user="user", password_env="CCA_SSH_PASSWORD"
    """
    global_ssh = infra.get("ssh", {})
    user = node.get("ssh_user") or global_ssh.get("user", "user")
    pass_env = node.get("ssh_password_env") or global_ssh.get("password_env", "CCA_SSH_PASSWORD")
    password = os.environ.get(pass_env, "")
    return user, password


# ---------------------------------------------------------------------------
# Docker access resolution
# ---------------------------------------------------------------------------


def _resolve_docker_access(
    infra: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], str, str] | None]:
    """Resolve docker_access node names to full node dicts with SSH creds.

    Returns dict with keys:
      - local: (node_dict, ssh_user, ssh_pass) or None
      - swarm: (node_dict, ssh_user, ssh_pass) or None
    """
    access = infra.get("docker_access", {})
    all_nodes = {
        n["name"]: n
        for n in infra.get("swarm_nodes", []) + infra.get("standalone_nodes", [])
    }

    result: dict[str, tuple[dict[str, Any], str, str] | None] = {
        "local": None,
        "swarm": None,
    }

    local_name = access.get("local_node")
    if local_name and local_name in all_nodes:
        node = all_nodes[local_name]
        user, pw = _get_node_ssh_creds(node, infra)
        result["local"] = (node, user, pw)

    swarm_name = access.get("swarm_manager")
    if swarm_name and swarm_name in all_nodes:
        node = all_nodes[swarm_name]
        user, pw = _get_node_ssh_creds(node, infra)
        result["swarm"] = (node, user, pw)

    return result


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
    """Build the local services table with quick-check commands.

    If [docker_access].local_node is configured, all docker commands are
    prefixed with SSH to the appropriate node (CCA runs in a container
    without docker socket access). Falls back to direct commands if not set.
    """
    services = infra.get("local_services", [])
    if not services:
        return ""

    # Resolve docker access — determines if we need SSH for docker commands
    docker = _resolve_docker_access(infra)
    local_access = docker["local"]

    if local_access:
        node, ssh_user, ssh_pass = local_access
        node_name = node.get("name", "?")
        node_ip = node["ip"]
        if ssh_pass:
            ssh_prefix = f"sshpass -p '{ssh_pass}' ssh -o StrictHostKeyChecking=no {ssh_user}@{node_ip}"
        else:
            ssh_prefix = f"ssh -o StrictHostKeyChecking=no {ssh_user}@{node_ip}"
        header_text = (
            f"These services run as Docker containers on **{node_name}** ({node_ip}). "
            f"Use SSH to check them:"
        )
    else:
        ssh_prefix = ""
        header_text = (
            "These services run as Docker containers on THIS machine. "
            "Check them directly without SSH:"
        )

    lines = [
        "## Local Services",
        "",
        header_text,
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

    if ssh_prefix:
        lines.append(
            f"Container status: `{ssh_prefix} "
            "\"docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'\"`"
        )
        lines.append(
            f"Container logs: `{ssh_prefix} \"docker logs --tail 50 <container-name>\"`"
        )
    else:
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


def _build_ssh_section(infra: dict[str, Any]) -> str:
    """Build the SSH access section with per-node credentials."""
    all_nodes = list(infra.get("swarm_nodes", [])) + list(infra.get("standalone_nodes", []))
    if not all_nodes:
        return ""

    # Collect unique (user, password) pairs to decide format
    creds_by_node: list[tuple[str, str, str]] = []  # (name, user, password)
    unique_creds: set[tuple[str, str]] = set()
    for node in all_nodes:
        user, password = _get_node_ssh_creds(node, infra)
        creds_by_node.append((node.get("name", "?"), user, password))
        unique_creds.add((user, password))

    # If all nodes share the same creds, show a single generic command
    if len(unique_creds) == 1:
        user, password = next(iter(unique_creds))
        if password:
            return (
                "### SSH Access to Remote Nodes\n"
                "```bash\n"
                f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no "
                f"{user}@<IP> \"command\"\n"
                "```"
            )
        return (
            "### SSH Access to Remote Nodes\n"
            "```bash\n"
            f"ssh -o StrictHostKeyChecking=no {user}@<IP> \"command\"\n"
            "```\n"
            "_Note: Set CCA_SSH_PASSWORD env var (or per-node ssh_password_env) "
            "for password-based SSH via sshpass._"
        )

    # Different creds per node — show per-node SSH commands
    lines = ["### SSH Access to Remote Nodes", ""]
    for name, user, password in creds_by_node:
        if password:
            lines.append(
                f"- **{name}**: "
                f"`sshpass -p '{password}' ssh -o StrictHostKeyChecking=no "
                f"{user}@<IP> \"command\"`"
            )
        else:
            lines.append(
                f"- **{name}**: "
                f"`ssh -o StrictHostKeyChecking=no {user}@<IP> \"command\"` "
                "_(no password configured)_"
            )
    return "\n".join(lines)


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
    docker = _resolve_docker_access(infra)
    sections.extend([
        "### Important",
        "- This container runs with host networking — `127.0.0.1` reaches host services directly.",
        "- `systemctl` is NOT available (no systemd in container). Use `docker ps`/`docker logs` instead.",
    ])

    # Docker access via SSH — tell the agent exactly how to run docker commands
    if docker["local"]:
        node, ssh_user, ssh_pass = docker["local"]
        if ssh_pass:
            ssh_cmd = f"sshpass -p '{ssh_pass}' ssh -o StrictHostKeyChecking=no {ssh_user}@{node['ip']}"
        else:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no {ssh_user}@{node['ip']}"
        sections.append(
            f"- **Docker is NOT available locally** — this container has no docker socket. "
            f"For container commands (docker ps, docker logs, docker restart), "
            f"SSH to **{node.get('name', node['ip'])}** ({node['ip']}): "
            f"`{ssh_cmd} \"docker ps\"`"
        )

    if docker["swarm"]:
        node, ssh_user, ssh_pass = docker["swarm"]
        if ssh_pass:
            ssh_cmd = f"sshpass -p '{ssh_pass}' ssh -o StrictHostKeyChecking=no {ssh_user}@{node['ip']}"
        else:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no {ssh_user}@{node['ip']}"
        sections.append(
            f"- Docker Swarm commands (service ls, node ls, stack deploy): "
            f"SSH to swarm manager **{node.get('name', node['ip'])}** ({node['ip']}): "
            f"`{ssh_cmd} \"docker node ls\"`"
        )

    # GPU check via SSH (per-node credentials)
    gpu_nodes = [
        n for n in infra.get("standalone_nodes", [])
        if "vllm" in n.get("purpose", "").lower()
        or "gpu" in n.get("purpose", "").lower()
    ]
    for n in gpu_nodes:
        ssh_user, ssh_pass = _get_node_ssh_creds(n, infra)
        if ssh_pass:
            sections.append(
                f"- GPU status on {n.get('name', n['ip'])}: "
                f"`sshpass -p '{ssh_pass}' ssh {ssh_user}@{n['ip']} nvidia-smi`"
            )
    sections.append("")

    # SSH section
    ssh_section = _build_ssh_section(infra)
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
        "| `str_replace_editor` | View, create, and edit config files, docker-compose.yml, scripts — use for ALL file changes (not bash cat/sed/echo) |",
        "| `bash` | Run commands (docker, ssh, curl, etc.) — for execution, not file editing |",
        "| `web_search` / `fetch_url_content` | Research docs, APIs, current information |",
        "| `write_memory` / `read_memory` | Track multi-step plans, save progress |",
        "| `search_codebase` | Find code, configs, or patterns — semantic search with graph context, richer than grep |",
        "| `query_call_graph` | Trace callers/callees from the code knowledge graph — start here for 'who calls X' questions (more complete than grep) |",
        "| `analyze_dependencies` | Map file/function dependencies from the knowledge graph — start here for 'dependencies of X' questions |",
        "| `trace_execution` / `assemble_traced_code` | Trace execution paths and assemble code across files — start here for 'what code runs when X starts' |",
        "| `find_orphan_functions` | Find dead/unused code from the graph — start here for 'find unused functions' |",
        "| `upload_document` / `search_documents` | Store and search user-provided documents and notes |",
        "| `create_rule` / `list_rules` | Define and manage persistent behavior rules |",
        "| `search_notes` | Check past session knowledge before re-investigating |",
        "",
        "**IMPORTANT**: All tools listed above are ALWAYS available. NEVER claim a tool is 'not available' or 'not supported'. If a tool call returns an error, retry with different parameters.",
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
