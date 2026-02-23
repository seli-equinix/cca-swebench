# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""Allowed commands for the Infrastructure expert.

Extends the base coding commands with Docker, Swarm, networking,
and system administration tools needed for full-stack operations.
"""
from __future__ import annotations

from typing import Dict

from ..code.commands import get_allowed_commands as get_code_commands


def get_infra_commands() -> Dict[str, str]:
    """Return allowed commands for infrastructure tasks.

    Includes all coding commands PLUS infrastructure-specific tools.
    """
    commands = get_code_commands()

    # Docker & container management
    commands.update({
        "docker": (
            "Docker CLI (ps, images, logs, inspect, exec, network, volume, "
            "system, compose, stack, service, node, secret, config)"
        ),
        "docker-compose": "Docker Compose (legacy, prefer 'docker compose')",
        # System administration
        "systemctl": "Manage systemd services (status, start, stop, restart, enable, disable)",
        "journalctl": "Query systemd journal logs",
        "ss": "Show socket statistics (replacement for netstat)",
        "ip": "Show/manipulate routing, network devices, interfaces",
        "ping": "Send ICMP echo requests to network hosts",
        "traceroute": "Trace the route packets take to a host",
        "dig": "DNS lookup utility",
        "nslookup": "Query DNS name servers",
        "host": "DNS lookup utility (simple)",
        # Remote operations
        "ssh": "Secure shell remote login (use sshpass for automated access)",
        "sshpass": "Non-interactive SSH password provider",
        "scp": "Secure copy between hosts",
        "rsync": "Remote (and local) file-copying tool",
        # Process management
        "ps": "Report process status",
        "top": "Display Linux processes (use -b for batch mode)",
        "htop": "Interactive process viewer (use -d for batch)",
        "kill": "Send signals to processes",
        "pkill": "Signal processes by name",
        # Monitoring & diagnostics
        "free": "Display amount of free and used memory",
        "uptime": "Show how long the system has been running",
        "lsblk": "List block devices",
        "mount": "Mount a filesystem or show mounts",
        "umount": "Unmount filesystems",
        "lsof": "List open files",
        "nc": "Netcat — arbitrary TCP/UDP connections",
        "nmap": "Network scanner (use responsibly, local network only)",
        # TLS / Certificates
        "openssl": "SSL/TLS utility (certificate operations, testing)",
        "step": "Step CLI for certificate authority operations",
    })

    return commands
