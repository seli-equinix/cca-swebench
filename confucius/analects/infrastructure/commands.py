# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""Allowed commands for the Infrastructure expert.

Extends the base coding commands with Docker, Swarm, networking,
system administration, and hardware diagnostics tools needed for
full-stack infrastructure operations.
"""
from __future__ import annotations

from typing import Dict

from ..code.commands import get_allowed_commands as get_code_commands


def get_infra_commands() -> Dict[str, str]:
    """Return allowed commands for infrastructure tasks.

    Includes all coding commands PLUS infrastructure-specific tools.
    Excludes destructive disk/partition tools (dd, mkfs, fdisk, parted)
    and system state changers (reboot, shutdown, poweroff, halt, init).
    """
    commands = get_code_commands()

    commands.update({
        # ── Docker & container management ──
        "docker": (
            "Docker CLI (ps, images, logs, inspect, exec, network, "
            "volume, system, compose, stack, service, node, secret, config)"
        ),
        "docker-compose": "Docker Compose (legacy, prefer 'docker compose')",
        "podman": "OCI container runtime (Docker-compatible)",
        # ── Container orchestration ──
        "kubectl": "Kubernetes command-line tool",
        "helm": "Kubernetes package manager",
        "k9s": "Kubernetes TUI dashboard (via SSH to cluster nodes)",
        "crictl": "CRI-compatible container runtime CLI (via SSH to container hosts)",
        "ctr": "Containerd CLI (via SSH to container hosts)",
        # ── System administration ──
        "sudo": "Execute a command as another user",
        "su": "Switch user",
        # ── systemd tools (via SSH — not available inside containers) ──
        "systemctl": "Manage systemd services (via SSH to host nodes — no systemd in container)",
        "journalctl": "Query systemd journal logs (via SSH to host nodes)",
        "hostnamectl": "Query/set hostname (via SSH to host nodes)",
        "timedatectl": "Query/set time and date (via SSH to host nodes)",
        "loginctl": "Control systemd login manager (via SSH to host nodes)",
        # ── Network tools ──
        "ss": "Show socket statistics (replacement for netstat)",
        "ip": "Show/manipulate routing, network devices, interfaces",
        "ping": "Send ICMP echo requests to network hosts",
        "traceroute": "Trace the route packets take to a host",
        "mtr": "Network diagnostic tool (traceroute + ping)",
        "dig": "DNS lookup utility",
        "nslookup": "Query DNS name servers",
        "host": "DNS lookup utility (simple)",
        "iptables": "IPv4 packet filter administration",
        "ip6tables": "IPv6 packet filter administration",
        "nft": "Nftables packet filter administration",
        "firewall-cmd": "FirewallD CLI (via SSH to hosts running firewalld)",
        "brctl": "Ethernet bridge administration",
        "ethtool": "Display and change ethernet device settings",
        "ifconfig": "Configure network interface (legacy)",
        "route": "Show/manipulate IP routing table (legacy)",
        "nmcli": "NetworkManager command-line interface",
        "resolvectl": "Resolve domain names, DNS records (systemd)",
        "tcpdump": "Dump traffic on a network (packet capture)",
        "arp": "Manipulate the system ARP cache",
        "iperf3": "Network bandwidth measurement tool",
        "nc": "Netcat — arbitrary TCP/UDP connections",
        "nmap": "Network scanner (use responsibly, local network only)",
        "whois": "Client for the WHOIS directory service",
        # ── Remote operations ──
        "ssh": "Secure shell remote login",
        "sshpass": "Non-interactive SSH password provider",
        "scp": "Secure copy between hosts",
        "rsync": "Remote (and local) file-copying tool",
        # ── Process management ──
        "ps": "Report process status",
        "top": "Display Linux processes (use -b for batch mode)",
        "htop": "Interactive process viewer (use -d for batch)",
        "kill": "Send signals to processes",
        "pkill": "Signal processes by name",
        "pgrep": "Look up processes by name",
        # ── Hardware & system diagnostics ──
        "free": "Display amount of free and used memory",
        "uptime": "Show how long the system has been running",
        "lscpu": "Display CPU architecture information",
        "lspci": "List PCI devices",
        "lsusb": "List USB devices",
        "lsblk": "List block devices",
        "nvidia-smi": "NVIDIA GPU info (via SSH: sshpass -p '...' ssh seli@<gpu-node> nvidia-smi)",
        "dmidecode": "Hardware BIOS/DMI info (via SSH — needs privileged access on host)",
        "sensors": "Hardware temperature/voltage sensors (via SSH — needs /sys access on host)",
        "dmesg": "Print kernel ring buffer messages",
        "who": "Show who is logged on",
        "w": "Show who is logged on and what they are doing",
        "last": "Show listing of last logged in users",
        # ── Disk & storage (read/inspect only) ──
        "blkid": "Locate and print block device attributes",
        "findmnt": "Find a filesystem in the mount table",
        "mount": "Mount a filesystem or show mounts",
        "umount": "Unmount filesystems",
        "lsof": "List open files",
        "fstrim": "Discard unused blocks on a mounted filesystem",
        "lvs": "Display LVM logical volumes",
        "vgs": "Display LVM volume groups",
        "pvs": "Display LVM physical volumes",
        # ── User & group management ──
        "useradd": "Create a new user",
        "userdel": "Delete a user account",
        "usermod": "Modify a user account",
        "groupadd": "Create a new group",
        "groupdel": "Delete a group",
        "groups": "Print group memberships for a user",
        "passwd": "Change user password",
        "chpasswd": "Batch update passwords",
        "getent": "Get entries from administrative databases",
        # ── Package management ──
        "apt": "APT package manager (Debian/Ubuntu)",
        "apt-get": "APT package handling utility (low-level)",
        "apt-cache": "APT package cache query tool",
        "dpkg": "Debian package manager",
        "snap": "Snap package manager (via SSH to host nodes — snapd not in container)",
        # ── Cron & scheduling ──
        "crontab": "Maintain crontab files for individual users",
        "at": "Schedule commands for later execution",
        # ── Log tools ──
        "logger": "Make entries in the system log",
        # ── TLS / certificates ──
        "openssl": "SSL/TLS utility (certificate operations, testing)",
        "certbot": "Let's Encrypt certificate management",
        "step": "Step CLI for certificate authority operations",
        # ── Database CLIs ──
        "redis-cli": "Redis command-line interface (redis-cli -h HOST -a PASS ping)",
        # ── Legacy network tools ──
        "netstat": "Print network connections, routing tables, interface statistics",
        # ── Cluster-specific ──
        "gluster": "GlusterFS CLI (via SSH to Swarm nodes — not installed locally)",
    })

    return commands
