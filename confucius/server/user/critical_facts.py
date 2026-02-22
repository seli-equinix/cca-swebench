# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Critical facts extraction and storage.

Ported from MCP server memory_manager.py CriticalFactsExtractor class.
Auto-extracts passwords, IPs, URLs, API keys, hostnames, usernames,
and ports from conversations. Stores in Redis for LLM context injection.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias for Redis — we accept either sync or async client
RedisClient = Any


class CriticalFactsExtractor:
    """Extracts and stores critical facts from conversations.

    Facts are stored permanently in Redis and included in the context
    prefix so the LLM never forgets them.
    """

    PATTERNS: dict[str, list[str]] = {
        "password": [
            r'password[:\s]+["\']?([^\s"\']+)["\']?',
            r'passwd[:\s]+["\']?([^\s"\']+)["\']?',
            r'pwd[:\s]+["\']?([^\s"\']+)["\']?',
            r'secret[:\s]+["\']?([^\s"\']+)["\']?',
            r'redis-cli.*-a\s+["\']?([^\s"\']+)["\']?',
        ],
        "ip_address": [
            r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?)",
        ],
        "url": [
            r'(https?://[^\s<>"\']+)',
        ],
        "api_key": [
            r'api[_-]?key[:\s]+["\']?([^\s"\']+)["\']?',
            r'token[:\s]+["\']?([A-Za-z0-9_\-\.]+)["\']?',
            r"(sk-[a-zA-Z0-9]{20,})",
            r"(github_pat_[a-zA-Z0-9_]+)",
        ],
        "hostname": [
            r"(?:ssh|connect|host)[:\s@]+([a-zA-Z0-9][\w\-\.]+\.[a-zA-Z]{2,})",
            r"(node\d+\.locallan\.com)",
            r"([a-zA-Z0-9\-]+\.locallan\.com)",
        ],
        "username": [
            r'(?:user|username|login)[:\s]+["\']?([^\s"\'@]+)["\']?',
            r"ssh\s+(\w+)@",
        ],
        "port": [
            r"port[:\s]+(\d+)",
            r":(\d{4,5})(?:\s|$|/)",
        ],
    }

    LABELS: dict[str, str] = {
        "password": "Password",
        "ip_address": "Server IP",
        "url": "URL",
        "api_key": "API Key",
        "hostname": "Hostname",
        "username": "Username",
        "port": "Port",
    }

    def __init__(self, redis_client: Optional[RedisClient] = None) -> None:
        self._redis = redis_client
        self._local_store: dict[str, dict[str, list[str]]] = {}

    def extract_facts(self, text: str) -> dict[str, list[str]]:
        """Extract all critical facts from a text string."""
        facts: dict[str, list[str]] = {}

        for fact_type, patterns in self.PATTERNS.items():
            matches: set[str] = set()
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(1) if match.lastindex else match.group(0)
                    if self._is_valid_fact(fact_type, value):
                        matches.add(value)

            if matches:
                facts[fact_type] = list(matches)

        return facts

    def _is_valid_fact(self, fact_type: str, value: str) -> bool:
        """Validate extracted fact to filter false positives."""
        if not value or len(value) < 3:
            return False

        common_words = {"the", "and", "for", "you", "are", "this", "that", "with"}
        if value.lower() in common_words:
            return False

        if fact_type == "ip_address":
            parts = value.split(":")[0].split(".")
            if len(parts) != 4:
                return False
            try:
                for p in parts:
                    if not 0 <= int(p) <= 255:
                        return False
            except ValueError:
                return False

        if fact_type == "port":
            try:
                port = int(value)
                if not 1 <= port <= 65535:
                    return False
            except ValueError:
                return False

        if fact_type == "password":
            placeholders = {"xxx", "password", "changeme", "***", "your_password"}
            if value.lower() in placeholders:
                return False

        return True

    async def store_facts(
        self,
        session_id: str,
        facts: dict[str, list[str]],
        source: str = "conversation",
    ) -> None:
        """Store extracted facts in Redis (or local fallback)."""
        if not facts:
            return

        timestamp = datetime.now().isoformat()

        if self._redis is not None:
            try:
                for fact_type, values in facts.items():
                    key = f"facts:{session_id}:{fact_type}"
                    for value in values:
                        fact_data = json.dumps(
                            {"source": source, "timestamp": timestamp, "value": value}
                        )
                        await self._redis.hset(key, value, fact_data)

                total = sum(len(v) for v in facts.values())
                logger.debug(f"Stored {total} critical facts for session {session_id[:12]}")
                return
            except Exception as e:
                logger.warning(f"Redis store_facts failed, using local: {e}")

        # Local fallback
        if session_id not in self._local_store:
            self._local_store[session_id] = {}
        for fact_type, values in facts.items():
            existing = set(self._local_store[session_id].get(fact_type, []))
            existing.update(values)
            self._local_store[session_id][fact_type] = list(existing)

    async def get_facts(self, session_id: str) -> dict[str, list[str]]:
        """Retrieve all stored facts for a session."""
        facts: dict[str, list[str]] = {}

        if self._redis is not None:
            try:
                for fact_type in self.PATTERNS:
                    key = f"facts:{session_id}:{fact_type}"
                    stored = await self._redis.hgetall(key)
                    if stored:
                        # Keys are the actual fact values (bytes in Redis)
                        decoded = [
                            k.decode() if isinstance(k, bytes) else k
                            for k in stored.keys()
                        ]
                        facts[fact_type] = decoded
                return facts
            except Exception as e:
                logger.warning(f"Redis get_facts failed, using local: {e}")

        # Local fallback
        return dict(self._local_store.get(session_id, {}))

    async def format_facts_for_context(self, session_id: str) -> str:
        """Format all facts for inclusion in LLM context."""
        facts = await self.get_facts(session_id)
        if not facts:
            return ""

        lines: list[str] = []
        for fact_type, values in facts.items():
            label = self.LABELS.get(fact_type, fact_type)
            for value in values:
                lines.append(f"  - {label}: {value}")

        return "\n".join(lines) if lines else ""

    async def process_conversation(
        self, session_id: str, user_message: str, assistant_response: str = ""
    ) -> None:
        """Process a conversation exchange and extract/store critical facts."""
        user_facts = self.extract_facts(user_message)
        assistant_facts = self.extract_facts(assistant_response) if assistant_response else {}

        all_facts: dict[str, list[str]] = {}
        for fact_type in set(list(user_facts.keys()) + list(assistant_facts.keys())):
            values = set(user_facts.get(fact_type, []) + assistant_facts.get(fact_type, []))
            if values:
                all_facts[fact_type] = list(values)

        if all_facts:
            await self.store_facts(session_id, all_facts)
