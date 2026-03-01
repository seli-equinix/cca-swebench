# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Rules storage layer — Qdrant-backed persistent rule management.

Ported from MCP server qdrant_adapter.py (Rule + RulesCollection classes).
Shares the existing `mcp_rules` Qdrant collection with MCP.

Rule types:
  always         — auto-injected into system prompt at conversation start
  auto_attached  — included when working with files matching globs/regex
  agent_requested — AI discovers via semantic search on description
  manual         — only applied when explicitly mentioned by name
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "mcp_rules"
EMBEDDING_DIMS = 4096


@dataclass
class Rule:
    """Rule data model for storing behavior rules."""

    id: str
    name: str
    rule_type: str  # always, auto_attached, agent_requested, manual
    rule: str
    description: Optional[str] = None
    globs: List[str] = field(default_factory=list)
    regex: Optional[str] = None
    user_id: str = "global"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type,
            "rule": self.rule,
            "description": self.description,
            "globs": self.globs,
            "regex": self.regex,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Rule:
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        return cls(
            id=data["id"],
            name=data["name"],
            rule_type=data["rule_type"],
            rule=data["rule"],
            description=data.get("description"),
            globs=data.get("globs", []),
            regex=data.get("regex"),
            user_id=data.get("user_id", "global"),
            created_at=created_at,
            updated_at=updated_at,
        )

    def matches_file(self, file_path: str) -> bool:
        """Check if this auto_attached rule matches a file path."""
        if self.rule_type != "auto_attached":
            return False
        import os

        for pattern in self.globs:
            if fnmatch.fnmatch(file_path, pattern):
                return True
            if fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        if self.regex:
            try:
                if re.search(self.regex, file_path):
                    return True
            except re.error:
                logger.warning("Invalid regex in rule %s: %s", self.name, self.regex)
        return False


VALID_RULE_TYPES = frozenset({"always", "auto_attached", "agent_requested", "manual"})


class RulesStore:
    """Async Qdrant-backed rules storage.

    Shares the `mcp_rules` collection with the MCP server.
    All methods are async (CCA uses AsyncQdrantClient).
    """

    def __init__(self, backend_clients: Any) -> None:
        self._clients = backend_clients

    def _point_id(self, rule_id: str) -> str:
        """Deterministic point ID from rule UUID."""
        return hashlib.md5(rule_id.encode()).hexdigest()

    async def _ensure_collection(self) -> None:
        qdrant = self._clients.qdrant
        try:
            await qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            from qdrant_client.models import VectorParams, Distance

            await qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMS, distance=Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)

    async def store_rule(self, rule: Rule) -> bool:
        """Embed and store a rule."""
        qdrant = self._clients.qdrant
        if not qdrant or not self._clients.available:
            return False
        try:
            await self._ensure_collection()
            embed_text = f"{rule.name}. {rule.description or ''} {rule.rule[:500]}"
            vectors = await self._clients.embed([embed_text])
            if not vectors:
                return False

            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=self._point_id(rule.id),
                vector=vectors[0],
                payload=rule.to_dict(),
            )
            await qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
            logger.info("Stored rule: %s (type=%s)", rule.name, rule.rule_type)
            return True
        except Exception as e:
            logger.error("Failed to store rule: %s", e)
            return False

    async def get_rule_by_name(
        self, name: str, user_id: Optional[str] = None
    ) -> Optional[Rule]:
        """Exact name lookup (prefers user's rule over global)."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return None
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results, _ = await qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="name", match=MatchValue(value=name))]
                ),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                if point.payload:
                    rule_user = point.payload.get("user_id", "global")
                    if user_id is None or rule_user in (user_id, "global"):
                        return Rule.from_dict(point.payload)
            return None
        except Exception as e:
            logger.error("get_rule_by_name failed: %s", e)
            return None

    async def get_rules_for_file(
        self, file_path: str, user_id: Optional[str] = None
    ) -> List[Rule]:
        """Get auto_attached rules matching a file path."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results, _ = await qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="rule_type",
                            match=MatchValue(value="auto_attached"),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            matching = []
            for point in results:
                if point.payload:
                    rule = Rule.from_dict(point.payload)
                    if user_id and rule.user_id not in (user_id, "global"):
                        continue
                    if rule.matches_file(file_path):
                        matching.append(rule)
            return matching
        except Exception as e:
            logger.error("get_rules_for_file failed: %s", e)
            return []

    async def search_rules_semantic(
        self, query: str, user_id: Optional[str] = None, n_results: int = 5
    ) -> List[Rule]:
        """Semantic search for agent_requested rule discovery."""
        qdrant = self._clients.qdrant
        if not qdrant or not self._clients.available:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vectors = await self._clients.embed([query])
            if not vectors:
                return []

            query_filter = None
            if user_id:
                query_filter = Filter(
                    should=[
                        FieldCondition(
                            key="user_id", match=MatchValue(value=user_id)
                        ),
                        FieldCondition(
                            key="user_id", match=MatchValue(value="global")
                        ),
                    ]
                )

            results = await qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vectors[0],
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
            )
            return [
                Rule.from_dict(pt.payload)
                for pt in results.points
                if pt.payload
            ]
        except Exception as e:
            logger.error("search_rules_semantic failed: %s", e)
            return []

    async def get_all_rules(
        self,
        user_id: Optional[str] = None,
        rule_type: Optional[str] = None,
        include_global: bool = True,
    ) -> List[Rule]:
        """Get all rules with optional filters."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            must = []
            if rule_type:
                must.append(
                    FieldCondition(key="rule_type", match=MatchValue(value=rule_type))
                )

            should = []
            if user_id:
                should.append(
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                )
            if include_global:
                should.append(
                    FieldCondition(key="user_id", match=MatchValue(value="global"))
                )

            # Build filter
            scroll_filter = None
            if must and should:
                if len(should) == 1:
                    must.append(should[0])
                    scroll_filter = Filter(must=must)
                else:
                    scroll_filter = Filter(must=must, should=should)
            elif must:
                scroll_filter = Filter(must=must)
            elif should:
                if len(should) == 1:
                    scroll_filter = Filter(must=should)
                else:
                    scroll_filter = Filter(should=should)

            all_rules: List[Rule] = []
            offset = None
            while True:
                results, next_offset = await qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    if point.payload:
                        all_rules.append(Rule.from_dict(point.payload))
                if next_offset is None:
                    break
                offset = next_offset
            return all_rules
        except Exception as e:
            logger.error("get_all_rules failed: %s", e)
            return []

    async def get_always_rules(self, user_id: Optional[str] = None) -> List[Rule]:
        """Get all 'always' rules for system prompt injection."""
        return await self.get_all_rules(
            user_id=user_id, rule_type="always", include_global=True
        )

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        qdrant = self._clients.qdrant
        if not qdrant:
            return False
        try:
            from qdrant_client.models import PointIdsList

            await qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=[self._point_id(rule_id)]),
            )
            logger.info("Deleted rule: %s", rule_id)
            return True
        except Exception as e:
            logger.error("Failed to delete rule: %s", e)
            return False
