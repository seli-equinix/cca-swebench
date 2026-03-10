# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ...core import types as cf
from ...core.analect import Analect, AnalectRunContext
from ...core.config import CCAConfigError, get_llm_params
from ...core.entry.base import EntryInput, EntryOutput
from ...core.entry.decorators import public
from ...core.entry.mixin import EntryAnalectMixin
from ...core.memory import CfMessage
from ...orchestrator.anthropic import AnthropicLLMOrchestrator
from ...orchestrator.extensions import Extension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.types import OrchestratorInput
from .tasks import NOTE_TAKER_PROMPT

logger = logging.getLogger(__name__)

from ...core.config import get_services_config


def _create_note_writer_extension(
    session_id: str = "",
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
) -> Optional[Extension]:
    """Create a NoteWriterExtension with Qdrant/Redis/embedding connections.

    Returns None if infrastructure is unavailable (graceful degradation).
    """
    try:
        from qdrant_client import QdrantClient
        import redis.asyncio as redis_async

        from ...orchestrator.extensions.note_writer import NoteWriterExtension
        from ...server.user.session_manager import _ProfileEmbeddingFunc

        svc = get_services_config()
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL") or svc.qdrant_url)
        redis_client = redis_async.from_url(
            os.getenv("REDIS_URL") or svc.redis_url, decode_responses=True
        )
        embedding_func = _ProfileEmbeddingFunc(os.getenv("EMBEDDING_URL") or svc.embedding_url)

        return NoteWriterExtension(
            qdrant=qdrant,
            redis_client=redis_client,
            embedding_func=embedding_func,
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
        )
    except Exception as e:
        logger.warning("Failed to create NoteWriterExtension: %s", e)
        return None


@public
class CCANoteTakerEntry(Analect[EntryInput, EntryOutput], EntryAnalectMixin):
    """Note Taking Analect

    Analyzes conversation trajectories and extracts key insights,
    storing them in Qdrant for semantic search in future sessions.

    Can be invoked via:
    - Deep analysis endpoint (POST /v1/notes/analyze)
    - Manual CLI (scripts/run_note_taker.py)
    """

    @classmethod
    def display_name(cls) -> str:
        return "NoteTaker"

    @classmethod
    def description(cls) -> str:
        return "LLM-powered trajectory observer that extracts insights to Qdrant"

    @classmethod
    def input_examples(cls) -> list[EntryInput]:
        return [EntryInput(question="Analyze session trajectories and extract insights.")]

    async def impl(self, inp: EntryInput, context: AnalectRunContext) -> EntryOutput:
        # Build extensions — NoteWriterExtension replaces file/CLI tools
        extensions: list[Extension] = [
            PlainTextExtension(),
        ]

        # Add Qdrant-backed note writer (graceful: skipped if infra unavailable)
        note_writer = _create_note_writer_extension(
            session_id=context.session,
        )
        if note_writer:
            extensions.insert(0, note_writer)
        else:
            logger.warning(
                "NoteWriterExtension unavailable — note-taker will run "
                "without Qdrant storage"
            )

        orchestrator = AnthropicLLMOrchestrator(
            llm_params=[
                get_llm_params("note_taker"),
            ],
            extensions=extensions,
            raw_output_parser=None,
        )

        # Use OrchestratorInput to run
        await context.invoke_analect(
            orchestrator,
            OrchestratorInput(
                messages=[
                    CfMessage(
                        type=cf.MessageType.HUMAN,
                        content=inp.question,
                        attachments=inp.attachments,
                    )
                ],
                task=NOTE_TAKER_PROMPT,
            ),
        )

        return EntryOutput()
