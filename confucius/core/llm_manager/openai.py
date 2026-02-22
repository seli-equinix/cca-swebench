# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from openai import AsyncOpenAI
from pydantic import PrivateAttr

from ..chat_models.openai.openai import OpenAIChat

from .base import LLMManager, LLMParams
from .constants import DEFAULT_INITIAL_MAX_TOKEN

logger: logging.Logger = logging.getLogger(__name__)


class OpenAILLMManager(LLMManager):
    """OpenAI manager using native AsyncOpenAI client.

    Environment variables used:
    - OPENAI_API_KEY: API key (defaults to "dummy" for local vLLM)
    - OPENAI_BASE_URL: Default base URL (auto-used by AsyncOpenAI SDK)

    Per-call overrides via LLMParams.additional_kwargs:
    - base_url: Override the endpoint for this specific model
    - use_responses_api: Use Responses API instead of chat.completions (default: False)
    """

    # Cache: maps base_url string -> client instance
    _clients: dict[str, AsyncOpenAI] = PrivateAttr(default_factory=dict)

    def get_client(self, base_url: str | None = None) -> AsyncOpenAI:
        """Create or retrieve a cached AsyncOpenAI client for the given base_url."""
        # Use a cache key that distinguishes explicit base_url from env default
        cache_key = base_url or os.environ.get("OPENAI_BASE_URL", "_default_")

        if cache_key not in self._clients:
            api_key = os.environ.get("OPENAI_API_KEY", "dummy")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._clients[cache_key] = AsyncOpenAI(**kwargs)

        return self._clients[cache_key]

    def _get_chat(self, params: LLMParams) -> BaseChatModel:
        """Get OpenAI chat model. Supports base_url override via additional_kwargs."""
        model = params.model
        if not model:
            raise ValueError("OpenAI model not specified. Set params.model.")

        # Extract routing kwargs before spreading the rest into OpenAIChat
        extra_kwargs = dict(params.additional_kwargs or {})
        base_url = extra_kwargs.pop("base_url", None)
        use_responses_api = extra_kwargs.pop("use_responses_api", False)

        client = self.get_client(base_url=base_url)

        return OpenAIChat(
            client=client,
            model=model,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=(
                params.max_tokens
                or params.initial_max_tokens
                or DEFAULT_INITIAL_MAX_TOKEN
            ),
            stop=params.stop,
            cache=params.cache,
            use_responses_api=use_responses_api,
            **extra_kwargs,
        )
