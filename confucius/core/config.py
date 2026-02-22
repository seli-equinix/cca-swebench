# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""CCA Configuration loader.

Reads ~/.confucius/config.toml (or CCA_CONFIG_PATH env var).
Falls back to hardcoded defaults when no config file exists.

Config format (TOML):
    [active]
    coder = "local"
    note_taker = "local"

    openai_model_prefixes = ["qwen", "/models/"]

    [providers.local.note_taker]
    model = "/models/Qwen3-8B-FP8"
    base_url = "http://192.168.4.205:8400/v1"
    initial_max_tokens = 4096
    temperature = 0.3
"""
from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .llm_manager.llm_params import LLMParams

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path.home() / ".confucius" / "config.toml"


class ProviderProfile(BaseModel):
    """A single LLM provider profile."""

    model: str
    provider: str = "openai"  # informational: "openai", "azure", "bedrock", "google"
    base_url: str | None = None
    api_key_env: str | None = None
    initial_max_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    thinking_budget: int | None = None
    use_responses_api: bool = False
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    class Config:
        extra = "ignore"

    def to_llm_params(self) -> LLMParams:
        """Convert to CCA's internal LLMParams format."""
        from .chat_models.bedrock.api.invoke_model import anthropic as ant

        additional_kwargs: dict[str, Any] = {}

        if self.base_url:
            additional_kwargs["base_url"] = self.base_url
        if self.use_responses_api:
            additional_kwargs["use_responses_api"] = True

        # Convert thinking_budget to Anthropic thinking format (for cloud providers)
        if self.thinking_budget and self.thinking_budget > 0:
            additional_kwargs["thinking"] = ant.Thinking(
                type=ant.ThinkingType.ENABLED,
                budget_tokens=self.thinking_budget,
            ).dict()

        return LLMParams(
            model=self.model,
            initial_max_tokens=self.initial_max_tokens,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            additional_kwargs=additional_kwargs or None,
        )


class CCAConfig(BaseModel):
    """Top-level CCA configuration."""

    active: dict[str, str] = Field(default_factory=dict)
    openai_model_prefixes: list[str] = Field(
        default_factory=lambda: ["qwen"],
    )
    providers: dict[str, dict[str, ProviderProfile]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"

    def get_profile(self, role: str) -> ProviderProfile | None:
        """Get the active profile for a role."""
        provider_set = self.active.get(role)
        if not provider_set:
            return None
        return self.providers.get(provider_set, {}).get(role)


# Module-level singleton
_config: CCAConfig | None = None


def _load_config() -> CCAConfig:
    """Load config from TOML file. Returns empty config if no file exists."""
    global _config
    if _config is not None:
        return _config

    config_path = Path(os.environ.get("CCA_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                raw = tomllib.load(f)
            _config = CCAConfig(**raw)
            logger.info(f"Loaded CCA config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load CCA config from {config_path}: {e}")
            _config = CCAConfig()
    else:
        logger.debug(f"No CCA config at {config_path}, using defaults")
        _config = CCAConfig()
    return _config


def reload_config() -> CCAConfig:
    """Force reload config from disk (for hot-reload / UI updates)."""
    global _config
    _config = None
    return _load_config()


def get_llm_params(role: str, default: LLMParams) -> LLMParams:
    """Get LLMParams for a named role from config, falling back to default.

    Usage in entry classes:
        from ...core.config import get_llm_params
        from ..code.llm_params import QWEN3_8B_NOTETAKER

        llm_params = get_llm_params("note_taker", default=QWEN3_8B_NOTETAKER)
    """
    config = _load_config()
    profile = config.get_profile(role)
    if profile is None:
        return default
    return profile.to_llm_params()


def get_openai_model_prefixes(default: list[str]) -> list[str]:
    """Get OpenAI model prefixes from config, falling back to default."""
    config = _load_config()
    return config.openai_model_prefixes or default
