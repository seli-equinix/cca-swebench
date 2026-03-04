# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""CCA Configuration loader.

Reads ~/.confucius/config.toml (or CCA_CONFIG_PATH env var).
Raises CCAConfigError if config is missing or incomplete — never silently
falls back. All errors are structured for UI consumption.

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

# NOTE: LLMParams is NOT imported at module level to avoid circular imports.
# config.py -> llm_manager.llm_params -> __init__ -> auto -> azure -> constants -> config.py
# Instead, LLMParams is imported lazily inside functions that need it.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_manager.llm_params import LLMParams

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path.home() / ".confucius" / "config.toml"


# ---------------------------------------------------------------------------
# Structured error — UI-consumable
# ---------------------------------------------------------------------------

class CCAConfigError(Exception):
    """Raised when CCA configuration is missing, invalid, or incomplete.

    Every field is designed for UI consumption:
        role        — which LLM role failed (e.g. "coder", "note_taker")
        detail      — human-readable explanation of what went wrong
        config_path — the config file path that was loaded (or expected)
        suggestion  — actionable fix the UI can display to the user

    Standard usage in CCA entry classes:
        try:
            params = get_llm_params("coder")
        except CCAConfigError as e:
            # e.role, e.detail, e.config_path, e.suggestion all available
            show_config_error(e)
    """

    def __init__(
        self,
        *,
        role: str,
        detail: str,
        config_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.role = role
        self.detail = detail
        self.config_path = config_path
        self.suggestion = suggestion
        super().__init__(f"[{role}] {detail}")

    def to_dict(self) -> dict[str, str | None]:
        """Serialize for JSON API / UI transport."""
        return {
            "error": "config_error",
            "role": self.role,
            "detail": self.detail,
            "config_path": self.config_path,
            "suggestion": self.suggestion,
        }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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
        from .llm_manager.llm_params import LLMParams as _LLMParams

        additional_kwargs: dict[str, Any] = {}

        if self.base_url:
            additional_kwargs["base_url"] = self.base_url
        if self.use_responses_api:
            additional_kwargs["use_responses_api"] = True

        # Convert thinking_budget — format depends on provider
        if self.thinking_budget and self.thinking_budget > 0:
            if self.provider in ("azure", "bedrock", "anthropic"):
                # Anthropic-compatible providers use the Thinking object
                from .chat_models.bedrock.api.invoke_model import anthropic as ant
                additional_kwargs["thinking"] = ant.Thinking(
                    type=ant.ThinkingType.ENABLED,
                    budget_tokens=self.thinking_budget,
                ).dict()
            else:
                # OpenAI-compatible providers (vLLM, etc.)
                additional_kwargs["thinking_budget"] = self.thinking_budget

        return _LLMParams(
            model=self.model,
            initial_max_tokens=self.initial_max_tokens,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            additional_kwargs=additional_kwargs or None,
        )


class RouterConfig(BaseModel):
    """Expert router configuration (Functionary-based classification)."""

    enabled: bool = False
    url: str = "http://192.168.4.204:8001"
    timeout_ms: int = 10000
    fallback_entry: str = "coder"
    temperature: float = 0.1

    class Config:
        extra = "ignore"


class ToolRouterConfig(BaseModel):
    """In-loop tool selection configuration (Phase 2).

    When enabled, the Functionary model selects which additional tool
    groups to enable mid-loop when the agent gets stuck without the
    right tools.
    """

    enabled: bool = False
    url: str = "http://192.168.4.204:8001"
    timeout_ms: int = 10000
    temperature: float = 0.1

    class Config:
        extra = "ignore"


class CCAConfig(BaseModel):
    """Top-level CCA configuration."""

    active: dict[str, str] = Field(default_factory=dict)
    openai_model_prefixes: list[str] = Field(
        default_factory=lambda: ["qwen"],
    )
    providers: dict[str, dict[str, ProviderProfile]] = Field(default_factory=dict)
    router: RouterConfig = Field(default_factory=RouterConfig)
    tool_router: ToolRouterConfig = Field(default_factory=ToolRouterConfig)

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
_config_path: str | None = None  # Track which path was loaded (for error messages)


def _resolve_config_path() -> Path:
    """Resolve the config file path from env or default."""
    return Path(os.environ.get("CCA_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))


def _load_config() -> CCAConfig:
    """Load config from TOML file. Raises CCAConfigError on failure."""
    global _config, _config_path
    if _config is not None:
        return _config

    config_path = _resolve_config_path()
    _config_path = str(config_path)

    if not config_path.exists():
        raise CCAConfigError(
            role="*",
            detail=f"Config file not found: {config_path}",
            config_path=str(config_path),
            suggestion=(
                f"Create {config_path} with [active] and [providers] sections. "
                "Set CCA_CONFIG_PATH env var to use a different location."
            ),
        )

    try:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        _config = CCAConfig(**raw)
        logger.info(f"Loaded CCA config from {config_path}")
    except CCAConfigError:
        raise
    except Exception as e:
        raise CCAConfigError(
            role="*",
            detail=f"Failed to parse config: {e}",
            config_path=str(config_path),
            suggestion="Check TOML syntax. Validate with: python -c \"import tomllib; tomllib.load(open('config.toml','rb'))\"",
        ) from e

    return _config


def reload_config() -> CCAConfig:
    """Force reload config from disk (for hot-reload / UI updates)."""
    global _config, _config_path
    _config = None
    _config_path = None
    return _load_config()


def get_llm_params(role: str) -> LLMParams:
    """Get LLMParams for a named role from config.

    Raises CCAConfigError if the role is not configured. No silent fallbacks.

    Usage in entry classes:
        from ...core.config import get_llm_params, CCAConfigError

        params = get_llm_params("coder")
    """
    config = _load_config()
    config_path = _config_path or str(_resolve_config_path())

    # Check [active] section has this role
    provider_set = config.active.get(role)
    if not provider_set:
        available = list(config.active.keys()) or ["(none)"]
        raise CCAConfigError(
            role=role,
            detail=f"Role '{role}' not found in [active] section",
            config_path=config_path,
            suggestion=f"Add '{role} = \"local\"' to the [active] section. Configured roles: {', '.join(available)}",
        )

    # Check the provider set has a profile for this role
    profile = config.providers.get(provider_set, {}).get(role)
    if profile is None:
        available_sets = list(config.providers.keys()) or ["(none)"]
        raise CCAConfigError(
            role=role,
            detail=f"No profile for role '{role}' in provider set '{provider_set}'",
            config_path=config_path,
            suggestion=f"Add [providers.{provider_set}.{role}] section with at least 'model' key. Available provider sets: {', '.join(available_sets)}",
        )

    return profile.to_llm_params()


def get_openai_model_prefixes() -> list[str]:
    """Get OpenAI model prefixes from config.

    Raises CCAConfigError if config cannot be loaded.
    """
    config = _load_config()
    return config.openai_model_prefixes


def get_router_config() -> RouterConfig:
    """Get expert router configuration.

    Returns RouterConfig (enabled=False if section is missing).
    Never raises — router is optional.
    """
    config = _load_config()
    return config.router


def get_tool_router_config() -> ToolRouterConfig:
    """Get in-loop tool router configuration (Phase 2).

    Returns ToolRouterConfig (enabled=False if section is missing).
    Never raises — tool router is optional.
    """
    config = _load_config()
    return config.tool_router
