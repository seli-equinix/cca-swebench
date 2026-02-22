# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

DEFAULT_INITIAL_MAX_TOKEN: int = 512
DEFAULT_MAX_MAX_TOKEN: int = 16384

AZURE_OPENAI_MODEL_PREFIXES = ["gpt", "o1", "o3", "o4"]

from ..config import get_openai_model_prefixes

OPENAI_MODEL_PREFIXES = get_openai_model_prefixes(default=["qwen"])
