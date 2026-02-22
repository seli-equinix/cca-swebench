# Copyright (c) Meta Platforms, Inc. and affiliates.
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...core.llm_manager.llm_params import LLMParams

CLAUDE_4_5_SONNET_THINKING = LLMParams(
    model="claude-sonnet-4-5",
    initial_max_tokens=16384,
    temperature=0.3,
    top_p=0.7,
    additional_kwargs={
        "thinking": ant.Thinking(
            type=ant.ThinkingType.ENABLED,
            budget_tokens=8192,
        ).dict(),
    },
)

CLAUDE_4_5_OPUS = LLMParams(
    model="claude-opus-4-5",
    initial_max_tokens=16384,
    temperature=0.3,
    top_p=None,
)

GPT5_1_THINKING = LLMParams(
    model="gpt-5.1",
    initial_max_tokens=8192,
    additional_kwargs={
        "thinking": ant.Thinking(
            type=ant.ThinkingType.ENABLED,
            budget_tokens=32768,
        ).dict(),
    },
)

GPT5_2_THINKING = LLMParams(
    model="gpt-5.2",
    initial_max_tokens=8192,
    additional_kwargs={
        "thinking": ant.Thinking(
            type=ant.ThinkingType.ENABLED,
            budget_tokens=32768,
        ).dict(),
    },
)

QWEN3_8B_NOTETAKER = LLMParams(
    model="/models/Qwen3-8B-FP8",
    initial_max_tokens=4096,
    max_tokens=8192,
    temperature=0.3,
    top_p=0.9,
    additional_kwargs={
        "base_url": "http://192.168.4.205:8400/v1",
    },
)
