"""OpenAI LLM driver.

Supports both standard chat models (gpt-4o, etc.) and reasoning models
(o1, o1-mini, o3-mini) which have different API constraints:
  - Reasoning models don't accept temperature/top_p
  - Reasoning models use max_completion_tokens instead of max_tokens
  - Reasoning models don't support system messages (use developer role)
"""

from __future__ import annotations

import time
from typing import Any

from cross_bench.models import LLMConfig, StepRecord
from cross_bench.systems.base import LLMDriver
from cross_bench.systems.registry import register_llm

# Models that use the reasoning API (no temperature/top_p, max_completion_tokens)
_REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3-mini", "o3"}


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model based on its name."""
    # Exact match or dated version (e.g. "o1-mini-2024-09-12")
    base = model.split("-2")[0]  # strip date suffix like "-2024-09-12"
    return base in _REASONING_MODELS


@register_llm("openai")
class OpenAIDriver(LLMDriver):
    def __init__(self, config: LLMConfig):
        import openai
        self._client = openai.AsyncOpenAI(
            api_key=config.extra.get("api_key"),
            base_url=config.extra.get("base_url"),
            timeout=float(config.extra.get("request_timeout", 120)),
            max_retries=0,
        )
        self._config = config

    async def complete(self, prompt: str, config: dict[str, Any]) -> tuple[str, StepRecord]:
        merged = self._merged_params(config)
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": merged["model"],
            "prompt": prompt,
            "temperature": merged["temperature"],
            "max_tokens": merged["max_tokens"],
            "top_p": merged.get("top_p", 1.0),
        }
        seed = merged.get("seed")
        if seed is not None:
            kwargs["seed"] = int(seed)
        resp = await self._client.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].text
        usage = resp.usage
        return text, StepRecord(
            step_index=0,
            action="completion",
            input_text=prompt,
            output_text=text,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )

    async def chat(self, messages: list[dict[str, str]], config: dict[str, Any]) -> tuple[str, StepRecord]:
        merged = self._merged_params(config)
        model = merged["model"]
        is_reasoning = _is_reasoning_model(model)

        # Reasoning models: convert system → developer, drop unsupported params
        if is_reasoning:
            messages = _adapt_messages_for_reasoning(messages)

        t0 = time.perf_counter()

        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if is_reasoning:
            # Reasoning models: no temperature/top_p, use max_completion_tokens
            kwargs["max_completion_tokens"] = merged["max_tokens"]
        else:
            kwargs["temperature"] = merged["temperature"]
            kwargs["max_tokens"] = merged["max_tokens"]
            kwargs["top_p"] = merged.get("top_p", 1.0)
        # Best-effort determinism. OpenAI accepts ``seed`` on both the
        # standard chat and the reasoning branches (the reasoning branch
        # silently ignores it on some snapshots but never rejects it).
        # ``None`` means "don't send" — preserves the pre-seed call shape
        # byte-for-byte for callers that haven't opted in.
        seed = merged.get("seed")
        if seed is not None:
            kwargs["seed"] = int(seed)

        resp = await self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, StepRecord(
            step_index=0,
            action="chat",
            input_text=str(messages),
            output_text=text,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )

    def _merged_params(self, override: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": override.get("model", self._config.model),
            "temperature": override.get("temperature", self._config.temperature),
            "max_tokens": override.get("max_tokens", self._config.max_tokens),
            "top_p": override.get("top_p", self._config.top_p),
            "seed": override.get("seed", self._config.seed),
        }


def _adapt_messages_for_reasoning(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reasoning models don't support 'system' role — convert to 'developer'."""
    adapted = []
    for msg in messages:
        if msg.get("role") == "system":
            adapted.append({"role": "developer", "content": msg["content"]})
        else:
            adapted.append(msg)
    return adapted
