"""Generic OpenAI-compatible API driver for any provider exposing the OpenAI API format."""

from __future__ import annotations

import time
from typing import Any

from cross_bench.models import LLMConfig, StepRecord
from cross_bench.systems.base import LLMDriver
from cross_bench.systems.registry import register_llm


@register_llm("generic_openai")
class GenericOpenAIDriver(LLMDriver):
    """Works with vLLM, Ollama, Together, Groq, or any OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        import openai
        self._client = openai.AsyncOpenAI(
            api_key=config.extra.get("api_key", "not-needed"),
            base_url=config.extra.get("base_url", "http://localhost:8000/v1"),
            timeout=float(config.extra.get("request_timeout", 120)),
            max_retries=0,
        )
        self._config = config

    async def complete(self, prompt: str, config: dict[str, Any]) -> tuple[str, StepRecord]:
        return await self.chat([{"role": "user", "content": prompt}], config)

    async def chat(self, messages: list[dict[str, str]], config: dict[str, Any]) -> tuple[str, StepRecord]:
        merged = self._merged_params(config)
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": merged["model"],
            "messages": messages,
            "temperature": merged["temperature"],
            "max_tokens": merged["max_tokens"],
        }
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
            "seed": override.get("seed", self._config.seed),
        }
