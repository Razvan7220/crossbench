"""Anthropic Claude LLM driver."""

from __future__ import annotations

import time
from typing import Any

from cross_bench.models import LLMConfig, StepRecord
from cross_bench.systems.base import LLMDriver
from cross_bench.systems.registry import register_llm


@register_llm("anthropic")
class AnthropicDriver(LLMDriver):
    def __init__(self, config: LLMConfig):
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=config.extra.get("api_key"),
            timeout=float(config.extra.get("request_timeout", 120)),
            max_retries=0,
        )
        self._config = config

    async def complete(self, prompt: str, config: dict[str, Any]) -> tuple[str, StepRecord]:
        # Anthropic uses messages API; wrap prompt as a user message
        return await self.chat([{"role": "user", "content": prompt}], config)

    async def chat(self, messages: list[dict[str, str]], config: dict[str, Any]) -> tuple[str, StepRecord]:
        merged = self._merged_params(config)
        t0 = time.perf_counter()

        # Extract system message if present
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": merged["model"],
            "messages": chat_messages,
            "max_tokens": merged["max_tokens"],
            "temperature": merged["temperature"],
        }
        if system:
            kwargs["system"] = system
        # NOTE: Anthropic's Messages API does NOT accept a ``seed``
        # parameter — it returns 400 ``seed: Extra inputs are not
        # permitted`` if sent. ``LLMConfig.seed`` is therefore IGNORED
        # for the anthropic provider. Use temperature=0 + multiple
        # reruns for variance management on this driver. The OpenAI
        # driver does honor ``seed`` as a best-effort hint.

        resp = await self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text

        return text, StepRecord(
            step_index=0,
            action="chat",
            input_text=str(messages),
            output_text=text,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_ms=latency,
        )

    def _merged_params(self, override: dict[str, Any]) -> dict[str, Any]:
        # ``seed`` intentionally omitted: Anthropic's Messages API rejects it.
        return {
            "model": override.get("model", self._config.model),
            "temperature": override.get("temperature", self._config.temperature),
            "max_tokens": override.get("max_tokens", self._config.max_tokens),
        }
