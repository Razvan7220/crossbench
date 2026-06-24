"""Anthropic Claude via AWS Bedrock.

Same Messages API surface as the direct Anthropic API
(``AsyncAnthropic``), but auth is IAM-based — uses the standard
boto3 credential chain (env vars / shared credentials file / EC2
instance metadata / ECS task role) instead of a static
``ANTHROPIC_API_KEY``. Recommended when:

- Running on AWS (ECS task role just works, no key secret needed)
- You can't or don't want to manage a separate Anthropic billing
  relationship (everything goes through the existing AWS account)
- An ``ANTHROPIC_API_KEY`` leaked and you need to remove the
  dependency entirely while you rotate

Prerequisites
-------------
1. **Enable Bedrock model access** for the specific Claude model
   in the AWS console (``console.aws.amazon.com/bedrock`` →
   "Model access" → request access). Access requests are typically
   approved in minutes for individual models.
2. **IAM permissions** for the caller (your local user or the ECS
   task role) need at minimum:
   - ``bedrock:InvokeModel`` on the model ARN
   - ``bedrock:InvokeModelWithResponseStream`` (for streaming;
     not used here but typically granted together)
3. **Model ID** uses Bedrock's naming, NOT the Anthropic direct-API
   id. E.g. direct API ``claude-haiku-4-5-20251001`` corresponds to
   Bedrock model id ``anthropic.claude-haiku-4-5-20251001-v1:0`` (or
   the regional inference profile equivalent, e.g.
   ``us.anthropic.claude-haiku-4-5-20251001-v1:0``).

Determinism note: same as the direct AnthropicDriver — Bedrock
proxies through to Anthropic's Messages API which doesn't accept a
``seed`` parameter. The driver intentionally drops ``LLMConfig.seed``
to avoid the ``400 invalid_request_error: seed: Extra inputs are not
permitted`` we hit on 2026-05-25.
"""

from __future__ import annotations

import time
from typing import Any

from cross_bench.models import LLMConfig, StepRecord
from cross_bench.systems.base import LLMDriver
from cross_bench.systems.registry import register_llm


@register_llm("anthropic_bedrock")
class AnthropicBedrockDriver(LLMDriver):
    """Claude via Bedrock — same Messages API as the direct provider."""

    def __init__(self, config: LLMConfig):
        # ``AsyncAnthropicBedrock`` reads AWS creds from the boto3 chain.
        # Explicit ``aws_region`` overrides the chain's region preference;
        # ``LLMConfig.extra["aws_region"]`` takes precedence over the
        # ``AWS_REGION`` env var (matches how OpenAI driver lets profiles
        # pin the base_url).
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:  # pragma: no cover — exercised only at install time
            raise RuntimeError(
                "anthropic_bedrock provider requires anthropic[bedrock] "
                "(or anthropic + boto3). Install with: pip install 'anthropic[bedrock]'"
            ) from e

        # ``max_retries`` is opt-in (default 0) — matches the workflow.md
        # "zero retries by default" invariant for the OpenAI/Anthropic
        # drivers. Bedrock specifically benefits from a nonzero value when
        # the IAM principal is throttled at the account TPM quota (429
        # responses); the anthropic SDK uses jittered exponential backoff
        # between attempts. Profiles like ``haiku45_bedrock`` set this
        # explicitly for SWE-bench fan-outs that bump up against TPM.
        kwargs: dict[str, Any] = {
            "timeout": float(config.extra.get("request_timeout", 120)),
            "max_retries": int(config.extra.get("max_retries", 0)),
        }
        aws_region = config.extra.get("aws_region")
        if aws_region:
            kwargs["aws_region"] = aws_region
        # ``aws_profile`` is occasionally useful for local dev with named
        # profiles in ~/.aws/credentials; not used on ECS where the task
        # role's creds come through the metadata service.
        aws_profile = config.extra.get("aws_profile")
        if aws_profile:
            kwargs["aws_profile"] = aws_profile

        self._client = AsyncAnthropicBedrock(**kwargs)
        self._config = config

    async def complete(self, prompt: str, config: dict[str, Any]) -> tuple[str, StepRecord]:
        return await self.chat([{"role": "user", "content": prompt}], config)

    async def chat(
        self, messages: list[dict[str, str]], config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        merged = self._merged_params(config)
        t0 = time.perf_counter()

        # Extract system message — Bedrock's Anthropic shim takes a
        # top-level ``system`` like the direct API.
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
        # NOTE: same as AnthropicDriver — ``seed`` is rejected by the
        # underlying Messages API (Bedrock proxies through it). Do NOT
        # add seed handling here. See 2026-05-25 incident in deviations.

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
        # ``seed`` intentionally omitted (Bedrock proxies to Anthropic
        # Messages API which rejects it). See AnthropicDriver for the
        # same constraint.
        return {
            "model": override.get("model", self._config.model),
            "temperature": override.get("temperature", self._config.temperature),
            "max_tokens": override.get("max_tokens", self._config.max_tokens),
        }
