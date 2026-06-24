"""AnthropicBedrockDriver — Claude via AWS Bedrock.

Same Messages-API surface as the direct AnthropicDriver, different
auth (IAM via boto3 chain instead of static API key). Tests pin:

- Driver registers under ``"anthropic_bedrock"`` provider name.
- ``aws_region`` is forwarded to the SDK when set on
  ``LLMConfig.extra``.
- ``aws_profile`` is forwarded when set (named-profile dev workflow).
- Same no-seed contract as the direct driver (Bedrock proxies through
  to Anthropic's Messages API which rejects ``seed``).
- System messages get pulled out of the messages list and into the
  top-level ``system`` kwarg (Bedrock + Anthropic both accept this
  shape).
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.models import LLMConfig


class _RecordingAsyncAnthropicBedrock:
    """Records construction kwargs + messages.create kwargs."""

    last_construct_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_construct_kwargs = kwargs
        self.last_create_kwargs: dict[str, Any] = {}
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, parent: "_RecordingAsyncAnthropicBedrock") -> None:
            self._parent = parent

        async def create(self, **kwargs: Any) -> Any:
            self._parent.last_create_kwargs = kwargs
            return _FakeResp()


class _FakeResp:
    class _Block:
        text = "ok"
    content = [_Block()]

    class _Usage:
        input_tokens = 1
        output_tokens = 1
    usage = _Usage()


@pytest.fixture
def patch_bedrock_sdk(monkeypatch):
    """Patch ``anthropic.AsyncAnthropicBedrock`` with a recorder."""
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "AsyncAnthropicBedrock", _RecordingAsyncAnthropicBedrock)
    return _RecordingAsyncAnthropicBedrock


def test_driver_registered_under_anthropic_bedrock_name():
    """The provider name profiles use to select this driver."""
    from cross_bench.systems.registry import _ensure_llm_loaded, _LLM_DRIVERS
    _ensure_llm_loaded()
    assert "anthropic_bedrock" in _LLM_DRIVERS


def test_construct_forwards_aws_region(patch_bedrock_sdk):
    """``LLMConfig.extra["aws_region"]`` lands on the SDK constructor."""
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        extra={"aws_region": "us-east-1"},
    )
    AnthropicBedrockDriver(config)
    assert patch_bedrock_sdk.last_construct_kwargs.get("aws_region") == "us-east-1"


def test_construct_forwards_aws_profile_when_set(patch_bedrock_sdk):
    """``aws_profile`` for named-profile dev workflows."""
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        extra={"aws_profile": "research-dev"},
    )
    AnthropicBedrockDriver(config)
    assert patch_bedrock_sdk.last_construct_kwargs.get("aws_profile") == "research-dev"


def test_construct_omits_aws_region_when_unset(patch_bedrock_sdk):
    """No aws_region → SDK falls back to boto3 chain. Avoid sending None."""
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    AnthropicBedrockDriver(config)
    assert "aws_region" not in patch_bedrock_sdk.last_construct_kwargs


@pytest.mark.asyncio
async def test_chat_does_not_send_seed_to_bedrock(patch_bedrock_sdk):
    """Bedrock proxies to Anthropic Messages API → seed rejected. Lock it.

    Same constraint as ``AnthropicDriver``; regression-locked here so a
    future contributor doesn't add seed forwarding to the Bedrock driver
    forgetting that Bedrock has the same downstream API.
    """
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        seed=42,
    )
    driver = AnthropicBedrockDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    kw = driver._client.last_create_kwargs
    assert "seed" not in kw
    assert "extra_body" not in kw


@pytest.mark.asyncio
async def test_chat_extracts_system_message_to_top_level(patch_bedrock_sdk):
    """System role messages are hoisted out of ``messages`` into ``system`` kwarg.

    Matches the direct AnthropicDriver's behaviour and is what Bedrock's
    Anthropic shim expects.
    """
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    driver = AnthropicBedrockDriver(config)
    await driver.chat([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ], {})
    kw = driver._client.last_create_kwargs
    assert kw["system"] == "be terse"
    # System message is removed from the messages list.
    assert {"role": "system", "content": "be terse"} not in kw["messages"]
    assert {"role": "user", "content": "hi"} in kw["messages"]


def test_construct_defaults_max_retries_to_zero(patch_bedrock_sdk):
    """Driver default preserves workflow.md "zero retries" invariant.

    Profiles opt into retries explicitly via ``extra["max_retries"]``;
    omitting the key must keep retries off.
    """
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    AnthropicBedrockDriver(config)
    assert patch_bedrock_sdk.last_construct_kwargs.get("max_retries") == 0


def test_construct_forwards_max_retries_when_set(patch_bedrock_sdk):
    """Bedrock TPM-throttling case: ``max_retries`` flows to the SDK."""
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        extra={"max_retries": 10},
    )
    AnthropicBedrockDriver(config)
    assert patch_bedrock_sdk.last_construct_kwargs.get("max_retries") == 10


def test_construct_forwards_request_timeout_override(patch_bedrock_sdk):
    """``request_timeout`` overrides the 120s default."""
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        extra={"request_timeout": 180},
    )
    AnthropicBedrockDriver(config)
    assert patch_bedrock_sdk.last_construct_kwargs.get("timeout") == 180.0


@pytest.mark.asyncio
async def test_chat_forwards_temperature_and_max_tokens(patch_bedrock_sdk):
    from cross_bench.systems.llm.anthropic_bedrock_driver import AnthropicBedrockDriver

    config = LLMConfig(
        provider="anthropic_bedrock",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        temperature=0.0,
        max_tokens=4096,
    )
    driver = AnthropicBedrockDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    kw = driver._client.last_create_kwargs
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == 4096
