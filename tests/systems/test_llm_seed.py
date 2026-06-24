"""Best-effort determinism: seed parameter forwarding for the LLM drivers.

Both OpenAI and Anthropic document ``seed`` as a hint, not a contract.
We don't test that the API returns deterministic output — that's a
provider behaviour. We test that the parameter flows from ``LLMConfig``
through ``_merged_params`` into the request kwargs, and that the
"don't send when None" rule is respected so legacy callers see the
exact same API call shape they did before the seed field was added.

Why this matters
----------------
A CrossBench cell value is one trajectory through 250+ LLM calls. Even
small variance per call compounds. Sending a stable ``seed`` cuts that
variance materially, which lets us produce tighter mean ± std with
fewer reruns. See deviations.md (seed addition) for the rationale.
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.models import LLMConfig


class _RecordingAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI`` that records the kwargs of
    the last ``chat.completions.create`` / ``completions.create`` call.

    Returned values are minimal but enough to satisfy the driver's
    response unpacking — choices[0], usage, etc.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_complete_kwargs: dict[str, Any] | None = None
        self.chat = self._ChatNamespace(self)
        self.completions = self._CompletionsNamespace(self)

    class _ChatNamespace:
        def __init__(self, parent: "_RecordingAsyncOpenAI") -> None:
            self._parent = parent
            self.completions = self._Completions(parent)

        class _Completions:
            def __init__(self, parent: "_RecordingAsyncOpenAI") -> None:
                self._parent = parent

            async def create(self, **kwargs: Any) -> Any:
                self._parent.last_chat_kwargs = kwargs
                return _FakeChatResp()

    class _CompletionsNamespace:
        def __init__(self, parent: "_RecordingAsyncOpenAI") -> None:
            self._parent = parent

        async def create(self, **kwargs: Any) -> Any:
            self._parent.last_complete_kwargs = kwargs
            return _FakeCompletionResp()


class _FakeChatResp:
    class _Choice:
        class _Msg:
            content = "ok"
        message = _Msg()
    choices = [_Choice()]

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
    usage = _Usage()


class _FakeCompletionResp:
    class _Choice:
        text = "ok"
    choices = [_Choice()]

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
    usage = _Usage()


@pytest.mark.asyncio
async def test_openai_chat_forwards_seed_when_set(monkeypatch):
    """``LLMConfig.seed=42`` → request kwargs include ``seed=42``."""
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    from cross_bench.systems.llm.openai_driver import OpenAIDriver

    config = LLMConfig(provider="openai", model="gpt-4o", seed=42)
    driver = OpenAIDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    assert driver._client.last_chat_kwargs is not None
    assert driver._client.last_chat_kwargs["seed"] == 42


@pytest.mark.asyncio
async def test_openai_chat_omits_seed_when_none(monkeypatch):
    """``LLMConfig.seed=None`` (default) → request has no ``seed`` key.

    Preserves the pre-seed call shape byte-for-byte for callers that
    haven't opted in.
    """
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    from cross_bench.systems.llm.openai_driver import OpenAIDriver

    config = LLMConfig(provider="openai", model="gpt-4o")  # seed=None default
    driver = OpenAIDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    assert driver._client.last_chat_kwargs is not None
    assert "seed" not in driver._client.last_chat_kwargs


@pytest.mark.asyncio
async def test_openai_chat_seed_override_per_call(monkeypatch):
    """Per-call override beats LLMConfig default — matches temperature/top_p flow."""
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    from cross_bench.systems.llm.openai_driver import OpenAIDriver

    config = LLMConfig(provider="openai", model="gpt-4o", seed=1)
    driver = OpenAIDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {"seed": 99})
    assert driver._client.last_chat_kwargs["seed"] == 99


@pytest.mark.asyncio
async def test_openai_reasoning_path_forwards_seed(monkeypatch):
    """Reasoning models (o1/o3) drop temperature but should still accept seed.

    The reasoning branch uses ``max_completion_tokens`` instead of
    ``max_tokens`` and omits temperature/top_p, but ``seed`` is still
    in the upstream contract.
    """
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    from cross_bench.systems.llm.openai_driver import OpenAIDriver

    config = LLMConfig(provider="openai", model="o3-mini", seed=7)
    driver = OpenAIDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    kw = driver._client.last_chat_kwargs
    assert kw["seed"] == 7
    assert "max_completion_tokens" in kw  # confirms reasoning branch took
    assert "temperature" not in kw


@pytest.mark.asyncio
async def test_openai_complete_path_forwards_seed(monkeypatch):
    """Legacy completions endpoint also gets ``seed`` when configured."""
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    from cross_bench.systems.llm.openai_driver import OpenAIDriver

    config = LLMConfig(provider="openai", model="gpt-4o", seed=11)
    driver = OpenAIDriver(config)
    await driver.complete("hi", {})
    assert driver._client.last_complete_kwargs["seed"] == 11


# --- Anthropic ---


class _RecordingAsyncAnthropic:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, parent: "_RecordingAsyncAnthropic") -> None:
            self._parent = parent

        async def create(self, **kwargs: Any) -> Any:
            self._parent.last_kwargs = kwargs
            return _FakeAnthropicResp()


class _FakeAnthropicResp:
    class _Block:
        text = "ok"
    content = [_Block()]

    class _Usage:
        input_tokens = 1
        output_tokens = 1
    usage = _Usage()


@pytest.mark.asyncio
async def test_anthropic_chat_ignores_seed_even_when_set(monkeypatch):
    """Anthropic's Messages API rejects ``seed`` — driver must NOT send it.

    Smoke run on 2026-05-25 against ``claude-haiku-4-5-20251001`` returned
    ``400 invalid_request_error: seed: Extra inputs are not permitted``
    when seed was forwarded via ``extra_body``. The contract is:
    ``LLMConfig.seed`` is honored by the OpenAI driver as a best-effort
    determinism hint, and IGNORED by Anthropic. Locks that behaviour so
    we don't reintroduce the regression.
    """
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _RecordingAsyncAnthropic)
    from cross_bench.systems.llm.anthropic_driver import AnthropicDriver

    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5", seed=42)
    driver = AnthropicDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    kw = driver._client.last_kwargs
    assert kw is not None
    # No top-level seed
    assert "seed" not in kw
    # No extra_body smuggling
    assert "extra_body" not in kw


@pytest.mark.asyncio
async def test_anthropic_chat_omits_seed_when_none(monkeypatch):
    """No seed configured → no seed sent (always true for Anthropic)."""
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _RecordingAsyncAnthropic)
    from cross_bench.systems.llm.anthropic_driver import AnthropicDriver

    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5")
    driver = AnthropicDriver(config)
    await driver.chat([{"role": "user", "content": "hi"}], {})
    kw = driver._client.last_kwargs
    assert "seed" not in kw
    assert "extra_body" not in kw


# --- LLMConfig dataclass ---


def test_llmconfig_seed_defaults_to_none():
    """``seed`` defaults to None so unconfigured callers don't change behaviour."""
    c = LLMConfig(provider="openai", model="gpt-4o")
    assert c.seed is None


def test_llmconfig_accepts_int_seed():
    c = LLMConfig(provider="openai", model="gpt-4o", seed=42)
    assert c.seed == 42
