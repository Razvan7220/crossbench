"""Tests for the ``MemoryMiddleware`` internal-token accounting contract.

Covers:
- ``count_tokens`` module-level helper (tiktoken + chars/4 fallback).
- ``_record_internal_llm`` prefers provider ``usage`` when present,
  falls back to tokenizer otherwise; handles both OpenAI
  (``prompt_tokens``/``completion_tokens``) and Anthropic
  (``input_tokens``/``output_tokens``) usage spellings.
- ``_record_io`` accumulates boundary tokens.
- ``_reset_internal_counters`` zeros everything.
- ``get_internal_token_stats`` returns the documented five-key dict.

See deviations.md "Platform / cross-cutting" entry on
internal-token accounting for the contract this locks.
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.systems.base import MemoryMiddleware, count_tokens


class _DummyMiddleware(MemoryMiddleware):
    """Concrete subclass for exercising the base-class accounting."""

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        return prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        return None

    async def reset(self) -> None:
        self._reset_internal_counters()


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_empty_returns_zero():
    assert count_tokens("") == 0


def test_count_tokens_nonempty_returns_positive():
    n = count_tokens("the quick brown fox jumps over the lazy dog")
    assert n >= 1


def test_count_tokens_proportional_to_text_length():
    short = count_tokens("hello")
    long = count_tokens("hello " * 200)
    assert long > short * 50


def test_count_tokens_fallback_when_tiktoken_unavailable(monkeypatch):
    """When tiktoken raises on import, helper falls back to chars/4."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    text = "x" * 40
    n = count_tokens(text)
    # chars/4 fallback ⇒ ~10 tokens; at least 1.
    assert n >= 1
    assert n <= len(text)


# ---------------------------------------------------------------------------
# _record_internal_llm: usage path
# ---------------------------------------------------------------------------


def test_record_internal_llm_prefers_openai_usage_over_tokenizer():
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="some prompt", response="some response",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] == 100
    assert stats["tokens_out"] == 50
    assert stats["llm_calls"] == 1


def test_record_internal_llm_handles_anthropic_usage_spellings():
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="x", response="y",
        usage={"input_tokens": 17, "output_tokens": 23},
    )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] == 17
    assert stats["tokens_out"] == 23


def test_record_internal_llm_falls_back_to_tokenizer_without_usage():
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="a longer-ish prompt that has several words",
        response="a response of moderate length",
    )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] > 0
    assert stats["tokens_out"] > 0
    assert stats["llm_calls"] == 1


def test_record_internal_llm_partial_usage_falls_back_per_side():
    """Missing ``completion_tokens`` only ⇒ in uses usage, out tokenizes."""
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="x", response="response text here",
        usage={"prompt_tokens": 42},
    )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] == 42
    assert stats["tokens_out"] > 0  # tokenizer fallback


def test_record_internal_llm_negative_usage_falls_back():
    """A defective ``usage`` payload (negative count) is ignored."""
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="prompt", response="response",
        usage={"prompt_tokens": -1, "completion_tokens": -5},
    )
    stats = m.get_internal_token_stats()
    # Both sides should fall back to tokenizer counts.
    assert stats["tokens_in"] > 0
    assert stats["tokens_out"] > 0


def test_record_internal_llm_accumulates_across_calls():
    m = _DummyMiddleware()
    for _ in range(3):
        m._record_internal_llm(
            prompt="x", response="y",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] == 30
    assert stats["tokens_out"] == 15
    assert stats["llm_calls"] == 3


def test_record_internal_llm_none_args_still_increments_calls():
    """A failed-then-retried call where neither side has text should
    still register the call attempt — ``llm_calls`` is the auditable
    "how many times did we ask the LLM" counter."""
    m = _DummyMiddleware()
    m._record_internal_llm(prompt=None, response=None, usage=None)
    stats = m.get_internal_token_stats()
    assert stats["llm_calls"] == 1
    assert stats["tokens_in"] == 0
    assert stats["tokens_out"] == 0


# ---------------------------------------------------------------------------
# _record_io: boundary path
# ---------------------------------------------------------------------------


def test_record_io_accumulates_independent_sides():
    m = _DummyMiddleware()
    m._record_io(request_text="hello world")
    m._record_io(response_text="something back")
    m._record_io(
        request_text="another request",
        response_text="another response",
    )
    stats = m.get_internal_token_stats()
    assert stats["io_tokens_in"] > 0
    assert stats["io_tokens_out"] > 0
    # IO doesn't touch the internal-LLM counters.
    assert stats["tokens_in"] == 0
    assert stats["tokens_out"] == 0
    assert stats["llm_calls"] == 0


def test_record_io_empty_strings_no_op():
    m = _DummyMiddleware()
    m._record_io(request_text="", response_text="")
    stats = m.get_internal_token_stats()
    assert stats["io_tokens_in"] == 0
    assert stats["io_tokens_out"] == 0


# ---------------------------------------------------------------------------
# reset + stats shape
# ---------------------------------------------------------------------------


def test_reset_internal_counters_zeroes_all_five_fields():
    m = _DummyMiddleware()
    m._record_internal_llm(
        prompt="x", response="y",
        usage={"prompt_tokens": 5, "completion_tokens": 3},
    )
    m._record_io(request_text="hello", response_text="world")
    m._reset_internal_counters()
    stats = m.get_internal_token_stats()
    assert stats == {
        "tokens_in": 0,
        "tokens_out": 0,
        "llm_calls": 0,
        "io_tokens_in": 0,
        "io_tokens_out": 0,
    }


def test_get_internal_token_stats_documented_keys():
    m = _DummyMiddleware()
    keys = set(m.get_internal_token_stats().keys())
    assert keys == {
        "tokens_in", "tokens_out", "llm_calls",
        "io_tokens_in", "io_tokens_out",
    }


def test_ensure_counters_initializes_lazily_when_init_skipped():
    """A subclass that doesn't chain ``super().__init__()`` still
    works because ``_ensure_token_counters`` initialises lazily on
    first record."""
    m = _DummyMiddleware.__new__(_DummyMiddleware)  # no __init__
    # Should not raise on first record.
    m._record_internal_llm(
        prompt="x", response="y",
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )
    stats = m.get_internal_token_stats()
    assert stats["tokens_in"] == 1
    assert stats["tokens_out"] == 2
    assert stats["llm_calls"] == 1
