"""Tests for the memos (MemoryOS) monkey-patch token hook.

Sibling of ``test_mem0_token_hook.py``. The hook patches the first of
``generate`` / ``generate_response`` / ``chat`` it finds on a known
MemoryOS LLM class. Real ``memos`` isn't required — these tests
fabricate a fake ``memos.llms.openai`` module with an ``OpenAILLM``
class exposing ``generate``, install the hook against it, and assert
the accountant fires (including across the ``run_in_executor`` worker
thread boundary — the 2026-06-15 fix).
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from cross_bench.systems.memory._memos_token_hook import (
    install_memos_token_hook,
    reset_active_accountant,
    run_in_executor_with_accountant,
    set_active_accountant,
)


@pytest.fixture
def fake_memos_openai(monkeypatch):
    """Inject a fake ``memos.llms.openai`` module with a patchable
    ``OpenAILLM.generate``. Yields the class so tests can assert the
    patch took effect.

    Uses ``monkeypatch.setitem`` so it only overrides the three keys it
    needs and auto-restores them on teardown. It deliberately does NOT
    wipe other ``memos.*`` entries — ``test_memos_middleware.py`` installs
    a ``memos.memories.textual.item`` stub at import time, and ``memos``
    is not a real installed package, so a broad delete here would
    permanently break that sibling test module's later cases."""
    pkg = types.ModuleType("memos")
    pkg.__path__ = []
    llms_pkg = types.ModuleType("memos.llms")
    llms_pkg.__path__ = []
    openai_mod = types.ModuleType("memos.llms.openai")

    class FakeOpenAILLM:
        def generate(self, messages, **kwargs):
            return "fake-memos-response"

    openai_mod.OpenAILLM = FakeOpenAILLM
    monkeypatch.setitem(sys.modules, "memos", pkg)
    monkeypatch.setitem(sys.modules, "memos.llms", llms_pkg)
    monkeypatch.setitem(sys.modules, "memos.llms.openai", openai_mod)

    yield FakeOpenAILLM


def test_install_patches_memos_llm(fake_memos_openai):
    assert getattr(fake_memos_openai, "_cb_token_hook_installed", False) is False
    assert install_memos_token_hook() is True
    assert fake_memos_openai._cb_token_hook_installed is True


def test_patched_method_invokes_active_accountant(fake_memos_openai):
    install_memos_token_hook()
    seen: list[tuple[str, str]] = []

    token = set_active_accountant(lambda p, r: seen.append((p, r)))
    try:
        instance = fake_memos_openai()
        result = instance.generate([{"role": "user", "content": "user-prompt"}])
    finally:
        reset_active_accountant(token)

    assert result == "fake-memos-response"
    assert len(seen) == 1
    assert "user-prompt" in seen[0][0]
    assert seen[0][1] == "fake-memos-response"


def test_set_before_executor_does_not_propagate(fake_memos_openai):
    """Same bug as mem0 (test_mem0_token_hook): the accountant set on the
    event-loop thread is invisible inside ``run_in_executor``. Locked so
    the naive pattern is never reintroduced for memos either."""
    install_memos_token_hook()
    seen: list[tuple[str, str]] = []

    async def run():
        loop = asyncio.get_running_loop()
        inst = fake_memos_openai()
        token = set_active_accountant(lambda p, r: seen.append((p, r)))
        try:
            return await loop.run_in_executor(
                None, lambda: inst.generate([{"role": "user", "content": "hi"}])
            )
        finally:
            reset_active_accountant(token)

    assert asyncio.run(run()) == "fake-memos-response"
    assert seen == []


def test_run_in_executor_with_accountant_records(fake_memos_openai):
    """The fix: ``run_in_executor_with_accountant`` sets the accountant on
    the worker thread so MemoryOS's in-thread LLM call is counted."""
    install_memos_token_hook()
    seen: list[tuple[str, str]] = []

    async def run():
        loop = asyncio.get_running_loop()
        inst = fake_memos_openai()
        return await run_in_executor_with_accountant(
            loop,
            lambda p, r: seen.append((p, r)),
            lambda: inst.generate([{"role": "user", "content": "user-prompt"}]),
        )

    assert asyncio.run(run()) == "fake-memos-response"
    assert len(seen) == 1
    assert "user-prompt" in seen[0][0]
    assert seen[0][1] == "fake-memos-response"
