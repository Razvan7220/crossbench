"""Tests for the mem0 monkey-patch token hook.

The hook patches ``mem0.llms.*.LLM.generate_response`` so each
internal mem0 LLM call invokes the active accountant. Real mem0
isn't required — these tests fabricate a fake ``mem0.llms.openai``
module with an ``OpenAILLM`` class exposing ``generate_response``,
install the hook against it, and assert the accountant gets called.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from cross_bench.systems.memory._mem0_token_hook import (
    install_mem0_token_hook,
    reset_active_accountant,
    run_in_executor_with_accountant,
    set_active_accountant,
)


@pytest.fixture
def fake_mem0_openai(monkeypatch):
    """Inject a fake ``mem0.llms.openai`` module with a patchable
    ``OpenAILLM.generate_response``.

    Yields the class so individual tests can assert the patch took
    effect by reading ``cls.generate_response`` and
    ``cls._cb_token_hook_installed``.
    """
    # Reset module cache so each test gets a fresh install.
    for key in list(sys.modules):
        if key.startswith("mem0"):
            del sys.modules[key]

    pkg = types.ModuleType("mem0")
    pkg.__path__ = []  # mark as package
    llms_pkg = types.ModuleType("mem0.llms")
    llms_pkg.__path__ = []
    openai_mod = types.ModuleType("mem0.llms.openai")

    class FakeOpenAILLM:
        def generate_response(self, messages, **kwargs):
            # Pretend mem0's real implementation returned a content
            # string. The accountant should see the messages we sent
            # and this returned value.
            return "fake-response-from-mem0"

    openai_mod.OpenAILLM = FakeOpenAILLM
    sys.modules["mem0"] = pkg
    sys.modules["mem0.llms"] = llms_pkg
    sys.modules["mem0.llms.openai"] = openai_mod

    yield FakeOpenAILLM

    for key in ("mem0.llms.openai", "mem0.llms", "mem0"):
        sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# install_mem0_token_hook
# ---------------------------------------------------------------------------


def test_install_patches_openai_llm(fake_mem0_openai):
    assert getattr(fake_mem0_openai, "_cb_token_hook_installed", False) is False
    ok = install_mem0_token_hook()
    assert ok is True
    assert fake_mem0_openai._cb_token_hook_installed is True


def test_install_idempotent(fake_mem0_openai):
    install_mem0_token_hook()
    method_after_first = fake_mem0_openai.generate_response
    install_mem0_token_hook()
    # Second install should NOT re-wrap (would lead to double-counting).
    assert fake_mem0_openai.generate_response is method_after_first


def test_install_returns_false_when_no_targets(monkeypatch):
    """No mem0 modules importable ⇒ install returns False but doesn't
    raise.

    Simulated via ``__import__`` patching because real ``mem0ai`` may
    well be installed in the dev environment — we want to lock the
    "loud-fail-silent-return-False" contract regardless."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mem0" or name.startswith("mem0."):
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for key in list(sys.modules):
        if key.startswith("mem0"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    assert install_mem0_token_hook() is False


# ---------------------------------------------------------------------------
# Accountant invocation
# ---------------------------------------------------------------------------


def test_patched_method_invokes_active_accountant(fake_mem0_openai):
    install_mem0_token_hook()

    seen: list[tuple[str, str]] = []

    def cb(prompt: str, response: str) -> None:
        seen.append((prompt, response))

    token = set_active_accountant(cb)
    try:
        instance = fake_mem0_openai()
        result = instance.generate_response(
            [
                {"role": "system", "content": "system-prompt"},
                {"role": "user", "content": "user-prompt"},
            ]
        )
    finally:
        reset_active_accountant(token)

    assert result == "fake-response-from-mem0"
    assert len(seen) == 1
    prompt_text, response_text = seen[0]
    assert "system-prompt" in prompt_text
    assert "user-prompt" in prompt_text
    assert response_text == "fake-response-from-mem0"


def test_no_active_accountant_is_a_noop(fake_mem0_openai):
    """When no accountant is registered, the patched method must
    behave identically to the original (no exception, return the
    same value)."""
    install_mem0_token_hook()
    instance = fake_mem0_openai()
    result = instance.generate_response([{"role": "user", "content": "hi"}])
    assert result == "fake-response-from-mem0"


def test_accountant_exception_swallowed(fake_mem0_openai):
    """A buggy accountant must NOT break mem0's call path —
    callback errors are swallowed at debug level."""
    install_mem0_token_hook()

    def cb(prompt: str, response: str) -> None:
        raise RuntimeError("buggy accountant")

    token = set_active_accountant(cb)
    try:
        instance = fake_mem0_openai()
        result = instance.generate_response([{"role": "user", "content": "x"}])
    finally:
        reset_active_accountant(token)

    assert result == "fake-response-from-mem0"


def test_set_then_reset_clears_accountant(fake_mem0_openai):
    install_mem0_token_hook()

    seen: list[Any] = []
    cb = lambda p, r: seen.append((p, r))

    token = set_active_accountant(cb)
    reset_active_accountant(token)

    # After reset, calls should not invoke cb.
    instance = fake_mem0_openai()
    instance.generate_response([{"role": "user", "content": "after-reset"}])
    assert seen == []


# ---------------------------------------------------------------------------
# Executor-thread propagation — the 2026-06-15 zero-internal-tokens bug
# ---------------------------------------------------------------------------


def test_set_before_executor_does_not_propagate(fake_mem0_openai):
    """Regression lock for the bug behind the all-zero
    ``total_memory_tokens_*`` on the 2026-06-15 ``mem0v2-round2-*`` LoCoMo
    runs: setting the accountant on the event-loop thread and then running
    the mem0 call via ``run_in_executor`` records NOTHING, because
    ContextVars do not cross the executor worker-thread boundary. This is
    exactly the pattern the middleware used until the fix; it is locked
    here so the naive pattern is never reintroduced as "equivalent"."""
    install_mem0_token_hook()
    seen: list[tuple[str, str]] = []

    async def run():
        loop = asyncio.get_running_loop()
        instance = fake_mem0_openai()
        token = set_active_accountant(lambda p, r: seen.append((p, r)))
        try:
            return await loop.run_in_executor(
                None,
                lambda: instance.generate_response(
                    [{"role": "user", "content": "hi"}]
                ),
            )
        finally:
            reset_active_accountant(token)

    result = asyncio.run(run())
    assert result == "fake-response-from-mem0"
    assert seen == []  # accountant never fired across the thread hop


def test_run_in_executor_with_accountant_records_across_thread(fake_mem0_openai):
    """The fix: ``run_in_executor_with_accountant`` sets the accountant on
    the worker thread, so the patched ``generate_response`` (which runs on
    that same thread, in-line with mem0's synchronous call) sees it and
    records the internal-LLM tokens."""
    install_mem0_token_hook()
    seen: list[tuple[str, str]] = []

    async def run():
        loop = asyncio.get_running_loop()
        instance = fake_mem0_openai()
        return await run_in_executor_with_accountant(
            loop,
            lambda p, r: seen.append((p, r)),
            lambda: instance.generate_response(
                [{"role": "user", "content": "user-prompt"}]
            ),
        )

    result = asyncio.run(run())
    assert result == "fake-response-from-mem0"
    assert len(seen) == 1
    prompt_text, response_text = seen[0]
    assert "user-prompt" in prompt_text
    assert response_text == "fake-response-from-mem0"


def test_executor_accountant_resets_after_call(fake_mem0_openai):
    """The accountant must not leak to a later naked call on the (pooled)
    worker thread — set→reset is balanced inside the thunk."""
    install_mem0_token_hook()
    seen: list[tuple[str, str]] = []

    async def run():
        loop = asyncio.get_running_loop()
        instance = fake_mem0_openai()
        await run_in_executor_with_accountant(
            loop,
            lambda p, r: seen.append((p, r)),
            lambda: instance.generate_response([{"role": "user", "content": "a"}]),
        )
        # A second call WITHOUT an accountant must record nothing, even if
        # it lands on the same pooled worker thread.
        return await loop.run_in_executor(
            None,
            lambda: instance.generate_response([{"role": "user", "content": "b"}]),
        )

    asyncio.run(run())
    assert len(seen) == 1  # only the first (accountant-scoped) call recorded
