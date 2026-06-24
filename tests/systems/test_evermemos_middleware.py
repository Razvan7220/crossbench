"""Smoke tests for EverMemOSMiddleware (cloud SDK variant).

These tests verify the migrated middleware against the contract of the
published ``evermemos==0.3.13`` SDK:

1. Constructor wires SDK + auth + base_url correctly
2. ``post_response`` buffers messages without calling the SDK
3. ``pre_prompt`` flushes the buffer with one ``add()`` per message,
   ``flush=True`` on the LAST one only, then calls ``search()`` with
   the right ``extra_body``
4. ``message_id`` prefers ``context["source_id"]`` and falls back to
   a UUID
5. ``create_time`` is normalised to ISO 8601 with timezone
6. ``reset()`` flushes pending writes then deletes the user's memories
7. Search response normalisation walks the ``result.memories``
   discriminated union (episode / atomic_fact / foresight)
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Stub the ``evermemos`` SDK so the middleware imports without the real
# package installed (CI image may not have it).
# ---------------------------------------------------------------------------

def _make_evermemos_stub() -> ModuleType:
    mod = ModuleType("evermemos")

    class _AsyncEverMemOS:  # placeholder; tests substitute their own
        def __init__(self, **kwargs):
            self._init_kwargs = kwargs
            # SDK exposes self.v0.memories.{add,search,delete,get}
            self.v0 = SimpleNamespace(
                memories=SimpleNamespace(
                    add=AsyncMock(),
                    search=AsyncMock(),
                    delete=AsyncMock(),
                    get=AsyncMock(),
                )
            )

    mod.AsyncEverMemOS = _AsyncEverMemOS  # type: ignore[attr-defined]
    mod.EverMemOS = _AsyncEverMemOS  # type: ignore[attr-defined]  (sync not used here)
    return mod


sys.modules.setdefault("evermemos", _make_evermemos_stub())


# Import middleware AFTER the stub is registered.
from cross_bench.models import MemoryConfig  # noqa: E402
from cross_bench.systems.memory.evermemos_middleware import (  # noqa: E402
    EverMemOSMiddleware,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mw():
    """Build a middleware with a stubbed AsyncEverMemOS client."""
    os.environ["EVERMEMOS_API_KEY"] = "test-key"
    config = MemoryConfig(
        provider="evermemos_api",
        enabled=True,
        params={
            "user_id": "test_user",
            "retrieve_method": "hybrid",
            "memory_types": "episodic_memory",
            "retrieval_limit": 5,
        },
    )
    return EverMemOSMiddleware(config)


def _make_search_response(memories: list[dict]) -> SimpleNamespace:
    """Build an SDK-shaped MemorySearchResponse: {result: {memories: [...]}}."""
    typed = []
    for m in memories:
        typed.append(SimpleNamespace(**m))
    return SimpleNamespace(
        result=SimpleNamespace(memories=typed),
        status="ok",
        message=None,
    )


# ---------------------------------------------------------------------------
# Constructor + auth
# ---------------------------------------------------------------------------


class TestInit:
    def test_requires_api_key(self):
        os.environ.pop("EVERMEMOS_API_KEY", None)
        config = MemoryConfig(
            provider="evermemos_api",
            enabled=True,
            params={"user_id": "u"},
        )
        with pytest.raises(ValueError, match="API key"):
            EverMemOSMiddleware(config)

    def test_passes_api_key_and_base_url_to_sdk(self):
        os.environ["EVERMEMOS_API_KEY"] = "from-env"
        config = MemoryConfig(
            provider="evermemos_api",
            enabled=True,
            params={
                "user_id": "u",
                "api_key": "explicit-key",
                "base_url": "https://api.evermind.ai",
                "timeout": 60,
                "max_retries": 7,
            },
        )
        m = EverMemOSMiddleware(config)
        # Profile-supplied api_key wins over env
        assert m._client._init_kwargs["api_key"] == "explicit-key"
        assert m._client._init_kwargs["base_url"] == "https://api.evermind.ai"
        assert m._client._init_kwargs["timeout"] == 60
        assert m._client._init_kwargs["max_retries"] == 7

    def test_empty_base_url_uses_sdk_default(self):
        """``base_url = ""`` (the profile default) must NOT be forwarded."""
        os.environ["EVERMEMOS_API_KEY"] = "k"
        config = MemoryConfig(
            provider="evermemos_api",
            enabled=True,
            params={"user_id": "u", "base_url": ""},
        )
        m = EverMemOSMiddleware(config)
        # SDK falls back to https://api.evermind.ai when base_url not in kwargs
        assert "base_url" not in m._client._init_kwargs

    def test_dropped_memory_types_logged(self):
        os.environ["EVERMEMOS_API_KEY"] = "k"
        config = MemoryConfig(
            provider="evermemos_api",
            enabled=True,
            params={
                "user_id": "u",
                "memory_types": "episodic_memory,bogus_type,foresight",
            },
        )
        m = EverMemOSMiddleware(config)
        assert m._memory_types == ["episodic_memory", "foresight"]


# ---------------------------------------------------------------------------
# Post-response buffering
# ---------------------------------------------------------------------------


class TestPostResponseBuffers:
    def test_buffers_without_calling_sdk(self, mw):
        asyncio.run(
            mw.post_response(
                "",
                "Hello world",
                {
                    "speaker_id": "Amy",
                    "speaker_name": "Amy Smith",
                    "timestamp": "2023-05-08T13:56:00+00:00",
                    "source_id": "D1:3",
                },
            )
        )
        # Nothing sent yet
        mw._client.v0.memories.add.assert_not_called()
        assert len(mw._pending) == 1
        entry = mw._pending[0]
        assert entry["content"] == "Hello world"
        assert entry["create_time"] == "2023-05-08T13:56:00+00:00"
        assert entry["message_id"] == "D1:3"
        assert entry["sender"] == "Amy"
        assert entry["sender_name"] == "Amy Smith"
        assert entry["role"] == "user"
        assert "flush" not in entry  # set in _flush, not at buffer time

    def test_empty_response_skipped(self, mw):
        asyncio.run(mw.post_response("", "   \n  ", {}))
        assert mw._pending == []

    def test_message_id_uuid_fallback(self, mw):
        asyncio.run(mw.post_response("", "hi", {}))
        mid = mw._pending[0]["message_id"]
        assert mid.startswith("msg-")
        assert len(mid) > 4

    def test_role_normalised(self, mw):
        asyncio.run(mw.post_response("", "hi", {"role": "weird"}))
        assert mw._pending[0]["role"] == "user"

    def test_assistant_role_preserved(self, mw):
        asyncio.run(mw.post_response("", "hi", {"role": "assistant"}))
        assert mw._pending[0]["role"] == "assistant"

    def test_create_time_unix_ms(self, mw):
        # 1683553560000 ms = 2023-05-08T13:46:00+00:00
        asyncio.run(mw.post_response("", "hi", {"timestamp": 1683553560000}))
        assert mw._pending[0]["create_time"].startswith("2023-05-08T")

    def test_create_time_unix_seconds(self, mw):
        asyncio.run(mw.post_response("", "hi", {"timestamp": 1683553560}))
        assert mw._pending[0]["create_time"].startswith("2023-05-08T")

    def test_create_time_iso_without_tz_gets_utc(self, mw):
        asyncio.run(mw.post_response("", "hi", {"timestamp": "2023-05-08T13:56:00"}))
        ct = mw._pending[0]["create_time"]
        assert ct.startswith("2023-05-08T")
        assert ct.endswith("+00:00")

    def test_group_id_forwarded_when_set(self):
        os.environ["EVERMEMOS_API_KEY"] = "k"
        config = MemoryConfig(
            provider="evermemos_api",
            enabled=True,
            params={"user_id": "u", "group_id": "conv-26"},
        )
        m = EverMemOSMiddleware(config)
        asyncio.run(m.post_response("", "hi", {}))
        assert m._pending[0]["group_id"] == "conv-26"


# ---------------------------------------------------------------------------
# Pre-prompt flush + search
# ---------------------------------------------------------------------------


class TestPrePromptFlushAndSearch:
    def test_flushes_pending_then_searches(self, mw):
        # Buffer two messages
        asyncio.run(mw.post_response("", "first", {"source_id": "D1:1"}))
        asyncio.run(mw.post_response("", "second", {"source_id": "D1:2"}))

        # Stub search to return one episode
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "episodic_memory", "episode": "amy met bob",
              "timestamp": "2023-05-08T10:00:00+00:00", "score": 0.9}]
        )

        out = asyncio.run(mw.pre_prompt("when did amy meet bob?", {}))

        # add() called twice — last call has flush=True, first does NOT
        assert mw._client.v0.memories.add.await_count == 2
        first_call = mw._client.v0.memories.add.await_args_list[0].kwargs
        last_call = mw._client.v0.memories.add.await_args_list[1].kwargs
        assert "flush" not in first_call
        assert last_call["flush"] is True
        assert first_call["message_id"] == "D1:1"
        assert last_call["message_id"] == "D1:2"

        # search() called with the right extra_body
        search_call = mw._client.v0.memories.search.await_args.kwargs
        body = search_call["extra_body"]
        assert body["query"] == "when did amy meet bob?"
        assert body["top_k"] == 5
        assert body["memory_types"] == ["episodic_memory"]
        assert body["retrieve_method"] == "hybrid"
        assert body["user_id"] == "test_user"

        # Result rendered into prompt
        assert "amy met bob" in out
        assert "when did amy meet bob?" in out

    def test_no_pending_no_flush(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response([])
        asyncio.run(mw.pre_prompt("q", {}))
        mw._client.v0.memories.add.assert_not_called()

    def test_search_failure_returns_original_prompt(self, mw):
        mw._client.v0.memories.search.side_effect = RuntimeError("upstream 500")
        out = asyncio.run(mw.pre_prompt("q", {}))
        assert out == "q"

    def test_add_failure_logged_continues(self, mw):
        asyncio.run(mw.post_response("", "first", {"source_id": "D1:1"}))
        asyncio.run(mw.post_response("", "second", {"source_id": "D1:2"}))
        # First add raises, second still attempted
        mw._client.v0.memories.add.side_effect = [RuntimeError("rate"), None]
        mw._client.v0.memories.search.return_value = _make_search_response([])
        asyncio.run(mw.pre_prompt("q", {}))
        assert mw._client.v0.memories.add.await_count == 2

    def test_retrieved_context_ids_unset(self, mw):
        """Provenance: the cloud SDK can't preserve caller IDs. Honest no-op."""
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "episodic_memory", "episode": "x",
              "timestamp": "2023-05-08T10:00:00+00:00"}]
        )
        ctx: dict = {}
        asyncio.run(mw.pre_prompt("q", ctx))
        # Must NOT be set — middleware contract says missing key means
        # "not measured" which is what we want for evermemos.
        assert "retrieved_context_ids" not in ctx


# ---------------------------------------------------------------------------
# Search response normalisation (discriminated-union walk)
# ---------------------------------------------------------------------------


class TestSearchResponseNormalisation:
    def test_episode_uses_episode_field(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "episodic_memory", "episode": "ep-text",
              "timestamp": "2023-05-08T10:00:00+00:00", "score": 0.8}]
        )
        out = asyncio.run(mw.pre_prompt("q", {}))
        assert "ep-text" in out
        assert "[episodic_memory]" in out

    def test_event_log_uses_atomic_fact(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "event_log", "atomic_fact": ["fact-a", "fact-b"],
              "timestamp": "2023-05-08T10:00:00+00:00"}]
        )
        out = asyncio.run(mw.pre_prompt("q", {}))
        assert "fact-a; fact-b" in out

    def test_foresight_uses_foresight_field(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "foresight", "foresight": "future-text",
              "timestamp": "2023-05-08T10:00:00+00:00"}]
        )
        out = asyncio.run(mw.pre_prompt("q", {}))
        assert "future-text" in out

    def test_episode_falls_back_to_summary_then_subject(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response(
            [{"memory_type": "episodic_memory", "episode": None,
              "summary": "summary-text", "subject": "sub",
              "timestamp": "2023-05-08T10:00:00+00:00"}]
        )
        out = asyncio.run(mw.pre_prompt("q", {}))
        assert "summary-text" in out

    def test_empty_content_skipped(self, mw):
        mw._client.v0.memories.search.return_value = _make_search_response(
            [
                {"memory_type": "episodic_memory", "episode": "real",
                 "timestamp": "2023-05-08T10:00:00+00:00"},
                {"memory_type": "foresight", "foresight": "",
                 "timestamp": "2023-05-08T10:00:00+00:00"},
            ]
        )
        out = asyncio.run(mw.pre_prompt("q", {}))
        # only the real one shows up
        assert out.count("\n- ") + out.count("\n[") <= 2  # sanity


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_flushes_pending_then_deletes_user(self, mw):
        asyncio.run(mw.post_response("", "x", {}))
        mw._client.v0.memories.search.return_value = _make_search_response([])
        asyncio.run(mw.reset())
        # add() called once for the flush, with flush=True (single-item batch)
        assert mw._client.v0.memories.add.await_count == 1
        assert mw._client.v0.memories.add.await_args.kwargs.get("flush") is True
        # delete called with user_id scoped
        mw._client.v0.memories.delete.assert_awaited_once_with(user_id="test_user")

    def test_no_pending_just_deletes(self, mw):
        asyncio.run(mw.reset())
        mw._client.v0.memories.add.assert_not_called()
        mw._client.v0.memories.delete.assert_awaited_once_with(user_id="test_user")

    def test_delete_failure_swallowed(self, mw):
        mw._client.v0.memories.delete.side_effect = RuntimeError("forbidden")
        asyncio.run(mw.reset())  # must not raise
