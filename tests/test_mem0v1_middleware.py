"""Smoke tests for Mem0V1EmbeddedMiddleware and Mem0V1ApiMiddleware.

These tests verify the v1.0.x API surface:
1. Local search uses top-level user_id + limit (NOT filters + top_k)
2. Cloud search uses top_k (per the v1 MemoryClient kwarg convention)
3. Response parsing handles both {"results": [...]} and bare list forms
4. metadata.timestamp is preferred over created_at in the rendered context
5. Reset on cloud uses scoped delete_all, never the global reset()
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Stub out the mem0 package so the middlewares import without mem0 installed
_mem0_stub = ModuleType("mem0")
_mem0_stub.Memory = MagicMock  # type: ignore[attr-defined]
_mem0_stub.MemoryClient = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("mem0", _mem0_stub)

from cross_bench.models import MemoryConfig  # noqa: E402
from cross_bench.systems.memory.mem0v1_middleware import (  # noqa: E402
    Mem0V1ApiMiddleware,
    Mem0V1EmbeddedMiddleware,
)


# ---------------------------------------------------------------------------
# Embedded (local Memory)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory():
    m = MagicMock()
    m.search.return_value = {"results": []}
    m.add.return_value = {"results": []}
    m.delete_all.return_value = None
    return m


@pytest.fixture
def embedded(mock_memory):
    mw = Mem0V1EmbeddedMiddleware.__new__(Mem0V1EmbeddedMiddleware)
    mw._config = MemoryConfig(
        provider="mem0v1_embedded",
        enabled=True,
        params={"user_id": "test_user"},
    )
    mw._user_id = "test_user"
    mw._agent_id = None
    mw._retrieval_limit = 20
    mw._memory = mock_memory
    return mw


class TestEmbeddedSearch:
    def test_uses_top_level_user_id_and_limit(self, embedded, mock_memory):
        asyncio.run(embedded.pre_prompt("Hello", {}))

        kwargs = mock_memory.search.call_args[1]
        # v1: top-level user_id
        assert kwargs["user_id"] == "test_user"
        assert "filters" not in kwargs
        # v1: "limit", not "top_k"
        assert kwargs["limit"] == 20
        assert "top_k" not in kwargs

    def test_agent_id_included_when_set(self, embedded, mock_memory):
        embedded._agent_id = "agent_42"
        asyncio.run(embedded.pre_prompt("Hello", {}))

        kwargs = mock_memory.search.call_args[1]
        assert kwargs["agent_id"] == "agent_42"

    def test_returns_memories(self, embedded, mock_memory):
        mock_memory.search.return_value = {
            "results": [{"memory": "Jon likes natural light"}]
        }
        out = asyncio.run(embedded.pre_prompt("Tell me about Jon", {}))
        assert "Jon likes natural light" in out
        assert "Tell me about Jon" in out

    def test_metadata_timestamp_preferred_over_created_at(self, embedded, mock_memory):
        mock_memory.search.return_value = {
            "results": [
                {
                    "memory": "Amy likes hiking",
                    "created_at": "2026-04-22T11:24:32+00:00",
                    "metadata": {"timestamp": "2023-05-08 13:56:00"},
                }
            ]
        }
        out = asyncio.run(embedded.pre_prompt("About Amy", {}))
        assert "[2023-05-08 13:56:00]" in out
        assert "2026-04-22" not in out

    def test_exception_returns_original_prompt(self, embedded, mock_memory):
        mock_memory.search.side_effect = RuntimeError("boom")
        ctx: dict = {}
        out = asyncio.run(embedded.pre_prompt("Hello", ctx))
        assert out == "Hello"
        # honest empty: search was attempted, got nothing
        assert ctx["retrieved_context_ids"] == []

    def test_source_id_extracted_from_search_hits(self, embedded, mock_memory):
        mock_memory.search.return_value = {
            "results": [
                {"memory": "fact 1", "metadata": {"source_id": "D1:3"}},
                {"memory": "fact 2", "metadata": {"source_id": "D1:5"}},
                {"memory": "fact 3", "metadata": {"source_id": "D1:3"}},  # dup
                {"memory": "fact 4", "metadata": {}},  # no source_id
            ]
        }
        ctx: dict = {}
        asyncio.run(embedded.pre_prompt("q", ctx))
        # rank-ordered, de-duplicated; missing source_ids skipped
        assert ctx["retrieved_context_ids"] == ["D1:3", "D1:5"]


class TestEmbeddedAdd:
    def test_empty_prompt_sends_single_user_message(self, embedded, mock_memory):
        asyncio.run(embedded.post_response("", "Jon: Hello", {}))
        kwargs = mock_memory.add.call_args[1]
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Jon: Hello"}
        assert kwargs["user_id"] == "test_user"

    def test_speaker_and_timestamp_metadata(self, embedded, mock_memory):
        asyncio.run(
            embedded.post_response(
                "",
                "Hello",
                {"speaker_id": "Amy", "timestamp": "2023-05-08 13:56:00"},
            )
        )
        kwargs = mock_memory.add.call_args[1]
        assert "[2023-05-08 13:56:00 | Amy]" in kwargs["messages"][0]["content"]
        assert kwargs["metadata"] == {
            "timestamp": "2023-05-08 13:56:00",
            "speaker_id": "Amy",
        }

    def test_source_id_written_to_metadata(self, embedded, mock_memory):
        asyncio.run(
            embedded.post_response(
                "",
                "Hello",
                {
                    "speaker_id": "Amy",
                    "timestamp": "2023-05-08 13:56:00",
                    "source_id": "D1:3",
                },
            )
        )
        kwargs = mock_memory.add.call_args[1]
        assert kwargs["metadata"]["source_id"] == "D1:3"

    def test_exception_is_swallowed(self, embedded, mock_memory):
        mock_memory.add.side_effect = RuntimeError("boom")
        asyncio.run(embedded.post_response("", "Hello", {}))  # must not raise


class TestEmbeddedReset:
    def test_reset_uses_scoped_delete_all(self, embedded, mock_memory):
        asyncio.run(embedded.reset())
        mock_memory.delete_all.assert_called_once_with(user_id="test_user")


# ---------------------------------------------------------------------------
# Cloud (MemoryClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    c = MagicMock()
    c.search.return_value = {"results": []}
    c.add.return_value = {"results": []}
    c.delete_all.return_value = None
    return c


@pytest.fixture
def api(mock_client):
    mw = Mem0V1ApiMiddleware.__new__(Mem0V1ApiMiddleware)
    mw._config = MemoryConfig(
        provider="mem0v1_api",
        enabled=True,
        params={"user_id": "cloud_user"},
    )
    mw._user_id = "cloud_user"
    mw._agent_id = None
    mw._app_id = None
    mw._retrieval_limit = 15
    mw._output_format = "v1.1"
    mw._async_mode = None
    mw._client = mock_client
    return mw


class TestApiSearch:
    def test_uses_top_k_and_scope_kwargs(self, api, mock_client):
        asyncio.run(api.pre_prompt("Hello", {}))

        kwargs = mock_client.search.call_args[1]
        assert kwargs["top_k"] == 15
        assert kwargs["user_id"] == "cloud_user"
        assert kwargs["output_format"] == "v1.1"
        # query is passed as a kwarg
        assert mock_client.search.call_args[1]["query"] == "Hello"

    def test_agent_and_app_scopes(self, api, mock_client):
        api._agent_id = "agent_1"
        api._app_id = "app_x"
        asyncio.run(api.pre_prompt("Hello", {}))

        kwargs = mock_client.search.call_args[1]
        assert kwargs["agent_id"] == "agent_1"
        assert kwargs["app_id"] == "app_x"

    def test_bare_list_response_is_handled(self, api, mock_client):
        mock_client.search.return_value = [{"memory": "Jon likes tea"}]
        out = asyncio.run(api.pre_prompt("About Jon", {}))
        assert "Jon likes tea" in out

    def test_exception_returns_original_prompt(self, api, mock_client):
        mock_client.search.side_effect = RuntimeError("boom")
        ctx: dict = {}
        out = asyncio.run(api.pre_prompt("Hello", ctx))
        assert out == "Hello"
        assert ctx["retrieved_context_ids"] == []

    def test_source_id_extracted_from_cloud_search(self, api, mock_client):
        mock_client.search.return_value = {
            "results": [
                {"memory": "fact A", "metadata": {"source_id": "D2:1"}},
                {"memory": "fact B", "metadata": {"source_id": "D2:2"}},
            ]
        }
        ctx: dict = {}
        asyncio.run(api.pre_prompt("q", ctx))
        assert ctx["retrieved_context_ids"] == ["D2:1", "D2:2"]


class TestApiAdd:
    def test_add_scope_kwargs(self, api, mock_client):
        asyncio.run(api.post_response("", "Hello", {}))
        args, kwargs = mock_client.add.call_args
        # messages is passed positionally
        assert args[0] == [{"role": "user", "content": "Hello"}]
        assert kwargs["user_id"] == "cloud_user"
        assert kwargs["output_format"] == "v1.1"
        # async_mode not set unless config enables it
        assert "async_mode" not in kwargs

    def test_async_mode_forwarded(self, api, mock_client):
        api._async_mode = False
        asyncio.run(api.post_response("", "Hello", {}))
        kwargs = mock_client.add.call_args[1]
        assert kwargs["async_mode"] is False

    def test_metadata_forwarded(self, api, mock_client):
        asyncio.run(
            api.post_response(
                "",
                "Hello",
                {"speaker_id": "Bob", "timestamp": "2024-01-01 09:00"},
            )
        )
        kwargs = mock_client.add.call_args[1]
        assert kwargs["metadata"] == {
            "timestamp": "2024-01-01 09:00",
            "speaker_id": "Bob",
        }

    def test_source_id_forwarded_via_metadata(self, api, mock_client):
        asyncio.run(
            api.post_response(
                "",
                "Hello",
                {
                    "speaker_id": "Bob",
                    "timestamp": "2024-01-01 09:00",
                    "source_id": "D7:11",
                },
            )
        )
        assert mock_client.add.call_args[1]["metadata"]["source_id"] == "D7:11"

    def test_exception_is_swallowed(self, api, mock_client):
        mock_client.add.side_effect = RuntimeError("boom")
        asyncio.run(api.post_response("", "Hello", {}))  # must not raise


class TestApiReset:
    def test_reset_uses_scoped_delete_all(self, api, mock_client):
        asyncio.run(api.reset())
        mock_client.delete_all.assert_called_once_with(user_id="cloud_user")
        # never call the nuclear reset()
        mock_client.reset.assert_not_called()

    def test_reset_exception_is_swallowed(self, api, mock_client):
        mock_client.delete_all.side_effect = RuntimeError("boom")
        asyncio.run(api.reset())  # must not raise


# ---------------------------------------------------------------------------
# Version-mismatch guard (Bug 1 from the 2026-05-21 locomo-mem0v1 diagnosis)
# ---------------------------------------------------------------------------


class TestMem0V1VersionGuard:
    """``_assert_mem0_v1_installed`` raises on v2 install.

    Background: ``mem0ai`` v1 and v2 share the import name ``mem0`` but
    have incompatible ``Memory.search()`` signatures (v1 takes top-level
    ``user_id`` / ``limit``; v2 requires ``filters={...}`` / ``top_k``).
    pyproject originally only shipped a ``memory-mem0`` extra pinning
    v2, so every ``mem0v1_embedded`` run on the orchestrator hit the
    v2 validator error inside the silent ``except Exception`` block in
    ``pre_prompt`` and produced empty retrieval on every task. See the
    locomo-mem0v1-20260521-200222 run.
    """

    def test_raises_when_v2_installed(self, monkeypatch):
        from cross_bench.systems.memory import mem0v1_middleware as mod

        monkeypatch.setattr(
            "importlib.metadata.version",
            lambda name: "2.0.0" if name == "mem0ai" else "0.0.0",
        )
        with pytest.raises(RuntimeError, match="found mem0ai==2.0.0"):
            mod._assert_mem0_v1_installed()

    def test_raises_when_pkg_not_installed(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        from cross_bench.systems.memory import mem0v1_middleware as mod

        def _raise(_name):
            raise PackageNotFoundError("mem0ai")

        monkeypatch.setattr("importlib.metadata.version", _raise)
        with pytest.raises(RuntimeError, match="requires the ``mem0ai`` package"):
            mod._assert_mem0_v1_installed()

    def test_accepts_v1(self, monkeypatch):
        from cross_bench.systems.memory import mem0v1_middleware as mod

        monkeypatch.setattr(
            "importlib.metadata.version",
            lambda name: "1.0.11" if name == "mem0ai" else "0.0.0",
        )
        # Should not raise.
        mod._assert_mem0_v1_installed()

    def test_error_message_points_at_separate_install(self, monkeypatch):
        """The error has to be actionable — mention the v1 extra and the
        coexistence rule so a future operator doesn't repeat the silent
        failure."""
        from cross_bench.systems.memory import mem0v1_middleware as mod

        monkeypatch.setattr(
            "importlib.metadata.version",
            lambda name: "2.0.5",
        )
        with pytest.raises(RuntimeError) as exc_info:
            mod._assert_mem0_v1_installed()
        msg = str(exc_info.value)
        assert "memory-mem0v1" in msg
        assert "separate" in msg.lower() or "cannot coexist" in msg.lower()
