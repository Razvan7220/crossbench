"""Unit tests for ``MemOSMiddleware`` (memos_embedded).

Covers the alignment-pass behaviours in ``deviations.md``:

- O4 / O15 — Path B falls back to ``TextualMemoryItem`` per message.
- O6 / O23 — MemReader auto-enable rule (only for ``tree_text``).
- O18 — ``reset()`` calls ``text_mem.delete_all()``.
- O21 — bench-LLM auto-follow contract presence.
- O24 — ``chat_time`` forwarded from ``context["timestamp"]``.
- O26 — profile TOML + ``BUILTIN_PROFILES`` wiring.
- O27 — MEMOS_ANSWER_PROMPT + source_id plumbing + [timestamp] rendering.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Stub out the upstream MemoryOS package so the middleware imports
# cleanly without it installed. Individual tests can swap in richer
# mocks via the fixtures.
_memos_stub = ModuleType("memos")
sys.modules.setdefault("memos", _memos_stub)
for sub in (
    "memos.mem_cube",
    "memos.mem_cube.general",
    "memos.configs",
    "memos.configs.mem_cube",
    "memos.configs.mem_reader",
    "memos.mem_reader",
    "memos.mem_reader.factory",
    "memos.memories",
    "memos.memories.textual",
    "memos.memories.textual.item",
):
    sys.modules.setdefault(sub, ModuleType(sub))

# Stub class symbols expected from upstream so import-time
# ``from memos.x import Y`` calls succeed inside the middleware.
_TextualMemoryItem = MagicMock(name="TextualMemoryItem")
_TextualMemoryMetadata = MagicMock(name="TextualMemoryMetadata")
sys.modules["memos.memories.textual.item"].TextualMemoryItem = _TextualMemoryItem
sys.modules["memos.memories.textual.item"].TextualMemoryMetadata = _TextualMemoryMetadata
sys.modules["memos.mem_cube.general"].GeneralMemCube = MagicMock(name="GeneralMemCube")
sys.modules["memos.configs.mem_cube"].GeneralMemCubeConfig = MagicMock(name="GeneralMemCubeConfig")
sys.modules["memos.configs.mem_reader"].MemReaderConfigFactory = MagicMock(name="MemReaderConfigFactory")
sys.modules["memos.mem_reader.factory"].MemReaderFactory = MagicMock(name="MemReaderFactory")

from cross_bench.config import BUILTIN_PROFILES, _MEMORY_LLM_KEY_MAP  # noqa: E402
from cross_bench.models import MemoryConfig  # noqa: E402
from cross_bench.systems.memory.memos_middleware import MemOSMiddleware  # noqa: E402


def _make_middleware(
    *,
    backend: str = "naive_text",
    use_mem_reader: bool | None = None,
    with_cube: bool = True,
    with_mem_reader: bool = False,
) -> MemOSMiddleware:
    """Bypass __init__ to inject mocked state directly."""
    mw = MemOSMiddleware.__new__(MemOSMiddleware)
    mw._config = MemoryConfig(provider="memos_embedded", enabled=True, params={})
    mw._backend = backend
    mw._retrieval_limit = 5
    mw._data_dir = None
    mw._persist_on_reset = False
    if use_mem_reader is None:
        mw._use_mem_reader = backend == "tree_text"
    else:
        mw._use_mem_reader = bool(use_mem_reader)
    mw._mem_reader_mode = "fine"
    mw._mem_reader_chunk_size = 512
    mw._mem_reader_chunk_overlap = 128
    mw._user_id = "test_user"
    mw._llm_provider = "openai"
    mw._llm_model = "gpt-4o-mini"
    mw._llm_base_url = "https://api.openai.com/v1"
    mw._llm_api_key = "sk-test"
    mw._embedder_provider = "openai"
    mw._embedder_model = "text-embedding-3-small"
    mw._embedder_base_url = "https://api.openai.com/v1"
    mw._embedder_dims = 1536
    mw._use_rest = False
    mw._rest_url = ""
    mw._fallback_memories = []
    mw._initialized = True  # skip _ensure_init

    if with_cube:
        cube = MagicMock()
        cube.text_mem = MagicMock()
        cube.text_mem.search.return_value = []
        cube.text_mem.add.return_value = None
        cube.text_mem.delete_all.return_value = None
        mw._mem_cube = cube
    else:
        mw._mem_cube = None

    mw._mem_reader = MagicMock() if with_mem_reader else None
    if mw._mem_reader is not None:
        mw._mem_reader.get_memory.return_value = [[]]  # one scene, no items

    return mw


# ---------------------------------------------------------------------------
# O26 — Profile TOML + BUILTIN_PROFILES wiring
# ---------------------------------------------------------------------------


class TestProfileWiring:
    """Confirm the profile name is discoverable + the inheritance contract."""

    def test_memos_embedded_listed_in_builtin_profiles(self):
        assert "memos_embedded" in BUILTIN_PROFILES

    def test_memos_embedded_in_key_map(self):
        """O21 — bench-LLM auto-follow must include memos_embedded."""
        assert "memos_embedded" in _MEMORY_LLM_KEY_MAP
        # The mapping must cover the canonical LLM + embedder keys so
        # stacking `-p online -p memos_embedded` inherits gpt-4o, and
        # `-p local -p memos_embedded` inherits Ollama.
        key_map = _MEMORY_LLM_KEY_MAP["memos_embedded"]
        for required in (
            "llm_provider",
            "llm_model",
            "llm_base_url",
            "emb_provider",
            "emb_model",
        ):
            assert required in key_map, f"missing {required} in MemOS key map"

    def test_profile_toml_exists(self):
        from pathlib import Path

        from cross_bench.config import PROFILES_DIR

        path = Path(PROFILES_DIR) / "memos_embedded.toml"
        assert path.is_file(), f"missing profile TOML: {path}"

    def test_profile_loads_with_online_stack(self):
        """`-p online -p memos_embedded` must merge cleanly and trigger
        bench-LLM inheritance into [system.memory]."""
        from cross_bench.config import load_run_config

        cfg = load_run_config(
            benchmark="locomo_original",
            profiles=["online", "memos_embedded"],
        )
        assert cfg.system.memory is not None
        assert cfg.system.memory.provider == "memos_embedded"
        # Inheritance should have populated llm_model from [system.llm]
        params = cfg.system.memory.params
        assert params.get("llm_model") == cfg.system.llm.model
        assert params.get("llm_provider") == cfg.system.llm.provider


# ---------------------------------------------------------------------------
# O6 / O23 — MemReader auto-enable rule
# ---------------------------------------------------------------------------


class TestMemReaderAutoEnable:
    """The init-time auto-rule must match upstream MOSCore.add semantics."""

    @pytest.mark.parametrize(
        "backend,expected",
        [
            ("naive_text", False),
            ("general_text", False),
            ("tree_text", True),
        ],
    )
    def test_auto_default_matches_upstream(self, backend, expected):
        cfg = MemoryConfig(
            provider="memos_embedded",
            enabled=True,
            params={"backend": backend, "llm_api_key": "sk-test"},
        )
        mw = MemOSMiddleware(cfg)
        assert mw._use_mem_reader is expected

    def test_explicit_true_overrides_auto(self):
        cfg = MemoryConfig(
            provider="memos_embedded",
            enabled=True,
            params={"backend": "naive_text", "use_mem_reader": True,
                    "llm_api_key": "sk-test"},
        )
        mw = MemOSMiddleware(cfg)
        assert mw._use_mem_reader is True

    def test_explicit_false_overrides_auto(self):
        cfg = MemoryConfig(
            provider="memos_embedded",
            enabled=True,
            params={"backend": "tree_text", "use_mem_reader": False,
                    "llm_api_key": "sk-test"},
        )
        mw = MemOSMiddleware(cfg)
        assert mw._use_mem_reader is False


# ---------------------------------------------------------------------------
# O24 — chat_time forwarding
# ---------------------------------------------------------------------------


class TestChatTimeForwarding:
    """The per-turn timestamp from context must flow to the MemReader."""

    def test_chat_time_forwarded_when_present(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        asyncio.run(
            mw.post_response(
                "What did Caroline do on May 8?",
                "She attended a support group.",
                {"timestamp": "1:56 pm on 8 May, 2023", "conversation_id": "conv-26"},
            )
        )
        mw._mem_reader.get_memory.assert_called_once()
        scene_data = mw._mem_reader.get_memory.call_args[0][0]
        # One scene, two messages (user + assistant)
        assert len(scene_data) == 1
        assert len(scene_data[0]) == 2
        for msg in scene_data[0]:
            assert msg["chat_time"] == "1:56 pm on 8 May, 2023"

    def test_chat_time_omitted_when_absent(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        asyncio.run(
            mw.post_response(
                "Hello",
                "Hi there",
                {"conversation_id": "conv-1"},  # no timestamp
            )
        )
        scene_data = mw._mem_reader.get_memory.call_args[0][0]
        for msg in scene_data[0]:
            assert "chat_time" not in msg

    def test_session_id_from_conversation_id(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        asyncio.run(
            mw.post_response("p", "r", {"conversation_id": "conv-42"})
        )
        info = mw._mem_reader.get_memory.call_args[1]["info"]
        assert info["session_id"] == "conv-42"
        assert info["user_id"] == "test_user"

    def test_session_id_default_when_missing(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        asyncio.run(mw.post_response("p", "r", {}))
        info = mw._mem_reader.get_memory.call_args[1]["info"]
        assert info["session_id"] == "default"

    def test_non_dict_context_does_not_crash(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        # Engine guarantees dict context, but defensive code path
        # should still survive a bad caller.
        asyncio.run(mw.post_response("p", "r", None))  # type: ignore[arg-type]
        info = mw._mem_reader.get_memory.call_args[1]["info"]
        assert info["session_id"] == "default"


# ---------------------------------------------------------------------------
# O4 / O15 — Path B fallback (TextualMemoryItem per message)
# ---------------------------------------------------------------------------


class TestFallbackTextualMemoryItem:
    """When MemReader is off (e.g. naive_text default), Path B wraps each
    message as a separate TextualMemoryItem and calls text_mem.add."""

    def test_naive_text_default_calls_add_with_two_items(self):
        mw = _make_middleware(backend="naive_text", with_mem_reader=False)
        asyncio.run(mw.post_response("Q?", "A.", {"conversation_id": "c"}))
        mw._mem_cube.text_mem.add.assert_called_once()
        items = mw._mem_cube.text_mem.add.call_args[0][0]
        assert len(items) == 2  # user + assistant

    def test_mem_reader_falls_through_to_path_b_on_failure(self):
        """If MemReader raises, the turn still lands via Path B so data
        isn't lost. Same resilience pattern as the other middlewares."""
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        mw._mem_reader.get_memory.side_effect = RuntimeError("LLM 500")
        asyncio.run(mw.post_response("Q", "A", {"conversation_id": "c"}))
        # Fallback Path B should still have called add.
        mw._mem_cube.text_mem.add.assert_called_once()

    def test_mem_reader_empty_falls_through_to_path_b(self):
        """When MemReader extracts zero items (filler turn), Path B kicks
        in so the raw exchange is still recorded."""
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        mw._mem_reader.get_memory.return_value = [[]]  # zero items extracted
        asyncio.run(mw.post_response("hi", "hello", {"conversation_id": "c"}))
        mw._mem_cube.text_mem.add.assert_called_once()


# ---------------------------------------------------------------------------
# pre_prompt — search path
# ---------------------------------------------------------------------------


class TestPreSearch:
    """pre_prompt must call cube.text_mem.search with top_k=retrieval_limit
    (O2), render the result as a memory block, and fall back gracefully on
    search failure."""

    def test_search_uses_top_k_kwarg(self):
        mw = _make_middleware()
        mw._mem_cube.text_mem.search.return_value = []
        asyncio.run(mw.pre_prompt("hi", {}))
        mw._mem_cube.text_mem.search.assert_called_once()
        _, kwargs = mw._mem_cube.text_mem.search.call_args
        assert kwargs.get("top_k") == mw._retrieval_limit

    def test_search_failure_falls_back_to_keyword_store(self):
        mw = _make_middleware()
        mw._mem_cube.text_mem.search.side_effect = RuntimeError("boom")
        mw._fallback_memories = [{"text": "Q: foo\nA: bar"}]
        result = asyncio.run(mw.pre_prompt("foo", {}))
        # O27: the middleware emits clean bullets — no "Retrieved memories:"
        # header (the answer template owns that header). The fallback bullet
        # must still appear so the question can ground in it.
        assert "- Q: foo\nA: bar" in result

    def test_no_memories_returns_prompt_verbatim(self):
        mw = _make_middleware()
        mw._mem_cube.text_mem.search.return_value = []
        result = asyncio.run(mw.pre_prompt("hi", {}))
        assert result == "hi"

    def test_pre_prompt_does_not_emit_retrieved_memories_header(self):
        """O27 — header lives in the answer template, not the middleware.
        Pre-O27 the middleware prefixed ``Retrieved memories:`` which
        then doubled when the template added its own header."""
        mw = _make_middleware()

        class FakeItem:
            memory = "fact_one"

        mw._mem_cube.text_mem.search.return_value = [FakeItem()]
        result = asyncio.run(mw.pre_prompt("question?", {}))
        # The bullet appears, but no second copy of the header.
        assert "- fact_one" in result
        assert "Retrieved memories:" not in result

    def test_format_memory_handles_string_and_item(self):
        """The _format_memory helper must handle both string returns
        (fallback path) and TextualMemoryItem-like objects (cube path)."""
        assert MemOSMiddleware._format_memory("raw text") == "raw text"

        class FakeItem:
            memory = "hello"

        assert MemOSMiddleware._format_memory(FakeItem()) == "hello"


# ---------------------------------------------------------------------------
# O18 — reset() semantics
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_calls_delete_all(self):
        mw = _make_middleware()
        asyncio.run(mw.reset())
        mw._mem_cube.text_mem.delete_all.assert_called_once()

    def test_reset_clears_fallback_store(self):
        mw = _make_middleware()
        mw._fallback_memories = [{"text": "x"}]
        asyncio.run(mw.reset())
        assert mw._fallback_memories == []

    def test_reset_no_cube_does_not_crash(self):
        mw = _make_middleware(with_cube=False)
        mw._fallback_memories = [{"text": "x"}]
        asyncio.run(mw.reset())
        assert mw._fallback_memories == []

    def test_persist_on_reset_false_skips_dump(self):
        mw = _make_middleware()
        mw._persist_on_reset = False
        asyncio.run(mw.reset())
        mw._mem_cube.dump.assert_not_called()


# ---------------------------------------------------------------------------
# _extract_via_mem_reader signature
# ---------------------------------------------------------------------------


class TestExtractViaMemReader:
    """Direct unit tests on the extraction helper, bypassing executors."""

    def test_returns_flat_list(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        item_a = MagicMock(name="item_a")
        item_b = MagicMock(name="item_b")
        mw._mem_reader.get_memory.return_value = [[item_a, item_b]]
        items = mw._extract_via_mem_reader("p", "r", "sess-1", None)
        assert items == [item_a, item_b]

    def test_passes_chat_kind(self):
        mw = _make_middleware(backend="tree_text", with_mem_reader=True)
        mw._mem_reader.get_memory.return_value = [[]]
        mw._extract_via_mem_reader("p", "r", "sess", None)
        kwargs = mw._mem_reader.get_memory.call_args[1]
        assert kwargs["type"] == "chat"
        assert kwargs["mode"] == "fine"  # O25 — sync default


# ---------------------------------------------------------------------------
# O27 — MEMOS_ANSWER_PROMPT + source_id plumbing + [timestamp] rendering
# ---------------------------------------------------------------------------


class TestAnswerPromptRegistry:
    """The provider-prompt registry must point ``memos_embedded`` at the
    MEMOS_ANSWER_PROMPT — NOT DEFAULT_ANSWER_PROMPT (the pre-O27 bug)."""

    def test_registry_points_at_memos_prompt(self):
        from cross_bench.evaluation.provider_prompts import (
            MEMOS_ANSWER_PROMPT,
            get_answer_prompt,
        )

        assert get_answer_prompt("memos_embedded") is MEMOS_ANSWER_PROMPT

    def test_memos_prompt_demands_final_answer_marker(self):
        """The ``FINAL ANSWER:`` marker is what
        ``_extract_final_answer`` strips before F1 grading. Without
        it, F1 drowns in CoT preamble (same trap as L1, M39, H47)."""
        from cross_bench.evaluation.provider_prompts import (
            MEMOS_ANSWER_PROMPT,
        )

        assert "FINAL ANSWER:" in MEMOS_ANSWER_PROMPT

    def test_memos_prompt_demands_cat5_abstention_substring(self):
        """Upstream LoCoMo accepts ``"no information available"`` as a
        cat-5 abstention. The prompt must instruct the model to emit
        that exact string."""
        from cross_bench.evaluation.provider_prompts import (
            MEMOS_ANSWER_PROMPT,
        )

        assert "no information available" in MEMOS_ANSWER_PROMPT

    def test_memos_prompt_names_bracketed_timestamp_contract(self):
        """The prompt must reference the bracketed-timestamp format so
        the LLM grounds temporal questions in the conversation's clock
        (anchoring fix; same shape as H46 / M38)."""
        from cross_bench.evaluation.provider_prompts import (
            MEMOS_ANSWER_PROMPT,
        )

        # We use "[<time>]" as the explicit contract; the prompt should
        # warn against substituting today's date.
        assert "bracketed timestamp" in MEMOS_ANSWER_PROMPT.lower()
        assert "today" in MEMOS_ANSWER_PROMPT.lower()


class TestMetaPrefixCodec:
    """Direct unit tests on the meta-prefix encode/decode helpers."""

    def test_encode_with_both_fields(self):
        from cross_bench.systems.memory.memos_middleware import (
            _encode_meta_prefix,
        )

        out = _encode_meta_prefix("D1:3", "1:56 pm on 8 May, 2023")
        assert out.startswith("<<MEMOS_META src=D1:3 | ts=")
        assert out.endswith(">>\n")
        assert "1:56 pm on 8 May, 2023" in out

    def test_encode_returns_empty_when_both_missing(self):
        from cross_bench.systems.memory.memos_middleware import (
            _encode_meta_prefix,
        )

        assert _encode_meta_prefix(None, None) == ""
        assert _encode_meta_prefix("", "") == ""

    def test_decode_roundtrip(self):
        from cross_bench.systems.memory.memos_middleware import (
            _decode_meta_prefix,
            _encode_meta_prefix,
        )

        prefix = _encode_meta_prefix("D5:42", "2023-05-08T13:56:00")
        body, src, ts = _decode_meta_prefix(prefix + "the actual body")
        assert body == "the actual body"
        assert src == "D5:42"
        assert ts == "2023-05-08T13:56:00"

    def test_decode_passes_through_when_no_prefix(self):
        """Items produced by MemReader (tree_text path) have no prefix
        — decoder must leave them alone."""
        from cross_bench.systems.memory.memos_middleware import (
            _decode_meta_prefix,
        )

        body, src, ts = _decode_meta_prefix("raw fact with no prefix")
        assert body == "raw fact with no prefix"
        assert src is None
        assert ts is None

    def test_decode_handles_timestamp_only(self):
        """source_id may be absent when LoCoMo's dia_id isn't supplied
        (other benchmarks). The encoder still emits the header so the
        timestamp survives."""
        from cross_bench.systems.memory.memos_middleware import (
            _decode_meta_prefix,
            _encode_meta_prefix,
        )

        prefix = _encode_meta_prefix(None, "2023-05-08")
        body, src, ts = _decode_meta_prefix(prefix + "body")
        assert src is None
        assert ts == "2023-05-08"
        assert body == "body"


class TestSourceIdPlumbing:
    """Path B ingest must encode source_id + timestamp into the
    ``TextualMemoryItem.memory`` string so retrieval can surface them."""

    def test_post_response_prepends_meta_on_path_b(self):
        mw = _make_middleware(backend="naive_text", with_mem_reader=False)
        item_ctor = sys.modules[
            "memos.memories.textual.item"
        ].TextualMemoryItem
        item_ctor.reset_mock()
        asyncio.run(
            mw.post_response(
                "what did Caroline do?",
                "she went to support group",
                {
                    "conversation_id": "conv-26",
                    "source_id": "D1:3",
                    "timestamp": "1:56 pm on 8 May, 2023",
                },
            )
        )
        assert item_ctor.call_count == 2
        memory_strs = [c.kwargs.get("memory", "") for c in item_ctor.call_args_list]
        for ms in memory_strs:
            assert ms.startswith("<<MEMOS_META src=D1:3 | ts=1:56 pm on 8 May, 2023>>\n")

    def test_post_response_emits_no_prefix_when_metadata_absent(self):
        """If the engine doesn't supply source_id or timestamp (e.g.
        other benchmarks), the memory string must NOT carry an empty
        sentinel — it would clutter retrieval and gain nothing."""
        mw = _make_middleware(backend="naive_text", with_mem_reader=False)
        item_ctor = sys.modules[
            "memos.memories.textual.item"
        ].TextualMemoryItem
        item_ctor.reset_mock()
        asyncio.run(mw.post_response("Q", "A", {"conversation_id": "c"}))
        memory_strs = [c.kwargs.get("memory", "") for c in item_ctor.call_args_list]
        for ms in memory_strs:
            assert not ms.startswith("<<MEMOS_META")

    def test_pre_prompt_populates_retrieved_context_ids(self):
        from cross_bench.systems.memory.memos_middleware import (
            _encode_meta_prefix,
        )

        mw = _make_middleware()

        class FakeItem:
            def __init__(self, mem):
                self.memory = mem

        mw._mem_cube.text_mem.search.return_value = [
            FakeItem(
                _encode_meta_prefix("D1:3", "1:56 pm on 8 May, 2023")
                + "I went to support group"
            ),
            FakeItem(
                _encode_meta_prefix("D1:5", "2:00 pm on 8 May, 2023")
                + "It was empowering"
            ),
        ]
        ctx: dict = {}
        result = asyncio.run(mw.pre_prompt("what did I do?", ctx))
        assert ctx["retrieved_context_ids"] == ["D1:3", "D1:5"]
        # Body is rendered with [ts] prefix, no double header.
        assert "- [1:56 pm on 8 May, 2023] I went to support group" in result
        assert "- [2:00 pm on 8 May, 2023] It was empowering" in result
        assert "Retrieved memories:" not in result

    def test_pre_prompt_dedupes_source_ids_in_rank_order(self):
        """Duplicates from over-retrieval shouldn't bloat the
        provenance list. Rank order is preserved."""
        from cross_bench.systems.memory.memos_middleware import (
            _encode_meta_prefix,
        )

        mw = _make_middleware()

        class FakeItem:
            def __init__(self, mem):
                self.memory = mem

        mw._mem_cube.text_mem.search.return_value = [
            FakeItem(_encode_meta_prefix("D1:3", "ts1") + "a"),
            FakeItem(_encode_meta_prefix("D1:5", "ts2") + "b"),
            FakeItem(_encode_meta_prefix("D1:3", "ts1") + "a-dup"),
        ]
        ctx: dict = {}
        asyncio.run(mw.pre_prompt("q?", ctx))
        assert ctx["retrieved_context_ids"] == ["D1:3", "D1:5"]

    def test_pre_prompt_empty_search_populates_empty_list(self):
        """Honest provenance: empty list signals 'retriever found
        nothing' (measured=True with zero hits). Absent key would
        signal 'not measured' — wrong here."""
        mw = _make_middleware()
        mw._mem_cube.text_mem.search.return_value = []
        ctx: dict = {}
        asyncio.run(mw.pre_prompt("q?", ctx))
        assert ctx.get("retrieved_context_ids") == []

    def test_pre_prompt_handles_items_without_prefix(self):
        """MemReader-produced items have no O27 prefix — they should
        render as plain ``- <body>`` lines (no timestamp), and their
        source_id is missing (None), so it just doesn't appear in
        retrieved_context_ids."""
        mw = _make_middleware()

        class FakeItem:
            def __init__(self, mem):
                self.memory = mem

        mw._mem_cube.text_mem.search.return_value = [
            FakeItem("plain mem reader fact, no prefix"),
        ]
        ctx: dict = {}
        result = asyncio.run(mw.pre_prompt("q?", ctx))
        assert ctx["retrieved_context_ids"] == []
        assert "- plain mem reader fact, no prefix" in result
        # No timestamp bracket since the prefix wasn't there.
        assert "- [" not in result

    def test_fallback_path_also_carries_meta(self):
        """When the cube is unavailable, the in-memory keyword store
        must still surface source_ids + timestamps via the same
        prefix scheme — otherwise the fallback would silently drop
        provenance and recall would appear 'not measured'."""
        from cross_bench.systems.memory.memos_middleware import (
            _encode_meta_prefix,
        )

        mw = _make_middleware()
        mw._mem_cube.text_mem.search.side_effect = RuntimeError("down")
        mw._fallback_memories = [
            {
                "text": _encode_meta_prefix("D1:3", "8 May 2023")
                + "Q: x\nA: support group"
            }
        ]
        ctx: dict = {}
        result = asyncio.run(mw.pre_prompt("support group", ctx))
        assert ctx["retrieved_context_ids"] == ["D1:3"]
        assert "[8 May 2023]" in result


class TestEmbedderConfig:
    """Ollama embedder plumbing (1536-dim GTE profile).

    MemOS' native Ollama embedder expects ``api_base`` without ``/v1``;
    OpenAI-compatible providers must keep using ``universal_api``.  The
    Qdrant collection size must follow ``embedder_dims`` rather than the
    old hardcoded 1536.
    """

    def test_ollama_uses_native_backend(self):
        mw = _make_middleware(backend="general_text")
        mw._embedder_provider = "ollama"
        mw._embedder_model = "gte-qwen2:8t"
        mw._embedder_base_url = "http://localhost:11434"
        mw._embedder_dims = 1536

        block = mw._embedder_config_block()

        assert block == {
            "backend": "ollama",
            "config": {
                "model_name_or_path": "gte-qwen2:8t",
                "api_base": "http://localhost:11434",
            },
        }

    def test_general_text_uses_native_embedder_and_dims(self):
        mw = _make_middleware(backend="general_text")
        mw._embedder_provider = "ollama"
        mw._embedder_model = "gte-qwen2:8t"
        mw._embedder_base_url = "http://localhost:11434"
        mw._embedder_dims = 1536

        cfg = mw._build_text_mem_inner_config()

        assert cfg["embedder"]["backend"] == "ollama"
        assert cfg["embedder"]["config"]["api_base"] == "http://localhost:11434"
        assert cfg["vector_db"]["config"]["vector_dimension"] == 1536

    def test_dims_follow_non_default_embedder(self):
        """A 768-dim embedder must not inherit Qdrant's old 1536 default."""
        mw = _make_middleware(backend="general_text")
        mw._embedder_provider = "ollama"
        mw._embedder_model = "nomic-embed-text:8t"
        mw._embedder_base_url = "http://localhost:11434"
        mw._embedder_dims = 768

        cfg = mw._build_text_mem_inner_config()

        assert cfg["vector_db"]["config"]["vector_dimension"] == 768

    def test_openai_keeps_universal_api_backend(self):
        mw = _make_middleware(backend="general_text")
        mw._embedder_provider = "openai"
        mw._embedder_model = "text-embedding-3-small"
        mw._embedder_base_url = "https://api.openai.com/v1"
        mw._embedder_dims = 1536

        block = mw._embedder_config_block()

        assert block["backend"] == "universal_api"
        assert block["config"]["provider"] == "openai"
        assert block["config"]["base_url"] == "https://api.openai.com/v1"
        assert block["config"]["embedding_dims"] == 1536
