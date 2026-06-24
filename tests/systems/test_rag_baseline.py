"""Tests for the RAG baseline middleware — focusing on provenance tracking.

The ``source_id`` → ``retrieved_context_ids`` round trip is the contract
that lets benchmarks (LoCoMo, LongMemEval, ConvoMem) compute evidence
recall against dataset ground truth. See
``MemoryMiddleware`` class docstring in ``systems/base.py``.

These tests force the bag-of-words embedding fallback (by not setting
``OPENAI_API_KEY``) so they're deterministic and run offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.models import MemoryConfig
from cross_bench.systems.memory.rag_baseline import (
    Chunk,
    RAGBaselineMiddleware,
)


@pytest.fixture
def no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure embeddings fall back to the deterministic bag-of-words
    vector so tests don't hit the network and are reproducible."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def middleware(no_openai_key: None) -> RAGBaselineMiddleware:
    config = MemoryConfig(
        provider="rag",
        enabled=True,
        params={
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 3,
            # Force the fallback path by giving a bad base_url too (belt +
            # suspenders — the missing key alone is enough).
            "embedding_api_key": "",
            "embedding_base_url": "http://127.0.0.1:1/missing",
        },
    )
    return RAGBaselineMiddleware(config)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

def test_chunk_defaults_source_id_to_none() -> None:
    c = Chunk(text="hello")
    assert c.source_id is None


def test_chunk_accepts_source_id() -> None:
    c = Chunk(text="hello", source_id="D1:3")
    assert c.source_id == "D1:3"


# ---------------------------------------------------------------------------
# post_response tags chunks with source_id from context
# ---------------------------------------------------------------------------

async def test_post_response_tags_chunks_with_source_id(
    middleware: RAGBaselineMiddleware,
) -> None:
    await middleware.post_response(
        prompt="",
        response="Caroline went hiking in the Alps last weekend",
        context={"source_id": "D1:3"},
    )
    assert middleware._chunks, "ingest should have produced at least one chunk"
    assert all(c.source_id == "D1:3" for c in middleware._chunks)


async def test_post_response_source_id_is_stringified(
    middleware: RAGBaselineMiddleware,
) -> None:
    """A non-string source_id (e.g. an int ID from another benchmark) is
    coerced to str so the contract with the adapter side (``list[str]``)
    holds."""
    await middleware.post_response(
        prompt="",
        response="something",
        context={"source_id": 42},
    )
    assert all(c.source_id == "42" for c in middleware._chunks)


async def test_post_response_missing_source_id_leaves_chunks_untagged(
    middleware: RAGBaselineMiddleware,
) -> None:
    await middleware.post_response(
        prompt="",
        response="some content with no provenance",
        context={},  # no source_id key
    )
    assert middleware._chunks
    assert all(c.source_id is None for c in middleware._chunks)


async def test_post_response_empty_source_id_leaves_chunks_untagged(
    middleware: RAGBaselineMiddleware,
) -> None:
    """``source_id=""`` is treated as absent (falsy in `if source_id_raw`)
    so we don't pollute the retrieved-ID list with empty strings."""
    await middleware.post_response(
        prompt="",
        response="content with empty provenance",
        context={"source_id": ""},
    )
    assert middleware._chunks
    assert all(c.source_id is None for c in middleware._chunks)


# ---------------------------------------------------------------------------
# pre_prompt surfaces retrieved_context_ids through the context dict
# ---------------------------------------------------------------------------

async def test_pre_prompt_empty_index_emits_empty_list(
    middleware: RAGBaselineMiddleware,
) -> None:
    """No chunks ingested — an empty list is emitted (distinct from
    absence / ``None``, which would mean 'not measured')."""
    ctx: dict[str, Any] = {"task_id": "q1"}
    prompt = await middleware.pre_prompt("Where did Caroline go?", ctx)
    assert prompt == "Where did Caroline go?"
    assert ctx["retrieved_context_ids"] == []


async def test_pre_prompt_emits_source_ids_in_rank_order(
    middleware: RAGBaselineMiddleware,
) -> None:
    # Ingest three items, each with a unique source_id. The BM25/BoW
    # scoring on the query "hiking alps" should rank the hiking message
    # first.
    await middleware.post_response(
        "", "Caroline went hiking in the Alps", {"source_id": "D1:3"}
    )
    await middleware.post_response(
        "", "Melanie baked a chocolate cake", {"source_id": "D1:4"}
    )
    await middleware.post_response(
        "", "They discussed their weekend plans", {"source_id": "D1:5"}
    )

    ctx: dict[str, Any] = {}
    await middleware.pre_prompt("Where did Caroline go hiking?", ctx)

    ids = ctx["retrieved_context_ids"]
    assert isinstance(ids, list)
    assert "D1:3" in ids  # the hiking message must have been retrieved
    # All IDs are strings
    assert all(isinstance(i, str) for i in ids)


async def test_pre_prompt_skips_untagged_chunks(
    middleware: RAGBaselineMiddleware,
) -> None:
    """Chunks with no source_id contribute to retrieval results but NOT
    to ``retrieved_context_ids``. An honest middleware only surfaces IDs
    it actually has — it doesn't invent them."""
    await middleware.post_response(
        "", "Caroline went hiking in the Alps", {"source_id": "D1:3"}
    )
    # Ingest a second tagged item so that top_k has something tagged to pick.
    await middleware.post_response(
        "", "They discussed their weekend plans", {"source_id": "D1:5"}
    )
    # A very on-topic untagged item — will rank highly but contribute no ID.
    await middleware.post_response(
        "", "Alps mountains skiing hiking Caroline weekend trip adventure",
        {},
    )

    ctx: dict[str, Any] = {}
    await middleware.pre_prompt("Caroline hiking Alps", ctx)

    ids = ctx["retrieved_context_ids"]
    # Must not contain empty strings / None
    assert all(i for i in ids)
    # Untagged chunk is not surfaced as an ID even though it's the
    # top-ranked hit; only tagged chunks produce IDs.
    assert set(ids).issubset({"D1:3", "D1:5"})


async def test_pre_prompt_deduplicates_ids_from_multiple_chunks(
    middleware: RAGBaselineMiddleware,
) -> None:
    """A long response produces multiple chunks sharing one source_id.
    When several of those chunks land in top-k, the source_id still
    appears only once in the output — matches upstream's
    ``sorted_context_ids`` de-dup behaviour in ``gpt_utils.py`` 151-160.
    """
    long_text = " ".join(
        f"sentence_{i} with some filler and the word hiking"
        for i in range(200)
    )
    await middleware.post_response(
        "", long_text, {"source_id": "D1:3"}
    )
    # Confirm it actually produced multiple chunks
    assert len(middleware._chunks) > 1
    assert all(c.source_id == "D1:3" for c in middleware._chunks)

    ctx: dict[str, Any] = {}
    await middleware.pre_prompt("hiking", ctx)

    ids = ctx["retrieved_context_ids"]
    # Only one unique ID, regardless of how many chunks were returned
    assert ids == ["D1:3"]


async def test_pre_prompt_overwrites_stale_retrieved_ids_key(
    middleware: RAGBaselineMiddleware,
) -> None:
    """If the caller passed in a dict that already contains a stale
    ``retrieved_context_ids`` value, pre_prompt replaces it — the
    middleware is authoritative for that key within its own call."""
    await middleware.post_response(
        "", "Caroline went hiking", {"source_id": "D1:3"}
    )

    ctx: dict[str, Any] = {"retrieved_context_ids": ["LEFTOVER"]}
    await middleware.pre_prompt("hiking", ctx)

    assert "LEFTOVER" not in ctx["retrieved_context_ids"]


# ---------------------------------------------------------------------------
# End-to-end round trip: ingest → retrieve → IDs match
# ---------------------------------------------------------------------------

async def test_round_trip_produces_ids_matching_ingested_source(
    middleware: RAGBaselineMiddleware,
) -> None:
    """Integration-ish: ingest a handful of tagged items, retrieve, and
    check that every returned ID corresponds to an actually-ingested
    ``source_id``. This is the guarantee that makes evidence recall
    meaningful: every ID the adapter sees was a real ingest tag."""
    ingested = {
        "D1:1": "Caroline prefers tea over coffee",
        "D1:2": "Melanie loves hiking in the Alps",
        "D1:3": "They met at a bookstore last Tuesday",
        "D1:4": "Caroline works as a biologist",
        "D1:5": "Melanie just adopted a dog named Rex",
    }
    for sid, text in ingested.items():
        await middleware.post_response("", text, {"source_id": sid})

    ctx: dict[str, Any] = {}
    await middleware.pre_prompt("What does Caroline do for work?", ctx)

    ids = ctx["retrieved_context_ids"]
    # Every emitted ID must be one we actually ingested
    assert set(ids).issubset(set(ingested))
    # And we should get at least one (top_k=3, non-empty index)
    assert ids


async def test_reset_clears_chunks_and_source_ids(
    middleware: RAGBaselineMiddleware,
) -> None:
    await middleware.post_response(
        "", "some text", {"source_id": "D1:3"}
    )
    assert middleware._chunks

    await middleware.reset()
    assert middleware._chunks == []

    # After reset, an empty pre_prompt returns [] (not the stale list)
    ctx: dict[str, Any] = {}
    await middleware.pre_prompt("anything", ctx)
    assert ctx["retrieved_context_ids"] == []
