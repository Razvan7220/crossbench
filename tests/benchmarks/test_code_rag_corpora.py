"""Tests for the CodeRAG-Bench corpus loader.

Covers ``cross_bench.benchmarks.code_rag_corpora.load_corpus`` and the
two helpers (``_library_prefix_of``, ``_matches_libraries``) without
touching the real HuggingFace dataset — the loader's HF-dataset
plumbing is exercised indirectly via the BigCodeBench integration
tests, while the filter logic is the algorithmic part worth locking.

The library-documentation corpus uses a specific naming scheme we
ported from CodeRAG-Bench (Wang et al. 2024, arXiv:2406.14497):

  - third-party libs use a top-level prefix: ``numpy.argsort``,
    ``pandas.DataFrame``;
  - Python stdlib lives under ``python.library.<module>``: e.g.
    ``python.library.random.choice``, ``python.library.itertools``.

Both prefix conventions must round-trip through ``_matches_libraries``
for the BigCodeBench wiring to filter by the run's task-set libs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cross_bench.benchmarks.code_rag_corpora import (
    _HF_DATASETS,
    _library_prefix_of,
    _matches_libraries,
    load_corpus,
)
from cross_bench.models import CorpusItem


# ---------------------------------------------------------------------------
# Library prefix extraction
# ---------------------------------------------------------------------------


class TestLibraryPrefixOf:
    def test_third_party_top_level(self):
        assert _library_prefix_of("numpy.argsort") == "numpy"
        assert _library_prefix_of("pandas.DataFrame.merge") == "pandas"

    def test_python_stdlib_keeps_python_prefix(self):
        # The stdlib is uniformly nested under "python.library.<module>".
        # The top-level prefix is "python", not the module name —
        # _matches_libraries handles the deeper match.
        assert _library_prefix_of("python.library.random") == "python"
        assert _library_prefix_of("python.library.random.choice") == "python"

    def test_handles_hash_anchor_suffix(self):
        # CodeRAG-Bench sometimes leaks devdocs.io anchors into the
        # doc_id (e.g. "torch#torch", "matplotlib_configuration_api#matplotlib").
        # Our extractor strips the anchor.
        assert _library_prefix_of("torch#torch") == "torch"
        assert _library_prefix_of(
            "matplotlib_configuration_api#matplotlib"
        ) == "matplotlib_configuration_api"

    def test_no_separators_returns_whole_string(self):
        assert _library_prefix_of("singletoken") == "singletoken"


# ---------------------------------------------------------------------------
# Library matching (the heart of the filter)
# ---------------------------------------------------------------------------


class TestMatchesLibraries:
    def test_third_party_top_level_match(self):
        assert _matches_libraries("numpy.argsort", {"numpy"})
        assert _matches_libraries("pandas.DataFrame", {"numpy", "pandas"})

    def test_third_party_no_match(self):
        assert not _matches_libraries("tensorflow.argmax", {"pytorch"})

    def test_stdlib_module_via_python_prefix(self):
        # ``random`` is a stdlib module. Asking for {"random"} should
        # match ``python.library.random.*`` doc_ids.
        assert _matches_libraries(
            "python.library.random", {"random"}
        )
        assert _matches_libraries(
            "python.library.random.choice", {"random"}
        )
        assert _matches_libraries(
            "python.library.itertools.chain", {"itertools"}
        )

    def test_stdlib_match_only_when_correct_module(self):
        # ``python.library.random`` should NOT match {"itertools"}.
        assert not _matches_libraries(
            "python.library.random.choice", {"itertools"}
        )

    def test_mixed_third_party_and_stdlib_filter(self):
        libs = {"numpy", "random"}
        assert _matches_libraries("numpy.zeros", libs)
        assert _matches_libraries("python.library.random.shuffle", libs)
        assert not _matches_libraries("pandas.DataFrame", libs)
        assert not _matches_libraries(
            "python.library.itertools", libs
        )

    def test_empty_filter_means_keep_everything(self):
        assert _matches_libraries("anything.at.all", set())

    def test_python_top_level_without_library_subprefix_doesnt_match_stdlib(self):
        """Edge case: a doc_id like ``python.something_else`` (not
        ``python.library.*``) — we don't know what stdlib module it
        refers to, so it shouldn't match a stdlib filter."""
        assert not _matches_libraries(
            "python.something_else", {"random"}
        )


# ---------------------------------------------------------------------------
# load_corpus dispatcher
# ---------------------------------------------------------------------------


class TestLoadCorpusDispatcher:
    @pytest.mark.asyncio
    async def test_unknown_corpus_name_raises(self):
        with pytest.raises(ValueError, match="Unknown CodeRAG-Bench corpus"):
            async for _ in load_corpus("does-not-exist"):
                pass

    @pytest.mark.asyncio
    async def test_unimplemented_corpus_raises_with_pointer(self):
        # ``programming-solutions`` is registered but not implemented
        # yet — the loader should raise NotImplementedError pointing
        # at the dataset card.
        with pytest.raises(
            NotImplementedError,
            match="programming-solutions",
        ):
            async for _ in load_corpus("programming-solutions"):
                pass

    def test_all_known_corpora_have_hf_ids(self):
        # Every key in _HF_DATASETS must have a real ``code-rag-bench/``
        # prefix — guards against typos in the registry.
        for name, hf_id in _HF_DATASETS.items():
            assert hf_id.startswith("code-rag-bench/"), (
                f"Corpus {name!r} has bogus HF id {hf_id!r}"
            )

    def test_library_documentation_is_registered(self):
        assert "library-documentation" in _HF_DATASETS
        assert (
            _HF_DATASETS["library-documentation"]
            == "code-rag-bench/library-documentation"
        )


# ---------------------------------------------------------------------------
# library-documentation loader (mocked HF dataset)
# ---------------------------------------------------------------------------


class _FakeHFDataset:
    """Minimal stand-in for ``datasets.load_dataset(...)`` returning a
    list-like of dicts. The real dataset is iterable, has __getitem__,
    and yields {"doc_id": str, "doc_content": str}; this fake mimics
    the iteration contract."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _patched_load_dataset(rows: list[dict]):
    """Patch ``datasets.load_dataset`` (the import path inside
    ``code_rag_corpora``) to return our fake dataset."""
    return patch(
        "datasets.load_dataset",
        return_value=_FakeHFDataset(rows),
    )


class TestLibraryDocumentationLoader:
    @pytest.mark.asyncio
    async def test_yields_corpus_items_with_library_prefixed_source_ids(self):
        rows = [
            {"doc_id": "numpy.argsort", "doc_content": "Sort along axis"},
            {"doc_id": "pandas.DataFrame", "doc_content": "Tabular data"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        assert len(items) == 2
        # source_id format: ``lib:<doc_id>`` so it doesn't collide with
        # per-task ``ctx:N`` IDs from RepoBench/CrossCodeEval.
        assert items[0].source_id == "lib:numpy.argsort"
        assert items[1].source_id == "lib:pandas.DataFrame"

    @pytest.mark.asyncio
    async def test_metadata_carries_library_and_doc_id(self):
        rows = [
            {"doc_id": "numpy.zeros", "doc_content": "Allocate zeros"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        assert items[0].metadata == {
            "library": "numpy",
            "doc_id": "numpy.zeros",
        }

    @pytest.mark.asyncio
    async def test_libraries_filter_includes_third_party(self):
        rows = [
            {"doc_id": "numpy.zeros", "doc_content": "..."},
            {"doc_id": "pandas.DataFrame", "doc_content": "..."},
            {"doc_id": "tensorflow.argmax", "doc_content": "..."},
        ]
        with _patched_load_dataset(rows):
            items = [
                item async for item in load_corpus(
                    "library-documentation", libraries={"numpy", "pandas"},
                )
            ]
        assert {i.metadata["doc_id"] for i in items} == {
            "numpy.zeros", "pandas.DataFrame",
        }

    @pytest.mark.asyncio
    async def test_libraries_filter_includes_stdlib(self):
        rows = [
            {"doc_id": "python.library.random", "doc_content": "Random module"},
            {"doc_id": "python.library.random.choice", "doc_content": "Pick one"},
            {"doc_id": "python.library.itertools.chain", "doc_content": "Chain iters"},
            {"doc_id": "numpy.argsort", "doc_content": "Sort along axis"},
        ]
        with _patched_load_dataset(rows):
            items = [
                item async for item in load_corpus(
                    "library-documentation", libraries={"random", "numpy"},
                )
            ]
        # ``random`` matches both python.library.random and
        # python.library.random.choice. ``numpy`` matches
        # numpy.argsort. ``python.library.itertools`` is dropped.
        ids = {i.metadata["doc_id"] for i in items}
        assert ids == {
            "python.library.random",
            "python.library.random.choice",
            "numpy.argsort",
        }

    @pytest.mark.asyncio
    async def test_skips_empty_doc_content(self):
        rows = [
            {"doc_id": "numpy.argsort", "doc_content": ""},
            {"doc_id": "numpy.zeros", "doc_content": "   "},
            {"doc_id": "numpy.ones", "doc_content": "Real content"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        # Only the row with non-empty content survives.
        assert len(items) == 1
        assert items[0].metadata["doc_id"] == "numpy.ones"

    @pytest.mark.asyncio
    async def test_skips_empty_doc_id(self):
        rows = [
            {"doc_id": "", "doc_content": "Orphaned content"},
            {"doc_id": "numpy.argsort", "doc_content": "Real entry"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        assert len(items) == 1
        assert items[0].metadata["doc_id"] == "numpy.argsort"

    @pytest.mark.asyncio
    async def test_limit_caps_yielded_items(self):
        rows = [
            {"doc_id": f"numpy.{i}", "doc_content": f"doc {i}"}
            for i in range(10)
        ]
        with _patched_load_dataset(rows):
            items = [
                item async for item in load_corpus(
                    "library-documentation", limit=3,
                )
            ]
        assert len(items) == 3
        assert items[0].metadata["doc_id"] == "numpy.0"
        assert items[2].metadata["doc_id"] == "numpy.2"

    @pytest.mark.asyncio
    async def test_no_filter_means_all_libraries(self):
        rows = [
            {"doc_id": "numpy.zeros", "doc_content": "x"},
            {"doc_id": "pandas.DataFrame", "doc_content": "y"},
            {"doc_id": "tensorflow.argmax", "doc_content": "z"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_corpus_item_content_is_byte_exact(self):
        """The doc body must hit memory verbatim — no rewriting,
        normalization, or trimming. Mem0 with infer=True will do its
        own summarization; the RAG baseline relies on byte-fidelity."""
        rows = [
            {"doc_id": "numpy.zeros", "doc_content": "  Allocate \nzeros\n"},
        ]
        with _patched_load_dataset(rows):
            items = [item async for item in load_corpus("library-documentation")]
        assert items[0].content == "  Allocate \nzeros\n"
