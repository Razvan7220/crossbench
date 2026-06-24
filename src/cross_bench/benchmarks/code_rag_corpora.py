"""Loaders for CodeRAG-Bench retrieval corpora (Wang et al. 2024).

Faithful integration of the corpora shipped at
https://github.com/code-rag-bench/code-rag-bench, paper at
https://arxiv.org/abs/2406.14497.

Five corpora are published on the HuggingFace Hub under
``code-rag-bench/*``. We expose a unified ``load_corpus`` streaming
loader that yields :class:`CorpusItem` records ready for ingestion via
:meth:`MemoryMiddleware.post_response`. The dominant use today is
``library-documentation`` (34,003 entries scraped from devdocs.io,
covering 13 third-party libraries plus the Python stdlib via the
``python.library.*`` prefix).

Verified upstream schemas (HF dataset cards, 2026-05-06):
  - ``library-documentation``  : ``{doc_id, doc_content}``
  - ``programming-solutions``  : ``{problem, solution}`` (TODO: not used yet)
  - ``online-tutorials``       : ``{title, content, url}``
  - ``stackoverflow-posts``    : large; not loaded by default
  - ``github-repos-python``    : large; not loaded by default

The library-documentation corpus contains exactly 34,003 rows split as:

    python (stdlib)             8,447
    matplotlib                  6,383
    tensorflow                  4,216
    django                      2,709
    sklearn                     2,677
    torch                       2,489
    pandas                      2,409
    numpy                       2,181
    skimage                       753
    pygame                        722
    flask                         438
    werkzeug                      425
    django_rest_framework         132

Plus a handful of malformed prefixes (``matplotlib_configuration_api``,
``torch#torch``) that we normalize on load.

Library prefix matching
-----------------------
``doc_id`` is dot-separated with the library name first, e.g.
``numpy.argsort`` or ``python.library.random.choice``. The
``libraries`` filter on :func:`load_corpus` accepts a set of
**logical** library names (``{"numpy", "random"}``) and matches them
against the corpus via two strategies:

* Third-party libs (numpy, pandas, …) are matched against the
  top-level prefix: ``doc_id.split(".")[0] == lib``.
* Python stdlib modules (``random``, ``itertools``, ``json``, …) are
  matched against the ``python.library.<lib>`` prefix.

Both strategies are tried for each requested library. A library that
doesn't match anything in the corpus simply contributes no docs;
it's not an error.

Example::

    from cross_bench.benchmarks.code_rag_corpora import load_corpus

    async for item in load_corpus(
        "library-documentation",
        libraries={"numpy", "pandas", "random"},
    ):
        await memory.post_response("", item.content, {
            "source_id": item.source_id,
            **item.metadata,
        })
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cross_bench.models import CorpusItem

# Frozen HF dataset IDs — pin so the corpus version is stable across
# CrossBench runs. The CodeRAG-Bench team rebuilds these occasionally;
# pinning a specific revision is a future TODO once HF exposes a
# convenient revision API for streaming reads.
_HF_DATASETS: dict[str, str] = {
    "library-documentation": "code-rag-bench/library-documentation",
    "programming-solutions": "code-rag-bench/programming-solutions",
    "online-tutorials": "code-rag-bench/online-tutorials",
    "stackoverflow-posts": "code-rag-bench/stackoverflow-posts",
    "github-repos-python": "code-rag-bench/github-repos-python",
}


def _library_prefix_of(doc_id: str) -> str:
    """Extract the canonical library prefix from a corpus ``doc_id``.

    Handles both dot- and ``#``-separators (the corpus has
    occasional ``torch#torch`` / ``matplotlib_configuration_api#matplotlib``
    leakage from devdocs.io's anchor-link rendering).
    """
    return doc_id.split(".")[0].split("#")[0]


def _matches_libraries(doc_id: str, libraries: set[str]) -> bool:
    """Does this ``doc_id`` belong to any of the requested libraries?

    Two matching strategies:
    1. Top-level prefix (``numpy.argsort`` matches ``numpy``).
    2. Python stdlib via ``python.library.<lib>`` (``python.library.random``
       matches ``random``).

    Falls through both — e.g. asking for ``{"numpy", "random"}``
    matches both ``numpy.argsort`` (strategy 1) and
    ``python.library.random.choice`` (strategy 2).
    """
    if not libraries:
        return True  # no filter → keep everything
    top = _library_prefix_of(doc_id)
    if top in libraries:
        return True
    if top == "python":
        # ``python.library.random`` → check if "random" is requested.
        # Format: ``python.library.<module>[.<symbol>][#<anchor>]``.
        parts = doc_id.split(".")
        if len(parts) >= 3 and parts[1] == "library":
            stdlib_module = parts[2].split("#")[0]
            if stdlib_module in libraries:
                return True
    return False


async def load_corpus(
    name: str,
    *,
    limit: int | None = None,
    libraries: set[str] | None = None,
) -> AsyncIterator[CorpusItem]:
    """Stream :class:`CorpusItem` records from a CodeRAG-Bench corpus.

    Args:
        name: One of the keys in :data:`_HF_DATASETS`. Today only
            ``"library-documentation"`` is wired through to a real
            schema; other corpora raise ``NotImplementedError`` with
            a pointer to the dataset card so a future contributor
            can extend.
        limit: Optional cap on the number of items yielded after
            filtering. Mainly for tests / smoke runs.
        libraries: Optional set of logical library names to filter by
            (see module docstring). ``None`` or empty set ⇒ no filter.

    Raises:
        ValueError: if ``name`` is not a known CodeRAG-Bench corpus.
        NotImplementedError: if ``name`` is known but the loader hasn't
            been written yet.
    """
    if name not in _HF_DATASETS:
        raise ValueError(
            f"Unknown CodeRAG-Bench corpus {name!r}. "
            f"Options: {sorted(_HF_DATASETS)}"
        )

    if name == "library-documentation":
        async for item in _load_library_documentation(
            limit=limit, libraries=libraries or set()
        ):
            yield item
        return

    raise NotImplementedError(
        f"CodeRAG-Bench corpus {name!r} is registered but no loader "
        f"is implemented yet. See the dataset card at "
        f"https://huggingface.co/datasets/{_HF_DATASETS[name]} and "
        f"add a ``_load_{name.replace('-', '_')}`` function with a "
        f"schema-mapping comment."
    )


async def _load_library_documentation(
    *, limit: int | None, libraries: set[str]
) -> AsyncIterator[CorpusItem]:
    """Stream the library-documentation corpus into :class:`CorpusItem` records.

    HF schema: ``{doc_id: str, doc_content: str}``. We map:

      * ``content``     → ``doc_content``
      * ``source_id``   → ``f"lib:{doc_id}"`` (the ``lib:`` prefix
                          disambiguates from per-task ``ctx:N`` IDs
                          used by RepoBench/CrossCodeEval; both share
                          the same memory store under the corpus +
                          per-task layered model).
      * ``metadata``    → ``{"library": <prefix>, "doc_id": <doc_id>}``
                          for vendor-side filtering.
    """
    # Lazy import — keep ``datasets`` out of the import path of
    # adapter code that doesn't actually load corpora at startup.
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASETS["library-documentation"], split="train")
    n_yielded = 0
    for row in ds:
        doc_id = row.get("doc_id", "") or ""
        if not doc_id:
            continue
        if not _matches_libraries(doc_id, libraries):
            continue
        content = row.get("doc_content", "") or ""
        if not content.strip():
            continue
        yield CorpusItem(
            content=content,
            source_id=f"lib:{doc_id}",
            metadata={
                "library": _library_prefix_of(doc_id),
                "doc_id": doc_id,
            },
        )
        n_yielded += 1
        if limit is not None and n_yielded >= limit:
            return
