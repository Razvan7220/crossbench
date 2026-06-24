"""MemOS memory middleware — local MemTensor/MemOS integration.

Wraps the ``MemoryOS`` PyPI package (``pip install MemoryOS``), which
provides the ``memos`` Python module — the MemOS project from MemTensor /
Shanghai Jiao Tong University / Renmin University (arXiv:2505.22101).

Audited against upstream `MemTensor/MemOS` at commit ``338d5e7b``
(2026-04-29). See ``deviations.md`` "memos_embedded" section for the
line-by-line audit + the "Can we claim to do MemOS evaluation?"
analysis. The 2026-05-07 alignment pass fixed the four critical
init/search/dump/add bugs (O1-O4); the follow-up pass on 2026-05-07
closes the headline ingestion gap by wiring upstream's
``SimpleStructMemReader`` into ``post_response``.

Pipeline (auto-selected per backend, matching upstream ``MOSCore.add``
at ``mos/core.py:741-770``):

    post_response(prompt, response, ctx)
      ↓
    backend == "tree_text" ?
      Y → SimpleStructMemReader.get_memory(
            scene_data=[[{"role": "user", ...}, {"role": "assistant", ...}]],
            type="chat",
            info={"user_id": ..., "session_id": ...},
            mode="fine",
          )                                 # LLM per-fact extraction
          ↓
          list[list[TextualMemoryItem]]
          ↓
          cube.text_mem.add(memories_flatten)
      N → wrap each message verbatim:
          [TextualMemoryItem(memory=prompt, ...),
           TextualMemoryItem(memory=response, ...)]
          ↓
          cube.text_mem.add(items)

Override via ``params["use_mem_reader"]``:
    - default (None): auto, matches upstream — MemReader only for
      ``tree_text``.
    - True: force MemReader on every backend (CrossBench-specific
      superset; useful when comparing the marginal benefit of per-
      fact extraction across backends — but the resulting
      ``naive_text + MemReader`` or ``general_text + MemReader``
      configurations DO NOT match what upstream's ``MOSCore.add``
      does for those backends).
    - False: force MemReader off (even for ``tree_text``).

Backends supported (set via ``params.backend``):
    - ``naive_text`` (default): in-process Python list,
      set-intersection retrieval (matches ``naive.py:120-128``).
      Lightweight, no external services. Pairs with MemReader to
      give paper-faithful ingestion + lightweight retrieval.
    - ``general_text``: vector search via Qdrant (Qdrant supports
      a local embedded mode when ``host`` / ``port`` are both
      ``None``). Requires ``pip install qdrant-client``.
    - ``tree_text``: graph-backed retrieval via Neo4j or Postgres
      (heavy; user must supply a ``data_dir`` pointing at a
      pre-built cube).

Config params (in ``MemoryConfig.params``):
    backend:            ``naive_text`` (default) / ``general_text`` / ``tree_text``
    retrieval_limit:    max memories returned (default 5)
    data_dir:           Optional pre-built cube directory
    persist_on_reset:   Dump cube to ``data_dir`` on reset (default False)
    use_mem_reader:     Run upstream's SimpleStructMemReader on every
                        ingest to extract per-fact items. Default
                        ``None`` = auto: matches upstream MOSCore.add
                        semantics — MemReader runs only for
                        ``tree_text`` backend. Override to ``True`` to
                        force MemReader on for ``naive_text`` /
                        ``general_text`` (CrossBench-specific superset)
                        or to ``False`` to skip extraction even for
                        ``tree_text``.
    mem_reader_mode:    ``"fine"`` (default; LLM-based extraction) or
                        ``"fast"`` (rule-based; cheaper, fewer extracted
                        facts). Matches upstream's
                        ``mos/core.py:763`` mode parameter.
    mem_reader_chunk_size:    Chunker chunk size in tokens (default 512,
                              matches upstream ``configs/chunker.py:14``).
    mem_reader_chunk_overlap: Chunker overlap in tokens (default 128,
                              matches upstream ``configs/chunker.py:15``).
    llm_provider:       (auto-inherited from bench LLM)
    llm_model:          (auto-inherited)
    llm_base_url:       (auto-inherited)
    llm_api_key:        Defaults to ``$OPENAI_API_KEY``
    embedder_provider:  (auto-inherited)
    embedder_model:     (auto-inherited; default ``text-embedding-3-small``)
    use_rest_api:       REST mode (drift-flagged ❌ in deviations.md O20).
    rest_api_url:       Default ``http://localhost:8001``

Source:
    https://github.com/MemTensor/MemOS
    https://memos-docs.openmem.net/

NOTE: When the MemoryOS package is not installed, the middleware falls
back to a simple in-memory keyword-overlap store so benchmarks remain
runnable. The fallback is also what kicks in if the cube fails to
init (e.g. missing API key, malformed config); errors are logged but
not raised.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.memory._memos_token_hook import (
    install_memos_token_hook,
    run_in_executor_with_accountant,
)
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline metadata-prefix scheme (deviations.md O27)
#
# Upstream MemTensor/MemOS's ``TextualMemoryMetadata`` is a strict Pydantic
# model whose field set varies by release. To plumb ``source_id`` /
# ``timestamp`` through the cube's ``add → search → return`` round-trip
# without coupling to upstream's schema, we prepend a sentinel header line
# to the ``memory`` string on the Path B ingest path (verbatim wrap of
# user + assistant messages; matches upstream ``MOSCore.add`` for the
# non-tree_text branch at ``mos/core.py:746-756``). The decoder strips
# the header on the way out so the rendered prompt only shows
# ``[<timestamp>] <body>`` lines.
#
# The sentinel format is intentionally angular-bracket-heavy so its
# tokens are vanishingly unlikely to match real conversation content
# under set-intersection retrieval (the default ``naive_text`` backend
# splits on whitespace). For ``general_text`` (vector cosine) the
# prefix contributes a fixed handful of tokens to each chunk's
# embedding — bounded perturbation, far below the per-chunk semantic
# signal.
#
# MemReader-produced items (``tree_text`` backend by default) do NOT
# carry our prefix because the MemReader extracts its own facts with
# its own metadata. They render as plain ``- <body>`` lines, which is
# upstream-faithful.
# ---------------------------------------------------------------------------

_META_TAG_RE = re.compile(
    r"^<<MEMOS_META\s+src=([^|>]*)\s*\|\s*ts=([^>]*)>>\n?",
    re.DOTALL,
)


def _encode_meta_prefix(source_id: str | None, timestamp: str | None) -> str:
    """Encode source_id + timestamp as a sentinel header line.

    Both fields are always emitted (empty string when missing) so the
    decoder regex has a stable shape and ``_decode_meta_prefix`` can
    distinguish "no prefix at all" (returns None/None) from "prefix
    with missing fields" (returns None/None for the missing ones).
    Returns an empty string when both are missing so callers can avoid
    a no-op prefix.
    """
    if not source_id and not timestamp:
        return ""
    src = source_id or ""
    ts = timestamp or ""
    return f"<<MEMOS_META src={src} | ts={ts}>>\n"


def _decode_meta_prefix(text: str) -> tuple[str, str | None, str | None]:
    """Strip the metadata prefix from ``text`` and return its components.

    Returns ``(body, source_id, timestamp)``. When no prefix is
    present (e.g., MemReader-produced items, items written before the
    O27 fix landed), returns ``(text, None, None)`` verbatim so the
    caller can render them as plain lines.
    """
    m = _META_TAG_RE.match(text)
    if not m:
        return text, None, None
    src = m.group(1).strip() or None
    ts = m.group(2).strip() or None
    body = text[m.end():]
    return body, src, ts


@register_memory("memos_embedded")
class MemOSMiddleware(MemoryMiddleware):
    """MemOS (MemTensor) memory middleware — paper-faithful pipeline.

    Lifecycle:
        - ``_ensure_init`` (lazy): instantiates the cube AND the
          ``SimpleStructMemReader`` if ``use_mem_reader`` is set.
          Either loads the cube from ``data_dir`` if it contains a
          ``config.json``, or builds a fresh ``GeneralMemCubeConfig``
          programmatically using the bench-inherited LLM/embedder
          params.
        - ``post_response``: when MemReader is available, runs
          ``mem_reader.get_memory(messages_list, type='chat',
          info={user_id, session_id}, mode=mem_reader_mode)`` to
          extract per-fact ``TextualMemoryItem``s, then
          ``cube.text_mem.add(items)``. When MemReader is disabled
          or fails to load, falls back to direct ``add(...)`` of a
          single ``Q: ...\\nA: ...`` blob.
        - ``pre_prompt``: calls ``cube.text_mem.search(query,
          top_k)`` — vector cosine for ``general_text``,
          set-intersection for ``naive_text``, graph search for
          ``tree_text``.
        - ``reset``: ``cube.text_mem.delete_all()`` (in-place, no
          filesystem manipulation).
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        params = config.params or {}

        # Public knobs.
        self._backend = params.get("backend", "naive_text")
        self._retrieval_limit = int(params.get("retrieval_limit", 5))
        self._data_dir = params.get("data_dir")  # Optional pre-built cube
        self._persist_on_reset = bool(params.get("persist_on_reset", False))

        # MemReader knobs — closes the headline ingestion gap (O6 / O15).
        # Default behaviour matches upstream `MOSCore.add` exactly
        # (`mos/core.py:741-770`):
        #   tree_text  → run MemReader (paper-faithful for the
        #                LoCoMo benchmark in the MemOS paper)
        #   naive_text → wrap each message verbatim (no extraction)
        #   general_text → wrap each message verbatim (no extraction)
        # Override via ``params["use_mem_reader"]``: set ``True`` to
        # force MemReader on every backend (CrossBench-specific superset
        # of upstream — useful when comparing the marginal benefit of
        # per-fact extraction across backends), or ``False`` to disable
        # extraction even for ``tree_text``.
        _override = params.get("use_mem_reader")
        if _override is None:
            self._use_mem_reader = self._backend == "tree_text"
        else:
            self._use_mem_reader = bool(_override)
        self._mem_reader_mode = str(params.get("mem_reader_mode", "fine"))
        self._mem_reader_chunk_size = int(
            params.get("mem_reader_chunk_size", 512)
        )
        self._mem_reader_chunk_overlap = int(
            params.get("mem_reader_chunk_overlap", 128)
        )
        self._user_id = str(params.get("user_id", "bench-user"))

        # Bench-inherited LLM config.
        self._llm_provider = params.get("llm_provider", "openai")
        self._llm_model = params.get("llm_model", "gpt-4o-mini")
        self._llm_base_url = params.get(
            "llm_base_url", "https://api.openai.com/v1"
        )
        self._llm_api_key = params.get(
            "llm_api_key", os.environ.get("OPENAI_API_KEY", "")
        )

        # Bench-inherited embedder config. Used by `general_text` AND by
        # the MemReader (which always embeds extracted items).
        self._embedder_provider = params.get("embedder_provider", "openai")
        self._embedder_model = params.get(
            "embedder_model", "text-embedding-3-small"
        )
        self._embedder_dims = int(params.get("embedder_dims", 1536))

        # REST mode (drift-flagged in deviations.md O20).
        self._use_rest = bool(params.get("use_rest_api", False))
        self._rest_url = params.get("rest_api_url", "http://localhost:8001")

        # State.
        self._mem_cube = None
        self._mem_reader = None  # SimpleStructMemReader instance (when enabled)
        self._fallback_memories: list[dict[str, Any]] = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation — programmatic GeneralMemCubeConfig construction
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if self._use_rest:
            return  # REST mode skips local-cube setup.

        try:
            from memos.mem_cube.general import GeneralMemCube
        except ImportError:
            logger.warning(
                "MemoryOS package not installed — falling back to "
                "in-memory keyword store. `pip install MemoryOS` to enable."
            )
            return

        # Patch MemoryOS LLM classes so their internal call sites surface
        # token counts to the active accountant. Idempotent + tolerant —
        # see ``_memos_token_hook.install_memos_token_hook`` for caveats.
        install_memos_token_hook()

        # Path 1: pre-built cube on disk. Only takes this path if the
        # directory exists AND contains a config.json — guards against
        # the historical bug where init_from_dir(empty_dir) raised
        # FileNotFoundError on the missing config.json.
        if self._data_dir:
            cube_path = Path(self._data_dir)
            cfg_path = cube_path / "config.json"
            if cube_path.is_dir() and cfg_path.is_file():
                try:
                    self._mem_cube = GeneralMemCube.init_from_dir(
                        str(cube_path)
                    )
                    logger.info(
                        "MemOS: loaded pre-built MemCube from %s", cube_path
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "MemOS: failed to load MemCube from %s (%s); "
                        "falling through to fresh build",
                        cube_path,
                        e,
                    )

        # Path 2: build a fresh cube programmatically.
        try:
            self._mem_cube = self._build_fresh_cube()
            if self._mem_cube is not None:
                logger.info(
                    "MemOS: built fresh in-process MemCube (backend=%s)",
                    self._backend,
                )
        except Exception as e:
            logger.warning(
                "MemOS: failed to build cube (%s); falling back to "
                "in-memory keyword store",
                e,
            )

        # Path 3: try to instantiate the MemReader. Closes the headline
        # ingestion gap (deviations.md O6 / O15). On failure (missing
        # `chonkie` chunker dep, missing embedder API key, etc.) we
        # silently fall through to direct cube.text_mem.add of raw
        # blobs in post_response — same as the pre-MemReader behaviour.
        if self._mem_cube is not None and self._use_mem_reader:
            try:
                self._mem_reader = self._build_fresh_mem_reader()
                if self._mem_reader is not None:
                    logger.info(
                        "MemOS: built SimpleStructMemReader (mode=%s)",
                        self._mem_reader_mode,
                    )
            except Exception as e:
                logger.warning(
                    "MemOS: failed to build MemReader (%s); will store "
                    "raw Q/A blobs without per-fact extraction",
                    e,
                )

    def _build_fresh_mem_reader(self):
        """Construct upstream's SimpleStructMemReader programmatically.

        The MemReader is what runs LLM-based per-fact extraction over a
        chat episode (`messages_list, type='chat', mode='fine'`). We
        feed each `post_response` `(prompt, response)` pair through it
        before calling `cube.text_mem.add(...)`, mirroring upstream's
        `MOSCore.add` path for the `tree_text` backend
        (`mos/core.py:759-766`) but applying it uniformly across all
        backends so naive / general also benefit from per-fact
        granularity.
        """
        from memos.configs.mem_reader import MemReaderConfigFactory
        from memos.mem_reader.factory import MemReaderFactory

        reader_factory = MemReaderConfigFactory.model_validate(
            {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": self._llm_provider,
                        "config": {
                            "model_name_or_path": self._llm_model,
                            "api_base": self._llm_base_url,
                            "api_key": self._llm_api_key,
                            "temperature": 0.0,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": self._embedder_provider,
                            "model_name_or_path": self._embedder_model,
                            "api_key": self._llm_api_key,
                            "base_url": self._llm_base_url,
                            "embedding_dims": self._embedder_dims,
                        },
                    },
                    "chunker": {
                        "backend": "sentence",
                        "config": {
                            "tokenizer_or_token_counter": "gpt2",
                            "chunk_size": self._mem_reader_chunk_size,
                            "chunk_overlap": self._mem_reader_chunk_overlap,
                        },
                    },
                },
            }
        )
        return MemReaderFactory.from_config(reader_factory)

    def _build_fresh_cube(self):
        """Construct a GeneralMemCube from scratch using bench-inherited config."""
        from memos.configs.mem_cube import GeneralMemCubeConfig
        from memos.mem_cube.general import GeneralMemCube

        # Build the textual memory config block. Mirrors what
        # `MemReader`/`MOS` would produce when `MOS.simple()` is called
        # with the OPENAI_API_KEY env var set.
        text_mem_config: dict[str, Any] = {
            "backend": self._backend,
            "config": self._build_text_mem_inner_config(),
        }

        cube_config = GeneralMemCubeConfig.model_validate(
            {
                "user_id": "bench-user",
                "cube_id": "bench-cube",
                "text_mem": text_mem_config,
                # Keep the other three memory types uninitialized — we
                # only consume textual. This mirrors upstream's "set
                # backend = 'uninitialized' to skip" pattern at
                # mem_cube/general.py:28-48.
                "act_mem": {"backend": "uninitialized", "config": {}},
                "para_mem": {"backend": "uninitialized", "config": {}},
                "pref_mem": {"backend": "uninitialized", "config": {}},
            }
        )
        return GeneralMemCube(cube_config)

    def _build_text_mem_inner_config(self) -> dict[str, Any]:
        """Build the inner `config:` block for the chosen backend."""
        llm_block = {
            "backend": self._llm_provider,
            "config": {
                "model_name_or_path": self._llm_model,
                "api_base": self._llm_base_url,
                "api_key": self._llm_api_key,
                "temperature": 0.0,
            },
        }

        if self._backend == "naive_text":
            # NaiveTextMemory only needs the LLM — search is in-process
            # set-intersection, no embedder, no vec_db.
            return {"extractor_llm": llm_block}

        if self._backend == "general_text":
            embedder_block = {
                "backend": self._embedder_provider,
                "config": {
                    "model_name_or_path": self._embedder_model,
                    "api_base": self._llm_base_url,
                    "api_key": self._llm_api_key,
                },
            }
            # Qdrant in local/embedded mode (host=None, port=None).
            vec_db_block = {
                "backend": "qdrant",
                "config": {
                    "collection_name": "memos_bench",
                    "vector_dimension": 1536,  # text-embedding-3-small
                    "distance_metric": "cosine",
                    "host": None,
                    "port": None,
                    "path": None,
                },
            }
            return {
                "extractor_llm": llm_block,
                "embedder": embedder_block,
                "vector_db": vec_db_block,
            }

        # tree_text or any other backend — defer to upstream defaults.
        # The user is expected to supply a `data_dir` with a working
        # config.json for non-naive/general backends.
        raise ValueError(
            f"Unsupported backend {self._backend!r} for programmatic "
            "construction; pass `data_dir` pointing to a pre-built "
            "MemCube directory instead."
        )

    # ------------------------------------------------------------------
    # MemoryMiddleware contract
    # ------------------------------------------------------------------

    def _on_memos_internal_llm(
        self, prompt_text: str, response_text: str
    ) -> None:
        """Callback fired by the patched MemoryOS LLM methods."""
        self._record_internal_llm(prompt=prompt_text, response=response_text)

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        self._ensure_init()
        loop = asyncio.get_event_loop()

        # Boundary: tokenize the query we hand to MemoryOS.
        self._record_io(request_text=prompt[:500])

        if self._use_rest:
            return await self._rest_search(prompt)

        if self._mem_cube is not None and self._mem_cube.text_mem is not None:
            # Accountant set inside the worker thread (it doesn't cross
            # the run_in_executor boundary — see
            # ``run_in_executor_with_accountant``) so the cube's in-thread
            # retrieval LLM calls are counted.
            try:
                memories = await run_in_executor_with_accountant(
                    loop,
                    self._on_memos_internal_llm,
                    lambda: self._mem_cube.text_mem.search(
                        prompt[:500], top_k=self._retrieval_limit
                    ),
                )
                rendered, source_ids = self._render_retrieved(memories)
                # Always populate ``retrieved_context_ids`` when we ran
                # the retriever — empty list signals "retriever found
                # nothing", absent key would signal "not measured" (per
                # the MemoryMiddleware contract). See deviations.md O27.
                if isinstance(context, dict):
                    context["retrieved_context_ids"] = source_ids
                if rendered:
                    block = "\n".join(rendered)
                    self._record_io(response_text=block)
                    # NB: no ``Retrieved memories:`` header here — the
                    # answer prompt template owns that header. The
                    # engine extracts everything before ``prompt`` as
                    # ``search_context`` and the MEMOS_ANSWER_PROMPT
                    # template wraps it. Including the header here was
                    # the source of the doubled-header bug in the
                    # 2026-05-21 ``locomo-memos-20260521-202641`` run.
                    return f"{block}\n\n{prompt}"
                return prompt
            except Exception as e:
                logger.debug("MemOS search failed (non-fatal): %s", e)

        # Fallback: simple keyword-overlap search.
        if self._fallback_memories:
            query_words = set(prompt.lower().split()[:10])
            scored: list[tuple[int, dict[str, Any]]] = []
            for mem in self._fallback_memories:
                text = mem.get("text", "")
                overlap = sum(1 for w in query_words if w in text.lower())
                if overlap > 0:
                    scored.append((overlap, mem))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [m for _, m in scored[: self._retrieval_limit]]
            rendered, source_ids = self._render_retrieved(
                [m.get("text", "") for m in top]
            )
            if isinstance(context, dict):
                context["retrieved_context_ids"] = source_ids
            if rendered:
                block = "\n".join(rendered)
                self._record_io(response_text=block)
                return f"{block}\n\n{prompt}"
            return prompt

        # No memories at all — still honest-empty-measured.
        if isinstance(context, dict):
            context["retrieved_context_ids"] = []
        return prompt

    @classmethod
    def _render_retrieved(
        cls, memories: list[Any]
    ) -> tuple[list[str], list[str]]:
        """Render a search result list into prompt lines + source_ids.

        Each entry becomes ``- [<timestamp>] <body>`` when a timestamp
        is present, or ``- <body>`` otherwise. Source IDs are
        deduplicated in rank order so the eventual
        ``retrieved_context_ids`` list matches the visual order of the
        prompt block. Items whose body is empty after decoding are
        dropped (defensive — upstream may return placeholder rows).
        """
        rendered: list[str] = []
        source_ids: list[str] = []
        seen: set[str] = set()
        for item in memories:
            if not item:
                continue
            body, src, ts = cls._extract_memory_components(item)
            body = body.strip()
            if not body:
                continue
            if src and src not in seen:
                source_ids.append(src)
                seen.add(src)
            rendered.append(f"- [{ts}] {body}" if ts else f"- {body}")
        return rendered, source_ids

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        self._ensure_init()
        loop = asyncio.get_event_loop()
        ctx = context if isinstance(context, dict) else {}
        source_id = (ctx.get("source_id") or "").strip() or None
        chat_time = (ctx.get("timestamp") or "").strip() or None
        meta_prefix = _encode_meta_prefix(source_id, chat_time)
        entry_text = f"Q: {prompt[:500]}\nA: {response[:500]}"

        # Boundary: tokenize the user/assistant exchange we hand to
        # MemoryOS, regardless of which path executes below.
        self._record_io(request_text=f"{prompt}\n{response}")

        if self._use_rest:
            await self._rest_add(entry_text)
            return

        if self._mem_cube is not None and self._mem_cube.text_mem is not None:
            session_id = ctx.get("conversation_id", "default")

            # Path A — MemReader is loaded: extract per-fact items first,
            # then add. Mirrors upstream's `MOSCore.add` for `tree_text`
            # backend (`mos/core.py:759-766`). Applies to all backends so
            # naive_text and general_text also benefit from per-fact
            # granularity. MemReader-produced items do NOT carry our
            # metadata prefix because the extractor builds its own
            # ``memory`` text from the LLM's structured output; the O27
            # plumbing only kicks in on Path B.
            if self._mem_reader is not None:
                # Accountant set inside each worker thread (it doesn't
                # cross the run_in_executor boundary — see
                # ``run_in_executor_with_accountant``) so MemReader's
                # extraction LLM calls and any cube-side add LLM calls
                # are counted.
                try:
                    items = await run_in_executor_with_accountant(
                        loop,
                        self._on_memos_internal_llm,
                        lambda: self._extract_via_mem_reader(
                            prompt, response, session_id, chat_time
                        ),
                    )
                    if items:
                        await run_in_executor_with_accountant(
                            loop,
                            self._on_memos_internal_llm,
                            lambda: self._mem_cube.text_mem.add(items),
                        )
                        # Boundary: tokenize what MemReader surfaced.
                        extracted_text = "\n".join(
                            getattr(it, "memory", "") or ""
                            for it in items
                        )
                        if extracted_text:
                            self._record_io(response_text=extracted_text)
                        return
                    # MemReader returned no extractable facts (e.g. pure
                    # filler turn). Fall through to direct-add of the raw
                    # blob so the turn isn't lost.
                except Exception as e:
                    logger.debug(
                        "MemOS MemReader extraction failed (%s); "
                        "falling back to direct add",
                        e,
                    )

            # Path B — wrap each message verbatim, no extraction.
            # Mirrors upstream ``MOSCore.add`` behaviour for the non-
            # tree_text branch (``mos/core.py:746-756``): one
            # TextualMemoryItem per role/content pair, each with the
            # same ``TextualMemoryMetadata(user_id, session_id,
            # source="conversation")`` block. Each ``memory`` string
            # is prefixed with a sentinel header carrying source_id +
            # timestamp so retrieval can surface them honestly (O27).
            try:
                from memos.memories.textual.item import (
                    TextualMemoryItem,
                    TextualMemoryMetadata,
                )

                metadata = TextualMemoryMetadata(
                    user_id=self._user_id,
                    session_id=session_id,
                    source="conversation",
                )
                items = [
                    TextualMemoryItem(
                        memory=meta_prefix + prompt[:4000], metadata=metadata
                    ),
                    TextualMemoryItem(
                        memory=meta_prefix + response[:4000], metadata=metadata
                    ),
                ]
                await loop.run_in_executor(
                    None, lambda: self._mem_cube.text_mem.add(items)
                )
                return
            except Exception as e:
                logger.debug("MemOS add failed (non-fatal): %s", e)

        # Fallback path — symmetric prefix so retrieval renders timestamps.
        self._fallback_memories.append({"text": meta_prefix + entry_text})

    def _extract_via_mem_reader(
        self,
        prompt: str,
        response: str,
        session_id: str,
        chat_time: str | None = None,
    ) -> list:
        """Run upstream's SimpleStructMemReader.get_memory on a single Q/A pair.

        Upstream signature (``mem_reader/simple_struct.py:542-551``):
            get_memory(scene_data, type, info, mode='fine', user_name=None)

        ``scene_data`` is a list of "scenes"; for chat-mode each scene is a
        list of message dicts. Each dict accepts an optional ``chat_time``
        which upstream prepends to the LLM-extractor input as
        ``[chat_time]: ...`` (``simple_struct.py:233-237``). We forward
        the engine-supplied per-turn timestamp so extracted facts anchor
        to the conversation's clock rather than today's wall-clock.

        Returns ``list[list[TextualMemoryItem]]`` — one inner list per scene.
        """
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": response}
        if chat_time:
            user_msg["chat_time"] = chat_time
            assistant_msg["chat_time"] = chat_time
        scene_data = [[user_msg, assistant_msg]]  # one scene
        extracted = self._mem_reader.get_memory(
            scene_data,
            type="chat",
            info={"user_id": self._user_id, "session_id": session_id},
            mode=self._mem_reader_mode,
        )
        # Flatten list[list[TextualMemoryItem]] → list[TextualMemoryItem].
        return [item for scene_items in extracted for item in scene_items]

    async def reset(self) -> None:
        # Clear fallback first.
        self._fallback_memories.clear()
        self._reset_internal_counters()

        if self._mem_cube is None or self._mem_cube.text_mem is None:
            return

        loop = asyncio.get_event_loop()

        # Optionally persist before reset.
        if self._persist_on_reset and self._data_dir:
            cube_path = Path(self._data_dir)
            cube_path.mkdir(parents=True, exist_ok=True)
            # `dump` raises if dir is non-empty (general.py:104-107) —
            # clear it first.
            for f in cube_path.iterdir():
                if f.is_file():
                    f.unlink()
            try:
                await loop.run_in_executor(
                    None, lambda: self._mem_cube.dump(str(cube_path))
                )
            except Exception as e:
                logger.warning("MemOS dump failed during reset: %s", e)

        # In-place clear — matches upstream's `delete_all` contract on
        # `BaseTextMemory` (base.py:87-88).
        try:
            await loop.run_in_executor(
                None, lambda: self._mem_cube.text_mem.delete_all()
            )
        except Exception as e:
            logger.warning("MemOS delete_all failed: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_memory(item: Any) -> str:
        """Render a memory item to a one-line string (no metadata).

        Strips the O27 sentinel header so callers that only want the
        body see clean text. For callers that need ``source_id`` /
        ``timestamp`` too, use ``_extract_memory_components`` instead.
        """
        body, _, _ = MemOSMiddleware._extract_memory_components(item)
        return body

    @staticmethod
    def _extract_memory_components(
        item: Any,
    ) -> tuple[str, str | None, str | None]:
        """Return ``(body, source_id, timestamp)`` for a memory item.

        Handles both ``TextualMemoryItem`` instances (the cube path)
        and raw strings (the fallback / REST paths). When no O27
        sentinel is present, returns the text verbatim with
        ``source_id=None`` and ``timestamp=None``.
        """
        if isinstance(item, str):
            return _decode_meta_prefix(item)
        memory_attr = getattr(item, "memory", None)
        if isinstance(memory_attr, str):
            return _decode_meta_prefix(memory_attr)
        return str(item), None, None

    # ------------------------------------------------------------------
    # REST API helpers (use_rest_api=True mode — see deviations.md O20)
    # ------------------------------------------------------------------

    async def _rest_search(self, prompt: str) -> str:
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._rest_url}/search",
                    json={
                        "query": prompt[:500],
                        "limit": self._retrieval_limit,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        lines = [
                            f"- {r.get('text', r.get('content', ''))}"
                            for r in results
                        ]
                        return (
                            "Retrieved memories:\n"
                            + "\n".join(lines)
                            + f"\n\n{prompt}"
                        )
        except Exception:
            pass
        return prompt

    async def _rest_add(self, text: str) -> None:
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self._rest_url}/add",
                    json={"text": text},
                    timeout=10,
                )
        except Exception:
            pass
