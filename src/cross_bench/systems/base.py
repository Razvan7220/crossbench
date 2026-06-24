"""Core protocols for system components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from cross_bench.models import StepRecord, Task


def count_tokens(text: str) -> int:
    """Token count under a single, model-agnostic encoder (``cl100k_base``).

    Used as the universal fallback when a memory system's internal LLM call
    does not surface a provider-reported ``usage`` object (or when the
    middleware delegates to an upstream package whose internal LLM client
    we cannot intercept). The same encoder is used across every memory
    system so the ``memory_io_tokens_*`` column means one thing across
    rows — see deviations.md "Platform / cross-cutting" entry on
    internal-token accounting.

    Falls back to ``len(text) // 4`` (the chars/4 heuristic) when
    ``tiktoken`` is not importable, so the metric stays populated on
    minimal-dependency installs.
    """
    if not text:
        return 0
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


class LLMDriver(ABC):
    """Interface for language model providers."""

    @abstractmethod
    async def complete(self, prompt: str, config: dict[str, Any]) -> tuple[str, StepRecord]:
        """Single completion. Returns (generated_text, step_record)."""

    @abstractmethod
    async def chat(
        self, messages: list[dict[str, str]], config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        """Chat-style call. Returns (generated_text, step_record)."""


class MemoryMiddleware(ABC):
    """Interface for memory systems that augment prompts with stored context.

    Retrieval provenance (optional contract)
        The ``context`` dict passed to :meth:`pre_prompt` and
        :meth:`post_response` is a mutable side-channel. The engine mutates
        / inspects a small, well-known set of keys so benchmarks can measure
        retrieval quality against dataset ground truth without changing the
        method signatures.

        Input keys (engine → middleware, read on ingest via
        :meth:`post_response`):

        * ``"source_id"`` — optional string identifying the ingested item
          (e.g. LoCoMo's ``dia_id = "D1:3"``). Middlewares that can preserve
          provenance (chunk-level indexes such as ``rag_baseline``) SHOULD
          remember this alongside stored content so it can be surfaced back
          on retrieval. Middlewares that extract / summarise / consolidate
          memories (Mem0, Zep, Letta, Graphiti, MemOS) MAY ignore it — in
          which case they also skip the output key below, and benchmarks
          fall back to their "no retrieval measured" default.

        * ``"code_workspace"`` — optional absolute workspace path string.
          When set (e.g. by RepoBench with ``synthetic_workspace_dir``), a
          filesystem-backed middleware can scope retrieval to that materialized
          tree on recall/store.

        Output key (middleware → engine, written by :meth:`pre_prompt`):

        * ``"retrieved_context_ids"`` — optional ``list[str]`` of
          ``source_id`` values for the items the middleware retrieved for
          this prompt, in rank order, de-duplicated. ``None`` / absent means
          "not measured" (NOT "retrieved nothing" — use ``[]`` for that).
          The engine copies this onto ``task.metadata`` so benchmark
          ``judge_answer`` implementations can compute recall against
          ground-truth evidence IDs.

        Middlewares that can't honestly produce stable per-item IDs SHOULD
        leave this key unset rather than inventing values — a missing key
        is meaningfully different from an empty list in downstream
        aggregations.

    Internal-LLM token accounting (optional contract)
        Many memory systems run their own internal LLM calls for fact
        extraction, dedup, consolidation, observer/reflector passes, etc.
        These tokens are invisible to the bench-LLM accounting that the
        engine derives from :class:`StepRecord`. To surface them on the
        leaderboard alongside the bench-LLM totals, middlewares accumulate
        two complementary counts per task:

        * ``_internal_tokens_in`` / ``_internal_tokens_out`` /
          ``_internal_llm_calls`` — exact counts pulled from each LLM
          response's ``usage`` object when the middleware owns the call
          site or wraps the upstream client.
        * ``_io_tokens_in`` / ``_io_tokens_out`` — model-agnostic
          boundary counts via :func:`count_tokens`. Used when the
          middleware delegates to an opaque upstream package (mem0,
          memos) and only the strings handed in / surfaced back are
          visible. Always available; never zero on a real call path.

        Call :meth:`_record_internal_llm` from every owned ``_llm`` /
        ``_llm_call`` / ``_llm_generate`` helper. Call :meth:`_record_io`
        wherever the middleware crosses a delegated package boundary.
        Both increment the relevant counters. :meth:`_reset_internal_counters`
        zeros them — subclasses' :meth:`reset` must call it (locked by
        ``tests/systems/test_reset_counter_contract.py``).
        :meth:`get_internal_token_stats` returns the five-key dict the
        engine reads at end of task.
    """

    def __init__(self) -> None:  # pragma: no cover — trivial, but locks the init contract
        self._internal_tokens_in: int = 0
        self._internal_tokens_out: int = 0
        self._internal_llm_calls: int = 0
        self._io_tokens_in: int = 0
        self._io_tokens_out: int = 0

    def _ensure_token_counters(self) -> None:
        """Lazy-init the counters if a subclass forgot to call ``super().__init__()``.

        Subclasses written before this contract landed don't necessarily
        chain to ``MemoryMiddleware.__init__``. Calling this from every
        ``_record_*`` helper keeps the contract working without forcing
        a coordinated refactor.
        """
        if not hasattr(self, "_internal_tokens_in"):
            self._internal_tokens_in = 0
            self._internal_tokens_out = 0
            self._internal_llm_calls = 0
            self._io_tokens_in = 0
            self._io_tokens_out = 0

    def _record_internal_llm(
        self,
        *,
        prompt: str | None = None,
        response: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one internal LLM call.

        Prefers provider-reported ``usage`` when present (exact, free) —
        looks for ``prompt_tokens``/``input_tokens`` (OpenAI / Anthropic
        spellings) and ``completion_tokens``/``output_tokens``. Falls
        back to :func:`count_tokens` over ``prompt`` and ``response``
        when ``usage`` is missing or partial. Always increments the
        call counter.
        """
        self._ensure_token_counters()
        self._internal_llm_calls += 1

        in_tokens: int | None = None
        out_tokens: int | None = None
        if usage:
            for key in ("prompt_tokens", "input_tokens"):
                val = usage.get(key)
                if isinstance(val, (int, float)) and val >= 0:
                    in_tokens = int(val)
                    break
            for key in ("completion_tokens", "output_tokens"):
                val = usage.get(key)
                if isinstance(val, (int, float)) and val >= 0:
                    out_tokens = int(val)
                    break

        if in_tokens is None and prompt is not None:
            in_tokens = count_tokens(prompt)
        if out_tokens is None and response is not None:
            out_tokens = count_tokens(response)

        if in_tokens:
            self._internal_tokens_in += in_tokens
        if out_tokens:
            self._internal_tokens_out += out_tokens

    def _record_io(
        self,
        *,
        request_text: str | None = None,
        response_text: str | None = None,
    ) -> None:
        """Record boundary tokens at a delegated-package call site.

        Used when the middleware can't intercept the internal LLM
        directly (mem0, memos) but knows the strings it hands across
        the package boundary. Counts both directions independently via
        :func:`count_tokens` so callers can record one side at a time
        when the other isn't materialised as text (e.g. ``Memory.search``
        returns objects, not a single response string).
        """
        self._ensure_token_counters()
        if request_text:
            self._io_tokens_in += count_tokens(request_text)
        if response_text:
            self._io_tokens_out += count_tokens(response_text)

    def _reset_internal_counters(self) -> None:
        """Zero all internal-token counters.

        Subclasses' :meth:`reset` must call this — uniform contract,
        locked by ``tests/systems/test_reset_counter_contract.py``.
        Engine-side per-task aggregation is snapshot-delta based
        (``engine._snapshot_memory_stats`` / ``_stamp_memory_delta``),
        so the summary columns stay correct even without the reset;
        the call keeps the raw counters conversation-scoped for any
        consumer reading :meth:`get_internal_token_stats` directly.
        """
        self._ensure_token_counters()
        self._internal_tokens_in = 0
        self._internal_tokens_out = 0
        self._internal_llm_calls = 0
        self._io_tokens_in = 0
        self._io_tokens_out = 0

    def get_internal_token_stats(self) -> dict[str, int]:
        """Return current internal-token counters.

        Keys: ``tokens_in``, ``tokens_out``, ``llm_calls``,
        ``io_tokens_in``, ``io_tokens_out``. Engine reads this at end
        of each task to stamp the values onto ``task.evaluation``.
        """
        self._ensure_token_counters()
        return {
            "tokens_in": self._internal_tokens_in,
            "tokens_out": self._internal_tokens_out,
            "llm_calls": self._internal_llm_calls,
            "io_tokens_in": self._io_tokens_in,
            "io_tokens_out": self._io_tokens_out,
        }

    @abstractmethod
    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Augment prompt with retrieved memories before the LLM call.

        Implementations MAY populate ``context["retrieved_context_ids"]``
        with the list of source IDs for retrieved items (see class
        docstring).
        """

    @abstractmethod
    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Store relevant information after the LLM responds.

        Implementations MAY read ``context.get("source_id")`` to tag the
        stored item for later provenance-aware retrieval (see class
        docstring).
        """

    @abstractmethod
    async def reset(self) -> None:
        """Clear session state (called between benchmark tasks)."""


class EmbeddingDriver(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for the given texts."""


class PromptStrategy(ABC):
    """Interface for prompt formatting strategies."""

    @abstractmethod
    def format(
        self,
        task: Task,
        benchmark_prompt: str,
        memory_context: str | None = None,
    ) -> str | list[dict[str, str]]:
        """Transform a task + benchmark prompt into a final prompt or message list."""
