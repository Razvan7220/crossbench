"""Mem0 memory middleware — v2.0.0 API (both embedded and cloud).

Two registrations live in this file, sharing config translation and
pre/post-prompt formatting:

* ``mem0v2_embedded`` — local ``Memory`` (open-source, in-process; Qdrant
  by default, configurable LLM + embedder + vector store).
* ``mem0v2_api``      — cloud ``MemoryClient`` (api.mem0.ai, v3 endpoints).

Install:
    pip install mem0ai>=2.0.0

v1 → v2 API deltas (LOCAL ``Memory``):
    - ``search()`` requires ``filters={"user_id": ...}`` — top-level
      ``user_id`` kwarg is rejected with ValueError.
    - ``search()`` uses ``top_k=`` (v1 used ``limit=``).
    - ``search()`` default ``threshold`` is 0.1 (v1 was None).
    - ``search()`` default ``rerank`` is False (v1 was True).
    - ``add()`` still accepts top-level ``user_id`` kwarg.
    - ``delete_all()`` still accepts top-level ``user_id`` kwarg.
    - ``MemoryConfig`` uses ``custom_instructions`` (v1 had separate
      ``custom_fact_extraction_prompt`` + ``custom_update_memory_prompt``).
    - Search results carry top-level ``created_at``/``updated_at`` per
      ``MemoryItem``.

v1 → v2 API deltas (CLOUD ``MemoryClient``):
    - ``MemoryClient()`` constructor no longer accepts ``org_id`` or
      ``project_id``; both are now resolved from the API key server-side.
    - ``search()`` **rejects** top-level entity kwargs
      (``user_id`` / ``agent_id`` / ``app_id`` / ``run_id``) — must go
      inside ``filters={...}``. (``ENTITY_PARAMS`` check in the client.)
    - ``add()`` and ``delete_all()`` also accept scoping via ``filters=``;
      we use ``filters=`` consistently for all three calls to match the
      v3 server convention.
    - Endpoints moved to ``/v3/memories/search/`` and
      ``/v3/memories/add/`` (delete stayed on ``/v1/memories/``).
    - Responses are always wrapped as ``{"results": [...]}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.memory._mem0_token_hook import (
    install_mem0_token_hook,
    run_in_executor_with_accountant,
)
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)

# Debug log file — set MEM0_DEBUG_LOG env var to enable detailed tracing
_DEBUG_LOG = os.environ.get("MEM0_DEBUG_LOG", "")

# Providers whose config class accepts ``openai_base_url`` (OpenAI-compatible).
# Anything else falls back to ``ollama_base_url``.
_OPENAI_BASE_URL_PROVIDERS = {
    "openai",
    "openai_structured",
    "azure_openai",
    "azure_openai_structured",
    "deepseek",
    "xai",
    "together",
    "groq",
    "litellm",
    "vllm",
    "lmstudio",
}


def _debug(event: str, data: dict[str, Any]) -> None:
    """Append a JSON-lines debug entry when MEM0_DEBUG_LOG is set."""
    if not _DEBUG_LOG:
        return
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **data}, default=str) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared helpers — prompt / response formatting
# ---------------------------------------------------------------------------


def _format_content(
    prompt: str, response: str, context: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the ``(messages, metadata)`` pair for ``add()``.

    Prepends speaker/timestamp prefix to the content so mem0's fact
    extraction produces speaker-attributed, temporally-grounded memories
    (critical for LoCoMo temporal questions).

    When ``prompt`` is empty (conversation ingestion mode), a single
    user-role message is emitted — an empty user content crashes OpenAI
    inside mem0's consolidation pipeline.
    """
    content = response
    speaker = context.get("speaker_id", "")
    timestamp = context.get("timestamp", "")
    if speaker or timestamp:
        prefix_parts = []
        if timestamp:
            prefix_parts.append(timestamp)
        if speaker:
            prefix_parts.append(speaker)
        content = f"[{' | '.join(prefix_parts)}] {content}"

    if prompt.strip():
        messages = [
            {"role": "user", "content": prompt[:8000]},
            {"role": "assistant", "content": content[:8000]},
        ]
    else:
        messages = [{"role": "user", "content": content[:8000]}]

    metadata: dict[str, Any] = {}
    if timestamp:
        metadata["timestamp"] = timestamp
    if speaker:
        metadata["speaker_id"] = speaker
    source_id = context.get("source_id")
    if source_id:
        metadata["source_id"] = str(source_id)
    return messages, metadata


def _extract_source_ids(memories: list[dict[str, Any]]) -> list[str]:
    """Pull ordered, de-duplicated ``source_id`` values from a search result.

    Mem0 consolidates multiple input messages into each stored memory, so the
    same source_id can appear repeatedly across hits — preserve rank order
    but emit each id once. Memories without metadata.source_id are skipped
    (they predate this plumbing or came from a no-source ingestion path).
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in memories:
        meta = m.get("metadata") or {}
        sid = meta.get("source_id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(str(sid))
    return out


def _format_memories_block(memories: list[dict[str, Any]], prompt: str) -> str:
    """Render retrieved memories as a prepended context block."""
    if not memories:
        return prompt

    lines: list[str] = []
    for m in memories:
        text = m.get("memory", "")
        meta = m.get("metadata") or {}
        # Prefer the conversation timestamp we stored in metadata over the
        # ingestion-time ``created_at`` — mem0 overrides ``created_at`` with
        # ``datetime.now(utc)`` in its consolidation path, which is never
        # what LoCoMo temporal questions need.
        ts = meta.get("timestamp") or m.get("created_at", "")
        if ts:
            lines.append(f"[{ts}] {text}")
        else:
            lines.append(text)

    memory_text = "\n".join(f"- {line}" for line in lines)
    return f"Relevant context from memory:\n{memory_text}\n\n{prompt}"


def _build_embedded_config(params: dict[str, Any]) -> dict[str, Any]:
    """Translate flat params into the v2 ``Memory.from_config`` dict.

    v2-specific: unified ``custom_instructions`` instead of the two v1
    prompts (``custom_fact_extraction_prompt`` + ``custom_update_memory_prompt``).
    """
    cfg: dict[str, Any] = {}

    # LLM
    llm_provider = params.get("llm_provider")
    if llm_provider:
        llm_cfg: dict[str, Any] = {"provider": llm_provider, "config": {}}
        if params.get("llm_model"):
            llm_cfg["config"]["model"] = params["llm_model"]
        if params.get("llm_base_url"):
            if llm_provider == "ollama":
                llm_cfg["config"]["ollama_base_url"] = params["llm_base_url"]
            elif llm_provider in _OPENAI_BASE_URL_PROVIDERS:
                llm_cfg["config"]["openai_base_url"] = params["llm_base_url"]
            else:
                raise ValueError(
                    f"llm_base_url is not supported for llm_provider "
                    f"'{llm_provider}' in this middleware"
                )
        if params.get("llm_temperature") is not None:
            llm_cfg["config"]["temperature"] = params["llm_temperature"]
        # AWS Bedrock auth: ``AWSBedrockConfig`` accepts ``aws_region`` +
        # ``aws_profile``. Inherited from the bench LLM block when the
        # bench driver is ``anthropic_bedrock`` (see config.py
        # ``_BENCH_TO_MEMORY_PROVIDER`` + ``_MEMORY_LLM_KEY_MAP``).
        if llm_provider == "aws_bedrock":
            if params.get("aws_region"):
                llm_cfg["config"]["aws_region"] = params["aws_region"]
            if params.get("aws_profile"):
                llm_cfg["config"]["aws_profile"] = params["aws_profile"]
        cfg["llm"] = llm_cfg

    # Embedder
    emb_provider = params.get("embedder_provider")
    if emb_provider:
        emb_cfg: dict[str, Any] = {"provider": emb_provider, "config": {}}
        if params.get("embedder_model"):
            emb_cfg["config"]["model"] = params["embedder_model"]
        if params.get("llm_base_url") and emb_provider == "ollama":
            emb_cfg["config"]["ollama_base_url"] = params["llm_base_url"]
        cfg["embedder"] = emb_cfg

    # Vector store
    vs_provider = params.get("vector_store_provider")
    if vs_provider:
        vs_cfg: dict[str, Any] = {"provider": vs_provider, "config": {}}
        if params.get("vector_store_path"):
            vs_cfg["config"]["path"] = params["vector_store_path"]
        if params.get("vector_store_host"):
            vs_cfg["config"]["host"] = params["vector_store_host"]
        if params.get("vector_store_port"):
            vs_cfg["config"]["port"] = params["vector_store_port"]
        if params.get("vector_store_collection"):
            vs_cfg["config"]["collection_name"] = params["vector_store_collection"]
        cfg["vector_store"] = vs_cfg

    if params.get("custom_instructions"):
        cfg["custom_instructions"] = params["custom_instructions"]
    if params.get("version"):
        cfg["version"] = params["version"]

    return cfg


# ---------------------------------------------------------------------------
# Local embedded — mem0v2_embedded
# ---------------------------------------------------------------------------


@register_memory("mem0v2_embedded")
class Mem0V2EmbeddedMiddleware(MemoryMiddleware):
    """Mem0 v2.0.0 local ``Memory`` middleware (in-process).

    Config params (all go in ``MemoryConfig.params``):
        user_id:           Scope memories to this user (default ``bench_default``)
        agent_id:          Optional agent scope
        retrieval_limit:   Max memories returned by search (default 20)

        # --- LLM for fact extraction ---
        llm_provider:      ``"openai"`` | ``"ollama"`` | ``"anthropic"`` | ...
        llm_model:         e.g. ``"gpt-4.1-nano"``
        llm_base_url:      e.g. ``"http://localhost:11434"``  (Ollama)
                           or   ``"https://api.openai.com/v1"`` (OpenAI-compat)
        llm_temperature:   e.g. 0.1

        # --- Embedder ---
        embedder_provider: ``"openai"`` | ``"ollama"`` | ``"huggingface"`` | ...
        embedder_model:    e.g. ``"text-embedding-3-small"``

        # --- Vector store ---
        vector_store_provider: ``"qdrant"`` (default) | ``"chroma"`` | ...
        vector_store_path:     local path for embedded Qdrant/ChromaDB
        vector_store_host:     host for remote Qdrant / pgvector etc.
        vector_store_port:     port for remote Qdrant / pgvector etc.

        # --- Extraction ---
        custom_instructions:   Unified v2 extraction/update instructions
                               (supersedes v1's split prompts).
        version:               ``"v1.1"`` (default, additive extraction)
        infer:                 When True (default, matches upstream
                               ``Memory.add(infer=True)``), each ingested
                               chunk is run through mem0's LLM fact-
                               extraction prompt before storage. The
                               stored memory is the natural-language
                               fact, not the original chunk. Set False
                               to store chunks verbatim — required for
                               code-RAG benchmarks (CrossCodeEval /
                               RepoBench) where the grader measures
                               literal identifier match. See
                               deviations.md C13 for the rationale and
                               ``configs/profiles/mem0_no_infer.toml``
                               for the stackable mixin.
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._user_id = config.params.get("user_id", "bench_default")
        self._agent_id = config.params.get("agent_id")
        self._retrieval_limit = config.params.get("retrieval_limit", 20)
        # Mem0's ``Memory.add`` default is ``infer=True``; we mirror it
        # so existing callers (LoCoMo and friends) keep producing the
        # same conversation-fact extraction. Set False from the profile
        # to store ingested chunks verbatim — required for code-RAG
        # benchmarks where literal identifiers must survive ingestion.
        self._infer = bool(config.params.get("infer", True))

        mem0_config = _build_embedded_config(config.params)

        from mem0 import Memory

        if mem0_config:
            self._memory = Memory.from_config(config_dict=mem0_config)
        else:
            self._memory = Memory()

        # Patch mem0's LLM classes so their internal ``generate_response``
        # surfaces token counts to the active accountant. Idempotent;
        # safe to call from every instance.
        install_mem0_token_hook()

        _debug("mem0v2_embedded.init", {
            "user_id": self._user_id,
            "retrieval_limit": self._retrieval_limit,
            "config_keys": list(mem0_config.keys()) if mem0_config else [],
        })

    def _on_mem0_internal_llm(self, prompt_text: str, response_text: str) -> None:
        """Callback installed via ContextVar before each ``Memory`` call.

        Records one internal LLM call's tokens via the base-class
        accountant. Tokenized under ``cl100k_base`` because mem0 drops
        the underlying response's ``usage`` before returning."""
        self._record_internal_llm(prompt=prompt_text, response=response_text)

    # ------------------------------------------------------------------
    # Search — v2.0.0: user_id must be in filters, not a top-level kwarg
    # ------------------------------------------------------------------

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()

        filters: dict[str, Any] = {"user_id": self._user_id}
        if self._agent_id:
            filters["agent_id"] = self._agent_id

        search_kwargs: dict[str, Any] = {
            "query": prompt[:500],
            "filters": filters,
            "top_k": self._retrieval_limit,
        }

        _debug("mem0v2_embedded.search.request", {
            "query_len": len(prompt),
            "query_preview": prompt[:200],
            "filters": filters,
            "top_k": self._retrieval_limit,
        })

        # Boundary: we hand mem0 a search query string.
        self._record_io(request_text=prompt[:500])

        # The accountant is set INSIDE the executor worker thread (it
        # doesn't propagate from here — see
        # ``run_in_executor_with_accountant``), so mem0's in-thread
        # ``llm.generate_response`` records its internal tokens.
        try:
            result = await run_in_executor_with_accountant(
                loop,
                self._on_mem0_internal_llm,
                lambda: self._memory.search(**search_kwargs),
            )
        except Exception as e:
            logger.warning("mem0v2 local search() failed (non-fatal): %s", e)
            _debug("mem0v2_embedded.search.error", {"error": str(e)})
            context["retrieved_context_ids"] = []
            return prompt

        memories = result.get("results", []) if isinstance(result, dict) else result

        _debug("mem0v2_embedded.search.result", {
            "count": len(memories),
            "top_3": [
                {"memory": m.get("memory", "")[:100], "score": m.get("score")}
                for m in memories[:3]
            ] if memories else [],
        })

        context["retrieved_context_ids"] = _extract_source_ids(memories)
        formatted = _format_memories_block(memories, prompt)
        # Boundary: mem0 returns memory items; the formatted block is
        # what we'd hand the bench LLM. Tokenize just the block (drop
        # the trailing ``prompt`` we re-attached).
        if isinstance(formatted, str) and formatted.endswith(prompt):
            block = formatted[: -len(prompt)]
        else:
            block = formatted if isinstance(formatted, str) else ""
        self._record_io(response_text=block)
        return formatted

    # ------------------------------------------------------------------
    # Add — v2.0.0: user_id is still a top-level kwarg on add()
    # ------------------------------------------------------------------

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        loop = asyncio.get_event_loop()
        messages, metadata = _format_content(prompt, response, context)

        add_kwargs: dict[str, Any] = {
            "messages": messages,
            "user_id": self._user_id,
            # Always forward ``infer`` explicitly (instead of relying on
            # the upstream default) so the per-profile choice is
            # auditable in the request log. ``self._infer`` defaults to
            # True to preserve historical conversational-benchmark
            # behavior; code-RAG benchmarks stack the
            # ``mem0_no_infer`` mixin to flip it.
            "infer": self._infer,
        }
        if self._agent_id:
            add_kwargs["agent_id"] = self._agent_id
        if metadata:
            add_kwargs["metadata"] = metadata

        _debug("mem0v2_embedded.add.request", {
            "msg_count": len(messages),
            "content_preview": messages[-1]["content"][:200],
            "speaker": metadata.get("speaker_id", ""),
            "timestamp": metadata.get("timestamp", ""),
            "infer": self._infer,
        })

        # Boundary: tokenize what we ship to mem0.
        request_text = "\n".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
        self._record_io(request_text=request_text)

        # Accountant set inside the worker thread (see search() above and
        # ``run_in_executor_with_accountant``) so mem0's fact-extraction
        # LLM calls during add() are counted.
        try:
            result = await run_in_executor_with_accountant(
                loop,
                self._on_mem0_internal_llm,
                lambda: self._memory.add(**add_kwargs),
            )
            results = result.get("results", []) if isinstance(result, dict) else []
            _debug("mem0v2_embedded.add.result", {
                "extracted_count": len(results),
                "memories": [r.get("memory", "")[:100] for r in results[:5]],
            })
            # Boundary: tokenize whatever mem0 surfaced back.
            response_text = "\n".join(
                r.get("memory", "") for r in results if isinstance(r, dict)
            )
            if response_text:
                self._record_io(response_text=response_text)
        except Exception as e:
            logger.warning("mem0v2 local add() failed (non-fatal): %s", e)
            _debug("mem0v2_embedded.add.error", {"error": str(e)})

    # ------------------------------------------------------------------
    # Reset — v2.0.0: delete_all still accepts top-level user_id
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        loop = asyncio.get_event_loop()
        _debug("mem0v2_embedded.reset", {"user_id": self._user_id})
        try:
            await loop.run_in_executor(
                None, lambda: self._memory.delete_all(user_id=self._user_id)
            )
        except Exception:
            # Fallback: full reset
            await loop.run_in_executor(None, self._memory.reset)
        self._reset_internal_counters()


# ---------------------------------------------------------------------------
# Cloud API — mem0v2_api
# ---------------------------------------------------------------------------


@register_memory("mem0v2_api")
class Mem0V2ApiMiddleware(MemoryMiddleware):
    """Mem0 v2.0.0 cloud ``MemoryClient`` middleware.

    Calls the v3 endpoints at api.mem0.ai. All three operations
    (``search``, ``add``, ``delete_all``) scope by ``filters=`` — the v3
    server rejects top-level entity kwargs (``user_id`` / ``agent_id`` /
    ``app_id`` / ``run_id``) with a ``ValueError`` raised inside the
    client.

    Config params:
        user_id:           Scope (default ``bench_default``)
        agent_id:          Optional scope
        app_id:            Optional scope
        run_id:            Optional scope (new in v2 — a per-session handle)
        retrieval_limit:   ``top_k`` for search (default 20)
        api_key:           Overrides the ``MEM0_API_KEY`` env var
        host:              Overrides default api.mem0.ai

    Notes:
        - ``MemoryClient()`` no longer accepts ``org_id`` or ``project_id``
          at construction — they are resolved server-side from the API key.
        - ``reset()`` always uses scoped ``delete_all(filters=...)`` to avoid
          wiping unrelated memories across the whole workspace.
        - Responses are always ``{"results": [...]}`` from the v3 endpoints.
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._user_id = config.params.get("user_id", "bench_default")
        self._agent_id = config.params.get("agent_id")
        self._app_id = config.params.get("app_id")
        self._run_id = config.params.get("run_id")
        self._retrieval_limit = config.params.get("retrieval_limit", 20)
        # See the embedded variant for ``infer`` rationale + the
        # ``mem0_no_infer`` mixin. Default True mirrors upstream
        # ``MemoryClient.add(infer=...)``'s default.
        self._infer = bool(config.params.get("infer", True))

        from mem0 import MemoryClient

        client_kwargs: dict[str, Any] = {}
        if config.params.get("api_key"):
            client_kwargs["api_key"] = config.params["api_key"]
        if config.params.get("host"):
            client_kwargs["host"] = config.params["host"]

        self._client = MemoryClient(**client_kwargs)

        _debug("mem0v2_api.init", {
            "user_id": self._user_id,
            "retrieval_limit": self._retrieval_limit,
            "host": client_kwargs.get("host", "default"),
            "infer": self._infer,
        })

    def _scope_filters(self) -> dict[str, Any]:
        """Build the ``filters`` dict used by search/add/delete_all.

        The v3 server expects entity IDs inside ``filters=``; supplying them
        top-level to ``search()`` raises ``ValueError`` in the client.
        """
        f: dict[str, Any] = {"user_id": self._user_id}
        if self._agent_id:
            f["agent_id"] = self._agent_id
        if self._app_id:
            f["app_id"] = self._app_id
        if self._run_id:
            f["run_id"] = self._run_id
        return f

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()

        filters = self._scope_filters()
        search_kwargs: dict[str, Any] = {
            "filters": filters,
            "top_k": self._retrieval_limit,
        }

        _debug("mem0v2_api.search.request", {
            "query_len": len(prompt),
            "filters": filters,
            "top_k": self._retrieval_limit,
        })

        # Boundary only — cloud variant's LLM runs server-side on
        # api.mem0.ai; ``_internal_tokens_*`` stay at zero because no
        # local LLM is invoked.
        self._record_io(request_text=prompt[:500])

        try:
            result = await loop.run_in_executor(
                None, lambda: self._client.search(query=prompt[:500], **search_kwargs)
            )
        except Exception as e:
            logger.warning("mem0v2 cloud search() failed (non-fatal): %s", e)
            _debug("mem0v2_api.search.error", {"error": str(e)})
            context["retrieved_context_ids"] = []
            return prompt

        if isinstance(result, dict):
            memories = result.get("results", [])
        elif isinstance(result, list):
            # Defensive: v3 endpoints wrap, but tolerate a bare list too.
            memories = result
        else:
            memories = []

        _debug("mem0v2_api.search.result", {"count": len(memories)})
        context["retrieved_context_ids"] = _extract_source_ids(memories)
        formatted = _format_memories_block(memories, prompt)
        if isinstance(formatted, str) and formatted.endswith(prompt):
            block = formatted[: -len(prompt)]
        else:
            block = formatted if isinstance(formatted, str) else ""
        self._record_io(response_text=block)
        return formatted

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        loop = asyncio.get_event_loop()
        messages, metadata = _format_content(prompt, response, context)

        add_kwargs: dict[str, Any] = {
            "filters": self._scope_filters(),
            "infer": self._infer,
        }
        if metadata:
            add_kwargs["metadata"] = metadata

        _debug("mem0v2_api.add.request", {
            "msg_count": len(messages),
            "filters": add_kwargs["filters"],
            "infer": self._infer,
        })

        request_text = "\n".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
        self._record_io(request_text=request_text)

        try:
            result = await loop.run_in_executor(
                None, lambda: self._client.add(messages, **add_kwargs)
            )
            if isinstance(result, dict):
                results = result.get("results", [])
            elif isinstance(result, list):
                results = result
            else:
                results = []
            response_text = "\n".join(
                r.get("memory", "") for r in results if isinstance(r, dict)
            )
            if response_text:
                self._record_io(response_text=response_text)
        except Exception as e:
            logger.warning("mem0v2 cloud add() failed (non-fatal): %s", e)
            _debug("mem0v2_api.add.error", {"error": str(e)})

    async def reset(self) -> None:
        loop = asyncio.get_event_loop()
        filters = self._scope_filters()
        _debug("mem0v2_api.reset", {"filters": filters})
        # Use scoped ``delete_all(filters=...)`` rather than
        # ``MemoryClient.reset()``. The global ``reset()`` exists
        # (``mem0/client/main.py:491``) and calls ``delete_users()`` which
        # would wipe the entire workspace — every user, every benchmark
        # run, on a shared API key. Scoped delete_all only clears the
        # current ``user_id`` scope, which is what per-conversation
        # isolation needs.
        try:
            await loop.run_in_executor(
                None, lambda: self._client.delete_all(filters=filters)
            )
        except Exception as e:
            logger.warning("mem0v2 cloud delete_all() failed (non-fatal): %s", e)
            _debug("mem0v2_api.reset.error", {"error": str(e)})
        self._reset_internal_counters()
