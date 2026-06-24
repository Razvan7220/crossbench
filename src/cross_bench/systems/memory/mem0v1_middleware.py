"""Mem0 memory middleware — pre-v2.0.0 API (v1.0.x).

Two registrations live in this file, sharing config translation and
pre/post-prompt formatting:

* ``mem0v1_embedded`` — local ``Memory`` (open-source, in-process).
* ``mem0v1_api``      — cloud ``MemoryClient`` (api.mem0.ai).

Install:
    pip install "mem0ai>=1.0,<2"

> Audited against ``mem0ai==1.0.11`` at commit ``144627c4`` (release
> tag ``v1.0.11``). Files inspected: ``mem0/memory/main.py``
> (``Memory.add``, ``Memory.search``, ``Memory.delete_all``,
> ``Memory.reset``), ``mem0/client/main.py`` (``MemoryClient.add``,
> ``.search``, ``.delete_all``, ``.reset``, ``__init__``),
> ``mem0/configs/base.py`` (``MemoryConfig.version`` default ``"v1.1"``).
> See ``deviations.md`` for the line-by-line audit table.

v1 → v2 API deltas relevant here (local ``Memory``):
    - ``search()`` takes top-level ``user_id`` / ``agent_id`` / ``run_id``
      kwargs (v2 requires them inside ``filters``).
    - ``search()`` uses ``limit=`` (default 100); v2 renamed this to
      ``top_k=`` and changed the default to 20.
    - ``search()`` default ``rerank=True`` (v2 default is False). We
      DELIBERATELY keep v1's default — paper-faithfulness means each
      version uses its own upstream defaults; rerank=True is what a v1
      user gets out-of-the-box, and that IS the v1 cell of the
      leaderboard. Cross-version comparison must account for this.
    - ``search()`` default ``threshold=None`` (v2 default 0.1).
    - ``MemoryConfig`` uses ``custom_fact_extraction_prompt`` and
      ``custom_update_memory_prompt`` (v2 replaced both with
      ``custom_instructions``).
    - v1 supports graphs (``enable_graph`` config); ``add()`` returns
      ``{"results": ..., "relations": ...}`` when graph is enabled. We
      surface only ``results`` to the answering LLM — graph relations
      are not part of the LoCoMo evaluation contract. See deviations.md.

For the cloud client:
    - ``__init__`` accepts ``org_id`` and ``project_id`` (REMOVED in
      v2 — server resolves them from API key).
    - ``add()``, ``search()``, ``delete_all()`` take everything via
      ``**kwargs`` (``user_id``, ``agent_id``, ``app_id``,
      ``metadata``, ``filters``, ``top_k``, ``async_mode``,
      ``output_format``). v1's add() defaults ``async_mode=True`` and
      forces ``output_format="v1.1"`` server-side.
    - Endpoints: ``add()`` → ``/v1/memories/`` (v2 uses
      ``/v3/memories/add/``); ``search()`` → ``/v2/memories/search/``
      (v2 uses ``/v3/memories/search/``); ``delete_all()`` →
      ``/v1/memories/`` DELETE.
    - Results are wrapped to ``{"results": [...]}`` (raw list is also
      tolerated and wrapped on the fly).

Provenance (LoCoMo evidence-recall plumbing): ``post_response`` writes
``metadata["source_id"]`` from the engine context; ``pre_prompt``
reads it back via ``_extract_source_ids`` and surfaces the rank-ordered
de-duplicated list as ``context["retrieved_context_ids"]``. Same
contract as the v2 middleware.
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

_DEBUG_LOG = os.environ.get("MEM0_DEBUG_LOG", "")


def _assert_mem0_v1_installed() -> None:
    """Verify the installed ``mem0ai`` is v1, not v2.

    v1 and v2 share the import name ``mem0`` but have incompatible
    ``Memory.search()`` signatures: v1 takes top-level ``user_id`` /
    ``limit``; v2 requires ``filters={"user_id": ...}`` / ``top_k``.
    With v2 installed, every v1 ``search()`` call raises::

        ValueError: Top-level entity parameters frozenset({'user_id'})
        are not supported in search().

    The middleware's broad ``except Exception`` would then swallow it
    as ``non-fatal`` and return an empty retrieval — leaving every
    task with no memory context. The 2026-05-21 ``locomo-mem0v1-...``
    run hit this exactly: 1986 tasks, 100% empty retrieval, F1=23%
    (all cat-5 abstentions), LLM judge=0.35%. See the diagnosis in
    deviations.md § ``mem0v1_embedded``.

    Raises:
        RuntimeError: if v2 (or any non-1.x version) is installed, OR
            if ``mem0ai`` is not installed at all.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version
    except ImportError:  # pragma: no cover — Python <3.8
        return
    try:
        installed = _pkg_version("mem0ai")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "mem0v1 middleware requires the ``mem0ai`` package "
            "(``pip install 'mem0ai>=1.0,<2'``). "
            "Install via ``pip install cross-bench[memory-mem0v1]`` in a "
            "venv that does NOT also have ``memory-mem0v2`` — the two "
            "versions share the import name ``mem0`` and cannot coexist."
        ) from exc
    major = installed.split(".")[0]
    if major != "1":
        raise RuntimeError(
            f"mem0v1 middleware found mem0ai=={installed}, expected 1.x. "
            "v1 and v2 share the import name ``mem0`` but have incompatible "
            "``Memory.search()`` signatures — v1 uses top-level ``user_id`` / "
            "``limit``; v2 requires ``filters={'user_id': ...}`` / ``top_k``. "
            "Running v1 middleware against v2 install will silently fail "
            "every retrieval (see the 2026-05-21 locomo-mem0v1 run "
            "diagnosed in deviations.md). "
            "Install ``cross-bench[memory-mem0v1]`` in a separate image/venv "
            "from ``cross-bench[memory-mem0v2]``."
        )


def _debug(event: str, data: dict[str, Any]) -> None:
    if not _DEBUG_LOG:
        return
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **data}, default=str) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_content(prompt: str, response: str, context: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the (messages, metadata) pair for add() from a response + context.

    Matches the prefix/metadata convention used by the v2 middleware so
    speaker/timestamp grounding AND source-id provenance are consistent
    across versions. Upstream v1 ``Memory.add()`` (``mem0/memory/main.py:370``)
    accepts arbitrary ``metadata=`` and deepcopies it through
    ``_build_filters_and_metadata`` and ``_create_memory``, so our extra
    keys (``timestamp``, ``speaker_id``, ``source_id``) survive untouched
    on every persisted memory.
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

    Mirrors the v2 helper. Memories without ``metadata.source_id`` are
    skipped (they predate this plumbing or came from a no-source ingestion
    path) — never fabricated.
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
        # ingestion-time ``created_at``, which mem0 sets to datetime.now().
        ts = meta.get("timestamp") or m.get("created_at", "")
        if ts:
            lines.append(f"[{ts}] {text}")
        else:
            lines.append(text)

    memory_text = "\n".join(f"- {line}" for line in lines)
    return f"Relevant context from memory:\n{memory_text}\n\n{prompt}"


def _build_embedded_config(params: dict[str, Any]) -> dict[str, Any]:
    """Translate flat params into the v1 ``MemoryConfig`` dict.

    Notable v1 differences from v2:
        - ``custom_fact_extraction_prompt`` / ``custom_update_memory_prompt``
          (two keys) replace v2's ``custom_instructions``.
    """
    cfg: dict[str, Any] = {}

    llm_provider = params.get("llm_provider")
    if llm_provider:
        llm_cfg: dict[str, Any] = {"provider": llm_provider, "config": {}}
        if params.get("llm_model"):
            llm_cfg["config"]["model"] = params["llm_model"]
        if params.get("llm_base_url"):
            if llm_provider == "ollama":
                llm_cfg["config"]["ollama_base_url"] = params["llm_base_url"]
            else:
                # All other providers in v1 use ``openai_base_url`` on their
                # OpenAI-compatible config class.
                llm_cfg["config"]["openai_base_url"] = params["llm_base_url"]
        if params.get("llm_temperature") is not None:
            llm_cfg["config"]["temperature"] = params["llm_temperature"]
        cfg["llm"] = llm_cfg

    emb_provider = params.get("embedder_provider")
    if emb_provider:
        emb_cfg: dict[str, Any] = {"provider": emb_provider, "config": {}}
        if params.get("embedder_model"):
            emb_cfg["config"]["model"] = params["embedder_model"]
        if params.get("llm_base_url") and emb_provider == "ollama":
            emb_cfg["config"]["ollama_base_url"] = params["llm_base_url"]
        cfg["embedder"] = emb_cfg

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

    if params.get("custom_fact_extraction_prompt"):
        cfg["custom_fact_extraction_prompt"] = params["custom_fact_extraction_prompt"]
    if params.get("custom_update_memory_prompt"):
        cfg["custom_update_memory_prompt"] = params["custom_update_memory_prompt"]
    if params.get("version"):
        cfg["version"] = params["version"]

    return cfg


# ---------------------------------------------------------------------------
# Local embedded — mem0v1_embedded
# ---------------------------------------------------------------------------


@register_memory("mem0v1_embedded")
class Mem0V1EmbeddedMiddleware(MemoryMiddleware):
    """Mem0 v1.0.x local ``Memory`` middleware.

    Config params (flat, under ``MemoryConfig.params``):
        user_id:                         scope (default ``bench_default``)
        agent_id:                        optional scope
        retrieval_limit:                 search() ``limit`` (default 100 in v1)
        llm_provider / llm_model / llm_base_url / llm_temperature
        embedder_provider / embedder_model
        vector_store_provider / vector_store_path / vector_store_host /
        vector_store_port / vector_store_collection
        custom_fact_extraction_prompt    (v1-specific; supersedes v2's
                                          ``custom_instructions`` for the
                                          extraction step)
        custom_update_memory_prompt      (v1-specific; supersedes v2's
                                          ``custom_instructions`` for the
                                          update step)
        version:                         ``"v1.1"`` (default)
    """

    def __init__(self, config: MemoryConfig):
        _assert_mem0_v1_installed()
        self._config = config
        self._user_id = config.params.get("user_id", "bench_default")
        self._agent_id = config.params.get("agent_id")
        # v1 default is 100; keep our existing 20 as a more conservative cap.
        self._retrieval_limit = config.params.get("retrieval_limit", 20)

        mem0_config = _build_embedded_config(config.params)

        from mem0 import Memory

        if mem0_config:
            self._memory = Memory.from_config(config_dict=mem0_config)
        else:
            self._memory = Memory()

        # Patch mem0's LLM classes so their internal ``generate_response``
        # surfaces token counts to the active accountant. Idempotent.
        install_mem0_token_hook()

        _debug("mem0v1_embedded.init", {
            "user_id": self._user_id,
            "retrieval_limit": self._retrieval_limit,
            "config_keys": list(mem0_config.keys()) if mem0_config else [],
        })

    def _on_mem0_internal_llm(self, prompt_text: str, response_text: str) -> None:
        """Callback installed via ContextVar before each ``Memory`` call.

        Records one internal mem0 LLM call's tokens (cl100k_base
        tokenized — mem0 drops the underlying response's ``usage``)."""
        self._record_internal_llm(prompt=prompt_text, response=response_text)

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()

        # v1 takes user_id / agent_id / run_id as top-level kwargs and uses
        # ``limit`` not ``top_k``.
        search_kwargs: dict[str, Any] = {
            "query": prompt[:500],
            "user_id": self._user_id,
            "limit": self._retrieval_limit,
        }
        if self._agent_id:
            search_kwargs["agent_id"] = self._agent_id

        _debug("mem0v1_embedded.search.request", {
            "query_len": len(prompt),
            "user_id": self._user_id,
            "limit": self._retrieval_limit,
        })

        self._record_io(request_text=prompt[:500])

        # Accountant set inside the worker thread (it doesn't cross the
        # run_in_executor boundary — see ``run_in_executor_with_accountant``)
        # so mem0's in-thread ``llm.generate_response`` is counted.
        try:
            result = await run_in_executor_with_accountant(
                loop,
                self._on_mem0_internal_llm,
                lambda: self._memory.search(**search_kwargs),
            )
        except Exception as e:
            logger.warning("mem0v1 local search() failed (non-fatal): %s", e)
            _debug("mem0v1_embedded.search.error", {"error": str(e)})
            context["retrieved_context_ids"] = []
            return prompt

        memories = result.get("results", []) if isinstance(result, dict) else result
        _debug("mem0v1_embedded.search.result", {"count": len(memories)})
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
            "messages": messages,
            "user_id": self._user_id,
        }
        if self._agent_id:
            add_kwargs["agent_id"] = self._agent_id
        if metadata:
            add_kwargs["metadata"] = metadata

        _debug("mem0v1_embedded.add.request", {"msg_count": len(messages)})

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
            logger.warning("mem0v1 local add() failed (non-fatal): %s", e)
            _debug("mem0v1_embedded.add.error", {"error": str(e)})

    async def reset(self) -> None:
        loop = asyncio.get_event_loop()
        _debug("mem0v1_embedded.reset", {"user_id": self._user_id})
        try:
            await loop.run_in_executor(
                None, lambda: self._memory.delete_all(user_id=self._user_id)
            )
        except Exception:
            await loop.run_in_executor(None, self._memory.reset)
        self._reset_internal_counters()


# ---------------------------------------------------------------------------
# Cloud API — mem0v1_api
# ---------------------------------------------------------------------------


@register_memory("mem0v1_api")
class Mem0V1ApiMiddleware(MemoryMiddleware):
    """Mem0 v1.0.x cloud ``MemoryClient`` middleware.

    Config params:
        user_id:           scope (default ``bench_default``)
        agent_id:          optional scope
        app_id:            optional scope
        retrieval_limit:   ``top_k`` for search (default 20)
        api_key:           overrides ``MEM0_API_KEY`` env var
        host:              overrides default api.mem0.ai
        org_id / project_id: optional; pinned to a specific Mem0 workspace
        output_format:     ``"v1.1"`` (default) — forces a uniform response
                           shape regardless of server-side default.
        async_mode:        forwarded to add() as-is; v1 server defaults this
                           to True.
    """

    def __init__(self, config: MemoryConfig):
        _assert_mem0_v1_installed()
        self._config = config
        self._user_id = config.params.get("user_id", "bench_default")
        self._agent_id = config.params.get("agent_id")
        self._app_id = config.params.get("app_id")
        self._retrieval_limit = config.params.get("retrieval_limit", 20)
        self._output_format = config.params.get("output_format", "v1.1")
        self._async_mode = config.params.get("async_mode")

        from mem0 import MemoryClient

        client_kwargs: dict[str, Any] = {}
        if config.params.get("api_key"):
            client_kwargs["api_key"] = config.params["api_key"]
        if config.params.get("host"):
            client_kwargs["host"] = config.params["host"]
        if config.params.get("org_id"):
            client_kwargs["org_id"] = config.params["org_id"]
        if config.params.get("project_id"):
            client_kwargs["project_id"] = config.params["project_id"]

        self._client = MemoryClient(**client_kwargs)

        _debug("mem0v1_api.init", {
            "user_id": self._user_id,
            "retrieval_limit": self._retrieval_limit,
            "host": client_kwargs.get("host", "default"),
        })

    def _scope_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"user_id": self._user_id}
        if self._agent_id:
            kw["agent_id"] = self._agent_id
        if self._app_id:
            kw["app_id"] = self._app_id
        return kw

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()

        kwargs: dict[str, Any] = {
            **self._scope_kwargs(),
            "top_k": self._retrieval_limit,
            "output_format": self._output_format,
        }

        _debug("mem0v1_api.search.request", {
            "query_len": len(prompt),
            "top_k": self._retrieval_limit,
            "scope": self._scope_kwargs(),
        })

        # Boundary only — cloud variant's LLM runs server-side.
        self._record_io(request_text=prompt[:500])

        try:
            result = await loop.run_in_executor(
                None, lambda: self._client.search(query=prompt[:500], **kwargs)
            )
        except Exception as e:
            logger.warning("mem0v1 cloud search() failed (non-fatal): %s", e)
            _debug("mem0v1_api.search.error", {"error": str(e)})
            context["retrieved_context_ids"] = []
            return prompt

        if isinstance(result, dict):
            memories = result.get("results", [])
        elif isinstance(result, list):
            # Server sometimes returns a bare list; client wraps it, but
            # defend against either shape.
            memories = result
        else:
            memories = []

        _debug("mem0v1_api.search.result", {"count": len(memories)})
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

        kwargs: dict[str, Any] = {
            **self._scope_kwargs(),
            "output_format": self._output_format,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if self._async_mode is not None:
            kwargs["async_mode"] = self._async_mode

        _debug("mem0v1_api.add.request", {"msg_count": len(messages)})

        request_text = "\n".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
        self._record_io(request_text=request_text)

        try:
            result = await loop.run_in_executor(
                None, lambda: self._client.add(messages, **kwargs)
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
            logger.warning("mem0v1 cloud add() failed (non-fatal): %s", e)
            _debug("mem0v1_api.add.error", {"error": str(e)})

    async def reset(self) -> None:
        loop = asyncio.get_event_loop()
        _debug("mem0v1_api.reset", {"user_id": self._user_id})
        # MemoryClient.reset() nukes the whole workspace — always use
        # scoped delete_all to avoid wiping unrelated memories.
        try:
            await loop.run_in_executor(
                None, lambda: self._client.delete_all(**self._scope_kwargs())
            )
        except Exception as e:
            logger.warning("mem0v1 cloud delete_all() failed (non-fatal): %s", e)
            _debug("mem0v1_api.reset.error", {"error": str(e)})
        self._reset_internal_counters()
