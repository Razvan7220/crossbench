"""EverMemOS API middleware — wraps the official ``evermemos`` Python SDK.

This is the *cloud-only* API variant. It targets the EverMind cloud
service at ``https://api.evermind.ai`` (or any base_url that exposes
the same ``/api/v0/`` paths) using the Stainless-generated SDK
``evermemos>=0.3.13``.

Why cloud-only:
    The previous incarnation of this middleware spoke to a local
    Docker server at ``http://localhost:1995`` via an internal
    ``/api/v1/`` API whose source isn't published in the public
    EverOS repo (``methods/evermemos/src/`` is gitignored). Targeting
    an unverifiable contract made the middleware a moving target and
    blocked any honest leaderboard claim. The official SDK uses
    ``/api/v0/``, which is a published OpenAPI contract; rewriting
    against the SDK closes deviations E.A1–E.A5 in
    ``deviations.md``. If you need to run against a self-hosted
    server, point ``base_url`` at it AND verify it exposes the same
    ``/api/v0/`` paths — see deviations.md for the audit trail.

Install:
    pip install evermemos>=0.3.13

API contract (per ``evermemos==0.3.13``):
    POST   /api/v0/memories          add (single message, ``flush`` field)
    GET    /api/v0/memories/search   search (params via JSON body)
    GET    /api/v0/memories          get
    DELETE /api/v0/memories          delete (filters: id/event_id/group_id/memory_id/user_id)
    Auth:  ``Authorization: Bearer <api_key>``

Memory-middleware contract:
    * ``post_response`` buffers each turn in middleware memory.
    * ``pre_prompt`` flushes the buffer (one ``add()`` per message,
      with ``flush=True`` on the last one to force the server's
      boundary detection to run before search), then runs ``search()``.
    * ``reset()`` deletes the user's memories scoped to ``user_id``.
    * ``retrieved_context_ids`` is left **unset** — see deviation
      E.A7. Server-side episodes don't preserve caller-supplied IDs,
      so honest no-op > inventing values.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


# Memory types the SDK supports (per types/v0/memory_type.py and the
# search response discriminated-union). Anything else gets dropped.
_VALID_MEMORY_TYPES: frozenset[str] = frozenset(
    {"episodic_memory", "event_log", "foresight", "profile"}
)


@register_memory("evermemos_api")
class EverMemOSMiddleware(MemoryMiddleware):
    """EverMemOS cloud SDK middleware (``evermemos==0.3.13``).

    Config params (all in ``MemoryConfig.params``):
        base_url:         API URL (default ``https://api.evermind.ai``)
                          — also reads ``EVER_MEM_OS_BASE_URL`` env var.
        api_key:          API key (defaults to ``EVERMEMOS_API_KEY`` env var).
        user_id:          Sender user ID for memory ownership
                          (default ``bench_user``). Passed as ``sender``
                          on every add (the SDK uses sender as user_id
                          internally).
        group_id:         Optional explicit group ID. If absent, the
                          server auto-generates ``hash(sender) +
                          '__group'`` (single-user mode).
        retrieve_method:  ``keyword`` | ``vector`` | ``hybrid`` |
                          ``rrf`` | ``agentic`` (default ``hybrid``).
        memory_types:     Comma-separated list (or list) drawn from
                          ``{episodic_memory, event_log, foresight,
                          profile}`` (default ``episodic_memory``).
        retrieval_limit:  ``top_k`` for search (default 10).
        timeout:          HTTP timeout in seconds (default 30).
        max_retries:      SDK-level retry count (default 3).
    """

    def __init__(self, config: MemoryConfig):
        from evermemos import AsyncEverMemOS  # type: ignore[import-not-found]

        self._config = config
        p = config.params or {}

        api_key = p.get("api_key") or os.environ.get("EVERMEMOS_API_KEY")
        if not api_key:
            raise ValueError(
                "evermemos_api requires an API key — set [system.memory] "
                "api_key=... in the profile or export EVERMEMOS_API_KEY."
            )

        # Resolve base_url. Empty string means "use SDK default", which
        # is api.evermind.ai (the published cloud).
        base_url_raw = p.get("base_url", "")
        base_url = base_url_raw.rstrip("/") if base_url_raw else None

        timeout = float(p.get("timeout", 30))
        max_retries = int(p.get("max_retries", 3))

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncEverMemOS(**client_kwargs)

        self._user_id = str(p.get("user_id", "bench_user"))
        self._group_id: str | None = p.get("group_id") or None
        self._retrieve_method = str(p.get("retrieve_method", p.get("method", "hybrid")))
        self._memory_types = self._parse_memory_types(
            p.get("memory_types", "episodic_memory")
        )
        self._retrieval_limit = int(p.get("retrieval_limit", p.get("top_k", 10)))

        # Pending writes. Each entry is a fully-shaped ``add()`` kwargs
        # dict — we buffer in middleware memory so we can stamp
        # ``flush=True`` on the last one in a batch.
        self._pending: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_memory_types(raw: Any) -> list[str]:
        if isinstance(raw, str):
            items = [t.strip() for t in raw.split(",") if t.strip()]
        elif isinstance(raw, (list, tuple)):
            items = [str(t).strip() for t in raw if str(t).strip()]
        else:
            items = []
        filtered = [t for t in items if t in _VALID_MEMORY_TYPES]
        dropped = [t for t in items if t not in _VALID_MEMORY_TYPES]
        if dropped:
            logger.warning(
                "evermemos_api: dropping unsupported memory_types %s "
                "(valid: %s)",
                dropped,
                sorted(_VALID_MEMORY_TYPES),
            )
        return filtered or ["episodic_memory"]

    @staticmethod
    def _to_iso8601(ts: Any) -> str:
        """Normalise ISO string / epoch-seconds / epoch-ms to ISO 8601 with TZ.

        The SDK requires ``create_time`` to be ISO 8601 *with timezone*.
        """
        if ts is None or ts == "":
            return datetime.now(timezone.utc).isoformat()
        if isinstance(ts, (int, float)):
            secs = ts / 1000.0 if ts >= 1e12 else float(ts)
            return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                pass
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _stable_message_id(context: dict[str, Any]) -> str:
        """Pick a stable per-message ID for the SDK's required ``message_id``.

        Prefers the engine-supplied ``source_id`` (LoCoMo's ``D1:3``-style
        provenance tag); falls back to a UUID4 if absent. The SDK rejects
        empty / missing message_id with a 4xx, so a fallback is mandatory.
        """
        sid = context.get("source_id")
        if sid:
            return str(sid)
        return f"msg-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # MemoryMiddleware protocol
    # ------------------------------------------------------------------

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Flush any buffered writes, then search and prepend.

        Provenance: EverMemOS consolidates raw messages into episodes
        server-side; the SDK has no slot for caller-supplied IDs on the
        ``add`` payload that survive consolidation. Per the
        ``MemoryMiddleware`` contract we leave
        ``retrieved_context_ids`` unset so downstream aggregators
        report ``evidence_recall_measured=False`` rather than mixing
        in invented IDs.
        """
        # Boundary (CrossBench↔middleware seam): tokenize the query.
        self._record_io(request_text=prompt[:500])

        if self._pending:
            await self._flush()

        results = await self._search(prompt)
        if not results:
            return prompt
        formatted = self._format_search_results(results, prompt)
        if isinstance(formatted, str) and formatted.endswith(prompt):
            self._record_io(response_text=formatted[: -len(prompt)])
        return formatted

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Buffer one SDK ``add()`` call per turn, flushed lazily.

        Each entry is the fully-resolved kwargs dict for
        ``client.v0.memories.add()``. We don't call the SDK here so we
        can stamp ``flush=True`` on the LAST entry of the batch in
        ``_flush``.
        """
        text = (response or "").strip()
        if not text:
            return

        # Boundary: tokenize the content we ship to the EverMemOS server.
        self._record_io(request_text=text)

        speaker = context.get("speaker_id") or self._user_id
        speaker_name = context.get("speaker_name") or speaker
        role = context.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"

        kwargs: dict[str, Any] = {
            "content": text,
            "create_time": self._to_iso8601(context.get("timestamp")),
            "message_id": self._stable_message_id(context),
            "sender": str(speaker),
            "sender_name": str(speaker_name),
            "role": role,
        }
        if self._group_id:
            kwargs["group_id"] = self._group_id

        self._pending.append(kwargs)

    async def reset(self) -> None:
        """Flush any pending writes, then delete this user's memories.

        ``client.v0.memories.delete(user_id=...)`` is the SDK's scoped
        deletion path. Equivalent to mem0v2's
        ``delete_all(filters={'user_id': ...})``: removes only this
        user's memories on the server, not the whole workspace.
        """
        if self._pending:
            try:
                await self._flush()
            except Exception as e:
                logger.warning("EverMemOS flush on reset failed: %s", e)
        try:
            await self._client.v0.memories.delete(user_id=self._user_id)
        except Exception as e:
            logger.warning("EverMemOS delete on reset failed: %s", e)
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # SDK call sites
    # ------------------------------------------------------------------

    async def _flush(self) -> None:
        """Drain the buffer, calling SDK.add() once per message.

        Sets ``flush=True`` on the last call only — that triggers
        immediate boundary detection on the server so the upcoming
        search sees freshly-extracted episodes. Earlier calls let the
        server batch them into its natural boundary detector.

        If a single call fails, log + continue: the cloud SDK has its
        own retries, so by the time an exception escapes, the request
        has already been retried. Losing one message is preferable to
        aborting the whole conversation.
        """
        if not self._pending:
            return
        batch = self._pending
        self._pending = []

        last = len(batch) - 1
        for i, kwargs in enumerate(batch):
            call_kwargs = dict(kwargs)
            if i == last:
                call_kwargs["flush"] = True
            try:
                await self._client.v0.memories.add(**call_kwargs)
            except Exception as e:
                logger.warning(
                    "EverMemOS add() failed for message_id=%s: %s",
                    call_kwargs.get("message_id"),
                    e,
                )

    async def _search(self, prompt: str) -> list[dict[str, Any]]:
        """Call SDK.search() with our query/top_k/method/etc. via extra_body.

        The SDK's ``search()`` signature has no body params (the
        endpoint is GET ``/api/v0/memories/search``); passing
        ``extra_body=`` puts the query into the request body, which is
        what the server expects per the response shape (the
        ``query_metadata`` it echoes back is exactly the keys we send).
        """
        body: dict[str, Any] = {
            "query": prompt[:1000],
            "top_k": self._retrieval_limit,
            "memory_types": self._memory_types,
            "retrieve_method": self._retrieve_method,
            "user_id": self._user_id,
            "include_metadata": False,
        }
        if self._group_id:
            body["group_id"] = self._group_id

        try:
            resp = await self._client.v0.memories.search(extra_body=body)
        except Exception as e:
            logger.warning("EverMemOS search() failed: %s", e)
            return []

        return self._normalise_search_response(resp)

    def _normalise_search_response(self, resp: Any) -> list[dict[str, Any]]:
        """Walk ``MemorySearchResponse.result.memories`` into a flat list.

        The SDK returns a discriminated union per memory: EpisodeMemory
        carries ``episode``, EventLog carries ``atomic_fact``, Foresight
        carries ``foresight``. We unify into ``{type, content, timestamp,
        score}`` for downstream rendering.
        """
        result = getattr(resp, "result", None)
        memories = getattr(result, "memories", None) if result is not None else None
        if not memories:
            return []

        out: list[dict[str, Any]] = []
        for mem in memories:
            mem_type = (
                getattr(mem, "memory_type", "")
                or ""
            )
            mem_type_str = str(mem_type) if mem_type else ""

            # Pick the type-specific content field, falling back to
            # summary / subject for episodes.
            content = (
                getattr(mem, "episode", None)
                or getattr(mem, "atomic_fact", None)
                or getattr(mem, "foresight", None)
                or getattr(mem, "summary", None)
                or getattr(mem, "subject", None)
                or ""
            )
            if isinstance(content, list):
                # atomic_fact may be a list[str] — join for display.
                content = "; ".join(str(x) for x in content if x)
            if not content:
                continue

            ts = getattr(mem, "timestamp", None)
            if isinstance(ts, datetime):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts) if ts else ""

            out.append(
                {
                    "type": mem_type_str,
                    "content": str(content),
                    "timestamp": ts_str,
                    "score": getattr(mem, "score", None),
                }
            )

        return out

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _format_search_results(
        self, results: list[dict[str, Any]], prompt: str
    ) -> str:
        lines: list[str] = []
        for r in results:
            mem_type = r.get("type") or ""
            ts = r.get("timestamp") or ""
            content = r.get("content") or ""
            if not content:
                continue
            prefix = f"[{mem_type}]" if mem_type else "-"
            if ts:
                prefix = f"{prefix} [{ts}]"
            lines.append(f"{prefix} {content}")
        if not lines:
            return prompt
        memory_text = "\n".join(lines)
        return f"Retrieved memories:\n{memory_text}\n\n{prompt}"
