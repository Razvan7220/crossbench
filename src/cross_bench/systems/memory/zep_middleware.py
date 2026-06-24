"""Zep memory middleware — v2.x SDK (modern API).

Migrated from v0.x on 2026-05-21. The v0.x ``zep-python`` package
carried a hard pin ``httpx>=0.24,<0.25`` in its setup.py that conflicted
with the rest of our stack (``openai`` / ``anthropic`` / ``mem0ai`` all
require ``httpx>=0.27``). zep-python v2.x drops that pin and uses a
redesigned API around the ``Zep`` / ``AsyncZep`` client. The migration
is documented under deviations.md (zep section, v0→v2 entry).

This middleware works with both the **self-hosted** Zep server (via
``base_url`` pointed at your local Docker Compose deployment) and the
**managed** Zep cloud service (``base_url=https://api.getzep.com``,
plus a real ``api_key``). The provider name ``zep_api`` is unchanged.

Install:
    pip install "zep-python>=2.0,<3"

Running Zep self-hosted:
    See https://github.com/getzep/zep for the current docker-compose
    setup. Zep CE is deprecated by upstream as of April 2025 in favor
    of Graphiti; we keep this adapter for backward compatibility and
    for the managed-cloud option, but Graphiti
    (graphiti_middleware.py) is the actively-maintained alternative.

v0.x → v2.x API mapping (for posterity):
    ZepClient(api_url=, api_key=)         → AsyncZep(base_url=, api_key=)
    memory.aadd_memory(session_id, Memory(messages=[Message(role=, content=)]))
        → memory.add(session_id, messages=[Message(role=, role_type=, content=)])
    memory.asearch_memory(session_id, MemorySearchPayload(text=), limit=)
        → memory.search_sessions(text=, session_ids=[session_id], limit=,
                                 search_scope='messages')
    memory.adelete_memory(session_id)     → memory.delete(session_id)

Relevant docs:
    https://help.getzep.com/   (v2 SDK reference)
"""

from __future__ import annotations

import uuid
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory


@register_memory("zep_api")
class ZepMiddleware(MemoryMiddleware):
    """Zep memory middleware (v2.x SDK).

    Config params (all go in ``MemoryConfig.params``):
        base_url:        Zep server URL (default ``http://localhost:8000``).
                         For Zep cloud: ``https://api.getzep.com``.
        api_key:         Auth key (required for cloud; optional for self-hosted).
        session_id:      Reuse a specific session (default: auto-generated UUID).
        user_id:         Zep user id (optional).
        retrieval_limit: Max results from search (default 5).
        search_scope:    'messages' | 'summary' | 'facts' (default 'messages').
                         'messages' returns raw message hits; 'summary'
                         returns session summaries; 'facts' returns the
                         consolidated entity facts the v2 engine extracts.
        search_type:     'similarity' | 'mmr' (default 'similarity').
                         'mmr' diversifies results via Maximum Marginal
                         Relevance — useful when many hits are near-duplicates.
    """

    def __init__(self, config: MemoryConfig):
        # Lazy import — keeps the orchestrator image happy even if the
        # ``memory-zep`` extra wasn't installed, as long as this provider
        # isn't actually constructed at runtime.
        from zep_python.client import AsyncZep

        self._client = AsyncZep(
            base_url=config.params.get("base_url", "http://localhost:8000"),
            api_key=config.params.get("api_key"),
        )
        self._session_id = config.params.get("session_id", str(uuid.uuid4()))
        self._user_id = config.params.get("user_id")
        self._retrieval_limit = config.params.get("retrieval_limit", 5)
        self._search_scope = config.params.get("search_scope", "messages")
        self._search_type = config.params.get("search_type", "similarity")
        # v2 requires the session to exist before ``memory.add(...)``.
        # We lazily create it on the first write, tolerating the conflict
        # error when the session is already there.
        self._session_ready = False
        self._config = config

    async def _ensure_session(self) -> None:
        """Idempotent session creation.

        v2's ``add_session`` raises a ``ConflictError`` when the session
        already exists — we swallow that so reset()-then-write cycles and
        warm restarts both work cleanly. Other errors (network, auth)
        will resurface on the next add/search call with a clearer
        context, so we silently swallow them here too.
        """
        if self._session_ready:
            return
        try:
            await self._client.memory.add_session(
                session_id=self._session_id,
                user_id=self._user_id,
            )
        except Exception:
            pass
        self._session_ready = True

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        # Boundary (CrossBench↔middleware seam): tokenize the query.
        self._record_io(request_text=prompt[:500])
        try:
            response = await self._client.memory.search_sessions(
                text=prompt[:500],
                session_ids=[self._session_id],
                limit=self._retrieval_limit,
                search_scope=self._search_scope,
                search_type=self._search_type,
            )
        except Exception:
            return prompt

        # v2 search returns a SessionSearchResponse; ``results`` is a list
        # of SessionSearchResult, each of which carries a ``message`` OR
        # a ``summary`` OR a ``fact`` (depending on search_scope).
        results = getattr(response, "results", None) or []
        lines: list[str] = []
        for r in results:
            msg = getattr(r, "message", None)
            if msg and getattr(msg, "content", None):
                lines.append(f"- {msg.content}")
                continue
            summary = getattr(r, "summary", None)
            if summary:
                lines.append(f"- {summary}")
                continue
            fact = getattr(r, "fact", None)
            if fact:
                lines.append(f"- {fact}")

        memory_text = "\n".join(lines)
        if not memory_text.strip():
            return prompt
        self._record_io(response_text=memory_text)
        return f"Session context:\n{memory_text}\n\n{prompt}"

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        from zep_python import Message

        await self._ensure_session()
        # Boundary: tokenize the user/assistant turn we ship to Zep.
        self._record_io(request_text=f"{prompt[:2000]}\n{response[:2000]}")
        # v2's Message has both ``role`` (display name) and ``role_type``
        # (user/assistant/system enum). Setting them identically keeps the
        # storage shape compatible with the v0.x flow.
        messages = [
            Message(role="user", role_type="user", content=prompt[:2000]),
            Message(role="assistant", role_type="assistant", content=response[:2000]),
        ]
        try:
            await self._client.memory.add(
                session_id=self._session_id,
                messages=messages,
            )
        except Exception:
            pass

    async def reset(self) -> None:
        try:
            await self._client.memory.delete(session_id=self._session_id)
        except Exception:
            pass
        # New session UUID + reset the lazy-create flag so the next write
        # re-creates the session container.
        self._session_id = str(uuid.uuid4())
        self._session_ready = False
        self._reset_internal_counters()
