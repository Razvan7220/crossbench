"""Mastra memory middleware — HTTP API client for the Mastra memory server.

Mastra is an open-source TypeScript framework for building AI agents with
persistent, multi-layered memory.  Its memory system provides:

  - **Message History** — raw conversation messages stored in threads, scoped
    to a resource (user) and thread (conversation)
  - **Semantic Recall** — RAG-based vector search over past messages using
    configurable embedding models and vector stores
  - **Working Memory** — persistent structured scratchpad (Markdown or JSON)
    that maintains key facts across conversations
  - **Observational Memory** — background observer/reflector agents that
    compress conversations into dense observation summaries

The REST API is exposed by the Mastra server package (default port 4111).
All routes require an ``agentId`` query parameter identifying which agent's
memory config to use.

Install & run:
    npx create-mastra@latest
    cd my-mastra-app && npm run dev   # starts on localhost:4111

    # Or with Docker:
    docker compose up -d

API reference (key endpoints):
    POST   /memory/threads                        — create thread
    GET    /memory/threads/:threadId/messages      — list messages
    POST   /memory/save-messages                   — save messages
    GET    /memory/threads/:threadId/working-memory — get working memory
    POST   /memory/threads/:threadId/working-memory — update working memory
    GET    /memory/search                          — semantic search
    GET    /memory/status                          — memory status
    GET    /memory/config                          — memory config

Source:
    https://github.com/mastra-ai/mastra

NOTE: When the Mastra server is unavailable, the middleware falls back to
a simple in-memory keyword store so benchmarks can still run.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


@register_memory("mastra_api")
class MastraMiddleware(MemoryMiddleware):
    """Mastra REST API memory middleware.

    Config params (all go in ``MemoryConfig.params``):
        base_url:           Mastra server URL (default ``http://localhost:4111``)
        agent_id:           Agent ID for memory scoping (default ``bench-agent``)
        resource_id:        Resource (user) ID (default ``bench-user``)
        thread_id:          Specific thread ID (default: auto-created per session)
        semantic_recall:    Enable semantic search (default True)
        semantic_top_k:     Number of semantic results (default 4)
        semantic_scope:     Scope for search: ``thread`` | ``resource``
                            (default ``resource``)
        message_range:      Number of surrounding messages to include with
                            semantic results (default 1)
        last_messages:      Number of recent messages to include (default 10,
                            0 to disable)
        working_memory:     Enable working memory retrieval (default True)
        timeout:            HTTP timeout in seconds (default 30)
        max_retries:        Max retries on failure (default 3)
        api_key:            Optional API key for authenticated deployments
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        p = config.params or {}
        self._base_url = p.get("base_url", "http://localhost:4111").rstrip("/")
        self._agent_id = p.get("agent_id", "bench-agent")
        self._resource_id = p.get("resource_id", "bench-user")
        self._thread_id = p.get("thread_id", "")
        self._semantic_recall = p.get("semantic_recall", True)
        self._semantic_top_k = p.get("semantic_top_k", 4)
        self._semantic_scope = p.get("semantic_scope", "resource")
        self._message_range = p.get("message_range", 1)
        self._last_messages = p.get("last_messages", 10)
        self._working_memory = p.get("working_memory", True)
        self._timeout = p.get("timeout", 30)
        self._max_retries = p.get("max_retries", 3)
        self._api_key = p.get("api_key", "")

        # State
        self._fallback_memories: list[dict[str, Any]] = []
        self._server_available: bool | None = None
        self._thread_created: bool = False

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _params(self, **extra: Any) -> dict[str, Any]:
        """Build common query parameters (agentId is required on all routes)."""
        p: dict[str, Any] = {"agentId": self._agent_id}
        p.update(extra)
        return p

    # ------------------------------------------------------------------
    # Server health + thread management
    # ------------------------------------------------------------------

    async def _check_server(self) -> bool:
        if self._server_available is not None:
            return self._server_available

        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._base_url}/memory/status",
                    params=self._params(),
                    headers=self._headers(),
                )
                self._server_available = resp.status_code == 200
        except Exception:
            self._server_available = False
            logger.warning(
                "Mastra server not available at %s — using in-memory fallback",
                self._base_url,
            )

        if self._server_available and not self._thread_created:
            await self._ensure_thread()

        return self._server_available

    async def _ensure_thread(self) -> None:
        """Create or verify the conversation thread."""
        import httpx

        if self._thread_id:
            # Verify existing thread
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        f"{self._base_url}/memory/threads/{self._thread_id}",
                        params=self._params(),
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        self._thread_created = True
                        return
            except Exception:
                pass

        # Create new thread
        if not self._thread_id:
            self._thread_id = str(uuid.uuid4())

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/memory/threads",
                    params=self._params(),
                    json={
                        "id": self._thread_id,
                        "title": "Bench Platform Evaluation",
                        "resourceId": self._resource_id,
                        "metadata": {"source": "cross_bench"},
                    },
                    headers=self._headers(),
                )
                if resp.status_code in (200, 201):
                    self._thread_created = True
                    logger.info(
                        "Mastra thread '%s' created for resource '%s'",
                        self._thread_id,
                        self._resource_id,
                    )
                else:
                    logger.warning(
                        "Failed to create Mastra thread: %s %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as e:
            logger.warning("Failed to create Mastra thread: %s", e)

    async def is_indexed(self) -> bool:
        """Check if semantic indexing is complete.

        Mastra indexes messages into the vector store on save, so this is
        typically instant.  For observational memory, check buffer status.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/memory/observational-memory/buffer-status",
                    params=self._params(),
                    json={"threadId": self._thread_id},
                    headers=self._headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # If observational memory is not running, we're indexed
                    return not data.get("isObserving", False)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # MemoryMiddleware interface
    # ------------------------------------------------------------------

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Retrieve memories and prepend to prompt.

        Combines three retrieval strategies (matching Mastra's approach):
          1. Semantic recall — vector search over past messages
          2. Working memory — structured scratchpad
          3. Recent message history — last N messages
        """
        # Boundary (CrossBench↔middleware seam): tokenize the query.
        self._record_io(request_text=prompt[:1000])

        if not await self._check_server():
            return self._fallback_search(prompt)

        import httpx

        parts: list[str] = []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # 1. Semantic recall
            if self._semantic_recall:
                try:
                    resp = await client.get(
                        f"{self._base_url}/memory/search",
                        params=self._params(
                            searchQuery=prompt[:1000],
                            resourceId=self._resource_id,
                            threadId=self._thread_id if self._semantic_scope == "thread" else "",
                            topK=self._semantic_top_k,
                        ),
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data if isinstance(data, list) else data.get("results", data.get("messages", []))
                        if results:
                            parts.append(self._format_semantic_results(results))
                except Exception as e:
                    logger.debug("Mastra semantic search failed: %s", e)

            # 2. Working memory
            if self._working_memory and self._thread_id:
                try:
                    resp = await client.get(
                        f"{self._base_url}/memory/threads/{self._thread_id}/working-memory",
                        params=self._params(),
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        wm_content = data.get("workingMemory", data.get("data", ""))
                        if wm_content and wm_content.strip():
                            parts.append(f"[Working Memory]\n{wm_content}")
                except Exception as e:
                    logger.debug("Mastra working memory fetch failed: %s", e)

            # 3. Recent message history
            if self._last_messages and self._thread_id:
                try:
                    resp = await client.get(
                        f"{self._base_url}/memory/threads/{self._thread_id}/messages",
                        params=self._params(
                            perPage=self._last_messages,
                            orderBy='{"field":"createdAt","direction":"DESC"}',
                        ),
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        messages = data if isinstance(data, list) else data.get("messages", data.get("data", []))
                        if messages:
                            parts.append(self._format_message_history(messages))
                except Exception as e:
                    logger.debug("Mastra message history fetch failed: %s", e)

        if parts:
            memory_text = "\n\n---\n\n".join(parts)
            self._record_io(response_text=memory_text)
            return f"Retrieved memories:\n{memory_text}\n\n{prompt}"
        return prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Store a message in Mastra via POST /memory/save-messages.

        Uses the Mastra message format (content.format: 2, content.parts[]).
        """
        # Boundary: tokenize the content we ship to the Mastra server.
        self._record_io(request_text=response[:4000])

        if not await self._check_server():
            self._fallback_memories.append({"text": response[:2000]})
            return

        import httpx

        # Build message in Mastra format (MastraDBMessage with format 2)
        message = {
            "id": str(uuid.uuid4()),
            "role": context.get("role", "user"),
            "threadId": self._thread_id,
            "resourceId": self._resource_id,
            "content": {
                "format": 2,
                "parts": [{"type": "text", "text": response[:4000]}],
            },
        }

        body = {
            "messages": [message],
        }

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/memory/save-messages",
                        params=self._params(),
                        json=body,
                        headers=self._headers(),
                    )
                    if resp.status_code in (200, 201):
                        return
                    else:
                        logger.debug(
                            "Mastra save-messages returned %s: %s",
                            resp.status_code,
                            resp.text[:200],
                        )
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("Mastra save-messages failed after retries: %s", e)
                    self._fallback_memories.append({"text": response[:2000]})

    async def reset(self) -> None:
        """Clear session state.

        Resets the thread ID so a new thread is created next session.
        Does NOT delete data from the Mastra server.
        """
        self._fallback_memories.clear()
        self._thread_id = ""
        self._thread_created = False
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_semantic_results(self, results: list[dict[str, Any]]) -> str:
        """Format semantic search results."""
        lines = []
        for r in results:
            # Mastra returns messages with content.parts or direct text
            content = r.get("content", {})
            if isinstance(content, dict):
                parts = content.get("parts", [])
                text = " ".join(
                    p.get("text", "") for p in parts if p.get("type") == "text"
                )
                if not text:
                    text = content.get("content", str(content))
            elif isinstance(content, str):
                text = content
            else:
                text = str(r)

            role = r.get("role", "")
            score = r.get("score", r.get("similarity", ""))
            prefix = f"[{role}]" if role else "-"
            suffix = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
            lines.append(f"{prefix} {text[:500]}{suffix}")

        return "[Semantic Recall]\n" + "\n".join(lines)

    def _format_message_history(self, messages: list[dict[str, Any]]) -> str:
        """Format recent message history."""
        lines = []
        # Messages come in DESC order, reverse for chronological
        for msg in reversed(messages[-self._last_messages:]):
            content = msg.get("content", {})
            if isinstance(content, dict):
                parts = content.get("parts", [])
                text = " ".join(
                    p.get("text", "") for p in parts if p.get("type") == "text"
                )
                if not text:
                    text = content.get("content", "")
            elif isinstance(content, str):
                text = content
            else:
                text = str(msg)

            role = msg.get("role", "unknown")
            lines.append(f"{role}: {text[:300]}")

        return "[Recent History]\n" + "\n".join(lines)

    def _fallback_search(self, prompt: str) -> str:
        """Simple keyword search over in-memory fallback store."""
        if not self._fallback_memories:
            return prompt

        query_words = set(prompt.lower().split()[:15])
        scored = []
        for mem in self._fallback_memories:
            text = mem.get("text", "")
            overlap = sum(1 for w in query_words if w in text.lower())
            if overlap > 0:
                scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[: self._semantic_top_k]]
        if top:
            memory_text = "\n".join(f"- {t}" for t in top)
            return f"Retrieved memories:\n{memory_text}\n\n{prompt}"
        return prompt
