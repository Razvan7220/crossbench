"""Letta memory middleware — local self-hosted server.

Uses the ``letta-client`` SDK (the official Python client), NOT the
deprecated ``from letta import create_client`` path.

Three-tier memory architecture:
    - **Core memory** (blocks): in-context working memory — always in the
      agent's prompt.  Blocks have a ``label`` (e.g. ``"human"``, ``"persona"``)
      and a ``value`` (free-form text).
    - **Archival memory** (passages): vector-store backed long-term storage
      (pgvector on self-hosted).  Inserted/searched via the passages API.
    - **Recall memory**: full conversation history, searchable by date/text.

Install:
    pip install letta-client

Running the Letta server locally (Docker, with bundled Postgres+pgvector):
    docker run -d --name letta-server \\
        -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \\
        -p 8283:8283 \\
        -e OPENAI_API_KEY="sk-..." \\
        letta/letta:latest

Relevant docs:
    https://docs.letta.com/api/python
    https://docs.letta.com/guides/docker/
"""

from __future__ import annotations

from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory


@register_memory("letta_api")
class LettaMiddleware(MemoryMiddleware):
    """Letta local memory middleware.

    Config params (all go in ``MemoryConfig.params``):
        base_url:       Letta server URL (default ``http://localhost:8283``)
        api_key:        Server password (only if ``SECURE=true`` on server)
        agent_id:       Reuse an existing agent (optional)
        agent_name:     Name for auto-created agents (default ``cross_bench_agent``)
        model:          LLM model string, e.g. ``"openai/gpt-4o"``
        embedding:      Embedding model, e.g. ``"openai/text-embedding-3-small"``
        archival_limit: Max passages from archival search (default 5)
        memory_blocks:  Initial core memory blocks (list of {label, value} dicts)
    """

    def __init__(self, config: MemoryConfig):
        # New SDK: letta_client.Letta (sync) or AsyncLetta (async)
        from letta_client import Letta

        self._client = Letta(
            base_url=config.params.get("base_url", "http://localhost:8283"),
            api_key=config.params.get("api_key"),
        )
        self._config = config
        self._archival_limit = config.params.get("archival_limit", 5)
        self._agent_id = config.params.get("agent_id")

        # Create or reuse an agent
        if not self._agent_id:
            self._agent_id = self._create_agent()

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        parts: list[str] = []

        # Boundary (CrossBench↔middleware seam): tokenize the query we run.
        self._record_io(request_text=prompt[:500])

        # 1. Read core memory blocks
        try:
            blocks = await loop.run_in_executor(
                None,
                lambda: self._client.agents.blocks.list(agent_id=self._agent_id),
            )
            core_lines = []
            for block in blocks:
                label = getattr(block, "label", "unknown")
                value = getattr(block, "value", "")
                if value:
                    core_lines.append(f"[{label}]: {value}")
            if core_lines:
                parts.append("Core memory:\n" + "\n".join(core_lines))
        except Exception:
            pass

        # 2. Search archival memory (passages)
        try:
            passages = await loop.run_in_executor(
                None,
                lambda: self._client.agents.passages.search(
                    agent_id=self._agent_id,
                    query=prompt[:500],
                ),
            )
            if passages:
                archival_lines = []
                for p in passages[: self._archival_limit]:
                    text = getattr(p, "content", getattr(p, "text", ""))
                    if text:
                        archival_lines.append(f"- {text}")
                if archival_lines:
                    parts.append("Archival context:\n" + "\n".join(archival_lines))
        except Exception:
            pass

        if parts:
            block = "\n".join(parts)
            self._record_io(response_text=block)
            return f"{block}\n\n{prompt}"
        return prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        # Boundary: tokenize the Q/A we ship to the Letta server.
        self._record_io(
            request_text=f"Q: {prompt[:500]}\nA: {response[:500]}"
        )
        # Insert into archival memory for long-term retrieval
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.agents.passages.insert(
                    agent_id=self._agent_id,
                    content=f"Q: {prompt[:500]}\nA: {response[:500]}",
                ),
            )
        except Exception:
            pass

    async def reset(self) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        # Delete the agent and create a fresh one
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.agents.delete(agent_id=self._agent_id),
            )
        except Exception:
            pass
        self._agent_id = self._create_agent()
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    def _create_agent(self) -> str:
        memory_blocks = self._config.params.get("memory_blocks", [
            {"label": "human", "value": "Benchmark evaluation user."},
            {"label": "persona", "value": "I am a helpful coding assistant."},
        ])
        agent_kwargs: dict[str, Any] = {
            "memory_blocks": memory_blocks,
        }
        model = self._config.params.get("model")
        if model:
            agent_kwargs["model"] = model
        embedding = self._config.params.get("embedding")
        if embedding:
            agent_kwargs["embedding"] = embedding

        agent_state = self._client.agents.create(**agent_kwargs)
        return agent_state.id
