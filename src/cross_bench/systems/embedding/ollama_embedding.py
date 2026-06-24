"""Ollama local embedding driver.

Requires a running Ollama server with an embedding model pulled:
    ollama pull nomic-embed-text:latest

Works with any Ollama-served embedding model (nomic-embed-text,
mxbai-embed-large, all-minilm, snowflake-arctic-embed, etc.).
"""

from __future__ import annotations

from typing import Any

from cross_bench.models import EmbeddingConfig
from cross_bench.systems.base import EmbeddingDriver
from cross_bench.systems.registry import register_embedding


@register_embedding("ollama")
class OllamaEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using a local Ollama server.

    Config params (via ``EmbeddingConfig.params``):
        base_url: Ollama API URL (default ``http://localhost:11434``)
    """

    def __init__(self, config: EmbeddingConfig):
        self._model = config.model or "nomic-embed-text:latest"
        self._base_url = config.params.get("base_url", "http://localhost:11434")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for text in texts:
                resp = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                # Ollama /api/embed returns {"embeddings": [[...], ...]}
                emb = data.get("embeddings", [[]])[0]
                embeddings.append(emb)
        return embeddings
