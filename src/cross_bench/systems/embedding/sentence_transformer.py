"""Sentence Transformers local embedding driver."""

from __future__ import annotations

from cross_bench.models import EmbeddingConfig
from cross_bench.systems.base import EmbeddingDriver
from cross_bench.systems.registry import register_embedding


@register_embedding("sentence_transformer")
class SentenceTransformerDriver(EmbeddingDriver):
    def __init__(self, config: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.model or "all-MiniLM-L6-v2")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers is synchronous; run in thread for async compat
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._model.encode, texts)
        return [e.tolist() for e in embeddings]
