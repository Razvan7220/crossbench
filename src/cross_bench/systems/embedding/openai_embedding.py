"""OpenAI embedding driver."""

from __future__ import annotations

from cross_bench.models import EmbeddingConfig
from cross_bench.systems.base import EmbeddingDriver
from cross_bench.systems.registry import register_embedding


@register_embedding("openai")
class OpenAIEmbeddingDriver(EmbeddingDriver):
    def __init__(self, config: EmbeddingConfig):
        import openai
        self._client = openai.AsyncOpenAI(api_key=config.params.get("api_key"))
        self._model = config.model or "text-embedding-3-small"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]
