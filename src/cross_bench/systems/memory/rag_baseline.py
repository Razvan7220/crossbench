"""RAG baseline memory provider — BM25 + vector hybrid search.

A non-memory baseline that uses standard RAG (chunking + retrieval) instead
of a dedicated memory system. Useful for comparison: any memory system should
outperform naive RAG on conversational tasks.

Follows MemoryBench's RAG provider implementation:
  - 400-token chunks with 50-token overlap
  - BM25 keyword search + vector cosine similarity
  - Hybrid scoring: 70% vector, 30% BM25
  - OpenAI text-embedding-3-small for embeddings (configurable)

Reference:
    https://github.com/supermemoryai/memorybench
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with its embedding.

    ``source_id`` is the provenance tag propagated from
    ``MemoryMiddleware.post_response`` via ``context["source_id"]`` (e.g.
    LoCoMo's ``dia_id`` like ``"D1:3"``). When multiple chunks are
    produced from a single ingested item they all share the same
    ``source_id``. Emitted back through ``context["retrieved_context_ids"]``
    on :meth:`RAGBaselineMiddleware.pre_prompt` so benchmarks can score
    retrieval against dataset ground-truth evidence IDs.
    """

    text: str
    embedding: list[float] = field(default_factory=list)
    token_count: int = 0
    source_id: str | None = None


@register_memory("rag")
class RAGBaselineMiddleware(MemoryMiddleware):
    """RAG baseline — BM25 + vector hybrid retrieval over conversation chunks.

    Config params (in MemoryConfig.params):
      - chunk_size: int = 400 (tokens per chunk)
      - chunk_overlap: int = 50 (token overlap between chunks)
      - top_k: int = 5 (number of chunks to retrieve)
      - vector_weight: float = 0.7 (weight for vector similarity)
      - bm25_weight: float = 0.3 (weight for BM25 score)
      - embedding_model: str = "text-embedding-3-small"
      - embedding_api_key: str (defaults to OPENAI_API_KEY env var)
      - embedding_base_url: str = "https://api.openai.com/v1"
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._params = config.params or {}
        self._chunk_size = self._params.get("chunk_size", 400)
        self._chunk_overlap = self._params.get("chunk_overlap", 50)
        self._top_k = self._params.get("top_k", 5)
        self._vector_weight = self._params.get("vector_weight", 0.7)
        self._bm25_weight = self._params.get("bm25_weight", 0.3)
        self._embedding_model = self._params.get("embedding_model", "text-embedding-3-small")

        self._chunks: list[Chunk] = []
        self._documents: list[str] = []  # raw documents for BM25
        self._bm25_index: _BM25Index | None = None
        self._embed_client = None

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Search chunks and prepend relevant context.

        Also writes ``context["retrieved_context_ids"]`` — the ordered,
        de-duplicated list of ``source_id`` values for the retrieved
        chunks. Chunks with no ``source_id`` are skipped. An empty list
        (retriever ran, no tagged hits) is distinct from an absent key
        (no retrieval attempted). See ``MemoryMiddleware`` docstring.
        """
        if not self._chunks:
            context["retrieved_context_ids"] = []
            return prompt

        results = await self._hybrid_search(prompt, self._top_k)
        if not results:
            context["retrieved_context_ids"] = []
            return prompt

        # Collect provenance IDs in rank order, de-duplicated (multiple
        # chunks from the same ingested item share a source_id).
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for chunk, _score in results:
            sid = chunk.source_id
            if sid is None or sid == "" or sid in seen:
                continue
            seen.add(sid)
            ordered_ids.append(sid)
        context["retrieved_context_ids"] = ordered_ids

        context_text = "\n\n---\n\n".join(
            f"[Chunk {i + 1}, score={score:.3f}]\n{chunk.text}"
            for i, (chunk, score) in enumerate(results)
        )
        return f"Retrieved context:\n{context_text}\n\n{prompt}"

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Ingest text by chunking and indexing.

        Reads ``context.get("source_id")`` and tags every produced chunk
        with it so :meth:`pre_prompt` can emit provenance IDs back. See
        ``MemoryMiddleware`` docstring.
        """
        text = response.strip()
        if not text:
            return

        source_id_raw = context.get("source_id")
        source_id = str(source_id_raw) if source_id_raw else None

        self._documents.append(text)
        new_chunks = self._chunk_text(text)
        for chunk in new_chunks:
            chunk.source_id = source_id

        # Compute embeddings
        embeddings = await self._get_embeddings([c.text for c in new_chunks])
        for chunk, emb in zip(new_chunks, embeddings):
            chunk.embedding = emb

        self._chunks.extend(new_chunks)

        # Rebuild BM25 index
        self._bm25_index = _BM25Index([c.text for c in self._chunks])

    async def reset(self) -> None:
        """Clear all indexed chunks."""
        self._chunks.clear()
        self._documents.clear()
        self._bm25_index = None
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[Chunk]:
        """Split text into overlapping token-based chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(text=chunk_text, token_count=len(chunk_words)))
            start += self._chunk_size - self._chunk_overlap
            if start >= len(words):
                break
        return chunks if chunks else [Chunk(text=text, token_count=len(words))]

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    async def _hybrid_search(
        self, query: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Combine BM25 and vector search with weighted scoring."""
        if not self._chunks:
            return []

        # Vector search
        query_embedding = (await self._get_embeddings([query]))[0]
        vector_scores = [
            _cosine_similarity(query_embedding, chunk.embedding)
            if chunk.embedding else 0.0
            for chunk in self._chunks
        ]

        # BM25 search
        bm25_scores = [0.0] * len(self._chunks)
        if self._bm25_index:
            bm25_scores = self._bm25_index.score(query)

        # Normalize scores to [0, 1]
        vector_scores = _normalize_scores(vector_scores)
        bm25_scores = _normalize_scores(bm25_scores)

        # Hybrid: weighted combination
        combined = [
            self._vector_weight * vs + self._bm25_weight * bs
            for vs, bs in zip(vector_scores, bm25_scores)
        ]

        # Top-K
        ranked = sorted(
            enumerate(combined), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [(self._chunks[idx], score) for idx, score in ranked if score > 0]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings via OpenAI API or fallback to TF-IDF-like vectors."""
        if not texts:
            return []

        try:
            import os
            from openai import AsyncOpenAI

            api_key = self._params.get(
                "embedding_api_key",
                os.environ.get("OPENAI_API_KEY", ""),
            )
            base_url = self._params.get(
                "embedding_base_url", "https://api.openai.com/v1"
            )

            if not self._embed_client:
                self._embed_client = AsyncOpenAI(
                    api_key=api_key, base_url=base_url
                )

            response = await self._embed_client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.debug("Embedding API unavailable (%s), using bag-of-words fallback", e)
            return [_bow_vector(t) for t in texts]


# ---------------------------------------------------------------------------
# BM25 index (Okapi BM25)
# ---------------------------------------------------------------------------

class _BM25Index:
    """Simple Okapi BM25 implementation for keyword search."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._docs = [self._tokenize(d) for d in documents]
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        self._n = len(self._docs)

        # Document frequency
        self._df: Counter = Counter()
        for doc in self._docs:
            self._df.update(set(doc))

    def score(self, query: str) -> list[float]:
        """Score all documents against a query."""
        query_terms = self._tokenize(query)
        scores = []
        for doc in self._docs:
            s = 0.0
            doc_len = len(doc)
            tf = Counter(doc)
            for term in query_terms:
                if term not in tf:
                    continue
                f = tf[term]
                df = self._df.get(term, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                numerator = f * (self._k1 + 1)
                denominator = f + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl)
                s += idf * numerator / denominator
            scores.append(s)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s
    if rng == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / rng for s in scores]


def _bow_vector(text: str, dim: int = 256) -> list[float]:
    """Simple bag-of-words hash vector as embedding fallback."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        idx = hash(word) % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec
