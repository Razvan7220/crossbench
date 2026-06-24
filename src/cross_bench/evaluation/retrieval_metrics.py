"""Retrieval quality metrics — Hit@K, Precision@K, Recall@K, F1@K, MRR, NDCG.

Evaluates how well the memory system's search results contain relevant
information for answering the question, using an LLM to judge relevance
of each retrieved item.

Follows MemoryBench's retrieval-eval.ts methodology:
  1. For each search result, ask an LLM if it's relevant to the question + answer
  2. Compute standard IR metrics from the binary relevance labels

Reference:
    https://github.com/supermemoryai/memorybench
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Information retrieval metrics for a single question."""

    hit_at_k: float = 0.0       # 1 if any result in top-K is relevant, else 0
    precision_at_k: float = 0.0  # fraction of top-K results that are relevant
    recall_at_k: float = 0.0     # fraction of relevant items found in top-K
    f1_at_k: float = 0.0         # harmonic mean of precision and recall
    mrr: float = 0.0             # 1 / rank of first relevant result
    ndcg_at_k: float = 0.0       # normalized discounted cumulative gain
    num_results: int = 0
    num_relevant: int = 0
    k: int = 5
    relevance_labels: list[int] = field(default_factory=list)


def compute_retrieval_metrics(
    relevance_labels: list[int],
    k: int = 5,
    total_relevant: int | None = None,
) -> RetrievalMetrics:
    """Compute IR metrics from binary relevance labels.

    Args:
        relevance_labels: List of 0/1 labels for each search result (in rank order).
        k: Number of top results to consider.
        total_relevant: Total number of relevant items in the collection.
                       If None, uses the count of relevant items in the list.
    """
    if not relevance_labels:
        return RetrievalMetrics(k=k)

    top_k = relevance_labels[:k]
    num_relevant_in_k = sum(top_k)
    total_rel = total_relevant or max(sum(relevance_labels), 1)

    # Hit@K: 1 if any result in top-K is relevant
    hit = 1.0 if num_relevant_in_k > 0 else 0.0

    # Precision@K
    precision = num_relevant_in_k / len(top_k) if top_k else 0.0

    # Recall@K
    recall = num_relevant_in_k / total_rel if total_rel > 0 else 0.0

    # F1@K
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for i, rel in enumerate(relevance_labels):
        if rel == 1:
            mrr = 1.0 / (i + 1)
            break

    # NDCG@K
    ndcg = _compute_ndcg(top_k, k)

    return RetrievalMetrics(
        hit_at_k=hit,
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        mrr=mrr,
        ndcg_at_k=ndcg,
        num_results=len(relevance_labels),
        num_relevant=sum(relevance_labels),
        k=k,
        relevance_labels=relevance_labels,
    )


def _compute_ndcg(relevance_labels: list[int], k: int) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    DCG = sum(rel_i / log2(i + 2)) for i in 0..k-1
    IDCG = DCG of the ideal ranking (all relevant first)
    NDCG = DCG / IDCG
    """
    if not relevance_labels:
        return 0.0

    # DCG
    dcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevance_labels[:k])
    )

    # IDCG (ideal: all relevant items first)
    ideal = sorted(relevance_labels[:k], reverse=True)
    idcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(ideal)
    )

    return dcg / idcg if idcg > 0 else 0.0


async def judge_relevance(
    question: str,
    golden_answer: str,
    search_results: list[str],
    llm_config: dict[str, Any] | None = None,
) -> list[int]:
    """Use an LLM to judge the relevance of each search result.

    Returns a list of 0/1 labels (1 = relevant, 0 = not relevant).

    Follows MemoryBench's AI-powered relevance judging approach.
    """
    if not search_results:
        return []

    if llm_config is None:
        llm_config = {}

    model = llm_config.get("model", "gpt-4o-mini")
    api_key = llm_config.get("api_key", "")
    base_url = llm_config.get("base_url", "https://api.openai.com/v1")

    labels: list[int] = []

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        for result_text in search_results:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You judge whether a retrieved memory is relevant to "
                                "answering a given question. Respond with only 1 (relevant) "
                                "or 0 (not relevant)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Expected answer: {golden_answer}\n"
                                f"Retrieved memory: {result_text[:2000]}\n\n"
                                "Is this memory relevant to answering the question? "
                                "Reply with 1 or 0."
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
                text = (response.choices[0].message.content or "").strip()
                labels.append(1 if "1" in text else 0)
            except Exception:
                labels.append(0)

    except ImportError:
        # Fallback: simple keyword overlap
        logger.warning("openai not installed, using keyword overlap for relevance")
        question_words = set(question.lower().split())
        answer_words = set(golden_answer.lower().split())
        query_words = question_words | answer_words
        for result_text in search_results:
            result_words = set(result_text.lower().split())
            overlap = len(query_words & result_words) / max(len(query_words), 1)
            labels.append(1 if overlap > 0.2 else 0)

    return labels


def aggregate_retrieval_metrics(
    all_metrics: list[RetrievalMetrics],
) -> dict[str, float]:
    """Average retrieval metrics across all questions."""
    if not all_metrics:
        return {}

    n = len(all_metrics)
    return {
        "retrieval_hit_at_k": sum(m.hit_at_k for m in all_metrics) / n,
        "retrieval_precision_at_k": sum(m.precision_at_k for m in all_metrics) / n,
        "retrieval_recall_at_k": sum(m.recall_at_k for m in all_metrics) / n,
        "retrieval_f1_at_k": sum(m.f1_at_k for m in all_metrics) / n,
        "retrieval_mrr": sum(m.mrr for m in all_metrics) / n,
        "retrieval_ndcg_at_k": sum(m.ndcg_at_k for m in all_metrics) / n,
    }
