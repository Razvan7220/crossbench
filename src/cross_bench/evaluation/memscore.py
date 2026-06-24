"""MemScore — composite metric for memory system evaluation.

Combines quality (accuracy), latency, and token efficiency into a single
comparable metric.  Follows the MemoryBench scoring methodology:

    MemScore = accuracy% / avg_latency_ms / avg_context_tokens

Higher is better: high accuracy, low latency, low token usage.

Reference:
    https://github.com/supermemoryai/memorybench
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Latency statistics across all search operations."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    total_ms: float = 0.0
    count: int = 0


@dataclass
class TokenMetrics:
    """Token usage statistics for context provided to the answering model."""

    total_context_tokens: int = 0
    avg_context_tokens: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass
class MemScore:
    """Composite memory evaluation score."""

    accuracy: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """Compute latency statistics from a list of latency measurements."""
    if not latencies_ms:
        return LatencyStats()

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    return LatencyStats(
        min_ms=sorted_lat[0],
        max_ms=sorted_lat[-1],
        mean_ms=sum(sorted_lat) / n,
        median_ms=sorted_lat[n // 2],
        p95_ms=sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[-1],
        p99_ms=sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[-1],
        total_ms=sum(sorted_lat),
        count=n,
    )


def compute_memscore(
    accuracy: float,
    avg_latency_ms: float,
    avg_context_tokens: float,
) -> MemScore:
    """Compute MemScore from accuracy, latency, and token metrics.

    Formula: accuracy / (latency_ms * context_tokens)
    Scaled for readability. Higher is better.

    Handles edge cases:
      - If latency or tokens are 0, uses 1.0 as floor to avoid division by zero
      - If accuracy is 0, MemScore is 0
    """
    # Floor values to avoid division by zero
    lat = max(avg_latency_ms, 1.0)
    tok = max(avg_context_tokens, 1.0)

    # MemScore: accuracy percentage scaled by efficiency
    # Multiply by 1e6 for readable numbers
    score = (accuracy * 100.0) / (lat * tok) * 1e6

    return MemScore(
        accuracy=accuracy,
        score=score,
        components={
            "accuracy_pct": accuracy * 100.0,
            "avg_latency_ms": avg_latency_ms,
            "avg_context_tokens": avg_context_tokens,
            "memscore": score,
        },
    )


def format_memscore(ms: MemScore) -> str:
    """Format MemScore as a human-readable string."""
    return (
        f"{ms.components.get('accuracy_pct', 0):.1f}% / "
        f"{ms.components.get('avg_latency_ms', 0):.0f}ms / "
        f"{ms.components.get('avg_context_tokens', 0):.0f}tok "
        f"= {ms.score:.2f}"
    )


def memscore_to_dict(ms: MemScore) -> dict[str, Any]:
    """Convert MemScore to a flat dict for metrics reporting."""
    d: dict[str, Any] = {
        "memscore": ms.score,
        "memscore_accuracy_pct": ms.components.get("accuracy_pct", 0),
        "memscore_avg_latency_ms": ms.components.get("avg_latency_ms", 0),
        "memscore_avg_context_tokens": ms.components.get("avg_context_tokens", 0),
    }
    # Latency breakdown
    lat = ms.latency
    if lat.count > 0:
        d.update({
            "latency_min_ms": lat.min_ms,
            "latency_max_ms": lat.max_ms,
            "latency_mean_ms": lat.mean_ms,
            "latency_median_ms": lat.median_ms,
            "latency_p95_ms": lat.p95_ms,
            "latency_p99_ms": lat.p99_ms,
        })
    return d
