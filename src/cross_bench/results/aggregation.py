"""Metric aggregation and cross-run comparison utilities."""

from __future__ import annotations

from typing import Any


def compare_metrics(runs: list[dict[str, Any]], metric_keys: list[str] | None = None) -> dict[str, Any]:
    """Build a comparison table across multiple runs.

    Returns a dict with:
      - columns: list of metric names
      - rows: list of {run_id, benchmark, system, metric1, metric2, ...}
    """
    if not runs:
        return {"columns": [], "rows": []}

    # Discover all metric keys if not specified
    if metric_keys is None:
        metric_keys = sorted({
            k
            for run in runs
            for k in (run.get("aggregate_metrics") or {}).keys()
        })

    rows = []
    for run in runs:
        metrics = run.get("aggregate_metrics") or {}
        sys_config = run.get("system_config", {})
        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "benchmark": run.get("benchmark", ""),
            "model": sys_config.get("llm", {}).get("model", "unknown"),
            "memory": sys_config.get("memory", {}).get("provider", "none") if sys_config.get("memory") else "none",
            "execution_mode": run.get("execution_mode", ""),
        }
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)

    return {"columns": metric_keys, "rows": rows}


def leaderboard(runs: list[dict[str, Any]], sort_by: str = "pass_rate") -> list[dict[str, Any]]:
    """Sort runs by a metric to create a leaderboard."""
    comparison = compare_metrics(runs)
    rows = comparison["rows"]
    rows.sort(key=lambda r: r.get(sort_by) or 0.0, reverse=True)
    return rows
