"""Subprocess-based parallel task execution with per-task isolation.

Usage (via CLI):
    cross-bench run -b swebench -p online -p no_memory \\
        --task-id astropy__astropy-12907 \\
        --task-id astropy__astropy-13033 \\
        --task-id astropy__astropy-13236 \\
        --parallel 3 \\
        --db-url none --artifacts-dir ./artifacts/par

Each task runs in its own ``cross-bench run`` subprocess so memory middleware
state is fully isolated between tasks.

Tasks are assigned round-robin across ``min(parallel, len(task_ids))`` lanes;
tasks on the same lane run sequentially, so at most ``parallel`` tasks execute
concurrently.  Per-task artifacts land under ``slot-{task_idx}/`` and the
per-slot ``summary.json`` files are merged into a parent ``summary.json`` at
the end.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

async def _run_subprocess(cmd: list[str], label: str) -> int:
    """Run cmd as an async subprocess, streaming its stdout/stderr with a label prefix."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        print(f"[{label}] {line.decode(errors='replace')}", end="", flush=True)
    return (await proc.wait())


# ---------------------------------------------------------------------------
# Slot-summary merge
# ---------------------------------------------------------------------------

# Metrics that are summed across slots (not averaged).
_SUM_KEYS = frozenset({
    "total_tasks", "passed", "failed", "errors", "timeouts",
    "total_tokens_in", "total_tokens_out", "total_latency_ms",
})

# Metrics re-derived from summed counts rather than averaged.
_DERIVED_KEYS = frozenset({"pass_rate", "accuracy"})


def _find_slot_summary(slot_dir: Path) -> dict | None:
    """Find summary.json inside slot_dir/{run-id}/summary.json."""
    if not slot_dir.exists():
        return None
    # The artifacts layout is slot_dir/{run_id}/summary.json
    for child in sorted(slot_dir.iterdir()):
        if child.is_dir():
            candidate = child / "summary.json"
            if candidate.exists():
                try:
                    with open(candidate) as fh:
                        return json.load(fh)
                except Exception:
                    pass
    return None


def _merge_slot_summaries(artifacts_dir: str, task_ids: list[str]) -> dict | None:
    """Merge per-slot summary.json files into a parent summary.json.

    Artifacts layout: {artifacts_dir}/slot-{idx}/{run_id}/summary.json
    Metrics live under the ``aggregate_metrics`` key inside each summary.

    Merge strategy for aggregate_metrics values:
    - _SUM_KEYS          → arithmetic sum across slots
    - _DERIVED_KEYS      → re-computed from summed counts
    - avg_* / p50_* / p95_* → weighted mean (weight = slot's total_tasks)
    - everything else    → first non-null value
    """
    raw_summaries: list[dict] = []
    for idx in range(len(task_ids)):
        s = _find_slot_summary(Path(artifacts_dir) / f"slot-{idx}")
        if s is not None:
            raw_summaries.append(s)

    if not raw_summaries:
        return None

    # Extract the aggregate_metrics sub-dict from each summary.
    metrics_list: list[dict] = [s.get("aggregate_metrics") or {} for s in raw_summaries]
    total_tasks = sum(float(m.get("total_tasks") or 0) for m in metrics_list)
    if total_tasks == 0:
        return None

    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    merged_metrics: dict = {}
    for key in sorted(all_keys):
        if key in _DERIVED_KEYS:
            continue  # re-derived after the loop

        vals = [m.get(key) for m in metrics_list]
        nums = [
            (float(v), float(m.get("total_tasks") or 0))
            for v, m in zip(vals, metrics_list)
            if isinstance(v, (int, float)) and v is not None
        ]

        if key in _SUM_KEYS:
            merged_metrics[key] = sum(v for v, _ in nums)
        elif key.startswith(("avg_", "p50_", "p95_")):
            total_w = sum(w for _, w in nums)
            merged_metrics[key] = (
                sum(v * w for v, w in nums) / total_w if (nums and total_w > 0) else None
            )
        else:
            non_null = [v for v in vals if v is not None]
            merged_metrics[key] = non_null[0] if non_null else None

    passed = merged_metrics.get("passed") or 0
    merged_metrics["pass_rate"] = passed / total_tasks
    merged_metrics["accuracy"]  = merged_metrics["pass_rate"]

    # Build the top-level summary in the same shape individual runs use.
    first = raw_summaries[0]
    merged: dict = {
        "benchmark":    first.get("benchmark"),
        "status":       "completed",
        "aggregate_metrics": merged_metrics,
        "_merged_from_slots": len(raw_summaries),
        "_slot_task_ids": task_ids,
    }

    out_path = Path(artifacts_dir) / "summary.json"
    with open(out_path, "w") as fh:
        json.dump(merged, fh, indent=2, default=str)

    # Return flat metrics dict for display; tag it so _print_merged_summary
    # can show the slot count without reading the outer dict.
    merged_metrics["_merged_from_slots"] = len(raw_summaries)
    return merged_metrics


def _print_merged_summary(merged: dict, artifacts_dir: str) -> None:
    """Print a summary table matching the per-slot run output style."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        skip = {"_merged_from_slots", "_slot_task_ids", "run_id", "profiles",
                "benchmark", "model", "memory", "embedding", "mode", "tags"}
        for k, v in sorted(merged.items()):
            if k in skip or k.startswith("_"):
                continue
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            elif v is not None:
                table.add_row(k, str(v))

        n = merged.get("_merged_from_slots", "?")
        console.print(f"\n[bold]Merged summary ({n} slots → {artifacts_dir}/summary.json)[/bold]")
        console.print(table)
    except Exception:
        # Fallback if rich isn't available or any rendering error.
        print(f"\n[parallel] merged summary ({artifacts_dir}/summary.json):")
        for k, v in sorted(merged.items()):
            if not k.startswith("_"):
                print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_parallel(
    *,
    task_ids: list[str],
    parallel: int,
    profiles: list[str],
    benchmark: Optional[str],
    artifacts_dir: str,
    db_url: str,
    extra_cli_args: list[str],
) -> None:
    """Fan out tasks across ``parallel`` concurrent subprocesses.

    Each task runs in its own ``cross-bench run`` subprocess so memory
    middleware state is fully isolated between tasks. Tasks are assigned
    round-robin across ``min(parallel, len(task_ids))`` lanes; tasks on the
    same lane run sequentially, so at most ``parallel`` tasks execute at once.

    Artifacts: each task writes to ``slot-{task_idx}/`` keyed by task position.
    The merge step at the end reads all per-task dirs into a parent summary.
    """
    n_tasks = len(task_ids)
    n_lanes = min(parallel, n_tasks) if n_tasks else 0
    if n_lanes <= 0:
        print("[parallel] no tasks to run", file=sys.stderr)
        return

    # Base CLI command shared across all tasks.
    base_cmd = [sys.executable, "-m", "cross_bench.cli", "run"]
    if benchmark:
        base_cmd += ["-b", benchmark]
    for p in profiles:
        base_cmd += ["-p", p]
    base_cmd += ["--db-url", db_url]
    base_cmd += extra_cli_args

    def _task_cmd(task_idx: int, tid: str) -> tuple[list[str], str]:
        cmd = list(base_cmd)
        cmd += ["--task-id", tid]
        cmd += ["--artifacts-dir", str(Path(artifacts_dir) / f"slot-{task_idx}")]
        return cmd, f"task-{task_idx} {tid}"

    # Assign tasks to lanes round-robin; tasks on the same lane run
    # sequentially so no more than ``n_lanes`` run concurrently.
    batches: list[list[tuple[int, str]]] = [[] for _ in range(n_lanes)]
    for task_idx, tid in enumerate(task_ids):
        batches[task_idx % n_lanes].append((task_idx, tid))

    per_batch = max(len(b) for b in batches)
    print(
        f"[parallel] launching {n_tasks} task(s) across {n_lanes} lane(s)"
        f" (~{per_batch} tasks/lane)…\n"
    )

    return_codes_by_task: dict[int, int] = {}

    async def _run_batch(batch: list[tuple[int, str]]) -> None:
        for task_idx, tid in batch:
            cmd, label = _task_cmd(task_idx, tid)
            rc = await _run_subprocess(cmd, label)
            return_codes_by_task[task_idx] = rc

    await asyncio.gather(*[_run_batch(batches[i]) for i in range(n_lanes)])

    return_codes = [return_codes_by_task.get(i, -1) for i in range(n_tasks)]
    print(f"\n[parallel] all done — return codes: {return_codes}")
    failed_tasks = [task_ids[i] for i, rc in enumerate(return_codes) if rc != 0]
    if failed_tasks:
        print(f"[parallel] FAILED tasks: {failed_tasks}", file=sys.stderr)

    # Merge per-slot summary.json files into a single parent summary.
    merged = _merge_slot_summaries(artifacts_dir, task_ids)
    if merged:
        _print_merged_summary(merged, artifacts_dir)
    else:
        print("[parallel] no slot summaries found — skipping merge", file=sys.stderr)
