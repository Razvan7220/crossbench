#!/usr/bin/env python3
"""
Live aggregation view for an in-progress (or completed) dispatch run.

Usage:
    # Watch the latest run, refresh every 30s
    python scripts/watch_run.py

    # Watch a specific run
    python scripts/watch_run.py swebench-online-mem0v2-swebenchretrievalbm25-20260522-234018

    # One-shot (no loop)
    python scripts/watch_run.py --once

    # Custom refresh interval
    python scripts/watch_run.py --interval 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import boto3

BUCKET = "cross-bench-dev-artifacts-3684f16"
REGION = "us-east-1"


def get_latest_run(s3, prefix_hint: str = "swebench") -> str:
    """Return the run_id of the most-recently-created dispatch (by S3 key date)."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix_hint, Delimiter="/")
    prefixes = [
        (cp["Prefix"].rstrip("/"), cp["Prefix"])
        for cp in resp.get("CommonPrefixes", [])
    ]
    # Filter to parent runs only (no -s0XX- shard suffix)
    import re
    parents = [(name, p) for name, p in prefixes if not re.search(r"-s\d{3}-", name)]
    if not parents:
        sys.exit(f"No runs found with prefix '{prefix_hint}'")
    # Sort by name (timestamp-encoded) and take last
    parents.sort(key=lambda x: x[0])
    return parents[-1][0]


def collect_shard_results(s3, run_id: str) -> dict:
    """
    Scan all shard prefixes under run_id and return a dict with:
      results    : list of {task_id, resolved, patch_applied, steps, tokens_in, elapsed_s, shard}
      total      : total shards found
      done       : shards with summary.json
    """
    import json, re

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=run_id + "-s", Delimiter="/")
    shard_prefixes = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    results = []
    done = 0

    for p in shard_prefixes:
        shard_label = re.search(r"-(s\d{3})-", p)
        shard = shard_label.group(1) if shard_label else "?"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=p + "summary.json")
            sm = json.loads(obj["Body"].read())
            done += 1

            m = sm.get("aggregate_metrics", {})
            task_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=p + "tasks/")
            task_id = "?"
            steps = None
            for o in task_resp.get("Contents", []):
                task_id = o["Key"].split("/tasks/")[1].replace(".json", "")
                try:
                    tj = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
                    step_list = tj.get("steps", [])
                    steps = len([s for s in step_list if s.get("action") == "chat"])
                except Exception:
                    pass

            # Elapsed from started_at → completed_at
            elapsed_s = None
            try:
                from datetime import datetime, timezone
                started = datetime.fromisoformat(sm["started_at"].replace("Z", "+00:00"))
                completed = datetime.fromisoformat(sm["completed_at"].replace("Z", "+00:00"))
                elapsed_s = int((completed - started).total_seconds())
            except Exception:
                pass

            results.append({
                "shard": shard,
                "task_id": task_id,
                "resolved": bool(m.get("avg_resolved", 0) > 0),
                "patch_applied": bool(m.get("avg_patch_successfully_applied", 0) > 0),
                "steps": steps,
                "tokens_in": int(m.get("total_tokens_in", 0)),
                "elapsed_s": elapsed_s,
                "status": sm.get("status", "?"),
            })
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            results.append({
                "shard": shard, "task_id": "ERROR", "resolved": False,
                "patch_applied": False, "steps": None, "tokens_in": 0,
                "elapsed_s": None, "status": f"err:{e}",
            })

    results.sort(key=lambda x: x["shard"])
    return {"results": results, "total": len(shard_prefixes), "done": done}


def render(run_id: str, data: dict) -> None:
    results = data["results"]
    total = data["total"]
    done = data["done"]

    resolved = sum(1 for r in results if r["resolved"])
    patch_ok = sum(1 for r in results if r["patch_applied"])
    errors = sum(1 for r in results if r["status"].startswith("err"))
    step_limit = sum(1 for r in results if (r["steps"] or 0) >= 250)

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # Clear screen
    print("\033[H\033[J", end="")

    print(f"  Run: {run_id}")
    print(f"  As of {now}   {done}/{total} shards done", end="")
    if done < total:
        print(f"  ({total - done} pending)", end="")
    print()
    print()

    # Summary bar
    pct = resolved / done if done else 0
    bar_w = 30
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    print(f"  Resolved    [{bar}] {resolved}/{done}  ({pct*100:.1f}%)")
    print(f"  Patch OK    {patch_ok}/{done}  ({patch_ok/done*100:.1f}%)" if done else "  Patch OK    —")
    print(f"  Step-limit  {step_limit}/{done}")
    if errors:
        print(f"  ⚠ Errors    {errors}")
    print()

    # Per-task table
    COL = 46
    print(f"  {'Task':<{COL}}  {'S'}  {'✓'}  {'Steps':>5}  {'Tok(K)':>7}  {'Time':>5}")
    print(f"  {'─'*COL}  {'─'}  {'─'}  {'─'*5}  {'─'*7}  {'─'*5}")
    for r in results:
        tid = r["task_id"][:COL]
        resolved_mark = "✓" if r["resolved"] else ("·" if r["patch_applied"] else " ")
        patch_mark = "P" if r["patch_applied"] else " "
        steps_str = str(r["steps"]) if r["steps"] is not None else "—"
        if r["steps"] and r["steps"] >= 250:
            steps_str = "250!"
        tok_str = f"{r['tokens_in']//1000:,}" if r["tokens_in"] else "—"
        time_str = f"{r['elapsed_s']//60}m{r['elapsed_s']%60:02d}s" if r["elapsed_s"] else "—"
        print(f"  {tid:<{COL}}  {patch_mark}  {resolved_mark}  {steps_str:>5}  {tok_str:>7}  {time_str:>5}")

    if done < total:
        pending = total - done
        print(f"  {'... ' + str(pending) + ' still running ...'}")

    print()
    print(f"  Legend: S=patch applied  ✓=resolved  Steps=250! hit step limit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?", default=None,
                        help="Run ID prefix (auto-detects latest if omitted)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--prefix", default="swebench", help="Prefix for auto-detect")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    run_id = args.run_id
    if not run_id:
        run_id = get_latest_run(s3, prefix_hint=args.prefix)
        print(f"Auto-detected run: {run_id}")
        if args.once:
            time.sleep(0.5)

    if args.once:
        data = collect_shard_results(s3, run_id)
        render(run_id, data)
        return

    print(f"Watching {run_id}  (Ctrl-C to stop, refreshing every {args.interval}s)")
    try:
        while True:
            data = collect_shard_results(s3, run_id)
            render(run_id, data)
            done, total = data["done"], data["total"]
            if done == total and total > 0:
                print(f"  All {total} shards complete.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
