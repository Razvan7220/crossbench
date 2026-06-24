#!/usr/bin/env python3
"""
Watch a set of CrossBench runs and post a combined Slack summary when all finish.

Polls S3 for summary.json on each run ID every --interval seconds.
As each run lands it posts a per-run Slack message; when all are done
it posts a combined comparison table with averages across runs.

Usage
-----
  # Watch two named runs
  python scripts/watch_runs.py \
      --run "baseline:pro-smoke-baseline-r1-20260527-170109" \
      --run "mem0v2:pro-smoke-mem0v2-r1-20260527-180000" \
      --benchmark swebench_pro_seal \
      --s3-prefix runs/swebench_pro_seal/

  # Watch all shards under a parent (discovers via shards.json)
  python scripts/watch_runs.py \
      --parent pro-smoke-baseline-r1-20260527-170109 \
      --s3-prefix runs/swebench_pro_seal/

SLACK_WEBHOOK_URL must be set in the environment (or loaded from .env).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.parse
import urllib.request
from typing import Optional

import boto3

# ── Defaults ──────────────────────────────────────────────────────────────────
REGION    = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET",  "cross-bench-dev-artifacts-3684f16")
_POLL_INTERVAL = 30  # seconds

# ── Metrics shown in the combined table ───────────────────────────────────────
_COLUMNS = [
    ("pass_rate",                        "Pass%",   ".0%"),
    ("avg_resolved",                     "Res%",    ".0%"),
    ("avg_patch_successfully_applied",   "Patch%",  ".0%"),
    ("avg_f2p_rate",                     "F2P%",    ".0%"),
    ("avg_p2p_rate",                     "P2P%",    ".0%"),
    ("avg_steps_taken",                  "Steps",   ".1f"),
    ("avg_wallclock_seconds",            "Wall(s)", ".0f"),
    ("total_tokens_in",                  "Tok↑",    ".0f"),
    ("total_tokens_out",                 "Tok↓",    ".0f"),
    ("total_tasks",                      "N",       ".0f"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    env = pathlib.Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _post_slack(webhook_url: str, payload: dict) -> None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                print(f"  [slack] HTTP {resp.status}", file=sys.stderr)
    except Exception as exc:
        print(f"  [slack] failed: {exc}", file=sys.stderr)


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fmt.endswith("%"):
        prec = int(fmt[:-1].lstrip(".") or "0")
        return f"{v * 100:.{prec}f}%"
    return f"{v:{fmt}}"


def _fetch_summary(s3, s3_prefix: str, run_id: str) -> Optional[dict]:
    key = f"{s3_prefix}{run_id}/summary.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def _s3_console_url(run_id: str, s3_prefix: str) -> str:
    prefix = f"{s3_prefix}{run_id}/"
    params = urllib.parse.urlencode({"region": REGION, "prefix": prefix})
    return f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}?{params}"


def _post_run_done(webhook_url: str, label: str, run_id: str,
                   metrics: dict, s3_prefix: str) -> None:
    """Post a single-run completion message."""
    pass_rate = metrics.get("pass_rate")
    resolved  = metrics.get("avg_resolved")
    tasks     = int(metrics.get("total_tasks") or 0)
    errors    = int(metrics.get("errors") or 0)

    if errors:
        icon = ":warning:"
    elif pass_rate and pass_rate > 0:
        icon = ":white_check_mark:"
    else:
        icon = ":x:"

    lines = [
        f"{icon} *{label}* — run complete",
        f"Pass {_fmt(pass_rate, '.0%')}  ·  Resolved {_fmt(resolved, '.0%')}  "
        f"·  N={tasks}  ·  errors={errors}",
        f"`{run_id}`",
    ]
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(lines)},
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":package: S3", "emoji": True},
                    "url": _s3_console_url(run_id, s3_prefix),
                },
            }
        ],
        "text": f"{label} complete — pass {_fmt(pass_rate, '.0%')}",
    }
    _post_slack(webhook_url, payload)


def _build_comparison_table(results: list[tuple[str, dict]]) -> str:
    """ASCII table: one row per run, one column per metric."""
    headers = ["Run"] + [h for _, h, _ in _COLUMNS]
    widths  = [len(h) for h in headers]

    rows: list[list[str]] = []
    for label, metrics in results:
        row = [label] + [_fmt(metrics.get(k), f) for k, _, f in _COLUMNS]
        rows.append(row)
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    # Average row (numeric columns only)
    avg_row = ["AVG"]
    for k, _, f in _COLUMNS:
        vals = [float(m.get(k)) for _, m in results if m.get(k) is not None]
        avg_row.append(_fmt(statistics.mean(vals), f) if vals else "—")
    rows.append(avg_row)
    for i, v in enumerate(avg_row):
        widths[i] = max(widths[i], len(v))

    sep  = "  ".join("-" * w for w in widths)
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines = [head, sep]
    for i, row in enumerate(rows):
        lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
        if i == len(rows) - 2:          # separator before avg row
            lines.append(sep)
    return "\n".join(lines)


def _post_combined_summary(webhook_url: str, results: list[tuple[str, dict]],
                           benchmark: str, s3_prefix: str) -> None:
    table = _build_comparison_table(results)
    n = len(results)
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text",
                         "text": f"All {n} runs complete — {benchmark}", "emoji": True},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```\n{table}\n```"},
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn",
                               "text": f"s3://{S3_BUCKET}/{s3_prefix}"}],
            },
        ],
        "text": f"All {n} runs complete — {benchmark}",
    }
    _post_slack(webhook_url, payload)


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", dest="runs", action="append", default=[],
                    metavar="LABEL:RUN_ID",
                    help="Label and run ID to watch (repeatable)")
    ap.add_argument("--parent", default=None,
                    help="Discover shard run IDs from shards.json under this parent run ID")
    ap.add_argument("--s3-prefix", default="runs/swebench_pro_seal/",
                    help="S3 key prefix for artifacts (default: runs/swebench_pro_seal/)")
    ap.add_argument("--benchmark", default="swebench_pro_seal",
                    help="Benchmark name shown in the Slack summary")
    ap.add_argument("--interval", type=int, default=_POLL_INTERVAL,
                    help=f"Poll interval in seconds (default: {_POLL_INTERVAL})")
    ap.add_argument("--timeout", type=int, default=4 * 3600,
                    help="Give up after this many seconds (default: 4h)")
    return ap


def main() -> None:
    _load_dotenv()
    args = _build_parser().parse_args()

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        sys.exit("ERROR: SLACK_WEBHOOK_URL not set")

    s3_prefix = args.s3_prefix
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"

    s3 = boto3.client("s3", region_name=REGION)

    # ── Build watch list ──────────────────────────────────────────────────────
    watch: list[tuple[str, str]] = []  # (label, run_id)

    if args.parent:
        key = f"{s3_prefix}{args.parent}/shards.json"
        try:
            data = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read())
            for i, sid in enumerate(data.get("shard_run_ids", []), 1):
                watch.append((f"shard-{i:02d}", sid))
            print(f"Discovered {len(watch)} shards under {args.parent}")
        except Exception as exc:
            sys.exit(f"ERROR: could not load shards.json: {exc}")

    for spec in args.runs:
        label, _, run_id = spec.partition(":")
        watch.append((label or run_id, run_id.strip()))

    if not watch:
        sys.exit("ERROR: provide at least one --run LABEL:RUN_ID or --parent RUN_ID")

    print(f"Watching {len(watch)} run(s)  (poll every {args.interval}s)")
    for label, rid in watch:
        print(f"  {label:<30} {rid}")
    print()

    # ── Poll loop ─────────────────────────────────────────────────────────────
    pending  = dict(watch)          # label → run_id (still waiting)
    results  = []                   # (label, metrics) in completion order
    deadline = time.monotonic() + args.timeout

    while pending and time.monotonic() < deadline:
        time.sleep(args.interval)
        for label, run_id in list(pending.items()):
            summary = _fetch_summary(s3, s3_prefix, run_id)
            if summary is None:
                continue
            metrics = summary.get("aggregate_metrics", {})
            results.append((label, metrics))
            del pending[label]
            done = len(results)
            total = len(watch)
            print(f"  [{done:2d}/{total}] {label} — pass={_fmt(metrics.get('pass_rate'), '.0%')}")
            if webhook_url:
                _post_run_done(webhook_url, label, run_id, metrics, s3_prefix)

    if pending:
        print(f"\nTIMEOUT — {len(pending)} run(s) never produced a summary: {list(pending)}")
    else:
        print(f"\nAll {len(watch)} runs complete.")

    # ── Combined Slack summary ────────────────────────────────────────────────
    if results:
        # Sort back into original order
        order = {label: i for i, (label, _) in enumerate(watch)}
        results.sort(key=lambda x: order.get(x[0], 999))
        if webhook_url:
            _post_combined_summary(webhook_url, results, args.benchmark, s3_prefix)
            print("Combined summary posted to Slack.")
        else:
            table = _build_comparison_table(results)
            print("\n" + table)


if __name__ == "__main__":
    main()
