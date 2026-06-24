#!/usr/bin/env python3
"""
Post a smoke-run results table to Slack.

Reads cells_summary.csv produced by smoke_pro.sh and formats a compact
Slack Block Kit message with one row per cell.

Usage:
    python scripts/post_smoke_to_slack.py \
        --csv /tmp/smoke-pro-<TS>/cells_summary.csv \
        --sha 6be9a79 \
        --benchmark swebench_pro_seal \
        --s3-root s3://cross-bench-dev-artifacts-3684f16/runs/swebench_pro_seal/

Slack webhook URL must be set in the SLACK_WEBHOOK_URL environment variable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Optional


# ── Defaults matching dispatch_swebench.py ────────────────────────────────────
_DEFAULT_REGION  = "us-east-1"
_DEFAULT_CLUSTER = "cross-bench-dev-cluster-a2d8750"

# ── Key metrics to show in the table ─────────────────────────────────────────
# Column spec: (csv_key, header, format_fn)
_COLUMNS: list[tuple[str, str, str]] = [
    ("pass_rate_mean",                   "Pass%",   ".0%"),
    ("avg_resolved_mean",                "Res%",    ".0%"),
    ("avg_patch_successfully_applied_mean", "Patch%", ".0%"),
    ("avg_steps_taken_mean",             "Steps",   ".1f"),
    ("total_tasks_mean",                 "Tasks",   ".0f"),
]


def _fmt(value: Optional[str], fmt: str) -> str:
    if value is None or value == "":
        return "—"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return str(value)
    if fmt.endswith("%"):
        prec_part = fmt[:-1].lstrip(".")
        prec = int(prec_part) if prec_part else 0
        return f"{v * 100:.{prec}f}%"
    return f"{v:{fmt}}"


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _build_table(rows: list[dict[str, str]]) -> str:
    """Return a preformatted ASCII table string."""
    headers = ["Cell"] + [h for _, h, _ in _COLUMNS]
    col_specs = [(_k, _h, _f) for _k, _h, _f in _COLUMNS]

    widths = [len(h) for h in headers]
    data_rows: list[list[str]] = []
    for row in rows:
        cell_name = row.get("cell", "?")
        cells_formatted = [cell_name] + [
            _fmt(row.get(k), f) for k, _, f in col_specs
        ]
        data_rows.append(cells_formatted)
        for i, v in enumerate(cells_formatted):
            widths[i] = max(widths[i], len(v))

    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines = [header_line, sep]
    for dr in data_rows:
        lines.append("  ".join(v.ljust(w) for v, w in zip(dr, widths)))
    return "\n".join(lines)


def _s3_console_url(s3_root: str, region: str) -> str:
    """Convert s3://bucket/prefix/ → S3 console URL."""
    if not s3_root.startswith("s3://"):
        return s3_root
    without = s3_root[len("s3://"):]
    bucket, _, prefix = without.partition("/")
    params = urllib.parse.urlencode({"region": region, "prefix": prefix})
    return f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}?{params}"


def _ecs_console_url(cluster: str, region: str) -> str:
    """Link to the ECS cluster tasks page."""
    return (
        f"https://{region}.console.aws.amazon.com/ecs/v2/clusters"
        f"/{cluster}/tasks?region={region}"
    )


def _emoji_bar(rate_str: str, fmt: str) -> str:
    """Render a 5-block bar from a formatted percentage string."""
    try:
        v = float(rate_str.rstrip("%")) / 100 if "%" in rate_str else float(rate_str)
    except (ValueError, TypeError):
        return "░░░░░"
    filled = round(v * 5)
    return "█" * filled + "░" * (5 - filled)


def _build_payload(
    table: str,
    sha: str,
    benchmark: str,
    s3_root: str,
    n_cells: int,
    started_by: str = "",
    region: str = _DEFAULT_REGION,
    cluster: str = _DEFAULT_CLUSTER,
    rows: list | None = None,
) -> dict:
    title = f":trophy: Smoke complete — {sha} · {benchmark}"

    # ── Header ───────────────────────────────────────────────────────────────
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f"🏁  Smoke done  ·  {sha}  ·  {benchmark}", "emoji": True},
        },
        {"type": "divider"},
    ]

    # ── Per-cell rows as rich fields ──────────────────────────────────────────
    if rows:
        for row in rows:
            cell = row.get("cell", "?")
            pass_v  = _fmt(row.get("pass_rate_mean"), ".0%")
            res_v   = _fmt(row.get("avg_resolved_mean"), ".0%")
            patch_v = _fmt(row.get("avg_patch_successfully_applied_mean"), ".0%")
            steps_v = _fmt(row.get("avg_steps_taken_mean"), ".1f")
            tasks_v = _fmt(row.get("total_tasks_mean"), ".0f")
            f2p_v   = _fmt(row.get("avg_f2p_rate_mean"), ".0%")
            p2p_v   = _fmt(row.get("avg_p2p_rate_mean"), ".0%")
            wall_v  = _fmt(row.get("avg_wallclock_seconds_mean"), ".0f")
            wall_s  = f"{int(float(wall_v))//60}m {int(float(wall_v))%60}s" \
                      if wall_v not in ("—", "") else "—"
            # resolved count: pass_rate × total_tasks
            try:
                resolved_n = round(float(row.get("pass_rate_mean", 0) or 0) *
                                   float(row.get("total_tasks_mean", 0) or 0))
                resolved_s = f"{resolved_n} / {tasks_v}"
            except Exception:
                resolved_s = "—"

            pass_bar  = _emoji_bar(pass_v, ".0%")
            patch_bar = _emoji_bar(patch_v, ".0%")

            tok_in_v  = _fmt(row.get("total_tokens_in_mean"),  ".0f")
            tok_out_v = _fmt(row.get("total_tokens_out_mean"), ".0f")
            def _tok_k(s):
                try: v = float(s); return f"{v/1000:.1f}k" if v >= 1000 else s
                except: return s
            # SEAL runs can't introspect the vendored sweagent subprocess
            # so steps + tokens come back as 0. Hide those fields when
            # they're not measurable to avoid showing misleading zeros.
            # Mini-swe-agent runs report both natively; this gate leaves
            # them visible there.
            try:
                steps_val = float(row.get("avg_steps_taken_mean") or 0)
            except (TypeError, ValueError):
                steps_val = 0.0
            try:
                tokens_val = float(row.get("total_tokens_in_mean") or 0) + \
                             float(row.get("total_tokens_out_mean") or 0)
            except (TypeError, ValueError):
                tokens_val = 0.0
            fields = [
                {"type": "mrkdwn", "text": f"*{cell}*\n`N={tasks_v}`"},
                {"type": "mrkdwn", "text": f"*Resolved*\n{pass_bar}  *{pass_v}*  `{resolved_s}`"},
                {"type": "mrkdwn", "text": f"*F2P  ·  P2P*\n`{f2p_v}`  ·  `{p2p_v}`"},
                {"type": "mrkdwn", "text": f"*Patch*\n{patch_bar}  {patch_v}"},
            ]
            if steps_val > 0:
                fields.append({"type": "mrkdwn", "text": f"*Steps  ·  Wall*\n`{steps_v}`  ·  `{wall_s}`"})
            else:
                # Keep the Wall column visible even when Steps is hidden;
                # SEAL still reports wallclock seconds.
                fields.append({"type": "mrkdwn", "text": f"*Wall*\n`{wall_s}`"})
            if tokens_val > 0:
                fields.append({"type": "mrkdwn", "text": f"*Tokens  ↑in / ↓out*\n`{_tok_k(tok_in_v)}` / `{_tok_k(tok_out_v)}`"})
            blocks.append({
                "type": "section",
                "fields": fields,
            })
            blocks.append({"type": "divider"})
    else:
        # Fallback: plain code-block table
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"```\n{table}\n```"},
        })
        blocks.append({"type": "divider"})

    # ── Footer context ────────────────────────────────────────────────────────
    footer_parts = [f"{n_cells} cell(s)"]
    if started_by:
        footer_parts.append(f"ECS label: `{started_by}-NN`")
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "  |  ".join(footer_parts)}],
    })

    # ── Action buttons ────────────────────────────────────────────────────────
    buttons = []
    if s3_root:
        buttons.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "📦  S3 artifacts", "emoji": True},
            "url": _s3_console_url(s3_root, region),
        })
    buttons.append({
        "type": "button",
        "text": {"type": "plain_text", "text": "🖥️  ECS tasks", "emoji": True},
        "url": _ecs_console_url(cluster, region),
    })
    blocks.append({"type": "actions", "elements": buttons})

    return {"blocks": blocks, "text": title}


def post_to_slack(webhook_url: str, payload: dict) -> None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        status = resp.status
        body_resp = resp.read().decode()
    if status != 200 or body_resp.strip() != "ok":
        raise RuntimeError(f"Slack returned HTTP {status}: {body_resp!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to cells_summary.csv")
    ap.add_argument("--sha", default="", help="Commit SHA (display only)")
    ap.add_argument("--benchmark", default="swebench_pro_seal", help="Benchmark name")
    ap.add_argument("--s3-root", default="", help="S3 URL for artifact browsing link")
    ap.add_argument("--started-by", default="",
                    help="ECS startedBy prefix (e.g. dispatch-seal) shown in the footer")
    ap.add_argument("--region", default=_DEFAULT_REGION,
                    help="AWS region (default: us-east-1)")
    ap.add_argument("--cluster", default=_DEFAULT_CLUSTER,
                    help="ECS cluster name for console link")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the payload instead of posting")
    args = ap.parse_args()

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url and not args.dry_run:
        sys.exit("ERROR: SLACK_WEBHOOK_URL environment variable is not set.")

    rows = _read_csv(args.csv)
    if not rows:
        sys.exit(f"ERROR: {args.csv} is empty or has no data rows.")

    # Skip cells with no usable metrics (task errored before any
    # ``total_tasks`` was recorded). Posting a row with every field
    # rendering as ``—`` is misleading — the ":trophy: Smoke complete"
    # header implies success when reality is "ECS task errored or
    # never ran". Drop those rows; if NONE survive, exit without
    # posting (operator will see the failure via CloudWatch).
    def _has_data(row: dict) -> bool:
        # ``total_tasks_mean`` is the canonical "this cell produced
        # at least one task result" signal in the summary CSV. Empty
        # string or None means the run errored / has no aggregate.
        v = row.get("total_tasks_mean", "")
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False

    usable_rows = [r for r in rows if _has_data(r)]
    errored = [r.get("cell", "?") for r in rows if not _has_data(r)]
    if errored:
        print(
            f"WARNING: dropping {len(errored)} errored cell(s) from Slack post: "
            f"{', '.join(errored)}",
            file=sys.stderr,
        )
    if not usable_rows:
        print(
            f"ERROR: every cell in {args.csv} errored before producing aggregate "
            f"metrics. Skipping Slack post — check CloudWatch for the failure.",
            file=sys.stderr,
        )
        sys.exit(2)

    table = _build_table(usable_rows)
    payload = _build_payload(
        table, args.sha, args.benchmark, args.s3_root, len(usable_rows),
        started_by=args.started_by,
        region=args.region,
        cluster=args.cluster,
        rows=usable_rows,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    print(f"Posting to Slack ({len(rows)} cell(s))…")
    post_to_slack(webhook_url, payload)
    print("  ✓ posted")


if __name__ == "__main__":
    main()
