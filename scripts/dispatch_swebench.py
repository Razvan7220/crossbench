#!/usr/bin/env python3
"""
Parallel SWE-bench fan-out across ECS — one ECS task per shard,
N instances per shard.  Mirrors the LoCoMo ``dispatch_shards`` pattern
but keyed on instance_id instead of conversation_id.

After all shards STOP the script merges per-shard results into a single
parent ``summary.json`` and prints a master table (pass/fail per shard +
aggregate pass-rate, avg resolve-rate, timing).

USAGE
-----
  # 50 instances, 1 per ECS task, mem0 + BM25:
  python scripts/dispatch_swebench.py \\
      --sha abc1234 \\
      --profiles online,mem0v2_embedded,swebench_retrieval_bm25 \\
      --limit 50

  # All 500 instances, 5 per ECS task:
  python scripts/dispatch_swebench.py \\
      --sha abc1234 \\
      --profiles online,mem0v2_embedded,swebench_retrieval_bm25 \\
      --tasks-per-shard 5

  # Explicit task IDs:
  python scripts/dispatch_swebench.py \\
      --sha abc1234 \\
      --profiles online,mem0v2_embedded,swebench_retrieval_bm25 \\
      --task-id astropy__astropy-12907 \\
      --task-id astropy__astropy-13033

ENVIRONMENT OVERRIDES
---------------------
  AWS_REGION, ECR_URL, CLUSTER, CAPACITY_PROVIDER, TASK_DEF_FAMILY, S3_BUCKET
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone

import urllib.parse
import urllib.request

import boto3  # pip install boto3

# ── Defaults (match dispatch_pinned.sh) ──────────────────────────────────────
import os

REGION            = os.environ.get("AWS_REGION",            "us-east-1")
ECR_URL           = os.environ.get("ECR_URL",               "564175165906.dkr.ecr.us-east-1.amazonaws.com")
CLUSTER           = os.environ.get("CLUSTER",               "cross-bench-dev-cluster-a2d8750")
CAPACITY_PROVIDER = os.environ.get("CAPACITY_PROVIDER",     "cross-bench-dev-cp-f3ceab1")
TASK_DEF_FAMILY   = os.environ.get("TASK_DEF_FAMILY",       "cross-bench-dev-orchestrator")
S3_BUCKET         = os.environ.get("S3_BUCKET",             "cross-bench-dev-artifacts-3684f16")
# --artifacts-dir is a LOCAL path on the orchestrator container; S3
# mirroring is driven by BENCH_ARTIFACTS_BACKEND=s3 + BENCH_ARTIFACTS_BUCKET
# on the ECS task definition (see infra/pulumi/ecs.py). Passing an
# ``s3://...`` URL here makes ArtifactStore write task JSONs under a
# literal ``./s3:/<bucket>/...`` directory on the container's ephemeral
# disk — which fills the disk after ~15-20 SEAL Pro tasks (each task
# JSON carries full sweagent stdout/stderr). Use a tmpfs-backed path so
# the local copy is a working cache; the S3 mirror is the source of truth.
ARTIFACTS_LOCAL_DIR = os.environ.get("ARTIFACTS_LOCAL_DIR",  "/tmp/artifacts")

_POLL_INTERVAL = 30   # seconds between ECS describe-tasks calls
_DEFAULT_TIMEOUT = 4 * 3600  # 4 h ceiling


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shard_run_id(
    parent: str,
    idx: int,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> str:
    """Stable deterministic shard ID (parent + index → short hash).

    When ``start_offset`` is given, the shard's HF dataset offset is baked
    into the name as ``offNNN`` (single-task shard) or ``offNNN-MMM``
    (multi-task shard), making per-shard failure triage possible without
    a SHA-fragment lookup. The trailing 12-char hash is unchanged (still
    derived from ``parent + idx``), so recovery paths that recompute the
    shard ID stay deterministic.
    """
    h = hashlib.sha256(f"{parent}-shard-{idx}".encode()).hexdigest()[:12]
    if start_offset is None:
        return f"{parent}-s{idx:03d}-{h}"
    if end_offset is None or end_offset == start_offset:
        off_part = f"off{start_offset:03d}"
    else:
        off_part = f"off{start_offset:03d}-{end_offset:03d}"
    return f"{parent}-s{idx:03d}-{off_part}-{h}"


_DATASET_BY_VARIANT = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite":     "princeton-nlp/SWE-bench_Lite",
    "full":     "princeton-nlp/SWE-bench",
    "pro":      "ScaleAI/SWE-bench_Pro",
}


def _load_task_ids(
    variant: str, limit: int | None, offset: int = 0
) -> list[str]:
    """Return up to ``limit`` instance_ids starting at ``offset``.

    ``offset`` is the zero-based index into the HF dataset. ``ids[offset:
    offset+limit]`` is the slice — same semantics as Python list slicing.
    ``offset=0`` (default) preserves the original "first N" behaviour.
    Useful for sweep dispatches that want to fan out chunks (e.g.
    ``--offset 50 --limit 50`` to run instances 50-99).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'datasets' package not installed.  Run: pip install datasets")
    ds_name = _DATASET_BY_VARIANT[variant]
    print(f"Loading {ds_name} task IDs from HuggingFace…")
    ds = load_dataset(ds_name, split="test")
    ids: list[str] = [row["instance_id"] for row in ds]
    if offset:
        ids = ids[offset:]
    return ids[:limit] if limit else ids


# Back-compat alias for callers that imported the old name.
_load_verified_ids = _load_task_ids


def _truncate_started_by(value: str, limit: int = 36) -> str:
    """Truncate to ECS ``startedBy`` 36-char cap while preserving the
    trailing ``-NNN`` numeric offset suffix.

    Naïve ``value[:limit]`` slices off the offset on long cell names
    (e.g. ``dispatch-abl_05_l1_first_ast_codeembed-set1-000`` at 47
    chars → ``dispatch-abl_05_l1_first_ast_codeemb`` (35), every shard
    identical). This helper keeps the final ``-<digits>`` segment and
    truncates the prefix instead so shards remain distinguishable in
    the ECS console.
    """
    if len(value) <= limit:
        return value
    head, sep, tail = value.rpartition("-")
    if sep and tail.isdigit():
        suffix = sep + tail  # e.g. "-000"
        if len(suffix) < limit:
            return head[: limit - len(suffix)] + suffix
    return value[:limit]


def _ecr_tag_exists(ecr_client, short_sha: str) -> bool:
    try:
        ecr_client.describe_images(
            repositoryName="cross-bench-orchestrator",
            imageIds=[{"imageTag": short_sha}],
        )
        return True
    except ecr_client.exceptions.ImageNotFoundException:
        return False


def _register_pinned_task_def(ecs_client, short_sha: str) -> str:
    """Clone the current task-def, swap image to :<sha>, register & return ARN."""
    pinned_image = f"{ECR_URL}/cross-bench-orchestrator:{short_sha}"
    td = ecs_client.describe_task_definition(taskDefinition=TASK_DEF_FAMILY)["taskDefinition"]

    for c in td.get("containerDefinitions", []):
        if c.get("name") == "orchestrator":
            c["image"] = pinned_image
            break
    else:
        sys.exit("ERROR: container 'orchestrator' not found in task definition")

    for k in ("taskDefinitionArn", "revision", "status", "requiresAttributes",
               "compatibilities", "registeredAt", "registeredBy"):
        td.pop(k, None)

    resp = ecs_client.register_task_definition(**td)
    rev  = resp["taskDefinition"]["revision"]
    full = f"{TASK_DEF_FAMILY}:{rev}"
    print(f"  Registered {full}  (image={pinned_image})")
    return full


def _dispatch_one_shard(
    ecs_client,
    task_def: str,
    shard_run_id: str,
    task_ids: list[str],
    profiles: list[str],
    extra_args: list[str],
    *,
    benchmark: str = "swebench",
    s3_prefix: str = "",
    started_by: str = "",
    extra_env: list[dict[str, str]] | None = None,
) -> str:
    """Fire a single ECS task for this shard.  Returns the taskArn."""
    cmd: list[str] = [
        "run",
        "--run-id",        shard_run_id,
        "--artifacts-dir", ARTIFACTS_LOCAL_DIR,
        "--db-url",        "none",
        "-b",              benchmark,
    ]
    for p in profiles:
        cmd += ["-p", p]
    for tid in task_ids:
        cmd += ["--task-id", tid]
    cmd.extend(extra_args)

    # Forward the S3 prefix so artifacts land under the navigable
    # ``s3://<bucket>/<s3_prefix>/<run_id>/...`` layout. Empty prefix
    # preserves the flat ``s3://<bucket>/<run_id>/...`` layout for
    # back-compat (Verified existing dispatches keep their behavior).
    container_env = []
    if s3_prefix:
        container_env.append({"name": "BENCH_ARTIFACTS_S3_PREFIX", "value": s3_prefix})
    if extra_env:
        # Caller-supplied --extra-env entries (e.g. CROSSBENCH_SLACK_PER_TASK=1,
        # SLACK_WEBHOOK_URL=...). These take precedence over the task
        # definition's environment block via the ECS overrides mechanism.
        container_env.extend(extra_env)

    run_task_kwargs: dict = dict(
        cluster=CLUSTER,
        taskDefinition=task_def,
        capacityProviderStrategy=[{"capacityProvider": CAPACITY_PROVIDER, "weight": 1}],
        overrides={"containerOverrides": [
            {
                "name": "orchestrator",
                "command": cmd,
                **({"environment": container_env} if container_env else {}),
            }
        ]},
        # ``distinctInstance`` forces one task per EC2 host. SEAL / SWE-bench
        # tasks spawn multi-GB sibling containers via the mounted docker.sock
        # and pull 3-5 GB per-instance images, so one task per host keeps disk
        # + memory headroom predictable. Forces the ASG to scale out instead
        # of bin-packing; capacity is already sized for this (max_size vs
        # typical demand ~30 shards).
        placementConstraints=[{"type": "distinctInstance"}],
    )
    # startedBy appears in the ECS console "Started by" column (max 36 chars
    # per the AWS API). Naïve [:36] truncation slices off the trailing
    # ``-NNN`` offset suffix on long cell names — every shard ends up with
    # identical truncated text, defeating the per-shard ID. Preserve the
    # final numeric suffix and truncate the cell-name middle instead.
    if started_by:
        run_task_kwargs["startedBy"] = _truncate_started_by(started_by, 36)

    # Retry loop for transient capacity failures. Background: when
    # ``smoke_pro.sh`` fans out parallel cell dispatches against a cold
    # ECS cluster (min_size=0 ASG), the first ``run-task`` grabs the
    # available instance, and the second one races against capacity-
    # provider scale-up. AWS returns a synchronous ``RESOURCE:MEMORY``
    # failure instead of queueing the task — same root cause documented
    # in workflow.md ("If you see TaskFailedToStart: RESOURCE:MEMORY ...
    # raise max_size"; but here it's run-task-time, not after-launch).
    # The first failed attempt triggers ASG scale-up; subsequent retries
    # succeed once the new c5.xlarge comes online (~3-5 min). Retry up
    # to 8 times with exponential backoff capped at 60s — total budget
    # ~6 min, well within smoke_pro.sh's wait_for-all-cells loop.
    import time as _time
    # Bumped from 8 → 40 for large fan-outs (731+ shards). At max_size=60
    # the 61st-731st shards must wait for predecessors to finish; with
    # ~10 min/task that's ~2 hours of waiting for the tail. 8 retries
    # (max delay 60s × 8 ≈ 8 min) wasn't enough.
    max_attempts = 40
    for attempt in range(1, max_attempts + 1):
        resp = ecs_client.run_task(**run_task_kwargs)
        failures = resp.get("failures", [])
        if not failures:
            return resp["tasks"][0]["taskArn"]
        # Treat RESOURCE:* AND placement-constraint failures as transient
        # and retry; bail loud on everything else (auth, invalid task-def,
        # missing image, …). DistinctInstance constraint fires when all
        # eligible instances are taken — same semantic as RESOURCE:MEMORY
        # for our purposes (wait for ASG / running tasks to free a slot).
        reason = (failures[0].get("reason") or "") if failures else ""
        detail = failures[0].get("detail") or "" if failures else ""
        is_capacity = (
            reason.startswith("RESOURCE:")
            or "Insufficient" in detail
            or "placement constraint" in reason.lower()
            or "DistinctInstance" in reason
        )
        if not is_capacity or attempt == max_attempts:
            sys.exit(
                f"ECS run-task failed for shard {shard_run_id} "
                f"(attempt {attempt}/{max_attempts}): {failures}"
            )
        delay = min(60, 2 ** attempt)
        print(
            f"  capacity unavailable (attempt {attempt}/{max_attempts}) — "
            f"sleeping {delay}s then retrying: {failures[0].get('reason')} "
            f"({failures[0].get('detail', '')[:80]})",
            flush=True,
        )
        _time.sleep(delay)


def _fetch_shard_summary(shard_run_id: str, s3_prefix: str) -> dict:
    """Fetch aggregate_metrics from the shard's summary.json (best-effort, returns {} on failure)."""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        key = f"{s3_prefix}{shard_run_id}/summary.json"
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        return data.get("aggregate_metrics", {})
    except Exception:
        return {}


def _fetch_shard_tasks(shard_run_id: str, s3_prefix: str) -> list[dict]:
    """Fetch per-task evaluation dicts from tasks/ in the shard (best-effort)."""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        prefix = f"{s3_prefix}{shard_run_id}/tasks/"
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        tasks = []
        for obj in resp.get("Contents", []):
            if not obj["Key"].endswith(".json"):
                continue
            body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
            d = json.loads(body)
            tasks.append({
                "task_id": d.get("task_id", ""),
                **d.get("evaluation", {}),
            })
        return tasks
    except Exception:
        return []


def _fmt_pct(v) -> str:
    return f"{v * 100:.0f}%" if v is not None else "—"

def _fmt_tokens(n) -> str:
    if n is None:
        return "—"
    n = int(n)
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)

def _fmt_duration(secs) -> str:
    if secs is None:
        return "—"
    secs = int(secs)
    return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"


def _emoji_bar(rate) -> str:
    """Render a mini 5-block progress bar from a 0-1 rate."""
    if rate is None:
        return "░░░░░"
    filled = round(rate * 5)
    return "█" * filled + "░" * (5 - filled)


def _post_shard_slack(
    webhook_url: str,
    label: str,
    shard_run_id: str,
    exit_code: str,
    s3_prefix: str,
) -> None:
    """Post a per-shard done notification to Slack (best-effort, never raises)."""
    # ── ECS level ────────────────────────────────────────────────────────────
    if exit_code == "TIMEOUT":
        ecs_icon, ecs_text = ":stopwatch:", "Timed out"
    elif exit_code == "0":
        ecs_icon, ecs_text = ":white_check_mark:", "Success"
    else:
        ecs_icon, ecs_text = ":x:", f"Error (exit {exit_code})"

    # ── Task level (from summary.json) ───────────────────────────────────────
    metrics = _fetch_shard_summary(shard_run_id, s3_prefix)

    passed = int(metrics.get("passed") or 0)
    failed = int(metrics.get("failed") or 0)
    errors = int(metrics.get("errors") or 0)

    if errors:
        task_icon = ":x:"
    elif failed:
        task_icon = ":warning:"
    elif metrics:
        task_icon = ":white_check_mark:"
    else:
        task_icon = ":white_circle:"

    bucket, _, prefix = f"{S3_BUCKET}/{s3_prefix}{shard_run_id}/".partition("/")
    params = urllib.parse.urlencode({"region": REGION, "prefix": prefix})
    console_url = f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}?{params}"

    # ── Per-task bullets (fetched from S3) ───────────────────────────────────
    task_list = _fetch_shard_tasks(shard_run_id, s3_prefix)

    def _task_bullet(t: dict) -> str:
        tid     = t.get("task_id", "?").replace("instance_", "")[:50]
        res     = t.get("resolved", False)
        err     = t.get("error", "")
        f2p     = t.get("avg_f2p_rate") or t.get("f2p_rate")
        p2p     = t.get("avg_p2p_rate") or t.get("p2p_rate")
        patch   = t.get("avg_patch_successfully_applied") or t.get("patch_successfully_applied")
        steps   = t.get("steps_taken")
        wall    = t.get("wallclock_seconds")
        tok_in  = _fmt_tokens(t.get("total_tokens_in"))
        tok_out = _fmt_tokens(t.get("total_tokens_out"))
        icon    = ":white_check_mark:" if res else (":x:" if err else ":warning:")
        stats   = []
        if f2p   is not None: stats.append(f"F2P {_fmt_pct(f2p)}")
        if p2p   is not None: stats.append(f"P2P {_fmt_pct(p2p)}")
        if patch is not None: stats.append(f"patch {_fmt_pct(patch)}")
        if steps is not None: stats.append(f"steps {int(steps)}")
        if wall:               stats.append(f":stopwatch: {_fmt_duration(wall)}")
        if tok_in and tok_in != "—": stats.append(f":abc: {tok_in} in / {tok_out} out")
        stat_s = "  ·  ".join(stats)
        return f"  • {icon} `{tid}`" + (f"\n      {stat_s}" if stat_s else "")

    # ── Build blocks ─────────────────────────────────────────────────────────
    task_summary = f"{task_icon} Task: {passed} passed · {failed} failed · {errors} errors"
    header_text  = f"{ecs_icon} *{label}* — ECS: {ecs_text}\n{task_summary}"

    blocks: list[dict] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": header_text},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": ":package: S3", "emoji": True},
                "url": console_url,
            },
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"`{shard_run_id}`"}],
        },
    ]

    # ── Stats line (from summary metrics) ────────────────────────────────────
    if metrics:
        f2p   = metrics.get("avg_f2p_rate")
        p2p   = metrics.get("avg_p2p_rate")
        patch = metrics.get("avg_patch_successfully_applied")
        steps = metrics.get("avg_steps_taken")
        wall  = metrics.get("avg_wallclock_seconds")
        tok_in  = _fmt_tokens(metrics.get("total_tokens_in"))
        tok_out = _fmt_tokens(metrics.get("total_tokens_out"))
        steps_s = f"{steps:.0f}" if steps is not None else "—"
        wall_s  = _fmt_duration(wall) if wall else "—"
        # Build the stats line segment-by-segment so SEAL runs (which can't
        # introspect the vendored sweagent subprocess) don't display ``0``
        # for fields the scaffold can't measure. Mini-swe-agent runs own
        # the agent loop in-process and report both steps + tokens
        # natively — those segments will appear there.
        segments = [
            f"F2P {_fmt_pct(f2p)}",
            f"P2P {_fmt_pct(p2p)}",
            f"patch {_fmt_pct(patch)}",
        ]
        if steps is not None and steps > 0:
            segments.append(f"steps {steps_s}")
        if wall and wall > 0:
            segments.append(f":stopwatch: {wall_s}")
        tokens_in_n = metrics.get("total_tokens_in") or 0
        tokens_out_n = metrics.get("total_tokens_out") or 0
        if tokens_in_n > 0 or tokens_out_n > 0:
            segments.append(f":abc: {tok_in} in / {tok_out} out")
        # Render the stats line whenever at least one of the optional
        # segments populated; otherwise the F2P/P2P/patch trio alone is
        # already useful as a status summary.
        has_data = (steps is not None or (wall and wall > 0)
                    or tokens_in_n > 0)
        if has_data:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "  ·  ".join(segments)}],
            })

    # ── Bullet per instance when multiple tasks per shard ─────────────────────
    if len(task_list) > 1:
        bullets = "\n".join(_task_bullet(t) for t in task_list)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": bullets},
        })

    payload = {
        "blocks": blocks,
        "text": f"{label} — ECS {ecs_text} · {passed}✓ {failed}✗ {errors}err",
    }

    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
    except Exception as exc:
        print(f"  [slack] per-shard notify failed (non-fatal): {exc}")


def _poll_until_stopped(
    ecs_client,
    arn_to_shard: dict[str, str],
    timeout: float,
    on_stopped=None,  # optional callback(shard_id: str, exit_code: str)
) -> dict[str, str]:
    """
    Block until every ECS task is STOPPED (or timeout).
    Returns {taskArn: exit_code_str | "TIMEOUT"}.
    on_stopped fires immediately when each individual task stops.
    """
    remaining = set(arn_to_shard)
    results: dict[str, str] = {}
    deadline = time.monotonic() + timeout
    total    = len(remaining)

    while remaining and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL)
        # describe_tasks is hard-capped at 100 task ARNs per call by AWS;
        # chunk the remaining set so large fan-outs (731-shard Pro runs)
        # don't blow up with InvalidParameterException.
        remaining_list = list(remaining)
        described: list[dict] = []
        for chunk_start in range(0, len(remaining_list), 100):
            chunk = remaining_list[chunk_start:chunk_start + 100]
            resp_chunk = ecs_client.describe_tasks(cluster=CLUSTER, tasks=chunk)
            described.extend(resp_chunk.get("tasks", []))
        resp = {"tasks": described}
        for t in resp.get("tasks", []):
            arn    = t["taskArn"]
            status = t.get("lastStatus", "")
            if status == "STOPPED":
                exit_code = None
                for c in t.get("containers", []):
                    if c.get("name") == "orchestrator":
                        exit_code = c.get("exitCode")
                        break
                code = str(exit_code) if exit_code is not None else "0"
                results[arn] = code
                remaining.discard(arn)
                done = total - len(remaining)
                sid  = arn_to_shard[arn]
                print(f"  [{done:3d}/{total}] STOPPED  {sid}  exit={code}")
                if on_stopped:
                    on_stopped(sid, code)

    for arn in remaining:
        results[arn] = "TIMEOUT"
        done = total - len(remaining) + 1
        sid  = arn_to_shard[arn]
        print(f"  [{done:3d}/{total}] TIMEOUT  {sid}")

    return results


def _merge_shards(
    parent_run_id: str,
    shard_run_ids: list[str],
    *,
    benchmark: str = "swebench",
    s3_prefix: str = "",
) -> None:
    """Call ``cross-bench merge-shards`` as a subprocess to produce summary.json."""
    cmd = [
        sys.executable, "-m", "cross_bench.cli",
        "merge-shards",
        "--parent",     parent_run_id,
        "--benchmark",  benchmark,
        "--artifacts-dir", ARTIFACTS_LOCAL_DIR,
    ]
    for sid in shard_run_ids:
        cmd += ["--shard", sid]
    env = os.environ.copy()
    # Force the merger to write to S3 (not local ./artifacts). When the
    # operator runs this script from their laptop without
    # BENCH_ARTIFACTS_BACKEND=s3 in their shell, the cross-bench merge-
    # shards subprocess falls through to ArtifactStore(backend="local")
    # and writes the parent summary.json to ./artifacts/<run_id>/ on the
    # operator's disk — never uploaded to S3. The shards-themselves were
    # written by their ECS containers which DO have these env vars, so
    # this only affects the merge step. Setting them here on the env
    # passed to the subprocess fixes the parent-summary-missing bug.
    env.setdefault("BENCH_ARTIFACTS_BACKEND", "s3")
    env.setdefault("BENCH_ARTIFACTS_BUCKET", S3_BUCKET)
    if s3_prefix:
        env["BENCH_ARTIFACTS_S3_PREFIX"] = s3_prefix
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"  [warn] merge-shards exited {result.returncode} — check output above")


# ── Master log ────────────────────────────────────────────────────────────────

def _print_master_log(
    parent_run_id: str,
    arn_to_shard: dict[str, str],
    shard_to_tids: dict[str, list[str]],
    poll_results: dict[str, str],
    *,
    s3_prefix: str = "",
) -> None:
    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  MASTER LOG — {parent_run_id}")
    print(sep)
    header = f"  {'SHARD-ID':<36}  {'TASKS':>5}  {'EXIT':>4}  STATUS"
    print(header)
    print(f"  {sep}")

    ok = fail = timeout = 0
    for arn, sid in sorted(arn_to_shard.items(), key=lambda x: x[1]):
        code   = poll_results.get(arn, "?")
        tids   = shard_to_tids.get(sid, [])
        status = "OK" if code == "0" else ("TIMEOUT" if code == "TIMEOUT" else f"FAIL({code})")
        print(f"  {sid:<36}  {len(tids):>5}  {code:>4}  {status}")
        if code == "0":
            ok += 1
        elif code == "TIMEOUT":
            timeout += 1
        else:
            fail += 1

    print(f"  {sep}")
    total = ok + fail + timeout
    print(f"  Total: {total}  ✓ {ok}  ✗ {fail}  ⏱ {timeout}")
    print(sep)
    parent_url = f"s3://{S3_BUCKET}/{s3_prefix}{parent_run_id}/"
    print(f"\n  Parent run     : s3://{S3_BUCKET}/{s3_prefix}{parent_run_id}/")
    print(f"  Summary        : aws s3 cp {parent_url}summary.json -")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fan out SWE-bench across ECS shards, poll, merge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
        # 50 tasks, 1 per shard, mem0 + BM25:
        python scripts/dispatch_swebench.py \\
            --sha abc1234 \\
            --profiles online,mem0v2_embedded,swebench_retrieval_bm25 \\
            --limit 50

        # All 500, 5 per shard:
        python scripts/dispatch_swebench.py \\
            --sha abc1234 \\
            --profiles online,mem0v2_embedded,swebench_retrieval_bm25 \\
            --tasks-per-shard 5

        # Dry-run to preview shards without launching:
        python scripts/dispatch_swebench.py \\
            --sha abc1234 \\
            --profiles online,mem0v2_embedded \\
            --limit 10 --dry-run
        """),
    )
    ap.add_argument("--sha", required=True,
                    help="Commit SHA (7+ chars) — must exist in ECR as :<sha> tag")
    ap.add_argument("--task-def", default=None,
                    help="Pre-registered task-def revision (e.g. cross-bench-dev-orchestrator:75). "
                         "Skips RegisterTaskDefinition — use when dispatching many runs in parallel "
                         "to avoid the concurrent-registration rate limit.")
    ap.add_argument("--run-id", default=None,
                    help="Parent run ID.  Default: swebench-<profiles>-YYYYMMDD-HHMMSS")
    ap.add_argument("--profiles", required=True,
                    help="Comma-separated profile names, "
                         "e.g. online,mem0v2_embedded,swebench_retrieval_bm25")
    ap.add_argument("--limit", type=int, default=None,
                    help="Take only N task IDs from the HF dataset (default: all)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Zero-based offset into the HF dataset before "
                         "taking --limit (default: 0). Combine with --limit "
                         "to fan out chunks: e.g. --offset 50 --limit 50 runs "
                         "instances 50-99. Ignored when --task-id is given.")
    ap.add_argument("--task-id", dest="explicit_task_ids", action="append", default=[],
                    metavar="INSTANCE_ID",
                    help="Explicit instance ID (repeatable).  Overrides --limit/--offset.")
    ap.add_argument("--tasks-per-shard", type=int, default=1,
                    help="Instances per ECS task (default: 1 — one task per instance)")
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT,
                    help=f"Poll ceiling in seconds (default: {_DEFAULT_TIMEOUT})")
    ap.add_argument("--max-agent-steps", type=int, default=None,
                    help="Override max agent steps per task (default: benchmark TOML, currently 250)")
    ap.add_argument("--variant", default="verified",
                    choices=list(_DATASET_BY_VARIANT),
                    help="SWE-bench variant (default: verified). For Pro use --variant pro; "
                         "the script picks the matching dataset, benchmark TOML "
                         "(swebench_pro for pro, swebench otherwise), and a navigable "
                         "default S3 prefix.")
    ap.add_argument("--benchmark", default=None,
                    help="Override the auto-derived benchmark TOML name "
                         "(e.g. swebench_pro_seal to use the SEAL scaffold instead of "
                         "mini-swe-agent). When set, this value is passed to "
                         "`cross-bench run -b <benchmark>` and also used as the default "
                         "S3 prefix root unless --s3-prefix is also given.")
    ap.add_argument("--s3-prefix", default=None,
                    help="S3 key prefix for artifacts (e.g. 'runs/swebench_pro/'). "
                         "Defaults to 'runs/<benchmark>/' so navigation by benchmark "
                         "works out of the box. Pass an empty string to opt back into "
                         "the flat <bucket>/<run_id>/ layout.")
    ap.add_argument("--started-by", default="",
                    help="Value for ECS startedBy field (shown in console 'Started by' column, "
                         "max 36 chars). Defaults to empty (ECS shows 'AWS console').")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="Added to the 1-based shard index when building the ECS startedBy label. "
                         "E.g. 20 makes shards 0-29 show as dispatch-seal-21..50.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview shards without dispatching anything")
    ap.add_argument("--extra-env", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Extra env var injected into the orchestrator container "
                         "(repeatable). Format: KEY=VALUE. Used for per-dispatch "
                         "flags like CROSSBENCH_SLACK_PER_TASK=1 + SLACK_WEBHOOK_URL=..."
                         " without baking into the task definition.")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="Extra args forwarded verbatim to every cross-bench run call")
    return ap


def _parse_extra_env(values: list[str]) -> list[dict[str, str]]:
    """Convert ``--extra-env KEY=VALUE`` repeats into ECS env entries.

    Raises ValueError on a malformed entry (no '='). Empty list in →
    empty list out, no env injection.
    """
    entries: list[dict[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"--extra-env must be KEY=VALUE, got {raw!r}"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--extra-env has empty key: {raw!r}")
        entries.append({"name": key, "value": value})
    return entries


def main() -> None:
    args = _build_parser().parse_args()

    try:
        extra_env_entries = _parse_extra_env(args.extra_env)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    short_sha = args.sha[:7]
    profiles  = [p.strip() for p in args.profiles.split(",") if p.strip()]
    ts        = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prof_slug = "-".join(p.replace("_", "") for p in profiles[:3])

    # ── Resolve benchmark TOML + S3 prefix from variant ──────────────────────
    # Pro routes through swebench_pro.toml (variant=pro inside); Verified /
    # Lite / Full route through swebench.toml (variant set in benchmark
    # params). The benchmark argument to ``cross-bench run -b ...`` decides
    # which TOML loads.
    benchmark_toml = args.benchmark or ("swebench_pro" if args.variant == "pro" else "swebench")
    # Default S3 prefix is ``runs/<benchmark>/`` — makes
    # ``aws s3 ls s3://<bucket>/runs/swebench_pro/`` list every Pro run.
    # Empty string opts back into the flat legacy layout.
    s3_prefix = (
        args.s3_prefix
        if args.s3_prefix is not None
        else f"runs/{benchmark_toml}/"
    )

    # ── Resolve task IDs (and their HF dataset offsets) ──────────────────────
    # ``task_offsets[i]`` is the zero-based HF position of ``task_ids[i]`` in
    # the variant's test split. None when the ID is not in the dataset
    # (custom/unknown instance) — those shards omit the ``offNNN`` segment.
    if args.explicit_task_ids:
        task_ids = args.explicit_task_ids
        # Look up each explicit ID's HF position so failure triage can
        # reference offsets without a SHA-fragment grep.
        try:
            all_ids = _load_task_ids(args.variant, limit=None, offset=0)
            id_to_off = {tid: i for i, tid in enumerate(all_ids)}
            task_offsets: list[int | None] = [id_to_off.get(tid) for tid in task_ids]
        except SystemExit:
            # _load_task_ids exits when `datasets` is unavailable; degrade
            # gracefully — names just won't carry the offset segment.
            task_offsets = [None] * len(task_ids)
    else:
        task_ids = _load_task_ids(args.variant, args.limit, args.offset)
        task_offsets = list(range(args.offset, args.offset + len(task_ids)))
    print(f"  {len(task_ids)} instance(s) to run")
    print(f"  Variant       : {args.variant}")
    print(f"  Benchmark TOML: {benchmark_toml}.toml")
    print(f"  S3 prefix     : {s3_prefix or '(flat — legacy layout)'}")

    # ── Build shards ──────────────────────────────────────────────────────────
    tps     = args.tasks_per_shard
    batches = [task_ids[i:i + tps] for i in range(0, len(task_ids), tps)]
    offsets_per_batch = [task_offsets[i:i + tps] for i in range(0, len(task_ids), tps)]

    def _shard_offset_endpoints(off_batch: list[int | None]) -> tuple[int | None, int | None]:
        known = [o for o in off_batch if o is not None]
        if not known:
            return (None, None)
        return (min(known), max(known))

    parent_run_id  = args.run_id or f"swebench-{prof_slug}-{ts}"
    shard_run_ids: list[str] = []
    for i, off_batch in enumerate(offsets_per_batch):
        start, end = _shard_offset_endpoints(off_batch)
        shard_run_ids.append(_shard_run_id(parent_run_id, i, start, end))
    shard_to_tids  = {sid: tids for sid, tids in zip(shard_run_ids, batches)}

    print(f"  Parent run ID : {parent_run_id}")
    print(f"  Shards        : {len(batches)} × {tps} task(s)/shard")

    # ── Dry-run preview ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] Would dispatch:")
        for i, (sid, tids) in enumerate(zip(shard_run_ids, batches)):
            print(f"  [{i:3d}] {sid}  tasks={tids}")
        return

    # ── AWS clients ───────────────────────────────────────────────────────────
    session    = boto3.Session(region_name=REGION)
    ecs_client = session.client("ecs")
    ecr_client = session.client("ecr")
    s3_client  = session.client("s3")

    # ── Verify ECR tag ────────────────────────────────────────────────────────
    print(f"\nVerifying ECR tag :{short_sha}…")
    if not _ecr_tag_exists(ecr_client, short_sha):
        sys.exit(
            f"ERROR: tag :{short_sha} not found in ECR.\n"
            f"Trigger a build first:\n"
            f"  aws codebuild start-build --project-name cross-bench-dev-build "
            f"--source-version {args.sha} --region {REGION}"
        )
    print("  ✓")

    # ── Register pinned task-def (once for all shards) ────────────────────────
    if args.task_def:
        task_def = args.task_def
        print(f"\nUsing pre-registered task-def: {task_def}")
    else:
        print("\nRegistering pinned task-def…")
        task_def = _register_pinned_task_def(ecs_client, short_sha)

    # ── Write upfront manifest to S3 (recovery safety net) ───────────────────
    manifest = {
        "parent_run_id":       parent_run_id,
        "shard_run_ids":       shard_run_ids,
        "task_ids_per_shard":  {sid: tids for sid, tids in zip(shard_run_ids, batches)},
        "profiles":            profiles,
        "sha":                 short_sha,
        "dispatched_at":       ts,
        "benchmark":           benchmark_toml,
        "variant":             args.variant,
        "s3_prefix":           s3_prefix,
    }
    manifest_key = f"{s3_prefix}{parent_run_id}/shards.json"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode(),
    )
    print(f"\nManifest → s3://{S3_BUCKET}/{manifest_key}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    print(f"\nDispatching {len(batches)} ECS task(s)…")
    arn_to_shard: dict[str, str] = {}
    extra = [a for a in (args.extra or []) if a != "--"]
    if args.max_agent_steps is not None:
        extra = ["--max-agent-steps", str(args.max_agent_steps)] + extra
    for i, (sid, tids) in enumerate(zip(shard_run_ids, batches)):
        # Use the shard's first-task HF dataset offset as the label so the
        # ECS console "Started by" column shows the position directly
        # (e.g. "dispatch-hf100-gold-300" for the shard starting at offset 300).
        # Falls back to 1-based sequence when no offset is available.
        off_batch = offsets_per_batch[i]
        known_offs = [o for o in off_batch if o is not None]
        if args.started_by and known_offs:
            shard_started_by = f"{args.started_by}-{min(known_offs):03d}"
        elif args.started_by:
            shard_started_by = f"{args.started_by}-{i+1:02d}"
        else:
            shard_started_by = ""
        arn = _dispatch_one_shard(
            ecs_client, task_def, sid, tids, profiles, extra,
            benchmark=benchmark_toml,
            s3_prefix=s3_prefix,
            started_by=shard_started_by,
            extra_env=extra_env_entries,
        )
        arn_to_shard[arn] = sid
        print(f"  [{i+1:3d}/{len(batches)}] {sid}  → {arn.split('/')[-1]}")

    # ── Poll ──────────────────────────────────────────────────────────────────
    print(f"\nPolling (timeout={args.timeout}s, interval={_POLL_INTERVAL}s)…")
    slack_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    on_stopped = None
    if slack_url and args.started_by:
        # Build a per-shard lookup: shard_run_id → label matching startedBy field
        def _slack_label(i: int, sid: str) -> str:
            off_b = offsets_per_batch[i]
            known = [o for o in off_b if o is not None]
            if known:
                return f"{args.started_by}-{min(known):03d}"
            return f"{args.started_by}-{args.index_offset + i + 1:02d}"
        shard_label = {sid: _slack_label(i, sid) for i, sid in enumerate(shard_run_ids)}
        def on_stopped(sid, code,
                       _url=slack_url, _labels=shard_label, _pfx=s3_prefix):
            _post_shard_slack(_url, _labels.get(sid, sid), sid, code, _pfx)

    poll_results = _poll_until_stopped(ecs_client, arn_to_shard, float(args.timeout),
                                       on_stopped=on_stopped)

    # ── Merge ─────────────────────────────────────────────────────────────────
    ok_shards = [
        arn_to_shard[arn]
        for arn, code in poll_results.items()
        if code == "0"
    ]
    if ok_shards:
        print(f"\nMerging {len(ok_shards)} shard(s) → {parent_run_id}/summary.json…")
        _merge_shards(
            parent_run_id, shard_run_ids,
            benchmark=benchmark_toml, s3_prefix=s3_prefix,
        )
    else:
        print("\nNo shards succeeded — skipping merge.")

    # ── Master log ────────────────────────────────────────────────────────────
    _print_master_log(
        parent_run_id, arn_to_shard, shard_to_tids, poll_results,
        s3_prefix=s3_prefix,
    )


if __name__ == "__main__":
    main()
