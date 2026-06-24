"""Opt-in per-task Slack notifications.

The dispatcher (``scripts/dispatch_swebench.py::_post_shard_slack``) already
emits a Slack message when each ECS shard STOPS. With ``--tasks-per-shard 1``
that's effectively one message per task; with ``--tasks-per-shard N>1`` it
collapses to one shard summary at the end. This module covers the missing
case: a live ping the moment each task finishes, regardless of shard packing.

Two env vars gate the behaviour:

- ``CROSSBENCH_SLACK_PER_TASK=1`` — opt-in. Off by default so long conversational
  benchmarks (LoCoMo with 200+ questions per shard) don't spam.
- ``SLACK_WEBHOOK_URL`` — destination. Loaded from ``.env`` by the CLI.

Optional context env vars (auto-injected by ``smoke_pro.sh``; absent is fine):

- ``CROSSBENCH_IMAGE_SHA``   — git short sha of the orchestrator image
- ``CROSSBENCH_TASK_SET``    — "1" / "2" / "3" / "all" / "hf"
- ``CROSSBENCH_CELL_NAME``   — cell label from ``smoke_pro.sh --cell NAME:tail``
- ``CROSSBENCH_LOG_GROUP``   — CloudWatch log group (default
  ``cross-bench-dev-logs-81b05c7``)

Best-effort: any failure (network, malformed payload, Slack outage) is logged
at WARNING and swallowed so a transient Slack issue cannot break a benchmark
run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Per-cell icon palette for the Slack header block. Hash of the cell
# name picks a stable square color so the operator can scan many
# concurrent cell pings at a glance — same cell always gets the same
# icon across all task messages of that run.
_CELL_EMOJIS: tuple[str, ...] = (
    "large_blue_square",
    "large_purple_square",
    "large_orange_square",
    "large_yellow_square",
    "large_green_square",
    "large_brown_square",
    "red_square",
    "black_large_square",
)


def _cell_emoji(cell: str) -> str:
    """Stable colored-square emoji shortcode for a cell name."""
    if not cell or cell == "—":
        return "test_tube"
    h = hashlib.md5(cell.encode()).digest()[0]
    return _CELL_EMOJIS[h % len(_CELL_EMOJIS)]


# Bar palette (severity-graded traffic light). Each cell of the
# per-shard progress bar renders one of these emojis. Keep this map
# small and stable — callers pass outcome strings; we look up here.
_OUTCOME_EMOJI: dict[str, str] = {
    "resolved": "large_green_square",
    "failed": "large_yellow_square",
    "errored": "large_red_square",
    "timeout": "large_red_square",
    "pending": "white_large_square",
}


def _outcome_emoji(outcome: str) -> str:
    return _OUTCOME_EMOJI.get(outcome, "white_large_square")


def _derive_outcome(result: Any) -> str:
    """Map a ``TaskResult`` to one of the bar-palette outcome strings."""
    if result is None:
        return "pending"
    ev = getattr(result, "evaluation", None) or {}
    status = getattr(result, "status", None)
    status_value = getattr(status, "value", str(status) if status else "")
    err = getattr(result, "error", "") or ""
    if err or status_value == "error":
        return "errored"
    if status_value == "timeout":
        return "timeout"
    if bool(ev.get("resolved")) or status_value == "pass":
        return "resolved"
    return "failed"


def _progress_bar(index: int, total: int, outcomes: list[str]) -> str:
    """Per-task outcome bar.

    ``outcomes[i]`` is the recorded outcome for the i-th task (one of
    ``resolved`` / ``failed`` / ``errored`` / ``timeout``); positions
    beyond the supplied list render as ``pending`` (⬜).
    """
    cells: list[str] = []
    cap = total if total <= 20 else 20
    # If total exceeds 20, sample proportionally so the bar stays
    # 20 chars wide while still reflecting per-task outcomes.
    scale = total / cap if cap else 1
    for i in range(cap):
        src_idx = int(i * scale) if total > cap else i
        if src_idx < len(outcomes):
            cells.append(f":{_outcome_emoji(outcomes[src_idx])}:")
        else:
            cells.append(":white_large_square:")
    return "".join(cells)


def _fmt_dur(seconds: Any) -> str:
    """Compact duration: ``612s`` → ``10m 12s``; ``84.0`` → ``1m 24s``."""
    if not seconds:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


def _fmt_compact_tokens(n: Any) -> str:
    """``184320`` → ``184k``; ``8900`` → ``8.9k``; ``312`` → ``312``."""
    if not n:
        return "0"
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return "—"
    if n_int < 1000:
        return str(n_int)
    if n_int < 10_000:
        return f"{n_int / 1000:.1f}k"
    if n_int < 1_000_000:
        return f"{n_int // 1000}k"
    return f"{n_int / 1_000_000:.1f}M"

# ECS-side metadata is fixed for the lifetime of the container, so resolve
# it once on first call and reuse. ``_ecs_meta_cache = None`` means
# "not yet attempted"; once attempted (even if the lookup failed) we store
# a dict so we don't keep retrying a 169.254.170.2 hop that isn't there.
_ecs_meta_cache: dict[str, str] | None = None


def _per_task_enabled() -> bool:
    return os.environ.get("CROSSBENCH_SLACK_PER_TASK", "0") == "1"


def _webhook_url() -> str:
    return os.environ.get("SLACK_WEBHOOK_URL", "")


def _resolve_ecs_metadata() -> dict[str, str]:
    """Best-effort lookup of ECS task ID + region + log group / stream.

    Returns a dict (possibly empty when off-AWS). Cached after first call.
    Never raises — network errors / missing env vars / unexpected response
    shapes all degrade to "skip this segment of the Slack message".
    """
    global _ecs_meta_cache
    if _ecs_meta_cache is not None:
        return _ecs_meta_cache

    meta: dict[str, str] = {}
    metadata_uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4", "").strip()
    if metadata_uri:
        try:
            req = urllib.request.Request(metadata_uri + "/task", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                task_meta = json.loads(resp.read().decode())
            task_arn = task_meta.get("TaskARN", "")
            if task_arn:
                meta["task_arn"] = task_arn
                meta["task_id"] = task_arn.rsplit("/", 1)[-1]
            cluster_arn = task_meta.get("Cluster", "")
            if cluster_arn:
                meta["cluster"] = cluster_arn.rsplit("/", 1)[-1]
            started_by = task_meta.get("StartedBy", "")
            if started_by:
                meta["started_by"] = started_by
        except Exception as exc:
            logger.debug("ecs_metadata_lookup_failed: %s", exc)

    meta.setdefault(
        "region",
        os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    )
    meta.setdefault(
        "log_group",
        os.environ.get("CROSSBENCH_LOG_GROUP", "cross-bench-dev-logs-81b05c7"),
    )
    meta.setdefault(
        "log_stream_prefix",
        os.environ.get("CROSSBENCH_LOG_STREAM_PREFIX", "orchestrator"),
    )

    _ecs_meta_cache = meta
    return meta


def _cloudwatch_url(meta: dict[str, str]) -> str:
    """CloudWatch console URL for this task's log stream.

    Empty string when we can't construct it (off-AWS path, missing task id).
    """
    region = meta.get("region")
    log_group = meta.get("log_group")
    task_id = meta.get("task_id")
    prefix = meta.get("log_stream_prefix", "orchestrator")
    if not (region and log_group and task_id):
        return ""
    log_stream = f"{prefix}/orchestrator/{task_id}"
    lg_enc = urllib.parse.quote(log_group, safe="").replace("%", "$25")
    ls_enc = urllib.parse.quote(log_stream, safe="").replace("%", "$25")
    return (
        f"https://console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{lg_enc}/log-events/{ls_enc}"
    )


def _ecs_console_url(meta: dict[str, str]) -> str:
    """ECS console URL for this task. Empty when off-AWS / missing pieces."""
    region = meta.get("region")
    cluster = meta.get("cluster")
    task_id = meta.get("task_id")
    if not (region and cluster and task_id):
        return ""
    return (
        f"https://console.aws.amazon.com/ecs/v2/clusters/{cluster}"
        f"/tasks/{task_id}?region={region}"
    )


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_duration(seconds: Any) -> str:
    if not seconds:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def _fmt_tokens(n: Any) -> str:
    if n is None:
        return "—"
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return "—"
    if n_int >= 1_000_000:
        return f"{n_int / 1_000_000:.1f}M"
    if n_int >= 1_000:
        return f"{n_int / 1_000:.1f}k"
    return str(n_int)


def _extract_llm_model(config: Any) -> str:
    """SEAL scaffold uses ``benchmark_params.seal_model_name`` (the LLM the
    sweagent subprocess actually invokes via LiteLLM). All other scaffolds
    read the bench LLM from ``system.llm.model``. Prefer the SEAL field
    when populated so the cell reads as e.g. ``bedrock/us.anthropic.
    claude-haiku-4-5-…`` for SEAL Pro runs."""
    if config is None:
        return ""
    seal_model = ""
    bench_params = getattr(config, "benchmark_params", None) or {}
    if isinstance(bench_params, dict):
        seal_model = str(bench_params.get("seal_model_name") or "")
    if seal_model:
        return seal_model
    sys_cfg = getattr(config, "system", None)
    llm = getattr(sys_cfg, "llm", None) if sys_cfg else None
    return getattr(llm, "model", "") if llm else ""


def _extract_embedding_model(config: Any) -> str:
    if config is None:
        return ""
    sys_cfg = getattr(config, "system", None)
    emb = getattr(sys_cfg, "embedding", None) if sys_cfg else None
    return getattr(emb, "model", "") if emb else ""


def _coerce_run_id(config_or_run_id: Any) -> tuple[str, Any]:
    """Accept either a RunConfig or a plain run_id string (back-compat
    for older callers/tests). Returns ``(run_id, config_or_None)``."""
    if isinstance(config_or_run_id, str):
        return config_or_run_id, None
    cfg = config_or_run_id
    run_id = getattr(cfg, "run_id", "") or ""
    return run_id, cfg


def _post_card(payload: dict, url: str) -> None:
    """POST the rendered Slack payload; never raises."""
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        logger.warning("slack_per_task_notify_failed: %s", exc)


def _render_card(
    *,
    config: Any,
    run_id: str,
    task: Any | None,
    result: Any | None,
    index: int = 1,
    total: int = 1,
    outcomes: list[str] | None = None,
) -> dict:
    """Build the Slack ``{"text": ..., "blocks": [...]}`` payload for a
    finished task.

    Layout:
      header     ✅ + raw cell name
      section    progress bar  ·  ``i/total`` code chip  ·  status emoji + verb
      section    📈 F2P · ⏱️ runtime · 👣 steps · 💬 LLM in/out
      context    (errors only) red-light + truncated error text
      divider

    Color-graded bar (``resolved=🟩 / failed=🟨 / errored/timeout=🟥``)
    encodes the actual outcome.

    Args:
        index: 1-based position of this task within the shard.
        total: total task count in the shard.
        outcomes: list of outcome strings for tasks completed so far
            (one of ``resolved`` / ``failed`` / ``errored`` / ``timeout``).
            ``outcomes[i-1]`` is this task; trailing positions render as
            ``pending`` (⬜).
    """
    outcomes = list(outcomes or [])
    outcome = _derive_outcome(result)
    verb = outcome  # "resolved" / "failed" / "errored" / "timeout"
    status_emoji = _outcome_emoji(outcome)
    header_glyph = "white_check_mark"

    ev = getattr(result, "evaluation", {}) or {} if result else {}
    err = getattr(result, "error", "") or "" if result else ""
    cell = os.environ.get("CROSSBENCH_CELL_NAME", "—")

    # Metrics (read only in done mode but defined unconditionally so the
    # fallback string below can reference them in either branch).
    f2p = ev.get("avg_f2p_rate") or ev.get("f2p_rate") if ev else None
    steps = ev.get("steps_taken") if ev else None
    wall = ev.get("wallclock_seconds") if ev else None
    tok_in = ev.get("total_tokens_in") or 0
    tok_out = ev.get("total_tokens_out") or 0

    ecs_meta = _resolve_ecs_metadata()
    cw_url = _cloudwatch_url(ecs_meta)

    bar = _progress_bar(index, total, outcomes)

    blocks: list[dict] = []
    if cell and cell != "—":
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":{header_glyph}: {cell}",
                    "emoji": True,
                },
            }
        )

    verb_upper = verb.upper()
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{bar}    `{index}/{total}`    "
                    f":{status_emoji}: *{verb_upper}*"
                ),
            },
        }
    )

    # Emoji-led metrics row — icons carry the meaning, only the
    # headline F2P number is bold so the eye lands there first.
    parts: list[str] = [
        f":chart_with_upwards_trend: F2P *{_fmt_pct(f2p)}*",
        f":stopwatch: {_fmt_dur(wall)}",
        f":footprints: {int(steps) if steps else 0}",
    ]
    if tok_in or tok_out:
        parts.append(
            f":speech_balloon: "
            f"{_fmt_compact_tokens(tok_in)}/{_fmt_compact_tokens(tok_out)}"
        )
    # Logs link only on failure/error outcomes — pass-rate signal alone
    # is enough for resolved tasks.
    if outcome != "resolved" and cw_url:
        parts.append(f"<{cw_url}|:mag: logs>")
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "  ·  ".join(parts)},
        }
    )
    if err:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f":rotating_light: `{err[:200]}`",
                    }
                ],
            }
        )

    # Trailing divider so consecutive cell pings don't visually run together.
    blocks.append({"type": "divider"})

    fallback = f":{status_emoji}: {cell} {index}/{total} — {verb}"
    return {"text": fallback, "blocks": blocks}


def notify_task_done(
    result: Any,
    config_or_run_id: Any,
    task: Any | None = None,
    *,
    index: int = 1,
    total: int = 1,
    outcomes: list[str] | None = None,
) -> None:
    """Post a per-task ``done`` Slack message (best-effort, never raises).

    Args:
        result: ``TaskResult`` from the engine.
        config_or_run_id: ``RunConfig`` (preferred — gives LLM/embedding
            context) or a bare run-id string (back-compat).
        task: Optional ``Task``.
        index: 1-based position of this task within the shard.
        total: total task count in the shard.
        outcomes: list of outcomes for tasks 1..index (current task's
            outcome is at ``outcomes[index-1]``).
    """
    if not _per_task_enabled():
        return
    url = _webhook_url()
    if not url:
        return
    run_id, config = _coerce_run_id(config_or_run_id)
    payload = _render_card(
        config=config, run_id=run_id, task=task, result=result,
        index=index, total=total, outcomes=outcomes,
    )
    _post_card(payload, url)
