"""Shard-based fan-out for conversational benchmarks.

Runs one ECS task per conversation so a LoCoMo run finishes in roughly
the time of its slowest conversation instead of the sum of all ten.

Pieces here:

* :func:`merge_shards` — reads per-shard task results from the
  :class:`~cross_bench.artifact_store.ArtifactStore` and writes a
  unified parent summary plus a final ``shards.json`` manifest with
  per-shard task counts.
* :func:`dispatch_shards` — calls ``ecs.run_task`` once per conversation
  with an appropriate ``--conversation-id`` / ``--run-id`` command
  override, then polls until every shard has stopped.
* :func:`write_shards_manifest_upfront` — writes a lightweight
  ``shards.json`` manifest BEFORE dispatch begins, so the parent
  prefix has a record of which shard run IDs were intended even if
  the orchestrator dies mid-run. Recovery is then ``cross-bench
  merge-shards --parent <id> --discover --benchmark <name>`` — no
  CloudWatch log mining required. See the 2026-05-11 Mastra incident.
* :func:`read_shards_manifest` — loads ``shards.json`` from a parent
  prefix; used by the ``--discover`` merge path.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from cross_bench.artifact_store import ArtifactStore
from cross_bench.execution.engine import RunEngine
from cross_bench.models import (
    RunConfig,
    RunResult,
    RunStatus,
    StepRecord,
    TaskResult,
    TaskStatus,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

def _dict_to_task_result(data: dict[str, Any]) -> TaskResult:
    """Reconstruct a :class:`TaskResult` from its ``asdict`` form."""
    status = data.get("status", "error")
    if isinstance(status, str):
        status = TaskStatus(status)
    steps_raw = data.get("steps") or []
    steps = [StepRecord(**s) for s in steps_raw]
    return TaskResult(
        task_id=data["task_id"],
        run_id=data.get("run_id", ""),
        status=status,
        steps=steps,
        final_output=data.get("final_output", ""),
        evaluation=dict(data.get("evaluation") or {}),
        error=data.get("error"),
    )


@dataclass
class MergeReport:
    parent_run_id: str
    shard_run_ids: list[str]
    total_tasks: int
    metrics: dict[str, float]


async def merge_shards(
    parent_run_id: str,
    shard_run_ids: list[str],
    store: ArtifactStore,
    benchmark: str = "",
) -> MergeReport:
    """Aggregate per-shard task results into ``{parent_run_id}/summary.json``.

    Task JSONs remain under their shard dirs for provenance; only the
    rolled-up summary is written to the parent.
    """
    all_results: list[TaskResult] = []
    per_shard_counts: dict[str, int] = {}
    for shard_id in shard_run_ids:
        shard_tasks = await store.list_task_results(shard_id)
        per_shard_counts[shard_id] = len(shard_tasks)
        all_results.extend(_dict_to_task_result(d) for d in shard_tasks)

    # Resolve the adapter so ``_aggregate`` can pre-seed
    # ``metrics_schema()`` keys with ``None`` — same stable column set
    # the engine writes for unsharded runs (see workflow.md "Metric
    # schema is benchmark-level, not memory-system-level").
    adapter = None
    if benchmark:
        try:
            from cross_bench.benchmarks import get_adapter
            adapter = get_adapter(benchmark)
        except Exception as exc:
            logger.warning("merge.adapter_unavailable", benchmark=benchmark, error=str(exc))

    metrics = RunEngine._aggregate(all_results, adapter=adapter)

    merged = RunResult(
        run_id=parent_run_id,
        config=RunConfig(benchmark=benchmark, system=_placeholder_system()),
        task_results=all_results,
        aggregate_metrics=metrics,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status=RunStatus.COMPLETED,
    )

    # Write the merged summary. We reuse the store so local + S3 paths
    # stay consistent with how a normal run would have written it.
    await store.store_run_summary(merged)

    # Also persist a small manifest documenting which shards were merged.
    # Lives next to summary.json; makes retries and debugging easier.
    manifest_path = store._ensure_dir(parent_run_id) / "shards.json"
    store._write_json(
        manifest_path,
        {
            "parent_run_id": parent_run_id,
            "shard_run_ids": shard_run_ids,
            "per_shard_task_counts": per_shard_counts,
            "total_tasks": len(all_results),
        },
    )

    logger.info(
        "shards.merged",
        parent=parent_run_id,
        shards=len(shard_run_ids),
        tasks=len(all_results),
    )

    return MergeReport(
        parent_run_id=parent_run_id,
        shard_run_ids=shard_run_ids,
        total_tasks=len(all_results),
        metrics=metrics,
    )


def _placeholder_system() -> Any:
    """Minimal SystemConfig so RunResult serializes cleanly.

    The merger doesn't know which system each shard used (shards may
    even differ) so we write a placeholder; the per-shard ``config.json``
    files remain the source of truth for system config.
    """
    from cross_bench.models import LLMConfig, PromptStrategyConfig, SystemConfig

    return SystemConfig(
        llm=LLMConfig(provider="merged", model="merged"),
        prompt_strategy=PromptStrategyConfig(type="zero_shot"),
    )


# ---------------------------------------------------------------------------
# Upfront manifest — lets recovery find the shards when the orch dies
# ---------------------------------------------------------------------------

def write_shards_manifest_upfront(
    parent_run_id: str,
    conversation_ids: list[str],
    store: ArtifactStore,
    benchmark: str = "",
    profiles: list[str] | None = None,
) -> list[str]:
    """Write ``{parent_run_id}/shards.json`` BEFORE dispatch starts.

    The 2026-05-11 Mastra run lost its orchestrator at minute ~420
    with 5 of 10 shards still running. The orch normally writes
    ``shards.json`` at the END of the run (inside :func:`merge_shards`),
    so when the orch dies before that point the manifest never lands —
    recovery requires hunting shard UUIDs in CloudWatch logs.

    Writing the manifest upfront means recovery is one command:
    ``cross-bench merge-shards --parent <id> --discover --benchmark
    <name>``. Shard run IDs are deterministic UUID5(parent + conv_id)
    so the upfront manifest matches what dispatch_shards will compute.

    Returns the list of shard run IDs (same order as
    ``conversation_ids``) for the caller's convenience.

    The manifest is later overwritten by :func:`merge_shards` with the
    richer end-of-run shape (adds ``per_shard_task_counts`` /
    ``total_tasks``). Both shapes are forward-compatible — see
    :func:`read_shards_manifest`.
    """
    shard_run_ids = [_shard_run_id(parent_run_id, cid) for cid in conversation_ids]
    manifest_path = store._ensure_dir(parent_run_id) / "shards.json"
    payload: dict[str, Any] = {
        "parent_run_id": parent_run_id,
        "shard_run_ids": shard_run_ids,
        "conversation_ids": conversation_ids,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }
    if benchmark:
        payload["benchmark"] = benchmark
    if profiles:
        payload["profiles"] = list(profiles)
    store._write_json(manifest_path, payload)
    logger.info(
        "shards.manifest_upfront",
        parent=parent_run_id,
        shards=len(shard_run_ids),
    )
    return shard_run_ids


async def read_shards_manifest(
    parent_run_id: str,
    store: ArtifactStore,
) -> dict[str, Any]:
    """Load ``{parent_run_id}/shards.json``.

    Used by the ``--discover`` path of ``cross-bench merge-shards``
    to auto-populate shard IDs from the manifest without requiring
    the user to track each shard's UUID. Accepts both the upfront-
    write shape (no ``per_shard_task_counts``) and the post-merge
    shape (with task counts).

    Raises :class:`FileNotFoundError` if no manifest exists at the
    parent prefix.
    """
    import json as _json

    text = await store.retrieve(f"{parent_run_id}/shards.json")
    data = _json.loads(text)
    if "shard_run_ids" not in data:
        raise ValueError(
            f"shards.json at {parent_run_id} is malformed "
            f"(missing 'shard_run_ids'): {list(data.keys())}"
        )
    return data


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

@dataclass
class DispatchConfig:
    cluster: str
    task_definition: str  # family or full ARN
    benchmark: str
    profiles: list[str]
    parent_run_id: str
    conversation_ids: list[str]
    container_name: str = "orchestrator"
    launch_type: str = "EC2"  # EC2 for the current task def; FARGATE if ported
    # When set, shards are launched with a capacityProviderStrategy
    # instead of a plain launchType. That's what triggers the ECS
    # capacity provider's managed scaling to grow the ASG — without it,
    # pending shards just sit there waiting for pre-warmed capacity.
    capacity_provider: str | None = None
    # Fargate-only (ignored for EC2):
    subnets: list[str] | None = None
    security_groups: list[str] | None = None
    assign_public_ip: bool = False
    # Polling
    poll_interval_seconds: float = 30.0
    timeout_seconds: float = 3 * 60 * 60  # 3h ceiling


@dataclass
class ShardHandle:
    conversation_id: str
    shard_run_id: str
    task_arn: str


@dataclass
class DispatchReport:
    parent_run_id: str
    shards: list[ShardHandle]
    failed_shards: list[ShardHandle]


_SHARD_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _shard_run_id(parent: str, conv_id: str) -> str:
    """Deterministic UUID5 derived from the parent + conversation.

    The ``runs.run_id`` column is typed as UUID, so a raw
    ``{parent}-{conv_id}`` string fails the insert. UUID5 keeps the id
    stable across retries (same parent + conv ⇒ same uuid) while
    satisfying the column type. Parent / conv_id are still logged next
    to the uuid so traceability isn't lost.
    """
    return str(uuid.uuid5(_SHARD_NAMESPACE, f"{parent}:{conv_id}"))


def _build_shard_command(
    benchmark: str,
    profiles: list[str],
    conversation_id: str,
    shard_run_id: str,
) -> list[str]:
    cmd: list[str] = ["run", "-b", benchmark]
    for p in profiles:
        cmd.extend(["-p", p])
    cmd.extend(["--conversation-id", conversation_id, "--run-id", shard_run_id])
    return cmd


def dispatch_shards(cfg: DispatchConfig) -> DispatchReport:
    """Launch one ECS task per conversation and wait for all to stop.

    Returns after every shard reaches ``lastStatus=STOPPED``. The caller
    is responsible for merging with :func:`merge_shards` — keeping the
    two steps separate makes retries cheap when a single shard fails.
    """
    import boto3

    ecs = boto3.client("ecs")

    shards: list[ShardHandle] = []
    for conv_id in cfg.conversation_ids:
        shard_id = _shard_run_id(cfg.parent_run_id, conv_id)
        cmd = _build_shard_command(
            benchmark=cfg.benchmark,
            profiles=cfg.profiles,
            conversation_id=conv_id,
            shard_run_id=shard_id,
        )
        overrides = {
            "containerOverrides": [
                {
                    "name": cfg.container_name,
                    "command": cmd,
                    "environment": [
                        {"name": "BENCH_RUN_ID", "value": shard_id},
                    ],
                }
            ],
        }

        run_task_kwargs: dict[str, Any] = {
            "cluster": cfg.cluster,
            "taskDefinition": cfg.task_definition,
            "count": 1,
            "overrides": overrides,
            "startedBy": f"dispatch-{cfg.parent_run_id}"[:36],
        }
        if cfg.capacity_provider:
            run_task_kwargs["capacityProviderStrategy"] = [{
                "capacityProvider": cfg.capacity_provider,
                "weight": 1,
                "base": 0,
            }]
        else:
            run_task_kwargs["launchType"] = cfg.launch_type
        if cfg.launch_type == "FARGATE":
            if not cfg.subnets:
                raise ValueError("FARGATE launch type requires DispatchConfig.subnets")
            run_task_kwargs["networkConfiguration"] = {
                "awsvpcConfiguration": {
                    "subnets": cfg.subnets,
                    "securityGroups": cfg.security_groups or [],
                    "assignPublicIp": "ENABLED" if cfg.assign_public_ip else "DISABLED",
                }
            }

        resp = ecs.run_task(**run_task_kwargs)
        failures = resp.get("failures") or []
        if failures:
            raise RuntimeError(
                f"ecs.run_task rejected shard {conv_id}: {failures}"
            )
        task_arn = resp["tasks"][0]["taskArn"]
        shards.append(ShardHandle(
            conversation_id=conv_id,
            shard_run_id=shard_id,
            task_arn=task_arn,
        ))
        logger.info(
            "shard.dispatched",
            parent=cfg.parent_run_id,
            conv_id=conv_id,
            shard_run_id=shard_id,
            task_arn=task_arn,
        )

    # Poll until every task is STOPPED.
    pending = {s.task_arn: s for s in shards}
    failed: list[ShardHandle] = []
    deadline = time.monotonic() + cfg.timeout_seconds

    while pending and time.monotonic() < deadline:
        time.sleep(cfg.poll_interval_seconds)
        # describe_tasks caps at 100 ARNs per call; 10 shards fit fine.
        resp = ecs.describe_tasks(
            cluster=cfg.cluster,
            tasks=list(pending.keys()),
        )
        for t in resp.get("tasks", []):
            if t.get("lastStatus") != "STOPPED":
                continue
            arn = t["taskArn"]
            shard = pending.pop(arn, None)
            if shard is None:
                continue
            exit_codes = [
                c.get("exitCode") for c in t.get("containers", [])
                if c.get("name") == cfg.container_name
            ]
            stopped_reason = t.get("stoppedReason") or ""
            ok = bool(exit_codes) and exit_codes[0] == 0
            logger.info(
                "shard.stopped",
                parent=cfg.parent_run_id,
                shard_run_id=shard.shard_run_id,
                exit_code=exit_codes[0] if exit_codes else None,
                reason=stopped_reason,
                ok=ok,
            )
            if not ok:
                failed.append(shard)

    if pending:
        # Timed out — treat remaining as failures.
        for shard in pending.values():
            logger.warning(
                "shard.timeout",
                parent=cfg.parent_run_id,
                shard_run_id=shard.shard_run_id,
            )
            failed.append(shard)

    return DispatchReport(
        parent_run_id=cfg.parent_run_id,
        shards=shards,
        failed_shards=failed,
    )


async def discover_conversation_ids(benchmark: str) -> list[str]:
    """List all conversation IDs known to ``benchmark``'s adapter.

    Lets the dispatcher auto-shard without the caller hand-writing the
    list. Loads the benchmark dataset via the adapter's own resolution
    (download + cache), so the first call may pay the one-time download
    cost.
    """
    from cross_bench.benchmarks import get_adapter
    from cross_bench.benchmarks.base import ConversationalBenchmarkAdapter

    adapter = get_adapter(benchmark)
    if not isinstance(adapter, ConversationalBenchmarkAdapter):
        raise ValueError(
            f"Benchmark {benchmark!r} is not conversational; per-conversation "
            "sharding only applies to conversational benchmarks."
        )
    # load_conversations is synchronous today; wrap in to_thread so a
    # long download doesn't block the event loop.
    convs = await asyncio.to_thread(adapter.load_conversations, adapter.get_default_params())
    return [str(c["conversation_id"]) for c in convs]
