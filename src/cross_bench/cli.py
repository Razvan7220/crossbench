"""CLI entry point for the cross-bench evaluation tool."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="cross-bench",
    help="Modular evaluation platform for coding benchmarks.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@app.command()
def run(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Path to run config TOML")] = None,
    profile: Annotated[Optional[list[str]], typer.Option("--profile", "-p", help="Profiles to stack (repeatable, e.g. -p online -p claude -p evermemos_api)")] = None,
    benchmark: Annotated[Optional[str], typer.Option("--benchmark", "-b", envvar="BENCH_BENCHMARK")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="LLM model (e.g. openai:gpt-4o, generic_openai:qwen3:7b)")] = None,
    mode: Annotated[Optional[str], typer.Option("--mode", help="Execution mode: single_shot | agentic. Defaults to whatever the benchmark TOML declares (e.g. swebench/terminalbench → agentic).")] = None,
    memory: Annotated[Optional[str], typer.Option("--memory", help="Memory provider: mem0 | zep | letta | memos | evermemos | hindsight | mastra | rag | none")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max tasks to run")] = None,
    conversation_id: Annotated[Optional[list[str]], typer.Option("--conversation-id", help="Restrict to these conversation IDs (repeatable). Used by fan-out sharding.")] = None,
    task_id: Annotated[Optional[list[str]], typer.Option("--task-id", help="Restrict to these task IDs (repeatable). SWE-bench: maps to instance_id.")] = None,
    parallel: Annotated[int, typer.Option("--parallel", help="Run this many tasks simultaneously. Each task runs in its own subprocess for full memory isolation.")] = 1,
    max_agent_steps: Annotated[Optional[int], typer.Option("--max-agent-steps", help="Override max agent steps (default comes from benchmark TOML)")] = None,
    run_id: Annotated[Optional[str], typer.Option("--run-id", envvar="BENCH_RUN_ID", help="Override run ID (auto-generated if omitted)")] = None,
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
    db_url: Annotated[str, typer.Option("--db-url", envvar="BENCH_DATABASE_URL")] = "none",
) -> None:
    """Execute a benchmark evaluation run.

    \b
    Profiles are stackable — use multiple -p flags to compose a config:
        # Fully local — Qwen3:7b via Ollama, zero cloud calls
        cross-bench run -p local -b humaneval -n 10

        # Cloud — GPT-4o via OpenAI API
        cross-bench run -p online -b humaneval -n 10

        # Cloud — Claude (stack base + LLM mixin)
        cross-bench run -p online -p claude -b humaneval -n 10

        # Cloud — Claude + EverMemOS memory
        cross-bench run -p online -p claude -p evermemos_embedded -b locomo_original

        # Disable memory on any profile
        cross-bench run -p online -p no_memory -b humaneval

        # Store results in PostgreSQL
        cross-bench run -p online -b humaneval --db-url postgresql+asyncpg://bench:bench_dev@localhost:5432/cross_bench
    """
    # Parallel fan-out — delegate before building a full run config.
    if parallel > 1:
        if not task_id:
            raise typer.BadParameter("--parallel requires at least one --task-id", param_hint="--task-id")
        from cross_bench.parallel_runner import run_parallel
        asyncio.run(run_parallel(
            task_ids=list(task_id),
            parallel=parallel,
            profiles=list(profile or []),
            benchmark=benchmark,
            artifacts_dir=artifacts_dir,
            db_url=db_url,
            extra_cli_args=[],
        ))
        return

    from cross_bench.config import load_run_config

    # Build overrides from CLI flags
    overrides: dict = {}
    if benchmark:
        overrides.setdefault("run", {})["benchmark"] = benchmark
    if mode:
        overrides.setdefault("run", {})["execution_mode"] = mode
    if limit:
        overrides.setdefault("benchmark", {})["limit"] = limit
    if conversation_id:
        overrides.setdefault("benchmark", {})["filter_conversation_ids"] = list(
            conversation_id
        )
    if task_id:
        overrides.setdefault("benchmark", {})["filter_task_ids"] = list(task_id)
    if max_agent_steps is not None:
        overrides.setdefault("run", {})["max_agent_steps"] = max_agent_steps
    if model:
        # Support  provider:model  syntax (e.g. "openai:gpt-4o")
        provider, _, model_name = model.partition(":")
        overrides.setdefault("system", {}).setdefault("llm", {}).update(
            {"provider": provider, "model": model_name or provider}
        )
    if memory is not None:
        if memory.lower() == "none":
            overrides.setdefault("system", {})["memory"] = {"provider": "", "enabled": False}
        else:
            overrides.setdefault("system", {}).setdefault("memory", {}).update(
                {"provider": memory, "enabled": True}
            )

    # Support BENCH_PROFILE env var (comma-separated for multiple profiles)
    profiles = profile
    if not profiles:
        env_profile = os.environ.get("BENCH_PROFILE")
        if env_profile:
            profiles = [p.strip() for p in env_profile.split(",") if p.strip()]

    run_config = load_run_config(
        config, benchmark=benchmark, profiles=profiles, overrides=overrides,
    )

    # Override run_id if provided via CLI or env var
    if run_id:
        run_config.run_id = run_id

    # Print run banner
    console.print()
    console.rule("[bold]Bench Platform — Evaluation Run[/bold]")
    console.print(f"  [bold]Run ID:[/bold]     {run_config.run_id}")
    console.print(f"  [bold]Profile:[/bold]    {' + '.join(profiles) if profiles else '(custom)'}")
    console.print(f"  [bold]Benchmark:[/bold]  {run_config.benchmark}")
    console.print(f"  [bold]Model:[/bold]      {run_config.system.llm.provider}:{run_config.system.llm.model}")
    mem_label = run_config.system.memory.provider if run_config.system.memory and run_config.system.memory.enabled else "none"
    console.print(f"  [bold]Memory:[/bold]     {mem_label}")
    emb_label = run_config.system.embedding.provider + ":" + run_config.system.embedding.model if run_config.system.embedding else "none"
    console.print(f"  [bold]Embedding:[/bold]  {emb_label}")
    console.print(f"  [bold]Mode:[/bold]       {run_config.execution_mode.value}")
    if run_config.benchmark_params.get("limit"):
        console.print(f"  [bold]Task limit:[/bold] {run_config.benchmark_params['limit']}")
    console.rule()
    console.print()

    asyncio.run(_execute_run(run_config, artifacts_dir, db_url))


async def _execute_run(run_config, artifacts_dir: str, db_url: str) -> None:
    from cross_bench.artifact_store import ArtifactStore
    from cross_bench.execution.engine import RunEngine
    from cross_bench.execution.sandbox import SandboxManager

    import os

    backend = os.environ.get("BENCH_ARTIFACTS_BACKEND", "local")
    bucket = os.environ.get("BENCH_ARTIFACTS_BUCKET", "")
    # S3 key prefix for navigable layout — e.g. "runs/swebench_pro/" so
    # `aws s3 ls s3://<bucket>/runs/swebench_pro/` lists every Pro run.
    # Empty preserves the flat <bucket>/<run_id>/ layout for back-compat.
    s3_prefix = os.environ.get("BENCH_ARTIFACTS_S3_PREFIX", "")
    artifact_store = ArtifactStore(
        backend=backend,
        bucket=bucket,
        local_dir=artifacts_dir,
        s3_prefix=s3_prefix,
    )

    if db_url and db_url.lower() != "none":
        from cross_bench.results.repository import ResultRepository
        result_repo = ResultRepository(db_url)
    else:
        from cross_bench.results.repository import NullResultRepository
        result_repo = NullResultRepository()
    await result_repo.init_db()
    sandbox_mgr = SandboxManager()

    engine = RunEngine(artifact_store, result_repo, sandbox_mgr)
    result = await engine.execute_run(run_config)

    # Print summary
    console.print()
    table = Table(title=f"Run Results: {result.run_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in sorted(result.aggregate_metrics.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# list-benchmarks
# ---------------------------------------------------------------------------
@app.command()
def list_benchmarks() -> None:
    """List all available benchmarks."""
    from cross_bench.benchmarks import list_benchmarks as _list

    table = Table(title="Available Benchmarks")
    table.add_column("Name")
    table.add_column("Granularity")
    for name in _list():
        from cross_bench.benchmarks import get_adapter
        adapter = get_adapter(name)
        table.add_row(name, adapter.granularity.value)
    console.print(table)


# ---------------------------------------------------------------------------
# list-profiles
# ---------------------------------------------------------------------------
@app.command()
def list_profiles() -> None:
    """List available configuration profiles."""
    from cross_bench.config import list_profiles as _list

    table = Table(title="Available Profiles")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    descriptions = {
        # Base profiles
        "local": "Base — Qwen3:7b via Ollama, Ollama embeddings",
        "online": "Base — GPT-4o (OpenAI), OpenAI embeddings",
        # LLM mixins
        "claude": "Mixin — switches LLM to Claude (Anthropic)",
        "o3_mini": "Mixin — switches LLM to o3-mini (OpenAI reasoning)",
        "gpt4o_mini": "Mixin — switches LLM to GPT-4o-mini",
        # Memory mixins
        "no_memory": "Mixin — disables memory",
        "mem0v1_api": "Mixin — Mem0 v1 cloud (MemoryClient, pre-v2 API)",
        "mem0v1_embedded": "Mixin — Mem0 v1 embedded (local, pre-v2 API)",
        "mem0v2_api": "Mixin — Mem0 v2 cloud (MemoryClient, v3 endpoints)",
        "mem0v2_embedded": "Mixin — Mem0 v2 embedded (local, v2.0.0 additive pipeline)",
        "evermemos_api": "Mixin — EverMemOS API memory (conversational)",
        "evermemos_embedded": "Mixin — EverMemOS embedded (conversational)",
        "graphiti": "Mixin — Graphiti knowledge-graph memory (conversational)",
        "hindsight_embedded": "Mixin — Hindsight memory (conversational)",
        "mastra_embedded": "Mixin — Mastra multi-layer memory (conversational)",
    }
    for name in _list():
        table.add_row(name, descriptions.get(name, ""))
    console.print(table)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@app.command()
def results(
    run_id: Annotated[Optional[str], typer.Option("--run-id")] = None,
    benchmark: Annotated[Optional[str], typer.Option("--benchmark", "-b")] = None,
    compare: Annotated[Optional[list[str]], typer.Option("--compare")] = None,
    format: Annotated[str, typer.Option("--format", "-f")] = "table",
    db_url: Annotated[str, typer.Option("--db-url", envvar="BENCH_DATABASE_URL")] = "postgresql+asyncpg://bench:bench_dev@localhost:5432/cross_bench",
) -> None:
    """View and compare evaluation results."""
    asyncio.run(_show_results(run_id, benchmark, compare, format, db_url))


async def _show_results(run_id, benchmark, compare, fmt, db_url):
    from cross_bench.results.aggregation import compare_metrics, leaderboard
    from cross_bench.results.exporters import to_csv, to_json, to_markdown_table
    from cross_bench.results.repository import ResultRepository

    repo = ResultRepository(db_url)

    if compare:
        runs = await repo.compare_runs(compare)
        data = compare_metrics(runs)
        if fmt == "json":
            console.print(to_json(data))
        elif fmt == "csv":
            console.print(to_csv(data["rows"]))
        else:
            console.print(to_markdown_table(data["rows"], ["run_id", "model", "memory", *data["columns"]]))
        return

    if run_id:
        run = await repo.get_run(run_id)
        if not run:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise typer.Exit(1)
        task_results = await repo.get_task_results(run_id)
        console.print(f"[bold]Run:[/bold] {run_id}")
        console.print(f"[bold]Benchmark:[/bold] {run['benchmark']}")
        console.print(f"[bold]Status:[/bold] {run['status']}")
        if run.get("aggregate_metrics"):
            table = Table(title="Aggregate Metrics")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for k, v in sorted(run["aggregate_metrics"].items()):
                table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            console.print(table)
        console.print(f"\n[bold]Tasks:[/bold] {len(task_results)}")
        return

    # Default: list recent runs
    runs = await repo.list_runs(benchmark=benchmark)
    table = Table(title="Recent Runs")
    table.add_column("Run ID")
    table.add_column("Benchmark")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Created")
    for r in runs:
        metrics = r.get("aggregate_metrics") or {}
        table.add_row(
            r["run_id"][:12] + "...",
            r["benchmark"],
            r["execution_mode"],
            r["status"],
            f"{metrics.get('pass_rate', 0):.2%}" if metrics.get("pass_rate") is not None else "-",
            str(r["created_at"])[:19],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
@app.command()
def replay(
    run_id: Annotated[str, typer.Option("--run-id")],
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
) -> None:
    """Re-evaluate a previous run from stored artifacts (without re-calling the LLM)."""
    asyncio.run(_replay(run_id, artifacts_dir))


async def _replay(run_id: str, artifacts_dir: str) -> None:
    import json

    from cross_bench.artifact_store import ArtifactStore

    store = ArtifactStore(backend="local", local_dir=artifacts_dir)

    try:
        config_json = await store.retrieve(f"runs/{run_id}/config.json")
        config = json.loads(config_json)
        console.print(f"[bold]Replaying run:[/bold] {run_id}")
        console.print(f"[bold]Benchmark:[/bold] {config.get('benchmark')}")
        console.print(f"[bold]Model:[/bold] {config.get('system', {}).get('llm', {}).get('model')}")

        summary_json = await store.retrieve(f"runs/{run_id}/summary.json")
        summary = json.loads(summary_json)
        console.print(f"\n[bold]Original metrics:[/bold]")
        for k, v in sorted(summary.get("aggregate_metrics", {}).items()):
            console.print(f"  {k}: {v}")
    except FileNotFoundError:
        console.print(f"[red]Artifacts for run {run_id} not found in {artifacts_dir}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# docker-build
# ---------------------------------------------------------------------------
@app.command()
def docker_build(
    benchmark: Annotated[Optional[str], typer.Option("--benchmark", "-b")] = None,
) -> None:
    """Build Docker sandbox images for benchmarks."""
    import subprocess

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    cmd = ["bash", str(scripts_dir / "build_docker_images.sh")]
    if benchmark:
        cmd.append(benchmark)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
@app.command()
def compare(
    run_ids: Annotated[list[str], typer.Argument(help="Two or more run IDs to compare")],
    metrics: Annotated[Optional[list[str]], typer.Option("--metric", "-m", help="Metrics to compare (default: all)")] = None,
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
) -> None:
    """Compare results from multiple runs side-by-side.

    \b
    Examples:
        cross-bench compare RUN_ID_1 RUN_ID_2
        cross-bench compare RUN_ID_1 RUN_ID_2 RUN_ID_3 -m pass_rate -m memscore
    """
    asyncio.run(_compare_runs(run_ids, metrics, artifacts_dir))


async def _compare_runs(
    run_ids: list[str],
    metric_filter: list[str] | None,
    artifacts_dir: str,
) -> None:
    import json

    from cross_bench.artifact_store import ArtifactStore

    store = ArtifactStore(backend="local", local_dir=artifacts_dir)

    # Load summaries for each run
    runs: list[dict] = []
    for rid in run_ids:
        try:
            summary_json = await store.retrieve(f"runs/{rid}/summary.json")
            summary = json.loads(summary_json)
            runs.append(summary)
        except FileNotFoundError:
            console.print(f"[yellow]Warning: Run {rid} not found in {artifacts_dir}[/yellow]")

    if len(runs) < 2:
        console.print("[red]Need at least 2 valid runs to compare.[/red]")
        raise typer.Exit(1)

    # Collect all metric keys
    all_keys: set[str] = set()
    for r in runs:
        all_keys.update(r.get("aggregate_metrics", {}).keys())

    if metric_filter:
        all_keys = {k for k in all_keys if k in metric_filter}

    # Build comparison table
    table = Table(title="Run Comparison")
    table.add_column("Metric", style="bold")
    for r in runs:
        run_id = r.get("run_id", "?")[:12]
        model = r.get("config", {}).get("system", {}).get("llm", {}).get("model", "?")
        mem = r.get("config", {}).get("system", {}).get("memory", {}).get("provider", "none")
        table.add_column(f"{run_id}\n{model}\n{mem}", justify="right")

    for key in sorted(all_keys):
        row = [key]
        values = []
        for r in runs:
            v = r.get("aggregate_metrics", {}).get(key)
            if v is not None:
                values.append(v)
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            else:
                values.append(None)
                row.append("-")

        # Highlight best value (highest for most metrics, lowest for latency/error)
        if len(values) >= 2 and all(v is not None for v in values):
            is_lower_better = any(
                sub in key for sub in ("latency", "error", "timeout", "failed")
            )
            best_idx = values.index(min(v for v in values if v is not None) if is_lower_better
                                    else max(v for v in values if v is not None))
            row[best_idx + 1] = f"[bold green]{row[best_idx + 1]}[/bold green]"

        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# dispatch (fan-out a conversational benchmark across ECS tasks)
# ---------------------------------------------------------------------------
@app.command()
def dispatch(
    benchmark: Annotated[str, typer.Option("--benchmark", "-b")],
    profile: Annotated[list[str], typer.Option("--profile", "-p", help="Profiles to stack (passed through to each shard)")],
    cluster: Annotated[str, typer.Option("--cluster", envvar="BENCH_ECS_CLUSTER", help="ECS cluster name or ARN")],
    task_definition: Annotated[str, typer.Option("--task-definition", envvar="BENCH_ECS_TASK_DEF", help="ECS task definition family or ARN")],
    run_id: Annotated[Optional[str], typer.Option("--run-id", envvar="BENCH_RUN_ID", help="Parent run ID; shards get suffixed")] = None,
    conversation_id: Annotated[Optional[list[str]], typer.Option("--conversation-id", help="Conversation IDs to shard on (repeatable). Auto-discovers from the benchmark if omitted.")] = None,
    container: Annotated[str, typer.Option("--container", help="Container name in the task definition")] = "orchestrator",
    launch_type: Annotated[str, typer.Option("--launch-type", help="EC2 or FARGATE (ignored if --capacity-provider is set)")] = "EC2",
    capacity_provider: Annotated[Optional[str], typer.Option("--capacity-provider", help="Capacity provider name. Triggers managed scaling instead of plain launchType placement.")] = None,
    subnet: Annotated[Optional[list[str]], typer.Option("--subnet", help="Subnet IDs (FARGATE only)")] = None,
    security_group: Annotated[Optional[list[str]], typer.Option("--security-group", help="Security group IDs (FARGATE only)")] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval", help="Seconds between describe_tasks polls")] = 30.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait for all shards to stop")] = 3 * 60 * 60,
    merge: Annotated[bool, typer.Option("--merge/--no-merge", help="Run merge-shards locally after all shards stop")] = True,
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
) -> None:
    """Fan out a conversational benchmark across ECS tasks, one per conversation.

    \b
    Example:
        cross-bench dispatch -b locomo_original \\
            -p online -p mem0v2_embedded \\
            --cluster cross-bench-cluster \\
            --task-definition cross-bench-orchestrator \\
            --run-id locomo-2026-04-23
    """
    asyncio.run(_dispatch(
        benchmark=benchmark,
        profiles=profile,
        cluster=cluster,
        task_definition=task_definition,
        run_id=run_id,
        conversation_ids=conversation_id,
        container=container,
        launch_type=launch_type,
        capacity_provider=capacity_provider,
        subnets=subnet,
        security_groups=security_group,
        poll_interval=poll_interval,
        timeout=timeout,
        merge=merge,
        artifacts_dir=artifacts_dir,
    ))


async def _dispatch(
    benchmark: str,
    profiles: list[str],
    cluster: str,
    task_definition: str,
    run_id: Optional[str],
    conversation_ids: Optional[list[str]],
    container: str,
    launch_type: str,
    capacity_provider: Optional[str],
    subnets: Optional[list[str]],
    security_groups: Optional[list[str]],
    poll_interval: float,
    timeout: float,
    merge: bool,
    artifacts_dir: str,
) -> None:
    import uuid

    from cross_bench.sharding import (
        DispatchConfig,
        discover_conversation_ids,
        dispatch_shards,
        merge_shards,
        write_shards_manifest_upfront,
    )

    parent_run_id = run_id or f"dispatch-{uuid.uuid4().hex[:12]}"

    if not conversation_ids:
        console.print("[dim]Discovering conversation IDs from benchmark...[/dim]")
        conversation_ids = await discover_conversation_ids(benchmark)
    if not conversation_ids:
        console.print(f"[red]No conversations found for benchmark {benchmark!r}.[/red]")
        raise typer.Exit(1)

    console.rule("[bold]Dispatch — Fan-Out[/bold]")
    console.print(f"  [bold]Parent run:[/bold] {parent_run_id}")
    console.print(f"  [bold]Benchmark:[/bold]  {benchmark}")
    console.print(f"  [bold]Profiles:[/bold]   {' + '.join(profiles)}")
    console.print(f"  [bold]Cluster:[/bold]    {cluster}")
    console.print(f"  [bold]Task def:[/bold]   {task_definition}")
    console.print(f"  [bold]Shards:[/bold]     {len(conversation_ids)} ({', '.join(conversation_ids)})")
    console.rule()

    # Write the upfront ``shards.json`` manifest BEFORE dispatching any
    # ECS tasks. If the orchestrator dies mid-run, this manifest is
    # what makes ``cross-bench merge-shards --parent <id> --discover``
    # work. See the 2026-05-11 Mastra incident.
    backend = os.environ.get("BENCH_ARTIFACTS_BACKEND", "local")
    bucket = os.environ.get("BENCH_ARTIFACTS_BUCKET", "")
    store_for_manifest = _build_store(backend, bucket, artifacts_dir)
    try:
        write_shards_manifest_upfront(
            parent_run_id=parent_run_id,
            conversation_ids=conversation_ids,
            store=store_for_manifest,
            benchmark=benchmark,
            profiles=profiles,
        )
    except Exception as exc:
        # Manifest write is non-critical — if it fails (e.g. transient
        # S3 hiccup), we still want the dispatch to proceed. The end-
        # of-run merge will overwrite the manifest anyway. Log loudly.
        console.print(
            f"[yellow]Warning: failed to write upfront shards.json "
            f"(continuing dispatch): {exc}[/yellow]"
        )

    cfg = DispatchConfig(
        cluster=cluster,
        task_definition=task_definition,
        benchmark=benchmark,
        profiles=profiles,
        parent_run_id=parent_run_id,
        conversation_ids=conversation_ids,
        container_name=container,
        launch_type=launch_type,
        capacity_provider=capacity_provider,
        subnets=subnets,
        security_groups=security_groups,
        poll_interval_seconds=poll_interval,
        timeout_seconds=timeout,
    )

    # run_task + polling is synchronous; offload so we don't block the loop.
    report = await asyncio.to_thread(dispatch_shards, cfg)

    console.print()
    console.print(f"[bold]Shards stopped:[/bold] {len(report.shards)} "
                  f"([green]ok={len(report.shards) - len(report.failed_shards)}[/green] "
                  f"[red]failed={len(report.failed_shards)}[/red])")
    for f in report.failed_shards:
        console.print(f"  [red]✗ {f.shard_run_id}[/red] ({f.conversation_id})")

    if not merge:
        return

    if report.failed_shards:
        console.print("[yellow]Skipping merge — some shards failed. "
                      "Rerun with --no-merge and merge manually after retrying.[/yellow]")
        raise typer.Exit(1)

    # Reuse the same store we built for the upfront manifest write.
    merge_report = await merge_shards(
        parent_run_id=parent_run_id,
        shard_run_ids=[s.shard_run_id for s in report.shards],
        store=store_for_manifest,
        benchmark=benchmark,
    )

    table = Table(title=f"Merged Run: {merge_report.parent_run_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in sorted(merge_report.metrics.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# merge-shards (roll up shard summaries into a parent summary)
# ---------------------------------------------------------------------------
# run-ablation
#
# Multi-cell-per-task orchestrator. Runs the same task(s) through N
# different cell profiles sequentially in one process / one ECS task,
# so the Docker image, workspace clone, and (where summary state
# matches) the workspace index get amortized across cells.
#
# Compared to the per-cell-per-shard fan-out used by ``dispatch``:
# - One image pull per task (vs N pulls)
# - One workspace clone per task (vs N clones)
# - Workspace index re-used across same-summary-state cells via the
#   cache marker + skip_if_indexed plumbing
#
# Tradeoffs:
# - Sequential within an ECS task → loses host-level parallelism.
#   For full ablation sweeps, fan-out remains correct; for laptop
#   iteration + single-host hot-loop comparisons, this is faster.
# - One STOPPED ECS task per (task-id × all-cells), so per-cell
#   shard granularity is lost. The per-task summary contains a
#   per-cell breakdown instead.
# ---------------------------------------------------------------------------
@app.command(name="run-ablation")
def run_ablation_cmd(
    benchmark: Annotated[str, typer.Option("--benchmark", "-b", envvar="BENCH_BENCHMARK", help="Benchmark name (e.g. swebench_pro_seal)")],
    base_profile: Annotated[list[str], typer.Option("--base-profile", "-p", help="Profiles applied to EVERY cell (e.g. -p seal_haiku45 -p mem0v2_embedded). Repeatable.")] = None,
    cell: Annotated[list[str], typer.Option("--cell", "-C", help="Profile names appended PER cell (e.g. -C swebench_memory_first_step -C swebench_memory_retrieve_only). Repeatable.")] = None,
    task_id: Annotated[list[str], typer.Option("--task-id", help="Task IDs to run (repeatable). All cells run against all task IDs.")] = None,
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
    db_url: Annotated[str, typer.Option("--db-url", envvar="BENCH_DATABASE_URL")] = "none",
    run_id: Annotated[Optional[str], typer.Option("--run-id", envvar="BENCH_RUN_ID", help="Parent run ID (auto-generated if omitted)")] = None,
    stop_on_error: Annotated[bool, typer.Option("--stop-on-error", help="Abort the remaining cells if any cell errors out (default: continue)")] = False,
) -> None:
    """Run multiple ablation cells against the same task(s) in one shot.

    \b
    Loops over the (task_ids × cells) outer product sequentially. For
    each (task_id, cell) pair, runs ``cross-bench run`` with
    ``-p {base_profile...} -p {cell} --task-id {task_id}`` and writes
    per-cell artifacts under ``{artifacts_dir}/{run_id}/{cell}/``.
    Writes a roll-up ``{artifacts_dir}/{run_id}/ablation_summary.json``
    at the end with per-cell metrics.

    \b
    Example (laptop, single task, multi-cell):
        cross-bench run-ablation -b swebench_pro_seal \\
            -p seal_haiku45 \\
            -C swebench_memory_first_step -C swebench_memory_retrieve_only \\
            --task-id instance_NodeBB__... \\
            --artifacts-dir ./artifacts/multi-cell-nodebb

    \b
    Example (AWS dispatch — run the whole cell set on one ECS task):
        aws ecs run-task --cluster ... --task-definition cross-bench-dev-orchestrator \\
            --overrides '{"containerOverrides":[{"name":"orchestrator","command":[
              "run-ablation","-b","swebench_pro_seal",
              "-p","seal_haiku45",
              "-C","swebench_memory_first_step","-C","swebench_memory_retrieve_only",
              "--task-id","instance_NodeBB__..."
            ]}]}'
    """
    if not cell:
        raise typer.BadParameter("--cell (-C) is required at least once", param_hint="--cell")
    if not task_id:
        raise typer.BadParameter("--task-id is required at least once", param_hint="--task-id")
    base = list(base_profile or [])
    cells = list(cell)
    tasks = list(task_id)
    parent_run_id = run_id or _default_ablation_run_id(benchmark)
    asyncio.run(_execute_run_ablation(
        benchmark=benchmark,
        base_profiles=base,
        cells=cells,
        task_ids=tasks,
        artifacts_dir=artifacts_dir,
        db_url=db_url,
        parent_run_id=parent_run_id,
        stop_on_error=stop_on_error,
    ))


def _default_ablation_run_id(benchmark: str) -> str:
    import time
    return f"ablation-{benchmark.replace('_', '-')}-{time.strftime('%Y%m%d-%H%M%S')}"


async def _execute_run_ablation(
    *,
    benchmark: str,
    base_profiles: list[str],
    cells: list[str],
    task_ids: list[str],
    artifacts_dir: str,
    db_url: str,
    parent_run_id: str,
    stop_on_error: bool,
) -> None:
    """Outer product of (task_ids × cells), sequential, per-cell subdir."""
    import json
    import time
    from pathlib import Path

    from cross_bench.config import load_run_config

    parent_dir = Path(artifacts_dir) / parent_run_id
    parent_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.rule(f"[bold]Ablation Run — {parent_run_id}[/bold]")
    console.print(f"  [bold]Benchmark:[/bold]  {benchmark}")
    console.print(f"  [bold]Base profiles:[/bold] {' + '.join(base_profiles) if base_profiles else '(none)'}")
    console.print(f"  [bold]Cells:[/bold]      {len(cells)}")
    console.print(f"  [bold]Tasks:[/bold]      {len(task_ids)}")
    console.print(f"  [bold]Total cells:[/bold] {len(cells) * len(task_ids)} (task × cell)")
    console.print(f"  [bold]Artifacts:[/bold]  {parent_dir}")
    console.rule()
    console.print()

    per_cell_results: list[dict] = []
    overall_started_at = time.monotonic()
    for cell_idx, cell_spec in enumerate(cells):
        # Accept smoke-style ``label:profile1,profile2,...`` so different
        # cells can have different profile stacks. The label is used for
        # artifact dirs / run-id; the profiles after the colon are
        # appended to the base profiles. When no colon present, the cell
        # name IS the profile (legacy single-profile form).
        if ":" in cell_spec:
            cell_name, _, profile_csv = cell_spec.partition(":")
            cell_profiles = [p.strip() for p in profile_csv.split(",") if p.strip()]
        else:
            cell_name = cell_spec
            cell_profiles = [cell_spec]
        for task_idx, t_id in enumerate(task_ids):
            cell_label = f"{cell_name}/{t_id}"
            cell_run_id = f"{parent_run_id}-{cell_name}-{task_idx:02d}"
            cell_artifacts = parent_dir / cell_name / t_id
            cell_artifacts.mkdir(parents=True, exist_ok=True)
            console.print()
            console.rule(
                f"[bold cyan]Cell {cell_idx * len(task_ids) + task_idx + 1}/"
                f"{len(cells) * len(task_ids)}: {cell_label}[/bold cyan]"
            )
            stack = list(base_profiles) + cell_profiles
            overrides = {
                "run": {"benchmark": benchmark},
                "benchmark": {"filter_task_ids": [t_id]},
            }
            try:
                run_config = load_run_config(
                    None, benchmark=benchmark, profiles=stack, overrides=overrides,
                )
                run_config.run_id = cell_run_id
                cell_started = time.monotonic()
                await _execute_run(run_config, str(cell_artifacts), db_url)
                cell_elapsed = time.monotonic() - cell_started
                summary_path = cell_artifacts / cell_run_id / "summary.json"
                metrics: dict = {}
                if summary_path.exists():
                    try:
                        metrics = json.loads(summary_path.read_text())
                    except Exception:
                        pass
                per_cell_results.append({
                    "cell": cell_name,
                    "task_id": t_id,
                    "cell_run_id": cell_run_id,
                    "artifacts_dir": str(cell_artifacts),
                    "elapsed_seconds": round(cell_elapsed, 1),
                    "status": "completed",
                    "metrics": metrics.get("aggregate_metrics", {}) if isinstance(metrics, dict) else {},
                })
                console.print(f"[green]✓[/green] {cell_label} — {cell_elapsed:.1f}s")
            except Exception as e:
                console.print(f"[red]✗[/red] {cell_label} — {type(e).__name__}: {e}")
                per_cell_results.append({
                    "cell": cell_name,
                    "task_id": t_id,
                    "cell_run_id": cell_run_id,
                    "artifacts_dir": str(cell_artifacts),
                    "elapsed_seconds": None,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "metrics": {},
                })
                if stop_on_error:
                    console.print("[yellow]--stop-on-error set; aborting remaining cells[/yellow]")
                    break
        else:
            continue
        break  # propagate the inner break when stop_on_error tripped

    overall_elapsed = time.monotonic() - overall_started_at
    rollup = {
        "parent_run_id": parent_run_id,
        "benchmark": benchmark,
        "base_profiles": base_profiles,
        "cells": cells,
        "task_ids": task_ids,
        "overall_elapsed_seconds": round(overall_elapsed, 1),
        "per_cell_results": per_cell_results,
    }
    rollup_path = parent_dir / "ablation_summary.json"
    rollup_path.write_text(json.dumps(rollup, indent=2, default=str))

    # Print a compact end-of-run table
    console.print()
    table = Table(title=f"Ablation Summary: {parent_run_id}")
    table.add_column("Cell", style="bold")
    table.add_column("Task", style="cyan")
    table.add_column("Status")
    table.add_column("Elapsed (s)", justify="right")
    table.add_column("pass_rate", justify="right")
    for r in per_cell_results:
        status = ("[green]✓[/green]" if r["status"] == "completed" else "[red]✗[/red]")
        pr = r.get("metrics", {}).get("pass_rate")
        pr_s = f"{pr:.4f}" if isinstance(pr, (int, float)) else "—"
        el = f"{r['elapsed_seconds']:.1f}" if r.get("elapsed_seconds") is not None else "—"
        table.add_row(r["cell"], r["task_id"][:32], status, el, pr_s)
    console.print(table)
    console.print(f"\n[bold]Rollup:[/bold] {rollup_path}")


# ---------------------------------------------------------------------------
@app.command(name="merge-shards")
def merge_shards_cmd(
    parent: Annotated[str, typer.Option("--parent", help="Parent run ID to write the merged summary under")],
    shard: Annotated[list[str], typer.Option("--shard", help="Shard run IDs to merge (repeatable). Ignored when --discover is set.")] = None,
    benchmark: Annotated[str, typer.Option("--benchmark", "-b", help="Benchmark name to record in the merged summary")] = "",
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
    discover: Annotated[bool, typer.Option("--discover", help="Read shard run IDs from {parent}/shards.json instead of --shard.")] = False,
) -> None:
    """Aggregate per-shard task results into ``{parent}/summary.json``.

    \b
    Example (explicit shard IDs):
        cross-bench merge-shards \\
            --parent locomo-2026-04-23 \\
            --shard locomo-2026-04-23-conv-00 \\
            --shard locomo-2026-04-23-conv-01 \\
            ...

    \b
    Example (recovery — orchestrator died before auto-merge):
        cross-bench merge-shards \\
            --parent locomo-mastra-20260511-093717 \\
            --discover \\
            --benchmark locomo_original
    """
    asyncio.run(_merge_shards_cli(parent, shard or [], benchmark, artifacts_dir, discover))


async def _merge_shards_cli(
    parent: str,
    shards: list[str],
    benchmark: str,
    artifacts_dir: str,
    discover: bool,
) -> None:
    from cross_bench.sharding import merge_shards, read_shards_manifest

    backend = os.environ.get("BENCH_ARTIFACTS_BACKEND", "local")
    bucket = os.environ.get("BENCH_ARTIFACTS_BUCKET", "")
    store = _build_store(backend, bucket, artifacts_dir)

    if discover:
        try:
            manifest = await read_shards_manifest(parent, store)
        except FileNotFoundError:
            console.print(
                f"[red]--discover: no shards.json found at "
                f"{parent}/shards.json. The orchestrator died "
                f"before the upfront-manifest write, OR this run "
                f"predates the upfront-manifest feature. Pass "
                f"--shard explicitly.[/red]"
            )
            raise typer.Exit(1)
        shards = list(manifest.get("shard_run_ids", []))
        # If the manifest carries a benchmark and the caller didn't
        # provide one on the CLI, use it. CLI takes precedence.
        if not benchmark and manifest.get("benchmark"):
            benchmark = manifest["benchmark"]
        console.print(
            f"[dim]--discover: found {len(shards)} shards "
            f"in {parent}/shards.json[/dim]"
        )
    elif not shards:
        console.print(
            "[red]Either --shard (one or more) or --discover is "
            "required.[/red]"
        )
        raise typer.Exit(1)

    report = await merge_shards(
        parent_run_id=parent,
        shard_run_ids=shards,
        store=store,
        benchmark=benchmark,
    )

    table = Table(title=f"Merged Run: {report.parent_run_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in sorted(report.metrics.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


def _build_store(backend: str, bucket: str, artifacts_dir: str):
    """Construct an ArtifactStore that honors ``BENCH_ARTIFACTS_S3_PREFIX``.

    Centralised so merge-shards / replay / compare all pick up the
    navigable-layout prefix without each command duplicating the env
    var read.
    """
    import os

    from cross_bench.artifact_store import ArtifactStore

    s3_prefix = os.environ.get("BENCH_ARTIFACTS_S3_PREFIX", "")
    return ArtifactStore(
        backend=backend,
        bucket=bucket,
        local_dir=artifacts_dir,
        s3_prefix=s3_prefix,
    )


# ---------------------------------------------------------------------------
# serve (Web UI)
# ---------------------------------------------------------------------------
@app.command()
def serve(
    artifacts_dir: Annotated[str, typer.Option("--artifacts-dir")] = "./artifacts",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
) -> None:
    """Launch a web UI for inspecting benchmark results.

    \b
    Example:
        cross-bench serve --artifacts-dir ./artifacts --port 8080
    """
    asyncio.run(_serve(artifacts_dir, port, host))


async def _serve(artifacts_dir: str, port: int, host: str) -> None:
    try:
        from aiohttp import web
    except ImportError:
        console.print(
            "[red]aiohttp is required for the web UI. "
            "Install with: pip install aiohttp[/red]"
        )
        raise typer.Exit(1)

    import json
    from pathlib import Path

    artifacts_path = Path(artifacts_dir)

    # Layout matches what ``ArtifactStore`` writes:
    # ``{artifacts_dir}/{run_id}/summary.json`` — flat, no ``runs/``
    # prefix. Each run lives in its own top-level directory.
    def _list_run_summaries() -> list[dict]:
        out: list[dict] = []
        if not artifacts_path.exists():
            return out
        for run_dir in sorted(artifacts_path.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    out.append(json.load(f))
        return out

    async def index(request):
        """Serve the dashboard HTML."""
        html = _render_dashboard(_list_run_summaries())
        return web.Response(text=html, content_type="text/html")

    async def api_runs(request):
        """JSON API: list all runs."""
        return web.json_response(_list_run_summaries())

    async def api_run_detail(request):
        """JSON API: single run detail."""
        run_id = request.match_info["run_id"]
        summary_file = artifacts_path / run_id / "summary.json"
        if not summary_file.exists():
            return web.json_response({"error": "not found"}, status=404)
        with open(summary_file) as f:
            data = json.load(f)
        return web.json_response(data)

    app_web = web.Application()
    app_web.router.add_get("/", index)
    app_web.router.add_get("/api/runs", api_runs)
    app_web.router.add_get("/api/runs/{run_id}", api_run_detail)

    console.print(f"[bold]Bench Platform UI[/bold] → http://{host}:{port}")
    runner = web.AppRunner(app_web)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.cleanup()


def _render_dashboard(runs: list[dict]) -> str:
    """Render a self-contained HTML dashboard."""
    rows_html = ""
    for r in runs:
        rid = r.get("run_id", "?")
        benchmark = r.get("config", {}).get("benchmark", "?")
        model = r.get("config", {}).get("system", {}).get("llm", {}).get("model", "?")
        mem = r.get("config", {}).get("system", {}).get("memory", {}).get("provider", "none")
        metrics = r.get("aggregate_metrics", {})
        pass_rate = metrics.get("pass_rate", metrics.get("accuracy", 0))
        memscore = metrics.get("memscore", "")
        started = r.get("started_at", "")[:19]
        memscore_str = f"{memscore:.2f}" if isinstance(memscore, (int, float)) and memscore else "-"

        rows_html += f"""
        <tr>
          <td><code>{rid[:12]}</code></td>
          <td>{benchmark}</td>
          <td>{model}</td>
          <td>{mem}</td>
          <td><strong>{pass_rate:.1%}</strong></td>
          <td>{memscore_str}</td>
          <td>{started}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bench Platform — Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
  th {{ background: #16213e; color: white; padding: 12px 16px; text-align: left; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f0f4ff; }}
  code {{ background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .empty {{ text-align: center; padding: 40px; color: #666; }}
</style>
</head>
<body>
<h1>Bench Platform — Run Dashboard</h1>
{"<table><thead><tr><th>Run ID</th><th>Benchmark</th><th>Model</th><th>Memory</th><th>Pass Rate</th><th>MemScore</th><th>Started</th></tr></thead><tbody>" + rows_html + "</tbody></table>" if runs else '<div class="empty">No runs found. Run a benchmark first.</div>'}
<p style="margin-top:20px;color:#888;font-size:0.85em">
  API: <a href="/api/runs">/api/runs</a> •
  Auto-refreshes on page load
</p>
</body>
</html>"""


if __name__ == "__main__":
    app()
