"""Engine hook that writes ``environment.json`` for every run.

The capture itself lives in :mod:`cross_bench.environment`; these tests
cover the integration contract:

* it runs once per ``execute_run``, next to ``config.json``;
* it is skipped when ``CROSS_BENCH_CAPTURE_ENVIRONMENT`` is false;
* probe failures never abort the benchmark;
* legacy artifact-store doubles without the new method keep working.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import cross_bench.execution.engine as engine_mod
from cross_bench import environment as env_mod
from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.execution.engine import RunEngine
from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    RunStatus,
    SystemConfig,
    Task,
    TaskGranularity,
    TaskResult,
    TaskStatus,
)


class _RecordingStore:
    def __init__(self) -> None:
        self.configs: list[Any] = []
        self.environments: list[tuple[str, dict[str, Any]]] = []
        self.tasks: list[Any] = []
        self.summaries: list[Any] = []

    async def store_run_config(self, config: Any) -> None:
        self.configs.append(config)

    async def store_run_environment(self, run_id: str, environment: dict[str, Any]) -> None:
        self.environments.append((run_id, environment))

    async def store_task_result(self, run_id: str, result: Any) -> None:
        self.tasks.append(result)

    async def store_run_summary(self, result: Any) -> None:
        self.summaries.append(result)


class _LegacyStore:
    """Pre-environment artifact-store double (no ``store_run_environment``)."""

    async def store_run_config(self, config: Any) -> None:
        pass

    async def store_task_result(self, run_id: str, result: Any) -> None:
        pass

    async def store_run_summary(self, result: Any) -> None:
        pass


class _Repo:
    async def create_run(self, config: Any) -> None:
        pass

    async def save_task_result(self, result: Any) -> None:
        pass

    async def complete_run(self, run_id: str, metrics: Any, completed_at: Any) -> None:
        pass


class _SandboxManager:
    """No ``prefetch_images`` — keeps the run path minimal."""


class _StubAdapter(BenchmarkAdapter):
    name = "stub"
    granularity = TaskGranularity.FUNCTION
    requires_sandbox = False

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        yield Task(
            task_id="t1",
            benchmark=self.name,
            granularity=self.granularity,
            prompt="say hi",
        )

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: Any,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        return {"pass": 1.0}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""


def _config() -> RunConfig:
    return RunConfig(
        run_id="env-hook-run",
        benchmark="stub",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.SINGLE_SHOT,
    )


def _engine(store: Any) -> RunEngine:
    return RunEngine(store, _Repo(), sandbox_mgr=_SandboxManager())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_capture_environment_stores_snapshot(monkeypatch):
    monkeypatch.setattr(
        env_mod,
        "collect_environment",
        lambda config: {"run_id": config.run_id, "sentinel": True},
    )
    store = _RecordingStore()
    engine = _engine(store)
    config = _config()

    await engine._capture_environment(config)

    assert store.environments == [(config.run_id, {"run_id": config.run_id, "sentinel": True})]


@pytest.mark.asyncio
async def test_capture_environment_honours_disable_flag(monkeypatch):
    monkeypatch.setenv("CROSS_BENCH_CAPTURE_ENVIRONMENT", "0")
    called: list[Any] = []
    monkeypatch.setattr(env_mod, "collect_environment", lambda config: called.append(config))
    store = _RecordingStore()

    await _engine(store)._capture_environment(_config())

    assert called == []
    assert store.environments == []


@pytest.mark.asyncio
async def test_capture_environment_swallows_probe_failures(monkeypatch):
    def boom(_config):
        raise RuntimeError("no /proc on this host")

    monkeypatch.setattr(env_mod, "collect_environment", boom)
    store = _RecordingStore()

    # The benchmark must keep running even if the diagnostics probe dies.
    await _engine(store)._capture_environment(_config())

    assert store.environments == []


@pytest.mark.asyncio
async def test_capture_environment_tolerates_legacy_store():
    """Duck-typed test fakes without the hook must not need an update."""
    await _engine(_LegacyStore())._capture_environment(_config())


@pytest.mark.asyncio
async def test_execute_run_writes_environment_before_tasks(monkeypatch):
    captured: list[tuple[str, dict[str, Any]]] = []

    def fake_collect(config: RunConfig) -> dict[str, Any]:
        captured.append((config.run_id, {"sentinel": True}))
        return {"run_id": config.run_id, "sentinel": True}

    monkeypatch.setattr(env_mod, "collect_environment", fake_collect)
    monkeypatch.setattr(engine_mod, "get_adapter", lambda _name: _StubAdapter())
    monkeypatch.setattr(engine_mod, "get_llm_driver", lambda _cfg: object())
    monkeypatch.setattr(engine_mod, "get_prompt_strategy", lambda _cfg: object())

    async def _stub_execute_single_shot(**kwargs):
        task = kwargs["task"]
        return TaskResult(
            task_id=task.task_id,
            run_id=kwargs["config"].run_id,
            status=TaskStatus.PASS,
            steps=[],
            final_output="hi",
            evaluation={"pass": 1.0},
        )

    monkeypatch.setattr(engine_mod, "execute_single_shot", _stub_execute_single_shot)

    store = _RecordingStore()
    config = _config()
    result = await _engine(store).execute_run(config)

    assert result.status is RunStatus.COMPLETED
    assert captured == [(config.run_id, {"sentinel": True})]
    assert store.environments == [(config.run_id, {"run_id": config.run_id, "sentinel": True})]
    assert len(store.tasks) == 1
    assert store.summaries
