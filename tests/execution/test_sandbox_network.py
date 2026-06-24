"""Sandbox network_mode wiring for ``requires_network`` (deviations.md S21 / S23)."""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.execution.sandbox import SandboxManager
from cross_bench.models import Task, TaskGranularity


class _FakeContainer:
    id = "fake-container-id"
    short_id = "fake"


class _RecordingContainersAPI:
    def __init__(self) -> None:
        self.run_kwargs: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> _FakeContainer:
        self.run_kwargs.append(kwargs)
        return _FakeContainer()


class _FakeDockerClient:
    def __init__(self) -> None:
        self.containers = _RecordingContainersAPI()


@pytest.fixture
def _patch_docker(monkeypatch):
    import sys

    fake_docker = type("DockerModule", (), {})()
    client = _FakeDockerClient()
    fake_docker.from_env = lambda: client
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    return client


def _task(*, requires_network: bool) -> Task:
    return Task(
        task_id="django__django-1",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="fix it",
        metadata={"requires_network": requires_network},
    )


@pytest.mark.asyncio
async def test_create_uses_bridge_when_task_requires_network(_patch_docker):
    mgr = SandboxManager()
    await mgr.create("jefzda/sweap-images:tag", _task(requires_network=True))
    assert _patch_docker.containers.run_kwargs[-1]["network_mode"] == "bridge"


@pytest.mark.asyncio
async def test_create_defaults_to_none_without_requires_network(_patch_docker):
    mgr = SandboxManager()
    await mgr.create("sweb.eval.x86_64.django__django-1:latest", _task(requires_network=False))
    assert _patch_docker.containers.run_kwargs[-1]["network_mode"] == "none"


@pytest.mark.asyncio
async def test_create_honours_explicit_network_mode_kwarg(_patch_docker):
    """Explicit ``network_mode`` kwarg wins unless ``requires_network`` overrides."""
    mgr = SandboxManager(network_mode="host")
    await mgr.create(
        "img:latest",
        _task(requires_network=False),
        network_mode="none",
    )
    assert _patch_docker.containers.run_kwargs[-1]["network_mode"] == "none"
