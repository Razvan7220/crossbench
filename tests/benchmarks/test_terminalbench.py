"""Terminal-Bench adapter tests — verifies delegation to ``terminal-bench``.

The tests build a small on-disk task directory matching Terminal-Bench's
expected structure (``task.yaml``, ``docker-compose.yaml``, ``tests/``,
``run-tests.sh``), then drive the adapter through it with a fake
:class:`SandboxManager`. Upstream classes ``Task``, ``TaskPaths``, and
``PytestParser`` are imported directly so any change to their public
contract breaks the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks.terminalbench import (
    TerminalBenchAdapter,
    _image_from_compose,
)
from cross_bench.execution.sandbox import ExecutionResult, Sandbox
from cross_bench.models import Task, TaskGranularity


# ---------------------------------------------------------------------------
# Fake SandboxManager — same shape as the one in test_swebench.py
# ---------------------------------------------------------------------------
@dataclass
class FakeSandboxManager:
    exec_responses: list[ExecutionResult] = field(default_factory=list)
    executed: list[tuple[str, str, int]] = field(default_factory=list)
    writes: list[tuple[str, str, str]] = field(default_factory=list)

    async def create(self, image: str, task: Any, **_: Any) -> Sandbox:
        return Sandbox(container_id=f"fake-{task.task_id}", image=image)

    async def execute(
        self, sandbox: Sandbox, command: str, timeout: int = 60
    ) -> ExecutionResult:
        self.executed.append((sandbox.container_id, command, timeout))
        if not self.exec_responses:
            raise AssertionError(f"Unexpected exec: {command}")
        return self.exec_responses.pop(0)

    async def write_file(self, sandbox: Sandbox, path: str, content: str) -> None:
        self.writes.append((sandbox.container_id, path, content))


def _result(exit_code: int, stdout: str = "", stderr: str = "") -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code, stdout=stdout, stderr=stderr, duration_ms=1.0
    )


# ---------------------------------------------------------------------------
# On-disk task fixture
# ---------------------------------------------------------------------------
_TASK_YAML = """instruction: |
  Install Python 3.12 and write a hello-world script.
difficulty: easy
category: software_engineering
tags: [quickstart]
parser_name: pytest
max_agent_timeout_sec: 300
max_test_timeout_sec: 60
"""

_COMPOSE_YAML = """services:
  client:
    image: ubuntu:22.04-example
    command: sleep infinity
"""

_RUN_TESTS_SH = """#!/bin/bash
cd /workspace/tests
python3 -m pytest -q 2>&1
"""

_TEST_PY = """def test_hello():
    assert 1 + 1 == 2
"""


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Build a minimal terminal-bench task directory layout."""
    d = tmp_path / "hello-world"
    d.mkdir()
    (d / "task.yaml").write_text(_TASK_YAML)
    (d / "docker-compose.yaml").write_text(_COMPOSE_YAML)
    (d / "run-tests.sh").write_text(_RUN_TESTS_SH)
    (d / "solution.sh").write_text("#!/bin/bash\necho hello")
    (d / "tests").mkdir()
    (d / "tests" / "test_hello.py").write_text(_TEST_PY)
    return d


def _tb_task(task_dir: Path) -> Task:
    """Task matching what ``load_tasks`` would emit for ``task_dir``."""
    return Task(
        task_id=task_dir.name,
        benchmark="terminalbench",
        granularity=TaskGranularity.TERMINAL,
        prompt="Install Python 3.12 and write a hello-world script.\n",
        metadata={
            "task_dir": str(task_dir),
            "docker_image": "ubuntu:22.04-example",
            "difficulty": "easy",
            "category": "software_engineering",
            "tags": ["quickstart"],
            "parser_name": "pytest",
            "max_agent_timeout_sec": 300.0,
            "max_test_timeout_sec": 60.0,
            "run_tests_in_same_shell": False,
            "disable_asciinema": False,
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("terminalbench")
    assert isinstance(a, TerminalBenchAdapter)
    assert a.name == "terminalbench"
    assert a.granularity == TaskGranularity.TERMINAL
    # Terminal tasks need Docker.
    assert a.requires_sandbox is True


def test_default_params_use_upstream_dataset_name():
    p = TerminalBenchAdapter().get_default_params()
    assert p["dataset_name"] == "terminal-bench-core"
    assert p["dataset_version"] == "head"


# ---------------------------------------------------------------------------
# Task loading — delegated to terminal_bench.dataset.Dataset
# ---------------------------------------------------------------------------


def test_load_tasks_builds_local_dataset(task_dir: Path, monkeypatch):
    """When ``data_dir`` is set, Dataset should be constructed with path=...
    and the adapter should read task.yaml + docker-compose.yaml from disk."""
    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._tasks = [task_dir]

        def __iter__(self):
            return iter(self._tasks)

    monkeypatch.setattr("terminal_bench.dataset.dataset.Dataset", FakeDataset)

    a = TerminalBenchAdapter()
    tasks = list(a.load_tasks({"data_dir": str(task_dir.parent)}))
    assert captured["path"] == task_dir.parent
    assert len(tasks) == 1
    t = tasks[0]
    assert t.task_id == "hello-world"
    assert t.prompt.startswith("Install Python 3.12")
    assert t.metadata["docker_image"] == "ubuntu:22.04-example"
    assert t.metadata["difficulty"] == "easy"
    assert t.metadata["category"] == "software_engineering"
    assert t.metadata["tags"] == ["quickstart"]
    assert t.metadata["parser_name"] == "pytest"
    assert t.metadata["max_agent_timeout_sec"] == 300.0


def test_load_tasks_uses_dataset_name_when_no_data_dir(task_dir: Path, monkeypatch):
    """Without ``data_dir``, Dataset gets name + version — the registry path."""
    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._tasks = [task_dir]

        def __iter__(self):
            return iter(self._tasks)

    monkeypatch.setattr("terminal_bench.dataset.dataset.Dataset", FakeDataset)

    a = TerminalBenchAdapter()
    list(a.load_tasks({}))
    assert captured["name"] == "terminal-bench-core"
    assert captured["version"] == "head"


def test_load_tasks_respects_limit(task_dir: Path, monkeypatch):
    # Duplicate the task dir three times in the fake dataset.
    extra_dirs = [task_dir, task_dir, task_dir]

    class FakeDataset:
        def __init__(self, **kwargs):
            self._tasks = extra_dirs

        def __iter__(self):
            return iter(self._tasks)

    monkeypatch.setattr("terminal_bench.dataset.dataset.Dataset", FakeDataset)
    a = TerminalBenchAdapter()
    tasks = list(a.load_tasks({"data_dir": str(task_dir.parent), "limit": 2}))
    assert len(tasks) == 2


# ---------------------------------------------------------------------------
# _image_from_compose
# ---------------------------------------------------------------------------


def test_image_from_compose(tmp_path: Path):
    f = tmp_path / "docker-compose.yaml"
    f.write_text(_COMPOSE_YAML)
    assert _image_from_compose(f) == "ubuntu:22.04-example"


def test_image_from_compose_missing_returns_empty(tmp_path: Path):
    assert _image_from_compose(tmp_path / "nope.yaml") == ""


def test_image_from_compose_malformed_returns_empty(tmp_path: Path):
    f = tmp_path / "compose.yaml"
    f.write_text("not: valid: yaml: [")
    assert _image_from_compose(f) == ""


def test_image_from_compose_no_services_returns_empty(tmp_path: Path):
    f = tmp_path / "compose.yaml"
    f.write_text("version: '3'\n")
    assert _image_from_compose(f) == ""


# ---------------------------------------------------------------------------
# Evaluate: uploads run-tests.sh, runs it, parses with PytestParser
# ---------------------------------------------------------------------------


_PYTEST_PASS_OUTPUT = """
============================= test session starts ==============================
collected 1 item

tests/test_hello.py .                                                    [100%]

============================== 1 passed in 0.02s ===============================
=========================== short test summary info ============================
PASSED tests/test_hello.py::test_hello
"""

_PYTEST_FAIL_OUTPUT = """
============================= test session starts ==============================
collected 2 items

tests/test_hello.py .F                                                   [100%]

=========================== short test summary info ============================
PASSED tests/test_hello.py::test_hello
FAILED tests/test_hello.py::test_world - assert False
"""


@pytest.mark.asyncio
async def test_evaluate_uploads_run_tests_and_tests_dir(task_dir: Path):
    mgr = FakeSandboxManager(exec_responses=[_result(0, _PYTEST_PASS_OUTPUT)])
    a = TerminalBenchAdapter()
    out = await a.evaluate(_tb_task(task_dir), "(agent output ignored)", mgr)

    # run-tests.sh + tests/test_hello.py both uploaded.
    paths = [w[1] for w in mgr.writes]
    assert "/workspace/run-tests.sh" in paths
    assert "/workspace/tests/test_hello.py" in paths

    # The exec command is the upstream run-tests invocation.
    cmd = mgr.executed[0][1]
    assert "bash run-tests.sh" in cmd
    assert out["pass"] is True
    assert out["passed_cases"] == 1
    assert out["total_cases"] == 1
    # PytestParser keys by the short test name, not the full file::func path.
    assert out["test_statuses"] == {"test_hello": "passed"}


@pytest.mark.asyncio
async def test_evaluate_failure_reports_per_test(task_dir: Path):
    mgr = FakeSandboxManager(exec_responses=[_result(1, _PYTEST_FAIL_OUTPUT)])
    a = TerminalBenchAdapter()
    out = await a.evaluate(_tb_task(task_dir), "bad", mgr)
    assert out["pass"] is False
    assert out["passed_cases"] == 1
    assert out["total_cases"] == 2
    assert out["test_statuses"] == {
        "test_hello": "passed",
        "test_world": "failed",
    }


@pytest.mark.asyncio
async def test_evaluate_parser_error_on_unrecognised_output(task_dir: Path):
    # No ``short test summary info`` block → PytestParser raises ValueError.
    mgr = FakeSandboxManager(exec_responses=[_result(0, "gibberish")])
    a = TerminalBenchAdapter()
    out = await a.evaluate(_tb_task(task_dir), "?", mgr)
    assert out["pass"] is False
    assert "parser_error" in out["error"]
    assert out["exit_code"] == 0


@pytest.mark.asyncio
async def test_evaluate_missing_run_tests_sh(task_dir: Path):
    (task_dir / "run-tests.sh").unlink()
    mgr = FakeSandboxManager()
    a = TerminalBenchAdapter()
    out = await a.evaluate(_tb_task(task_dir), "?", mgr)
    assert out["pass"] is False
    assert out["error"] == "no_run_tests_sh"
    # No sandbox commands issued.
    assert mgr.executed == []


@pytest.mark.asyncio
async def test_evaluate_honors_test_timeout(task_dir: Path):
    mgr = FakeSandboxManager(exec_responses=[_result(0, _PYTEST_PASS_OUTPUT)])
    a = TerminalBenchAdapter()
    t = _tb_task(task_dir)
    t.metadata["max_test_timeout_sec"] = 99
    await a.evaluate(t, "?", mgr)
    assert mgr.executed[0][2] == 99


@pytest.mark.asyncio
async def test_evaluate_respects_explicit_timeout_override(task_dir: Path):
    mgr = FakeSandboxManager(exec_responses=[_result(0, _PYTEST_PASS_OUTPUT)])
    a = TerminalBenchAdapter()
    t = _tb_task(task_dir)
    t.metadata["_timeout_seconds"] = 5
    await a.evaluate(t, "?", mgr)
    assert mgr.executed[0][2] == 5


# ---------------------------------------------------------------------------
# Task.from_yaml is what we're using upstream — lock that contract
# ---------------------------------------------------------------------------


def test_task_from_yaml_round_trip(task_dir: Path):
    """Ensures the upstream ``Task`` pydantic model still parses the task.yaml
    format we ship in fixtures — catches upstream schema drift."""
    from terminal_bench.handlers.trial_handler import Task as TbTask
    from terminal_bench.handlers.trial_handler import TaskPaths

    paths = TaskPaths(task_dir)
    tb_task = TbTask.from_yaml(paths.task_config_path)
    assert tb_task.instruction.startswith("Install Python 3.12")
    assert tb_task.difficulty.value == "easy"
    assert tb_task.parser_name.value == "pytest"
