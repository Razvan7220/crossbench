"""Terminal-Bench adapter — delegates to the ``terminal-bench`` package.

Terminal-Bench is Stanford/Laude Institute's benchmark for terminal-native
agents (https://github.com/laude-institute/terminal-bench). Each task ships
a Docker Compose environment, an ``instruction`` in ``task.yaml``, and a
pytest-based verification suite under ``tests/``. An agent "passes" a task
iff every test in ``tests/`` passes after its session ends.

Delegation
----------
Instead of hand-rolling task loading and YAML parsing, this adapter reuses
three upstream primitives:

* :class:`terminal_bench.dataset.Dataset` — enumerates the canonical task
  paths for a named dataset / version (or a local directory). The dataset
  class handles registry lookups, caching under
  ``~/.cache/terminal-bench/<name>/<version>``, and ``task_ids`` / ``n_tasks``
  filtering. This is the authoritative task list — the previous adapter's
  hand-maintained ``tasks.json`` drifts from upstream.
* :class:`terminal_bench.handlers.trial_handler.Task` (pydantic model) /
  :class:`~terminal_bench.handlers.trial_handler.TaskPaths` — parse
  ``task.yaml`` with all upstream fields (``instruction``, ``difficulty``,
  ``category``, ``tags``, ``max_agent_timeout_sec``, ``max_test_timeout_sec``,
  ``parser_name``, ``run_tests_in_same_shell``, ``expert_time_estimate_min``,
  etc.) so downstream code sees the same metadata the reference harness does.
* :class:`terminal_bench.parsers.pytest_parser.PytestParser` — parses the
  ``short test summary info`` block of ``run-tests.sh``'s output into
  ``{test_name: UnitTestStatus}``. A task resolves iff every test reports
  ``PASSED`` — matching the upstream harness's pass/fail semantics.

Evaluation flow
---------------
The CrossBench engine is responsible for placing the generated output into
the task container (as a solution script, or as a tmux session for agentic
mode). This adapter's ``evaluate`` does the same final step the upstream
harness performs:

1. Read ``run-tests.sh`` from the task directory and write it to
   ``/workspace/run-tests.sh`` inside the sandbox.
2. Copy the ``tests/`` directory content into ``/workspace/tests`` (a tarball
   shuttle through :meth:`SandboxManager.write_file`).
3. Execute ``bash /workspace/run-tests.sh``; its combined stdout/stderr is
   the pytest output.
4. Parse via ``PytestParser.parse()``; flag ``pass`` iff all tests are
   ``PASSED``.

Sandbox image
-------------
``requires_sandbox = True``. The Docker image is task-specific and pulled
from the task's ``docker-compose.yaml`` — we parse it at load time rather
than asking the engine to guess. Tasks that rely on multi-service compose
graphs (most of Terminal-Bench) still need docker-compose support to run
end-to-end; single-service tasks work through CrossBench's flat sandbox.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

if TYPE_CHECKING:
    from terminal_bench.handlers.trial_handler import Task as TbTask

# Keep in sync with upstream's own cache convention —
# ``~/.cache/terminal-bench/<name>/<version>/tasks``.
_DEFAULT_DATASET_NAME = "terminal-bench-core"
_DEFAULT_DATASET_VERSION = "head"


@register_adapter
class TerminalBenchAdapter(BenchmarkAdapter):
    name = "terminalbench"
    granularity = TaskGranularity.TERMINAL

    def __init__(self) -> None:
        # Cache parsed task.yaml objects keyed by task_id so evaluate doesn't
        # reparse YAML on each run.
        self._task_paths: dict[str, Path] = {}

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        from terminal_bench.handlers.trial_handler import Task as TbTask
        from terminal_bench.handlers.trial_handler import TaskPaths

        dataset = _build_dataset(params)

        limit = params.get("limit")
        for i, task_path in enumerate(dataset):
            if limit and i >= limit:
                break

            paths = TaskPaths(task_path)
            tb_task: TbTask = TbTask.from_yaml(paths.task_config_path)

            self._task_paths[task_path.name] = task_path
            image = _image_from_compose(paths.docker_compose_path)

            yield Task(
                task_id=task_path.name,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=tb_task.instruction,
                metadata={
                    "task_dir": str(task_path),
                    "docker_image": image,
                    "difficulty": tb_task.difficulty.value,
                    "category": tb_task.category,
                    "tags": list(tb_task.tags),
                    "parser_name": tb_task.parser_name.value,
                    "max_agent_timeout_sec": tb_task.max_agent_timeout_sec,
                    "max_test_timeout_sec": tb_task.max_test_timeout_sec,
                    "run_tests_in_same_shell": tb_task.run_tests_in_same_shell,
                    "disable_asciinema": tb_task.disable_asciinema,
                    "expert_time_estimate_min": tb_task.expert_time_estimate_min,
                    "junior_time_estimate_min": tb_task.junior_time_estimate_min,
                    "estimated_duration_sec": tb_task.effective_estimated_duration_sec,
                    "requires_network": True,  # Many tasks need package installs
                },
                ground_truth=_read_solution(paths),
            )

    def format_prompt(self, task: Task) -> str:
        return (
            "You are operating in a Linux terminal environment.\n"
            f"Task: {task.prompt}\n\n"
            "Issue bash commands one at a time. When done, output TASK_COMPLETE."
        )

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        """Run the task's ``run-tests.sh`` and grade with PytestParser."""
        from terminal_bench.parsers.parser_factory import ParserFactory, ParserName

        task_dir = Path(task.metadata["task_dir"])
        run_tests_path = task_dir / "run-tests.sh"
        if not run_tests_path.exists():
            return {"pass": False, "error": "no_run_tests_sh"}

        if sandbox is None:
            sandbox = await sandbox_mgr.create(self.get_sandbox_image(task), task)

        # Upload run-tests.sh and the tests directory.
        await sandbox_mgr.write_file(
            sandbox, "/workspace/run-tests.sh", run_tests_path.read_text()
        )
        for test_file in (task_dir / "tests").rglob("*"):
            if test_file.is_file():
                rel = test_file.relative_to(task_dir / "tests")
                await sandbox_mgr.write_file(
                    sandbox,
                    f"/workspace/tests/{rel.as_posix()}",
                    test_file.read_text(errors="replace"),
                )

        timeout = int(
            (task.metadata or {}).get(
                "_timeout_seconds", task.metadata.get("max_test_timeout_sec", 60)
            )
        )
        result = await sandbox_mgr.execute(
            sandbox,
            "cd /workspace && bash run-tests.sh",
            timeout=timeout,
        )

        combined = result.stdout + "\n" + result.stderr
        parser_name = ParserName(task.metadata.get("parser_name", "pytest"))
        parser = ParserFactory.get_parser(parser_name)
        try:
            statuses = parser.parse(combined)
        except ValueError as exc:
            return {
                "pass": False,
                "error": f"parser_error: {exc}",
                "exit_code": result.exit_code,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:4000],
            }

        from terminal_bench.parsers.base_parser import UnitTestStatus

        total = len(statuses)
        passed = sum(1 for s in statuses.values() if s == UnitTestStatus.PASSED)
        resolved = total > 0 and passed == total

        return {
            "pass": resolved,
            "resolved": resolved,
            "passed_cases": passed,
            "total_cases": total,
            "pass_rate": passed / total if total else 0.0,
            "test_statuses": {k: v.value for k, v in statuses.items()},
            "exit_code": result.exit_code,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
        }

    def get_sandbox_image(self, task: Task | None = None) -> str:
        """Per-task image, parsed from each task's ``docker-compose.yaml``.

        ``load_tasks`` stashes the parsed image under
        ``task.metadata["docker_image"]``; the engine pre-creates the
        sandbox with that tag so the container actually has the
        per-task workspace the harness expects. The fallback string is
        only used when the metadata is missing (defensive — the
        upstream tasks all carry an image).
        """
        if task is not None:
            image = (task.metadata or {}).get("docker_image")
            if image:
                return str(image)
        return "cross-bench-terminal:latest"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "dataset_name": _DEFAULT_DATASET_NAME,
            "dataset_version": _DEFAULT_DATASET_VERSION,
        }


def _build_dataset(params: dict[str, Any]):
    """Construct a ``terminal_bench.dataset.Dataset`` from CrossBench params.

    Supports both the "pull from registry" (``dataset_name`` + ``dataset_version``)
    and "local directory" (``data_dir``) paths. A local ``data_dir`` wins if
    set — useful for air-gapped or pinned test runs.
    """
    from terminal_bench.dataset.dataset import Dataset

    data_dir = params.get("data_dir")
    if data_dir:
        return Dataset(
            path=Path(data_dir),
            task_ids=params.get("task_ids"),
            n_tasks=params.get("n_tasks"),
            exclude_task_ids=params.get("exclude_task_ids"),
        )

    return Dataset(
        name=params.get("dataset_name", _DEFAULT_DATASET_NAME),
        version=params.get("dataset_version", _DEFAULT_DATASET_VERSION),
        task_ids=params.get("task_ids"),
        n_tasks=params.get("n_tasks"),
        exclude_task_ids=params.get("exclude_task_ids"),
        registry_url=params.get("registry_url"),
        local_registry_path=params.get("local_registry_path"),
    )


def _image_from_compose(compose_path: Path) -> str:
    """Extract the primary service image from ``docker-compose.yaml``.

    Tasks occasionally declare multiple services (e.g. a DB + an agent
    workspace). We pick the first service entry with an ``image`` key — this
    is the convention in the terminal-bench-core dataset.
    """
    if not compose_path.exists():
        return ""
    try:
        data = yaml.safe_load(compose_path.read_text()) or {}
    except yaml.YAMLError:
        return ""
    for _name, svc in (data.get("services") or {}).items():
        image = svc.get("image") if isinstance(svc, dict) else None
        if image:
            return str(image)
    return ""


def _read_solution(paths) -> str:
    """Fetch the oracle solution script if one ships with the task."""
    try:
        return paths.solution_path.read_text()
    except FileNotFoundError:
        return ""
