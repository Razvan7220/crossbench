"""Tests for parallel_runner public surfaces.

Covers:
- run_parallel subprocess fan-out shape (mocked subprocess)
- Concurrency bounded by ``parallel``
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# run_parallel — generic per-task subprocess fan-out
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_parallel_launches_one_subprocess_per_task():
    """run_parallel calls _run_subprocess once per task with --task-id."""
    from cross_bench import parallel_runner

    call_log: list[tuple[list[str], str]] = []

    async def fake_run_subprocess(cmd: list[str], label: str) -> int:
        call_log.append((cmd, label))
        return 0

    with patch.object(parallel_runner, "_run_subprocess", side_effect=fake_run_subprocess):
        await parallel_runner.run_parallel(
            task_ids=["task-A", "task-B"],
            parallel=2,
            profiles=["online", "no_memory"],
            benchmark="swebench",
            artifacts_dir="/tmp/test-artifacts",
            db_url="none",
            extra_cli_args=[],
        )

    assert len(call_log) == 2
    cmds = [c for c, _ in call_log]
    ids_found = {c[c.index("--task-id") + 1] for c in cmds}
    assert ids_found == {"task-A", "task-B"}


@pytest.mark.asyncio
async def test_run_parallel_bounds_concurrency():
    """parallel=1 should serialise tasks (only 1 concurrent at a time)."""
    from cross_bench import parallel_runner

    active: list[int] = []
    max_concurrent: list[int] = []

    async def fake_run_subprocess(cmd: list[str], label: str) -> int:
        active.append(1)
        max_concurrent.append(len(active))
        await asyncio.sleep(0.01)
        active.pop()
        return 0

    with patch.object(parallel_runner, "_run_subprocess", side_effect=fake_run_subprocess):
        await parallel_runner.run_parallel(
            task_ids=["A", "B", "C"],
            parallel=1,
            profiles=["online", "no_memory"],
            benchmark="swebench",
            artifacts_dir="/tmp/test-artifacts",
            db_url="none",
            extra_cli_args=[],
        )

    assert max(max_concurrent) == 1
