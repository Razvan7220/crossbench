"""HumanEval adapter tests — verifies delegation to the ``human-eval`` package.

No Docker here. The tests stub :class:`SandboxManager` to record the exact
commands the adapter issues, and feed back canned stdout/stderr so the full
scoring path runs end-to-end. Data-loading tests call the real
``read_problems`` from the bundled ``HumanEval.jsonl.gz`` so any upstream
dataset change is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cross_bench.benchmarks.humaneval import (
    HumanEvalAdapter,
    _find_code,
    _shell_quote,
)
from cross_bench.execution.sandbox import ExecutionResult, Sandbox
from cross_bench.models import Task, TaskGranularity


# ---------------------------------------------------------------------------
# Fake sandbox that records every interaction and returns canned responses.
# ---------------------------------------------------------------------------
@dataclass
class FakeSandboxManager:
    exec_responses: list[ExecutionResult] = field(default_factory=list)
    executed: list[tuple[str, str, int]] = field(default_factory=list)

    async def create(self, image: str, task: Any, **_: Any) -> Sandbox:
        return Sandbox(container_id=f"fake-{task.task_id}", image=image)

    async def execute(
        self, sandbox: Sandbox, command: str, timeout: int = 60
    ) -> ExecutionResult:
        self.executed.append((sandbox.container_id, command, timeout))
        if not self.exec_responses:
            raise AssertionError(f"Unexpected exec: {command}")
        return self.exec_responses.pop(0)


def _ok() -> ExecutionResult:
    return ExecutionResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)


def _fail() -> ExecutionResult:
    return ExecutionResult(
        exit_code=1, stdout="", stderr="AssertionError", duration_ms=1.0
    )


def _task(
    *,
    task_id: str = "HumanEval/0",
    prompt: str = "def add(a, b):\n    '''Add.'''\n",
    test: str = "def check(c):\n    assert c(1, 2) == 3\n",
    entry_point: str = "add",
) -> Task:
    return Task(
        task_id=task_id,
        benchmark="humaneval",
        granularity=TaskGranularity.FUNCTION,
        prompt=prompt,
        ground_truth=test,
        metadata={"entry_point": entry_point, "canonical_solution": "    return a + b\n"},
    )


# ---------------------------------------------------------------------------
# Registration + loading
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("humaneval")
    assert isinstance(a, HumanEvalAdapter)
    assert a.name == "humaneval"
    assert a.granularity == TaskGranularity.FUNCTION
    # Docker sandbox needed for isolated execution
    assert a.requires_sandbox is True


def test_load_tasks_from_human_eval_package():
    """Dataset must come from the bundled ``HumanEval.jsonl.gz``, not HF."""
    a = HumanEvalAdapter()
    tasks = list(a.load_tasks({"limit": 3}))
    assert len(tasks) == 3
    # Upstream always begins with HumanEval/0.
    assert tasks[0].task_id == "HumanEval/0"
    assert tasks[0].metadata["entry_point"]
    assert tasks[0].ground_truth  # the packaged test string
    assert tasks[0].prompt


def test_load_tasks_full_size():
    """Upstream HumanEval has exactly 164 problems."""
    a = HumanEvalAdapter()
    tasks = list(a.load_tasks({}))
    assert len(tasks) == 164


def test_load_tasks_respects_limit():
    a = HumanEvalAdapter()
    tasks = list(a.load_tasks({"limit": 5}))
    assert len(tasks) == 5


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_prompt_prepends_simple_evals_instruction():
    a = HumanEvalAdapter()
    t = _task(prompt="PROMPT_BODY")
    out = a.format_prompt(t)
    assert out.startswith("Read the following function signature")
    assert out.endswith("PROMPT_BODY")


# ---------------------------------------------------------------------------
# _find_code — byte-for-byte port of simple-evals, including its quirk
# ---------------------------------------------------------------------------


def test_find_code_extracts_python_block():
    """First fenced ```python block wins."""
    completion = "blah\n```python\ndef f():\n    return 1\n```\ntrailing"
    # The simple-evals quirk: `:\n    ` exists inside `def f():\n    return`,
    # so everything up to and including the ":" is stripped.
    assert "return 1" in _find_code(completion)


def test_find_code_quirk_drops_first_char_when_no_sig():
    """When ``:\\n    `` is absent, ``find`` returns -1 and the slice is [1:].

    This is the published simple-evals behavior; preserved to keep scores
    comparable.
    """
    completion = "abcdef"  # no colon-newline-indent sequence
    # -1 + 2 = 1, so result is completion[1:] = "bcdef".
    assert _find_code(completion) == "bcdef"


def test_find_code_falls_back_to_raw_completion():
    """No fenced block => use the completion itself (still subject to quirk)."""
    completion = "def g():\n    return 2\n"
    out = _find_code(completion)
    assert "return 2" in out


# ---------------------------------------------------------------------------
# check_program construction mirrors human_eval.execution.unsafe_execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_pass_builds_expected_check_program():
    a = HumanEvalAdapter()
    mgr = FakeSandboxManager(exec_responses=[_ok()])
    t = _task(
        prompt="def add(a, b):\n    ",
        test="def check(c):\n    assert c(1, 2) == 3\n",
        entry_point="add",
    )
    # Completion contains fenced code with the signature: `:\n    ` present,
    # so _find_code strips up to ":" and leaves "\n    return a + b".
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    out = await a.evaluate(t, completion, mgr)
    assert out["pass"] is True
    assert out["pass@1"] == 1.0
    assert out["exit_code"] == 0

    # One exec call: `python3 -c '<quoted check_program>'`
    assert len(mgr.executed) == 1
    cmd = mgr.executed[0][1]
    assert cmd.startswith("python3 -c '")
    # Check program must include the prompt, the test body, and the check() call.
    assert "def add(a, b):" in cmd
    assert "return a + b" in cmd
    assert "def check(c):" in cmd
    assert "check(add)" in cmd


@pytest.mark.asyncio
async def test_evaluate_fail_returns_pass_zero():
    a = HumanEvalAdapter()
    mgr = FakeSandboxManager(exec_responses=[_fail()])
    out = await a.evaluate(_task(), "```python\n:\n    return 0\n```", mgr)
    assert out["pass"] is False
    assert out["pass@1"] == 0.0
    assert "AssertionError" in out["stderr"]


@pytest.mark.asyncio
async def test_evaluate_honors_timeout_from_metadata():
    a = HumanEvalAdapter()
    mgr = FakeSandboxManager(exec_responses=[_ok()])
    t = _task()
    t.metadata["_timeout_seconds"] = 99
    await a.evaluate(t, "```python\n:\n    return 3\n```", mgr)
    assert mgr.executed[0][2] == 99


@pytest.mark.asyncio
async def test_evaluate_creates_sandbox_when_none_provided():
    a = HumanEvalAdapter()
    mgr = FakeSandboxManager(exec_responses=[_ok()])
    await a.evaluate(_task(), "x", mgr, sandbox=None)
    assert mgr.executed[0][0] == "fake-HumanEval/0"


@pytest.mark.asyncio
async def test_evaluate_pass_at_1_uses_estimate_pass_at_k():
    """``pass@1`` must come from human_eval.evaluation.estimate_pass_at_k.

    With n=1, c=1, k=1 the estimator is exactly 1.0; with c=0 it is 0.0.
    Verifying both also guarantees we're not hand-computing the metric.
    """
    from human_eval.evaluation import estimate_pass_at_k

    assert float(estimate_pass_at_k(1, [1], 1)[0]) == 1.0
    assert float(estimate_pass_at_k(1, [0], 1)[0]) == 0.0

    a = HumanEvalAdapter()
    mgr = FakeSandboxManager(exec_responses=[_ok(), _fail()])
    ok = await a.evaluate(_task(), "x", mgr)
    bad = await a.evaluate(_task(), "x", mgr)
    assert ok["pass@1"] == 1.0
    assert bad["pass@1"] == 0.0


# ---------------------------------------------------------------------------
# Shell quoting
# ---------------------------------------------------------------------------


def test_shell_quote_plain():
    assert _shell_quote("abc") == "'abc'"


def test_shell_quote_embedded_single_quote():
    assert _shell_quote("a'b") == "'a'\\''b'"


def test_get_sandbox_image():
    assert HumanEvalAdapter().get_sandbox_image() == "cross-bench-function:latest"
