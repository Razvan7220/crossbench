"""HumanEval+ adapter tests — verifies delegation to the ``evalplus`` package.

EvalPlus's ``untrusted_check`` forks a subprocess that imports the Unix-only
``resource`` module via ``reliability_guard``; on Windows it always times out.
The tests therefore monkey-patch ``untrusted_check`` and ``sanitize`` so the
adapter's scoring logic runs end-to-end without invoking the real sandbox.
``get_groundtruth`` is likewise patched to avoid the ~minutes of canonical-
solution replay on first call.
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.benchmarks.humaneval_plus import HumanEvalPlusAdapter
from cross_bench.models import Task, TaskGranularity


def _task(
    *,
    task_id: str = "HumanEval/0",
    entry_point: str = "add",
    base_input: list[Any] | None = None,
    plus_input: list[Any] | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        benchmark="humaneval_plus",
        granularity=TaskGranularity.FUNCTION,
        prompt="def add(a, b):\n    '''Add.'''\n",
        ground_truth="def check(c):\n    assert c(1, 2) == 3\n",
        metadata={
            "entry_point": entry_point,
            "canonical_solution": "    return a + b\n",
            "atol": 0.0,
            "base_input": base_input if base_input is not None else [[1, 2], [3, 4]],
            "plus_input": plus_input if plus_input is not None else [[100, 200]],
            "contract": "",
        },
    )


def _expected() -> dict[str, Any]:
    return {
        "base": [3, 7],
        "base_time": [0.01, 0.01],
        "plus": [300],
        "plus_time": [0.01],
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("humaneval_plus")
    assert isinstance(a, HumanEvalPlusAdapter)
    assert a.name == "humaneval_plus"
    assert a.granularity == TaskGranularity.FUNCTION
    # In-process via evalplus's multiprocessing sandbox — no Docker.
    assert a.requires_sandbox is False


def test_get_sandbox_image_is_empty():
    """No Docker image; evalplus does its own sandboxing."""
    assert HumanEvalPlusAdapter().get_sandbox_image() == ""


# ---------------------------------------------------------------------------
# Task loading mirrors evalplus.data.get_human_eval_plus
# ---------------------------------------------------------------------------


def test_load_tasks_preserves_plus_fields():
    """Each Task must carry base_input / plus_input / atol — the grading fuel."""
    a = HumanEvalPlusAdapter()
    tasks = list(a.load_tasks({"limit": 3}))
    assert len(tasks) == 3
    t = tasks[0]
    assert t.task_id.startswith("HumanEval/")
    md = t.metadata
    for key in ("entry_point", "base_input", "plus_input", "atol", "canonical_solution"):
        assert key in md, f"missing: {key}"
    # The whole point of HumanEval+ is the augmented inputs.
    assert len(md["plus_input"]) > 0


def test_load_tasks_full_size():
    """HumanEval+ preserves the 164 problems of the original HumanEval."""
    a = HumanEvalPlusAdapter()
    tasks = list(a.load_tasks({}))
    assert len(tasks) == 164


def test_load_tasks_respects_limit():
    a = HumanEvalPlusAdapter()
    tasks = list(a.load_tasks({"limit": 2}))
    assert len(tasks) == 2


# ---------------------------------------------------------------------------
# Prompt formatting — simple-evals style (unchanged from humaneval)
# ---------------------------------------------------------------------------


def test_format_prompt_prepends_simple_evals_instruction():
    a = HumanEvalPlusAdapter()
    out = a.format_prompt(_task())
    assert out.startswith("Read the following function signature")
    assert "def add" in out


# ---------------------------------------------------------------------------
# Evaluate: sanitize + untrusted_check calls delegated to evalplus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_calls_sanitize_with_entry_point(monkeypatch):
    """``sanitize`` must be called with the task's entrypoint — that's how
    evalplus knows which function to keep and trim the rest."""
    captured: dict[str, Any] = {}

    def fake_sanitize(code: str, entrypoint: str | None = None) -> str:
        captured["code"] = code
        captured["entrypoint"] = entrypoint
        return "def add(a, b): return a + b"

    monkeypatch.setattr("evalplus.sanitize.sanitize", fake_sanitize)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: ("pass", [True] * len(args[2])),
    )

    a = HumanEvalPlusAdapter()
    a._expected_cache = {"HumanEval/0": _expected()}
    await a.evaluate(_task(), "```python\ndef add(a, b): return a + b\n```", None)
    assert captured["entrypoint"] == "add"
    assert "def add" in captured["code"]


@pytest.mark.asyncio
async def test_evaluate_runs_both_base_and_plus(monkeypatch):
    """Two ``untrusted_check`` calls: one for base_input, one for plus_input.

    Arg order: (dataset, code, inputs, entry_point, expected, atol, ref_time).
    """
    calls: list[tuple[Any, ...]] = []

    def fake_untrusted_check(
        dataset, code, inputs, entry_point, expected, atol, ref_time, **kwargs
    ):
        calls.append((dataset, entry_point, tuple(tuple(i) for i in inputs), tuple(expected)))
        return ("pass", [True] * len(inputs))

    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr("evalplus.eval.untrusted_check", fake_untrusted_check)

    a = HumanEvalPlusAdapter()
    a._expected_cache = {"HumanEval/0": _expected()}
    result = await a.evaluate(_task(), "code", None)

    assert len(calls) == 2
    # First call = base, second = plus.
    assert calls[0][0] == "humaneval"  # dataset
    assert calls[0][1] == "add"        # entry_point
    assert calls[0][2] == ((1, 2), (3, 4))
    assert calls[1][2] == ((100, 200),)

    assert result["pass_base"] is True
    assert result["pass_plus"] is True
    assert result["pass"] is True
    assert result["pass@1"] == 1


@pytest.mark.asyncio
async def test_evaluate_plus_fail_still_base_pass(monkeypatch):
    """base passes, plus fails → pass_plus is False; overall ``pass`` is False.

    This is the whole point of HumanEval+: the harder score drives reporting.
    """
    seq = iter([("pass", [True, True]), ("fail", [False])])
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: next(seq),
    )

    a = HumanEvalPlusAdapter()
    a._expected_cache = {"HumanEval/0": _expected()}
    out = await a.evaluate(_task(), "code", None)

    assert out["pass_base"] is True
    assert out["pass_plus"] is False
    assert out["pass"] is False
    assert out["pass@1"] == 0


@pytest.mark.asyncio
async def test_evaluate_base_fail_short_circuits_plus(monkeypatch):
    """If base fails, plus cannot salvage it regardless of plus's status."""
    seq = iter([("fail", [False, False]), ("pass", [True])])
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: next(seq),
    )

    a = HumanEvalPlusAdapter()
    a._expected_cache = {"HumanEval/0": _expected()}
    out = await a.evaluate(_task(), "code", None)

    assert out["pass_base"] is False
    assert out["pass_plus"] is False
    assert out["pass"] is False


# ---------------------------------------------------------------------------
# Groundtruth memoization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expected_cache_is_loaded_once(monkeypatch):
    """``get_groundtruth`` must only be called the first time _get_expected runs."""
    call_count = {"n": 0}

    def fake_get_groundtruth(problems, hashcode, tasks_only_output_not_none):
        call_count["n"] += 1
        # HumanEval passes an empty set — check that.
        assert tasks_only_output_not_none == set()
        return {k: _expected() for k in problems.keys()}

    # Monkey-patch at the place the adapter imports from.
    monkeypatch.setattr("evalplus.evaluate.get_groundtruth", fake_get_groundtruth)
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: ("pass", [True] * len(args[2])),
    )

    a = HumanEvalPlusAdapter()
    # Freshly constructed: cache is empty.
    assert a._expected_cache is None

    await a.evaluate(_task(task_id="HumanEval/0"), "x", None)
    await a.evaluate(_task(task_id="HumanEval/0"), "x", None)

    assert call_count["n"] == 1  # computed exactly once
    assert a._expected_cache is not None
