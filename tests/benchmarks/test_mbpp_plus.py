"""MBPP+ adapter tests — verifies delegation to the ``evalplus`` package.

Same pattern as ``test_humaneval_plus.py``: ``untrusted_check`` and
``sanitize`` are monkey-patched so the scoring logic runs end-to-end without
spawning the reliability-guarded subprocess (which needs Unix ``resource``).
``get_groundtruth`` is patched as well to avoid minutes of canonical-solution
replay.
"""

from __future__ import annotations

from typing import Any

import pytest

from cross_bench.benchmarks.mbpp_plus import MBPPPlusAdapter
from cross_bench.models import Task, TaskGranularity


def _task(
    *,
    task_id: str = "Mbpp/2",
    entry_point: str = "similar_elements",
    assertion: str = "assert similar_elements((3,4,5), (5,7,3)) == (3,5)",
) -> Task:
    return Task(
        task_id=task_id,
        benchmark="mbpp_plus",
        granularity=TaskGranularity.FUNCTION,
        prompt="Write a function to find the similar elements.",
        ground_truth="def check(c): pass",
        metadata={
            "entry_point": entry_point,
            "canonical_solution": "    return tuple(set(a) & set(b))\n",
            "atol": 0.0,
            "base_input": [[(3, 4, 5), (5, 7, 3)]],
            "plus_input": [[(1,), (1,)]],
            "contract": "",
            "assertion": assertion,
        },
    )


def _expected() -> dict[str, Any]:
    return {
        "base": [(3, 5)],
        "base_time": [0.01],
        "plus": [(1,)],
        "plus_time": [0.01],
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("mbpp_plus")
    assert isinstance(a, MBPPPlusAdapter)
    assert a.name == "mbpp_plus"
    assert a.granularity == TaskGranularity.FUNCTION
    assert a.requires_sandbox is False


def test_get_sandbox_image_is_empty():
    assert MBPPPlusAdapter().get_sandbox_image() == ""


# ---------------------------------------------------------------------------
# Task loading — MBPP+ has 378 problems and each needs base/plus fuel
# ---------------------------------------------------------------------------


def test_load_tasks_full_size():
    """MBPP+ has exactly 378 problems from the sanitized subset."""
    a = MBPPPlusAdapter()
    tasks = list(a.load_tasks({}))
    assert len(tasks) == 378


def test_load_tasks_respects_limit():
    a = MBPPPlusAdapter()
    tasks = list(a.load_tasks({"limit": 7}))
    assert len(tasks) == 7


def test_load_tasks_preserves_plus_fields():
    a = MBPPPlusAdapter()
    tasks = list(a.load_tasks({"limit": 2}))
    t = tasks[0]
    md = t.metadata
    for key in (
        "entry_point",
        "base_input",
        "plus_input",
        "atol",
        "canonical_solution",
        "assertion",
    ):
        assert key in md, f"missing: {key}"
    assert len(md["plus_input"]) > 0


# ---------------------------------------------------------------------------
# Prompt formatting — appends the concrete assertion example
# ---------------------------------------------------------------------------


def test_format_prompt_appends_assertion():
    """Upstream EvalPlus CLI prompts include the first assertion as an example."""
    a = MBPPPlusAdapter()
    out = a.format_prompt(_task(assertion="assert similar_elements((3,4,5), (5,7,3)) == (3,5)"))
    assert "Write a function" in out
    assert "assert similar_elements((3,4,5), (5,7,3)) == (3,5)" in out
    # Terminates the docstring-style prompt so downstream code completes cleanly.
    assert out.endswith('"""\n')


def test_format_prompt_no_assertion():
    """If somehow assertion is missing, prompt falls back to the raw text."""
    t = _task()
    t.metadata["assertion"] = ""
    a = MBPPPlusAdapter()
    out = a.format_prompt(t)
    assert out == t.prompt


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_calls_untrusted_check_with_mbpp_dataset(monkeypatch):
    """The ``dataset`` arg must be ``"mbpp"`` — it selects the grading policy
    for MBPP_OUTPUT_NOT_NONE_TASKS and atol handling inside evalplus."""
    datasets_seen: list[str] = []

    def fake_untrusted_check(
        dataset, code, inputs, entry_point, expected, atol, ref_time, **kwargs
    ):
        datasets_seen.append(dataset)
        return ("pass", [True] * len(inputs))

    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr("evalplus.eval.untrusted_check", fake_untrusted_check)

    a = MBPPPlusAdapter()
    a._expected_cache = {"Mbpp/2": _expected()}
    await a.evaluate(_task(), "code", None)

    assert datasets_seen == ["mbpp", "mbpp"]


@pytest.mark.asyncio
async def test_evaluate_pass_plus(monkeypatch):
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: ("pass", [True] * len(args[2])),
    )

    a = MBPPPlusAdapter()
    a._expected_cache = {"Mbpp/2": _expected()}
    out = await a.evaluate(_task(), "code", None)
    assert out["pass"] is True
    assert out["pass@1"] == 1
    assert out["pass_base"] is True
    assert out["pass_plus"] is True


@pytest.mark.asyncio
async def test_evaluate_plus_fail(monkeypatch):
    seq = iter([("pass", [True]), ("fail", [False])])
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: next(seq),
    )

    a = MBPPPlusAdapter()
    a._expected_cache = {"Mbpp/2": _expected()}
    out = await a.evaluate(_task(), "code", None)
    assert out["pass_base"] is True
    assert out["pass_plus"] is False
    assert out["pass"] is False


# ---------------------------------------------------------------------------
# Groundtruth memoization passes MBPP_OUTPUT_NOT_NONE_TASKS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expected_cache_passes_mbpp_output_not_none_tasks(monkeypatch):
    """MBPP+ must pass ``MBPP_OUTPUT_NOT_NONE_TASKS`` so the 3 mutating-arg
    tasks (``check_str``, ``text_match_three``, ``text_starta_endb``) are
    graded on non-None output rather than equality."""
    from evalplus.eval import MBPP_OUTPUT_NOT_NONE_TASKS

    call_count = {"n": 0}
    captured_set: list[Any] = []

    def fake_get_groundtruth(problems, hashcode, tasks_only_output_not_none):
        call_count["n"] += 1
        captured_set.append(tasks_only_output_not_none)
        return {k: _expected() for k in problems.keys()}

    monkeypatch.setattr("evalplus.evaluate.get_groundtruth", fake_get_groundtruth)
    monkeypatch.setattr("evalplus.sanitize.sanitize", lambda c, entrypoint=None: c)
    monkeypatch.setattr(
        "evalplus.eval.untrusted_check",
        lambda *args, **kwargs: ("pass", [True] * len(args[2])),
    )

    a = MBPPPlusAdapter()
    await a.evaluate(_task(task_id="Mbpp/2"), "x", None)
    await a.evaluate(_task(task_id="Mbpp/2"), "x", None)

    assert call_count["n"] == 1  # memoized
    assert captured_set[0] == MBPP_OUTPUT_NOT_NONE_TASKS
