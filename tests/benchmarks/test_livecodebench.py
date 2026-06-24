"""LiveCodeBench adapter tests — verifies delegation to ``livecodebench``.

The upstream ``codegen_metrics`` forks a pool of worker processes and, on
Unix, activates ``reliability_guard`` which imports the ``resource`` module.
The tests therefore monkey-patch ``codegen_metrics`` and the dataset loader
so the adapter's orchestration runs end-to-end without touching the real
grader. A couple of tests also stub ``format_prompt_generation`` so we can
assert delegation without requiring an LLM template library.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import pytest

from cross_bench.benchmarks.livecodebench import (
    LiveCodeBenchAdapter,
    _extract_code,
)
from cross_bench.models import Task, TaskGranularity


class _Platform(Enum):
    LEETCODE = "leetcode"


class _Difficulty(Enum):
    EASY = "easy"


class _StubProblem:
    """Minimal stand-in for ``CodeGenerationProblem`` — has the fields the
    adapter touches but no base64/zlib decoding requirement."""

    def __init__(self, question_id: str = "qid-1"):
        self.question_id = question_id
        self.question_title = f"Title {question_id}"
        self.question_content = f"Solve problem {question_id}."
        self.platform = _Platform.LEETCODE
        self.contest_id = "contest-42"
        self.contest_date = datetime(2024, 3, 15)
        self.starter_code = "class Solution:\n    def solve(self):\n        pass"
        self.difficulty = _Difficulty.EASY
        self.metadata = {"func_name": "solve"}

    def get_evaluation_sample(self) -> dict[str, str]:
        # The real method returns {"input_output": json.dumps(...)}.
        return {
            "input_output": (
                '{"inputs":["1\\n2\\n"],"outputs":["3\\n"],"fn_name":"solve"}'
            )
        }


def _task(qid: str = "qid-1") -> Task:
    """Task matching what ``load_tasks`` yields for a stub problem."""
    stub = _StubProblem(qid)
    return Task(
        task_id=qid,
        benchmark="livecodebench",
        granularity=TaskGranularity.CONTEST,
        prompt=stub.question_content,
        metadata={
            "question_title": stub.question_title,
            "platform": stub.platform.value,
            "contest_id": stub.contest_id,
            "contest_date": stub.contest_date.isoformat(),
            "difficulty": stub.difficulty.value,
            "starter_code": stub.starter_code,
            "evaluation_sample": stub.get_evaluation_sample(),
            "func_name": "solve",
            "release_version": "release_v1",
        },
        ground_truth=stub.starter_code,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("livecodebench")
    assert isinstance(a, LiveCodeBenchAdapter)
    assert a.name == "livecodebench"
    assert a.granularity == TaskGranularity.CONTEST
    # Upstream handles its own subprocess sandboxing.
    assert a.requires_sandbox is False


def test_get_sandbox_image_is_empty():
    assert LiveCodeBenchAdapter().get_sandbox_image() == ""


def test_default_params():
    p = LiveCodeBenchAdapter().get_default_params()
    assert p["release_version"] == "release_v1"
    assert p["lm_style"] == "OpenAIChat"


# ---------------------------------------------------------------------------
# Task loading — delegates to load_code_generation_dataset
# ---------------------------------------------------------------------------


def test_load_tasks_delegates_to_upstream_loader(monkeypatch):
    captured: dict[str, Any] = {}
    stubs = [_StubProblem("q1"), _StubProblem("q2")]

    def fake_loader(release_version, start_date=None, end_date=None):
        captured["release"] = release_version
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return stubs

    monkeypatch.setattr(
        "livecodebench.benchmarks.code_generation.load_code_generation_dataset",
        fake_loader,
    )

    a = LiveCodeBenchAdapter()
    tasks = list(a.load_tasks({
        "release_version": "release_v3",
        "start_date": "2024-05-01",
        "end_date": "2024-06-30",
    }))

    assert captured == {
        "release": "release_v3",
        "start_date": "2024-05-01",
        "end_date": "2024-06-30",
    }
    assert len(tasks) == 2
    assert tasks[0].task_id == "q1"
    assert tasks[1].task_id == "q2"


def test_load_tasks_respects_limit(monkeypatch):
    stubs = [_StubProblem(f"q{i}") for i in range(5)]
    monkeypatch.setattr(
        "livecodebench.benchmarks.code_generation.load_code_generation_dataset",
        lambda **kwargs: stubs,
    )
    a = LiveCodeBenchAdapter()
    tasks = list(a.load_tasks({"limit": 2}))
    assert len(tasks) == 2


def test_load_tasks_preserves_evaluation_sample(monkeypatch):
    monkeypatch.setattr(
        "livecodebench.benchmarks.code_generation.load_code_generation_dataset",
        lambda **kwargs: [_StubProblem("q1")],
    )
    a = LiveCodeBenchAdapter()
    tasks = list(a.load_tasks({}))
    md = tasks[0].metadata
    assert "input_output" in md["evaluation_sample"]
    assert md["platform"] == "leetcode"
    assert md["difficulty"] == "easy"
    assert md["contest_date"] == "2024-03-15T00:00:00"


# ---------------------------------------------------------------------------
# Prompt formatting — delegates to format_prompt_generation
# ---------------------------------------------------------------------------


def test_format_prompt_delegates_to_upstream(monkeypatch):
    captured = {}

    def fake_formatter(question, LanguageModelStyle):  # noqa: N803 - upstream kwarg name
        captured["question_id"] = question.question_id
        captured["style"] = LanguageModelStyle.value
        return f"<prompt for {question.question_id}>"

    monkeypatch.setattr(
        "livecodebench.prompts.code_generation.format_prompt_generation",
        fake_formatter,
    )
    monkeypatch.setattr(
        "livecodebench.benchmarks.code_generation.load_code_generation_dataset",
        lambda **kwargs: [_StubProblem("q1")],
    )

    a = LiveCodeBenchAdapter()
    list(a.load_tasks({"lm_style": "Claude3"}))
    out = a.format_prompt(_task("q1"))
    assert out == "<prompt for q1>"
    assert captured["question_id"] == "q1"
    assert captured["style"] == "Claude3"


# ---------------------------------------------------------------------------
# _extract_code: first ```python block wins; raw fallback
# ---------------------------------------------------------------------------


def test_extract_code_python_fence():
    out = _extract_code("blah\n```python\ndef f():\n    return 1\n```\ntrailing")
    assert "return 1" in out


def test_extract_code_bare_fence():
    out = _extract_code("intro\n```\nprint(2)\n```\nfoo")
    assert "print(2)" in out


def test_extract_code_no_fence_falls_back():
    assert _extract_code("def g(): return 3") == "def g(): return 3"


# ---------------------------------------------------------------------------
# Evaluate — calls codegen_metrics with the stashed sample + k_list=[1]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_calls_codegen_metrics_with_sample(monkeypatch):
    captured = {}

    def fake_metrics(samples_list, generations_list, k_list, num_process_evaluate, timeout, debug):
        captured["samples"] = samples_list
        captured["generations"] = generations_list
        captured["k_list"] = k_list
        captured["timeout"] = timeout
        # Two test cases, both passed.
        return ({"pass@1": 100.0, "detail": {}}, {0: [[True, True]]}, [])

    monkeypatch.setattr(
        "livecodebench.evaluation.compute_code_generation_metrics.codegen_metrics",
        fake_metrics,
    )

    a = LiveCodeBenchAdapter()
    t = _task()
    out = await a.evaluate(t, "```python\nprint(1)\n```", None)

    assert captured["k_list"] == [1]
    assert captured["samples"] == [t.metadata["evaluation_sample"]]
    # Code was extracted from the fence before being passed in.
    assert captured["generations"] == [["print(1)\n"]]
    assert out["pass"] is True
    assert out["pass@1"] == 1
    assert out["pass_at_1_fraction"] == 1.0
    assert out["passed_cases"] == 2
    assert out["total_cases"] == 2


@pytest.mark.asyncio
async def test_evaluate_fail(monkeypatch):
    def fake_metrics(samples_list, generations_list, k_list, num_process_evaluate, timeout, debug):
        return ({"pass@1": 0.0, "detail": {}}, {0: [[False, True]]}, [])

    monkeypatch.setattr(
        "livecodebench.evaluation.compute_code_generation_metrics.codegen_metrics",
        fake_metrics,
    )

    a = LiveCodeBenchAdapter()
    out = await a.evaluate(_task(), "```python\nprint(0)\n```", None)
    assert out["pass"] is False
    assert out["pass@1"] == 0
    assert out["pass_at_1_fraction"] == 0.0
    assert out["passed_cases"] == 1
    assert out["total_cases"] == 2


@pytest.mark.asyncio
async def test_evaluate_honors_timeout_override(monkeypatch):
    captured = {}

    def fake_metrics(samples_list, generations_list, k_list, num_process_evaluate, timeout, debug):
        captured["timeout"] = timeout
        return ({"pass@1": 100.0, "detail": {}}, {0: [[True]]}, [])

    monkeypatch.setattr(
        "livecodebench.evaluation.compute_code_generation_metrics.codegen_metrics",
        fake_metrics,
    )

    a = LiveCodeBenchAdapter()
    t = _task()
    t.metadata["_timeout_seconds"] = 42
    await a.evaluate(t, "code", None)
    assert captured["timeout"] == 42
