"""LiveCodeBench adapter — delegates to NVIDIA's packaged ``livecodebench``.

LiveCodeBench is a contamination-free contest-programming benchmark (Jain
et al., 2024 — https://arxiv.org/abs/2403.07974). Tasks are LeetCode,
AtCoder, and Codeforces problems timestamped by ``contest_date`` so new
releases can be filtered to exclude the model's training window.

Upstream publishes only a GitHub repo
(https://github.com/LiveCodeBench/LiveCodeBench) without a PyPI release;
NVIDIA packages the same code as ``nvidia-livecodebench`` (imports as
``livecodebench``). We delegate to that package so any upstream change to
prompt construction, evaluation, or test decoding is picked up — the Python
interface is identical.

Data
----
``livecodebench.benchmarks.code_generation.load_code_generation_dataset(
    release_version, start_date, end_date)`` returns ``list[CodeGenerationProblem]``.
Each problem's ``public_test_cases`` and ``private_test_cases`` arrive
base64 + zlib + pickle encoded; the loader decodes them. Date filtering
uses the problem's ``contest_date`` and is applied inside the adapter so
params like ``{"start_date": "2024-05-01"}`` match the upstream ``--start_date``
CLI flag.

Prompt
------
``livecodebench.prompts.code_generation.format_prompt_generation(problem, lm_style)``.
``lm_style`` defaults to :class:`~livecodebench.lm_styles.LMStyle.OpenAIChat`
— the chat-style template used in the reference leaderboard. Override via
``params["lm_style"]`` (any valid ``LMStyle`` name).

Evaluation
----------
``pass@1`` is the published metric at this release. We drive it through the
upstream batch grader so it stays exact:

1. Rebuild the grading sample via ``CodeGenerationProblem.get_evaluation_sample()``
   — a dict containing ``input_output`` (JSON-encoded inputs / outputs /
   optional ``fn_name``).
2. Call ``codegen_metrics([sample], [[code]], k_list=[1], num_process_evaluate=1)``.
   This forks into the upstream ``testing_util.run_test`` which handles both
   stdin/stdout and functional (``fn_name``) problem types, plus AtCoder
   stdin-via-argv quirks.
3. Extract ``metrics["pass@1"]`` → per-task ``pass@1 ∈ {0, 1}``. The engine
   averages these over tasks for the reported score.

Sandbox
-------
``requires_sandbox = False``. ``run_test`` spawns its own subprocess with a
per-test timeout and, on Unix, applies ``reliability_guard`` to block
destructive syscalls. Wrapping it in Docker would double-sandbox the run and
fight with the subprocess's own signal handling.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

if TYPE_CHECKING:
    from livecodebench.benchmarks.code_generation import CodeGenerationProblem


@register_adapter
class LiveCodeBenchAdapter(BenchmarkAdapter):
    name = "livecodebench"
    granularity = TaskGranularity.CONTEST
    # Upstream ``run_test`` sandboxes in its own subprocess; no Docker needed.
    requires_sandbox = False

    def __init__(self) -> None:
        # Maps question_id → CodeGenerationProblem. Populated by load_tasks and
        # reused by format_prompt / evaluate. Memoized so date-filter lookups
        # don't reparse the dataset.
        self._problems: dict[str, CodeGenerationProblem] = {}
        # Remember last-used params so format_prompt / evaluate don't need
        # them re-threaded through — the engine calls these after load_tasks.
        self._lm_style_name: str = "OpenAIChat"

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        from livecodebench.benchmarks.code_generation import (
            load_code_generation_dataset,
        )

        release_version = params.get("release_version", "release_v1")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        self._lm_style_name = params.get("lm_style", "OpenAIChat")

        problems = load_code_generation_dataset(
            release_version=release_version,
            start_date=start_date,
            end_date=end_date,
        )

        limit = params.get("limit")
        for i, problem in enumerate(problems):
            if limit and i >= limit:
                break
            self._problems[problem.question_id] = problem
            yield Task(
                task_id=problem.question_id,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=problem.question_content,
                metadata={
                    "question_title": problem.question_title,
                    "platform": problem.platform.value,
                    "contest_id": problem.contest_id,
                    "contest_date": problem.contest_date.isoformat()
                    if hasattr(problem.contest_date, "isoformat")
                    else str(problem.contest_date),
                    "difficulty": problem.difficulty.value,
                    "starter_code": problem.starter_code,
                    # Pre-serialised grading sample — survives task persistence
                    # without needing to pickle the full CodeGenerationProblem.
                    "evaluation_sample": problem.get_evaluation_sample(),
                    "func_name": problem.metadata.get("func_name"),
                    "release_version": release_version,
                },
                # Ground truth is the first canonical test, stashed here as
                # a stable stringified reference. Real grading uses the full
                # public + private suite inside ``evaluation_sample``.
                ground_truth=problem.starter_code,
            )

    def format_prompt(self, task: Task) -> str:
        from livecodebench.lm_styles import LMStyle
        from livecodebench.prompts.code_generation import format_prompt_generation

        problem = self._get_problem(task)
        lm_style = LMStyle[self._lm_style_name]
        return format_prompt_generation(problem, lm_style)

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        from livecodebench.evaluation.compute_code_generation_metrics import (
            codegen_metrics,
        )

        sample = task.metadata["evaluation_sample"]
        code = _extract_code(generated_output)

        timeout = int((task.metadata or {}).get("_timeout_seconds", 6))
        metrics, results, metadatas = codegen_metrics(
            [sample],
            [[code]],
            k_list=[1],
            num_process_evaluate=1,
            timeout=timeout,
            debug=False,
        )

        pass_at_1 = float(metrics.get("pass@1", 0.0)) / 100.0
        # ``results`` maps sample-index → list-of-lists of per-test statuses.
        # For k=1 there is exactly one sub-list; flatten to a [0/1] summary.
        passed_cases = 0
        total_cases = 0
        if results.get(0):
            run = results[0][0]
            if isinstance(run, list):
                total_cases = len(run)
                passed_cases = sum(1 for r in run if r is True)

        resolved = pass_at_1 >= 1.0
        return {
            "pass": resolved,
            "pass@1": 1 if resolved else 0,
            "pass_at_1_fraction": pass_at_1,
            "passed_cases": passed_cases,
            "total_cases": total_cases,
            "extracted_code": code[:4000],
            "upstream_metrics": metrics,
        }

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""

    def get_default_params(self) -> dict[str, Any]:
        return {"release_version": "release_v1", "lm_style": "OpenAIChat"}

    def _get_problem(self, task: Task) -> CodeGenerationProblem:
        """Look up the original ``CodeGenerationProblem`` for this task_id.

        Reloads the dataset on a cache miss — happens when ``format_prompt``
        or ``evaluate`` is called on a Task from a different run (e.g.
        rehydrated from disk). Keyed by ``question_id``.
        """
        problem = self._problems.get(task.task_id)
        if problem is not None:
            return problem

        from livecodebench.benchmarks.code_generation import (
            load_code_generation_dataset,
        )

        release_version = task.metadata.get("release_version", "release_v1")
        for p in load_code_generation_dataset(release_version=release_version):
            self._problems[p.question_id] = p
        return self._problems[task.task_id]


_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(completion: str) -> str:
    """Pull the first fenced ```python block; fall back to the raw text.

    Matches the extraction used by the upstream LiveCodeBench runner before
    calling ``run_test``.
    """
    matches = _CODE_FENCE_RE.findall(completion)
    return matches[0] if matches else completion
