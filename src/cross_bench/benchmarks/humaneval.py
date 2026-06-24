"""HumanEval benchmark adapter — delegates to the ``human-eval`` package.

Faithful port of the functional-correctness evaluation:
  https://github.com/openai/human-eval          — reference evaluator
  https://github.com/openai/simple-evals        — chat-model wrapper & find_code

The 164 problems are loaded via :func:`human_eval.data.read_problems` from the
``HumanEval.jsonl.gz`` bundled with the package, so the task set is
byte-identical to the upstream dataset. The check-program construction mirrors
:func:`human_eval.execution.check_correctness` — same concatenation order, same
``check(entry_point)`` trailer. Scoring uses
:func:`human_eval.evaluation.estimate_pass_at_k` with ``n=1, c∈{0,1}, k=1``,
which reduces to the identity ``pass@1 = c``.

Prompt
------
simple-evals instruction prefix (verbatim) concatenated with the raw prompt::

    "Read the following function signature and docstring, and fully implement
    the function described. Your response should only contain the code for
    this function.\\n" + prompt

Code extraction (``_find_code``) — byte-for-byte port of simple-evals::

    pattern = re.compile(r"```python\\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted = matches[0] if matches else completion
    extracted = extracted[extracted.find(":\\n    ") + 2 :]  # strip signature

The upstream quirk where ``find`` returns ``-1`` (dropping the first character
of the completion) is preserved — changing it would diverge from published
simple-evals scores.

Execution
---------
Matches ``human_eval.execution.unsafe_execute`` exactly::

    check_program = prompt + completion + "\\n" + test + "\\n" + f"check({entry_point})"

Upstream runs this under ``exec`` in a guarded subprocess with a SIGALRM
timeout and ``reliability_guard``. We run it inside a Docker sandbox
(``cross-bench-function:latest``) — OS-level isolation replaces the per-call
monkey-patching. Non-zero exit or timeout = failure, same as upstream.

Metrics
-------
Per task: ``pass`` (bool) and ``pass@1`` (0/1, computed via
``estimate_pass_at_k(1, [c], 1)``). The engine averages ``pass@1`` over tasks
to produce the standard reported metric.

Simplifications vs. upstream:
  - Single sample per task. Upstream simple-evals draws ``num_samples_per_task=5``
    and reports ``pass@{1,2,5}`` via the unbiased Codex estimator
    ``1 − ∏_{i=n−c+1..n}(1 − k/i)``. With ``n=1`` the estimator reduces to
    ``pass@1 = c``, which the flat mean over tasks yields exactly.
  - Sandbox: Docker container instead of guarded subprocess. The
    ``check_program`` construction and assertion semantics are identical.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

# Matches simple-evals ``humaneval_eval.py`` line 66 verbatim (ends with "\n").
_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
)


@register_adapter
class HumanEvalAdapter(BenchmarkAdapter):
    name = "humaneval"
    granularity = TaskGranularity.FUNCTION

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        from human_eval.data import read_problems

        problems = read_problems()
        limit = params.get("limit")
        for i, (task_id, problem) in enumerate(problems.items()):
            if limit and i >= limit:
                break
            yield Task(
                task_id=task_id,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=problem["prompt"],
                metadata={
                    "entry_point": problem["entry_point"],
                    "canonical_solution": problem["canonical_solution"],
                },
                ground_truth=problem["test"],
            )

    def format_prompt(self, task: Task) -> str:
        """Wrap the raw prompt with the simple-evals instruction."""
        return _INSTRUCTION + task.prompt

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        from human_eval.evaluation import estimate_pass_at_k

        entry_point = task.metadata["entry_point"]
        test_code = task.ground_truth

        completion = _find_code(generated_output)

        # Matches ``human_eval.execution.unsafe_execute`` verbatim.
        check_program = (
            task.prompt
            + completion
            + "\n"
            + test_code
            + "\n"
            + f"check({entry_point})"
        )

        if sandbox is None:
            sandbox = await sandbox_mgr.create(self.get_sandbox_image(task), task)

        timeout = int((task.metadata or {}).get("_timeout_seconds", 30))
        result = await sandbox_mgr.execute(
            sandbox, f"python3 -c {_shell_quote(check_program)}", timeout=timeout
        )

        passed = result.exit_code == 0
        pass_at_1 = float(estimate_pass_at_k(1, [1 if passed else 0], 1)[0])
        return {
            "pass": passed,
            "pass@1": pass_at_1,
            "exit_code": result.exit_code,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
        }

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return "cross-bench-function:latest"

    def get_default_params(self) -> dict[str, Any]:
        return {}


def _find_code(completion: str) -> str:
    """Byte-for-byte port of simple-evals ``humaneval_eval.py`` ``find_code``.

    Extracts the first ``\u0060\u0060\u0060python\\n...\u0060\u0060\u0060`` block (falling back to
    the raw completion), then strips the repeated function signature by
    slicing after ``:\\n    ``.

    Upstream quirk preserved: when ``:\\n    `` is absent, ``str.find``
    returns ``-1`` and the slice becomes ``[1:]`` — silently dropping the
    first character. Fixing it would diverge from published simple-evals
    scores on completions that omit the signature.
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    extracted_answer = extracted_answer[extracted_answer.find(":\n    ") + 2:]
    return extracted_answer


def _shell_quote(s: str) -> str:
    """POSIX single-quote escaping: ``'…'\\''…'``."""
    return "'" + s.replace("'", "'\\''") + "'"
