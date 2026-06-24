"""MBPP benchmark adapter — 3-shot port of inspect_evals MBPP.

Faithful port of the inspect_evals scorer and prompt:
  https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py

Paper: Austin et al., 2021 — https://arxiv.org/abs/2108.07732
Upstream repo: https://github.com/google-research/google-research/tree/master/mbpp

Dataset
-------
HuggingFace: ``google-research-datasets/mbpp`` with config ``sanitized``,
split ``test`` (~257 tasks) pinned to revision
``4bb6404fdc6cacfda99d4ac4205087b89d32030c``. Schema:
  ``task_id``, ``prompt``, ``code``, ``source_file``,
  ``test_imports`` (list[str]), ``test_list`` (list[str]).

Few-shot examples come from config ``full`` split ``prompt``, filtered to
``task_id in [2, 3, 4]`` per AgentCoder's MBPP prompt
(https://github.com/huangd1999/AgentCoder/blob/main/prompts/mbpp_prompt.txt).

Prompt
------
Template is built once at load time and cached. Structure (byte-identical
to inspect_evals)::

    <system instruction>

    For example:

    ## Prompt 1
    ```python
    <text>
    ```

    ## Test Case 1
    ```python
    <joined test_list>
    ```

    ## Completion 1
    ```python
    <code>
    ```

    ... (x3 few-shot blocks) ...

    # Now, do it for the following task.

    ## Prompt:
    ```python
    {prompt}
    ```

    ## Test Case:
    ```python
    {test_list_str}
    ```

    ## Completion:

Substitution uses ``str.replace`` rather than ``str.format`` so stray
braces inside few-shot code can't break it. Upstream relies on inspect_ai's
``prompt_template`` which uses ``.format`` — the task_ids 2/3/4 happen to
contain no literal braces, so outputs agree byte-for-byte.

Code extraction (``_find_code``) — byte-for-byte port::

    pattern = re.compile(r"```python\\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted = matches[0] if matches else completion

Execution
---------
For each assertion in ``test_list``, upstream appends a companion tuple
that surfaces the failing condition when the assert raises::

    code += f"\\n{test_case}, {repr(test_case[len('assert '):])}"

The combined snippet is run under ``python3 -c`` inside a Docker sandbox
(``cross-bench-function:latest``). Non-zero exit or timeout = failure.

Metrics
-------
Per task: ``pass`` (bool) and ``pass@1`` (0/1). The engine averages
``pass@1`` over tasks to produce the reported metric.

Simplifications vs. upstream:
  - Single sample per task. Upstream replays the dataset ``NUM_EPOCHS=5``
    times and reports ``mean / pass@1 / pass@2 / pass@5``. With n=1 these
    all collapse to pass@1, which is what the flat mean over tasks yields.
  - Sandbox: Docker ``python3 -c`` instead of inspect_ai's sandbox API;
    the executed code is identical.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

DATASET_PATH = "google-research-datasets/mbpp"
# Pinned upstream revision — matches inspect_evals for reproducibility.
MBPP_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

# System instruction — resolves byte-identical to inspect_evals
# PROMPT_TEMPLATE.strip(). Split across implicit-concat string literals so
# the source line length stays under the lint limit; newlines in the rendered
# string are explicit "\n".
_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a task, "
    "and the tests that your code must pass.\n"
    "Write the Python function to solve the task. "
    "Do not give additional explanations, just output the\n"
    "Python function. Only use imports that are included in "
    "Python's standard library."
)

_FEW_SHOT_TASK_IDS = [2, 3, 4]

# Per-task tail appended after the few-shot blocks. Matches inspect_evals
# textwrap.dedent call (leading 4-space indent stripped).
_TASK_TAIL = textwrap.dedent(
    """
    # Now, do it for the following task.

    ## Prompt:
    ```python
    {prompt}
    ```

    ## Test Case:
    ```python
    {test_list_str}
    ```

    ## Completion:
    """
)

_template_cache: str | None = None


def _build_template() -> str:
    """Build the 3-shot template once and cache it module-wide."""
    global _template_cache
    if _template_cache is not None:
        return _template_cache

    template = _SYSTEM_PROMPT + "\n\nFor example:\n\n"
    few_shot = load_dataset(
        DATASET_PATH,
        "full",
        split="prompt",
        revision=MBPP_DATASET_REVISION,
    )
    few_shot = few_shot.filter(lambda row: row["task_id"] in _FEW_SHOT_TASK_IDS)
    for i, row in enumerate(few_shot):
        test_cases = "\n".join(row["test_list"])
        template += (
            f"## Prompt {i + 1}\n"
            "```python\n"
            f"{row['text']}\n"
            "```\n\n"
            f"## Test Case {i + 1}\n"
            "```python\n"
            f"{test_cases}\n```\n\n"
            f"## Completion {i + 1}\n"
            "```python\n"
            f"{row['code']}\n```\n\n"
        )
    template += _TASK_TAIL

    _template_cache = template
    return template


@register_adapter
class MBPPAdapter(BenchmarkAdapter):
    name = "mbpp"
    granularity = TaskGranularity.FUNCTION

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        ds = load_dataset(
            DATASET_PATH,
            "sanitized",
            split="test",
            revision=MBPP_DATASET_REVISION,
        )
        limit = params.get("limit")
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            yield Task(
                task_id=str(row["task_id"]),
                benchmark=self.name,
                granularity=self.granularity,
                prompt=row["prompt"],
                ground_truth=row["test_list"],
                metadata={
                    "source_file": row.get("source_file"),
                    "test_imports": row.get("test_imports"),
                    "code": row["code"],
                    "test_list_str": "\n".join(row["test_list"]),
                },
            )

    def format_prompt(self, task: Task) -> str:
        """Substitute {prompt} and {test_list_str} into the cached template."""
        template = _build_template()
        return template.replace(
            "{prompt}", task.prompt
        ).replace(
            "{test_list_str}", task.metadata["test_list_str"]
        )

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        code = _find_code(generated_output)

        # Append each assert plus a repr() of the condition for better
        # traceback messages. Byte-identical to inspect_evals verify().
        for test_case in task.ground_truth:
            code += f"\n{test_case}, {repr(test_case[len('assert '):])}"

        if sandbox is None:
            sandbox = await sandbox_mgr.create(self.get_sandbox_image(task), task)

        timeout = int((task.metadata or {}).get("_timeout_seconds", 30))
        result = await sandbox_mgr.execute(
            sandbox,
            f"python3 -c {_shell_quote(code)}",
            timeout=timeout,
        )
        passed = result.exit_code == 0
        return {
            "pass": passed,
            "pass@1": 1 if passed else 0,
            "exit_code": result.exit_code,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
        }

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return "cross-bench-function:latest"

    def get_default_params(self) -> dict[str, Any]:
        return {}


def _find_code(completion: str) -> str:
    """Byte-for-byte port of inspect_evals ``find_code``.

    Extracts the first ``\u0060\u0060\u0060python\\n...\u0060\u0060\u0060`` block; falls back to the
    raw completion when no fenced block is present.
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    return str(extracted_answer)


def _shell_quote(s: str) -> str:
    """POSIX single-quote escaping: ``'…'\\''…'``."""
    return "'" + s.replace("'", "'\\''") + "'"
