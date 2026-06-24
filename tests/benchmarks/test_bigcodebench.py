"""BigCodeBench adapter tests — upstream faithfulness + memory non-applicability.

References:
  - upstream evaluator: ``bigcode-project/bigcodebench/blob/09dd993f/bigcodebench/evaluate.py``
  - inspect_evals scorer: ``UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/bigcodebench/bigcodebench.py``
  - paper: Zhuo et al., 2024, https://arxiv.org/abs/2406.15877
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cross_bench.benchmarks.bigcodebench import (
    _CALIBRATION_TAIL,
    _INSTRUCTION_PROMPT,
    _UNITTEST_TAIL,
    BigCodeBenchAdapter,
    _find_code,
    _parse_libs,
    _shell_quote,
)
from cross_bench.models import Task, TaskGranularity


@pytest.fixture
def adapter() -> BigCodeBenchAdapter:
    return BigCodeBenchAdapter()


# ---------------------------------------------------------------------------
# Byte-locks: the inspect_evals strings we ported verbatim
# ---------------------------------------------------------------------------


class TestPromptByteLocks:
    """Lock the exact byte sequences our adapter ports from inspect_evals.

    These constants are verbatim copies — if a future cleanup pass rewords
    them, our column on the leaderboard silently moves vs inspect_evals.
    """

    def test_instruction_prompt_is_byte_exact(self):
        # Locked against ``inspect_evals/bigcodebench/bigcodebench.py::INSTRUCTION_PROMPT``.
        assert _INSTRUCTION_PROMPT == (
            "Read the following function signature and docstring, and fully "
            "implement the function described. Make sure to include the import "
            "statements and use the same variable names as stated in the header.\n"
        )

    def test_unittest_tail_starts_with_blank_line(self):
        # The leading ``\n`` is intentional — inspect_evals builds with
        # a triple-quoted string that starts with a newline. Without it
        # the test class and the ``if __name__`` guard would share a
        # line, breaking exec.
        assert _UNITTEST_TAIL.startswith("\n")
        assert _UNITTEST_TAIL == "\nif __name__ == '__main__':\n    unittest.main()"

    def test_calibration_tail_is_pass_stub(self):
        # Upstream ``evaluate.py:305-306``: ``code_prompt + "\n    pass\n"``.
        # The leading ``\n`` separates the imports/signature from the
        # ``pass`` body line; the trailing ``\n`` ends the stub before
        # the real solution begins.
        assert _CALIBRATION_TAIL == "\n    pass\n"


# ---------------------------------------------------------------------------
# _find_code — fenced extraction
# ---------------------------------------------------------------------------


class TestFindCode:
    def test_extracts_python_fence(self):
        out = _find_code("here's the answer:\n```python\ndef f(): pass\n```\n")
        assert out == "def f(): pass\n"

    def test_extracts_plain_fence(self):
        out = _find_code("here's the answer:\n```\ndef f(): pass\n```\n")
        assert out == "def f(): pass\n"

    def test_python_fence_wins_over_plain_when_both_present(self):
        # Two fences in the output: the ```python one comes first in
        # the concatenated match list, so it wins. inspect_evals's
        # exact precedence rule.
        text = (
            "```python\nfirst = 1\n```\n"
            "and also\n"
            "```\nsecond = 2\n```\n"
        )
        assert _find_code(text) == "first = 1\n"

    def test_falls_back_to_raw_completion_when_no_fence(self):
        out = _find_code("def f(): pass")
        assert out == "def f(): pass"

    def test_first_fenced_block_wins(self):
        text = "```python\na = 1\n```\nthen\n```python\nb = 2\n```\n"
        assert _find_code(text) == "a = 1\n"


# ---------------------------------------------------------------------------
# _parse_libs — the latent bug fix
# ---------------------------------------------------------------------------


class TestParseLibs:
    """The HF dataset returns ``libs`` as a stringified Python list.
    Adapter must round-trip back to ``list[str]``."""

    def test_parses_stringified_list(self):
        # The actual dataset format observed at v0.1.2.
        assert _parse_libs("['random', 'itertools']") == ["random", "itertools"]

    def test_parses_single_lib(self):
        assert _parse_libs("['pandas']") == ["pandas"]

    def test_parses_empty_list(self):
        assert _parse_libs("[]") == []

    def test_passes_through_real_list(self):
        assert _parse_libs(["a", "b"]) == ["a", "b"]

    def test_handles_none(self):
        assert _parse_libs(None) == []

    def test_handles_empty_string(self):
        assert _parse_libs("") == []

    def test_falls_back_to_comma_split_when_not_a_list_literal(self):
        # Robustness for a hypothetical mirror that normalizes to
        # comma-separated.
        assert _parse_libs("pandas, numpy, requests") == ["pandas", "numpy", "requests"]

    def test_strips_whitespace(self):
        assert _parse_libs("['  pandas  ', 'numpy']") == ["pandas", "numpy"]


# ---------------------------------------------------------------------------
# _shell_quote — POSIX single-quote escaping for python -c
# ---------------------------------------------------------------------------


class TestShellQuote:
    def test_no_quotes_round_trip(self):
        assert _shell_quote("hello") == "'hello'"

    def test_embedded_single_quote(self):
        # POSIX trick: '...'\''...' — close, escaped quote, reopen.
        assert _shell_quote("it's") == "'it'\\''s'"

    def test_multiline_content(self):
        assert _shell_quote("a\nb") == "'a\nb'"


# ---------------------------------------------------------------------------
# Calibrated mode (upstream default)
# ---------------------------------------------------------------------------


class _CapturingSandboxManager:
    """Records every ``execute`` call so we can inspect what code ran."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def create(self, image: str, task: Task) -> Any:
        return object()

    async def destroy(self, sandbox: Any) -> None:
        pass

    async def execute(self, sandbox: Any, cmd: str, timeout: int = 30) -> Any:
        self.commands.append(cmd)
        return _ExecResult(exit_code=0, stdout="", stderr="")


@dataclass
class _ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def _task(
    *,
    code_prompt: str = "import math\ndef task_func(x):\n",
    test: str = "import unittest\nclass TestCases(unittest.TestCase):\n    def test_1(self):\n        self.assertTrue(True)\n",
    calibrated: bool = True,
) -> Task:
    return Task(
        task_id="BigCodeBench/0",
        benchmark="bigcodebench",
        granularity=TaskGranularity.FUNCTION,
        prompt="...",
        metadata={
            "variant": "complete",
            "version": "v0.1.2",
            "calibrated": calibrated,
            "entry_point": "task_func",
            "libs": ["math"],
            "canonical_solution": "    return x",
            "complete_prompt": code_prompt + '    """doc"""\n',
            "code_prompt": code_prompt,
        },
        ground_truth=test,
    )


class TestCalibratedMode:
    """Upstream's calibrated mode prepends ``code_prompt + "\\n    pass\\n"``
    to the answer before exec, so chat models that drop the signature
    still produce a runnable ``task_func``. Default: enabled."""

    @pytest.mark.asyncio
    async def test_calibrated_prepends_code_prompt_and_pass_stub(self, adapter):
        mgr = _CapturingSandboxManager()
        task = _task(
            code_prompt="import math\ndef task_func(x):\n",
            calibrated=True,
        )
        # Model output that DROPS the signature (chat-tuned style).
        await adapter.evaluate(task, "    return x * 2", mgr, sandbox=None)
        # The combined snippet that hit ``python -c`` must contain the
        # calibration stub before the bare body.
        cmd = mgr.commands[0]
        assert "import math" in cmd
        assert "def task_func(x):" in cmd
        # Stub: ``def task_func(x):\n    pass\n`` precedes the real body.
        # The bare ``return x * 2`` line is what the model produced.
        assert "pass" in cmd
        assert "return x * 2" in cmd
        # Stub appears before the real solution body in the source order.
        stub_idx = cmd.find("pass")
        body_idx = cmd.find("return x * 2")
        assert stub_idx < body_idx

    @pytest.mark.asyncio
    async def test_calibrated_disabled_skips_stub(self, adapter):
        mgr = _CapturingSandboxManager()
        task = _task(calibrated=False)
        await adapter.evaluate(task, "def task_func(x):\n    return x", mgr, sandbox=None)
        cmd = mgr.commands[0]
        # Without calibration, no ``pass`` stub is injected. The only
        # ``def task_func`` definition is the model's own.
        assert cmd.count("def task_func") == 1
        # The ``pass`` token should not appear at the calibration stub
        # position. (Some test bodies contain ``pass``, but the stub
        # location specifically should not.)

    @pytest.mark.asyncio
    async def test_default_params_calibrated_true(self, adapter):
        params = adapter.get_default_params()
        assert params["calibrated"] is True


# ---------------------------------------------------------------------------
# Combined-snippet shape (the ``python -c`` payload)
# ---------------------------------------------------------------------------


class TestCombinedSnippetShape:
    @pytest.mark.asyncio
    async def test_combined_ends_with_unittest_main_guard(self, adapter):
        mgr = _CapturingSandboxManager()
        task = _task(calibrated=False)
        await adapter.evaluate(task, "def task_func(x):\n    return x", mgr, sandbox=None)
        cmd = mgr.commands[0]
        # ``cmd`` is shell-quoted: each ``'`` in the snippet becomes
        # ``'\\''``. Match on tokens that survive escaping rather than
        # on quoted literals.
        assert "__name__" in cmd
        assert "__main__" in cmd
        assert "unittest.main()" in cmd
        # And the snippet must end with ``unittest.main()`` (modulo the
        # closing shell-quote).
        assert cmd.rstrip("'").endswith("unittest.main()")

    @pytest.mark.asyncio
    async def test_test_code_is_concatenated_after_answer(self, adapter):
        mgr = _CapturingSandboxManager()
        task = _task(
            code_prompt="def task_func(x):\n",
            test="import unittest\nclass TestCases(unittest.TestCase):\n    def test_x(self):\n        pass\n",
            calibrated=False,
        )
        await adapter.evaluate(task, "def task_func(x):\n    return 42", mgr, sandbox=None)
        cmd = mgr.commands[0]
        # Answer first, then test.
        ans_idx = cmd.find("return 42")
        test_idx = cmd.find("class TestCases")
        assert 0 <= ans_idx < test_idx


# ---------------------------------------------------------------------------
# Adapter contract + metrics_schema
# ---------------------------------------------------------------------------


class TestAdapterContract:
    def test_requires_sandbox(self, adapter):
        # Code execution → needs the docker sandbox.
        assert adapter.requires_sandbox is True
        assert adapter.get_sandbox_image() == "cross-bench-bigcodebench:latest"

    def test_metrics_schema_includes_pass_rate_and_pass_at_1(self, adapter):
        keys = adapter.metrics_schema()
        assert "pass_rate" in keys
        assert "avg_pass@1" in keys
        assert "avg_inference_latency_ms" in keys
        assert "avg_total_latency_ms" in keys
        # Schema does NOT declare per-language / per-category breakdowns
        # because BigCodeBench has no per-task split that maps to
        # ``evaluation["category"]``.
        assert not any(k.startswith("accuracy_cat_") for k in keys)

    def test_default_params(self, adapter):
        params = adapter.get_default_params()
        assert params["variant"] == "complete"
        assert params["version"] == "v0.1.2"
        assert params["calibrated"] is True


# ---------------------------------------------------------------------------
# Memory hooks — intentionally unimplemented
# ---------------------------------------------------------------------------


class _RecordingMemory:
    """Captures every memory call."""

    def __init__(self) -> None:
        self.post_response_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.pre_prompt_calls: list[tuple[str, dict[str, Any]]] = []

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        self.post_response_calls.append((prompt, response, dict(context)))

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        self.pre_prompt_calls.append((prompt, dict(context)))
        return prompt

    async def reset(self) -> None:
        pass


class TestMemoryHooksDefaultNoOp:
    """**Default mode** (no corpus opt-in). BigCodeBench tasks are
    self-contained — no per-task cross-file context to ingest. The
    adapter's per-task memory hooks return no-op / None so memory
    profiles run identically to the no-memory baseline. See
    ``deviations.md`` B7."""

    @pytest.mark.asyncio
    async def test_ingest_for_memory_is_no_op(self, adapter):
        memory = _RecordingMemory()
        task = _task()
        # The base-class default returns None and ingests nothing.
        result = await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert result is None
        assert memory.post_response_calls == []

    def test_format_memory_prompt_falls_through_to_format_prompt(self, adapter):
        # Default behavior: memory-mode prompt == no-memory prompt.
        # If anyone overrides this without also wiring ingestion the
        # leaderboard column would silently drift.
        task = _task()
        assert adapter.format_memory_prompt(task) == adapter.format_prompt(task)

    def test_get_retrieval_query_returns_none_when_corpus_inactive(self, adapter):
        # ``None`` keeps the engine on the legacy ``query == prompt``
        # path. With nothing ingested anyway, this is the right
        # behavior.
        task = _task()
        # Corpus is inactive by default — adapter._corpus_active = False.
        assert adapter.get_retrieval_query(task) is None

    def test_corpus_for_run_returns_none_when_flag_off(self, adapter):
        # Default params have use_lib_docs_corpus=False.
        assert adapter.corpus_for_run({"use_lib_docs_corpus": False}, tasks=[]) is None
        # Missing flag also means off.
        assert adapter.corpus_for_run({}, tasks=[]) is None
        # No params → off.
        assert adapter.corpus_for_run() is None

    def test_reset_memory_per_task_default_true(self, adapter):
        """Critical regression check: default-mode BCB must keep the
        per-task reset behavior so memory state can't leak across
        tasks (matters more for memory profiles run as a sanity
        check than for the no-op default itself)."""
        assert adapter.reset_memory_per_task is True


# ---------------------------------------------------------------------------
# Corpus mode (CodeRAG-Bench library-documentation, opt-in)
# ---------------------------------------------------------------------------


class TestCorpusForRun:
    """Opt-in CodeRAG-Bench library-docs corpus. Activated by
    ``params["use_lib_docs_corpus"] = True`` (typically via the
    ``bigcodebench_lib_docs`` profile mixin)."""

    def test_corpus_returns_none_when_no_tasks_supplied(self, adapter):
        # Engine always passes the materialized task list; if it's
        # empty, the lib-union is empty → return None to suppress
        # the corpus altogether (avoids loading the entire 34k
        # corpus by accident).
        out = adapter.corpus_for_run(
            {"use_lib_docs_corpus": True}, tasks=[]
        )
        assert out is None

    def test_corpus_active_flag_flips_when_corpus_returned(self, adapter):
        tasks = [
            Task(
                task_id="t",
                benchmark="bigcodebench",
                granularity=TaskGranularity.FUNCTION,
                prompt="...",
                metadata={"libs": ["numpy"]},
                ground_truth="ok",
            ),
        ]
        # Before: inactive.
        assert adapter._corpus_active is False
        assert adapter.reset_memory_per_task is True

        out = adapter.corpus_for_run(
            {"use_lib_docs_corpus": True}, tasks=tasks
        )
        # After: active, suppress reset.
        assert out is not None
        assert adapter._corpus_active is True
        assert adapter.reset_memory_per_task is False

    def test_corpus_returns_none_when_flag_off_even_with_tasks(self, adapter):
        tasks = [
            Task(
                task_id="t",
                benchmark="bigcodebench",
                granularity=TaskGranularity.FUNCTION,
                prompt="...",
                metadata={"libs": ["numpy"]},
                ground_truth="ok",
            ),
        ]
        out = adapter.corpus_for_run(
            {"use_lib_docs_corpus": False}, tasks=tasks
        )
        assert out is None
        assert adapter._corpus_active is False

    def test_corpus_returns_none_when_all_tasks_have_no_libs(self, adapter):
        tasks = [
            Task(
                task_id="t",
                benchmark="bigcodebench",
                granularity=TaskGranularity.FUNCTION,
                prompt="...",
                metadata={"libs": []},
                ground_truth="ok",
            ),
        ]
        # Empty lib-union → no corpus to load.
        out = adapter.corpus_for_run(
            {"use_lib_docs_corpus": True}, tasks=tasks
        )
        assert out is None


class TestGetRetrievalQueryCorpusMode:
    @pytest.mark.asyncio
    async def test_query_includes_libs_hint_when_corpus_active(self, adapter):
        # Activate corpus by calling corpus_for_run with the right shape.
        tasks = [
            Task(
                task_id="t",
                benchmark="bigcodebench",
                granularity=TaskGranularity.FUNCTION,
                prompt="...",
                metadata={"libs": ["numpy"]},
                ground_truth="ok",
            ),
        ]
        adapter.corpus_for_run({"use_lib_docs_corpus": True}, tasks=tasks)

        task = Task(
            task_id="t1",
            benchmark="bigcodebench",
            granularity=TaskGranularity.FUNCTION,
            prompt="...",
            metadata={
                "libs": ["numpy", "random"],
                "complete_prompt": "def task_func(x):\n    \"\"\"Sort\"\"\"\n",
            },
            ground_truth="ok",
        )
        q = adapter.get_retrieval_query(task)
        assert q is not None
        assert "def task_func(x):" in q
        assert "libraries:" in q
        assert "numpy" in q
        assert "random" in q

    def test_query_returns_none_when_corpus_inactive(self, adapter):
        # Without activating the corpus the query path returns None
        # so the engine falls back to ``query == prompt``.
        task = Task(
            task_id="t1",
            benchmark="bigcodebench",
            granularity=TaskGranularity.FUNCTION,
            prompt="...",
            metadata={"libs": ["numpy"], "complete_prompt": "..."},
            ground_truth="ok",
        )
        # adapter._corpus_active is False (fresh adapter).
        assert adapter.get_retrieval_query(task) is None
