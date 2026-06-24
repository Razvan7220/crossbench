"""BigCodeBench benchmark adapter — 1140 function-level tasks with diverse libraries.

Memory integration via CodeRAG-Bench library-documentation corpus
-----------------------------------------------------------------
**Optional opt-in**, off by default. When
``params["use_lib_docs_corpus"] = True`` is set (or the
``bigcodebench_lib_docs`` profile mixin is stacked), the adapter:

  1. Loads the CodeRAG-Bench ``library-documentation`` corpus (Wang
     et al. 2024, arXiv:2406.14497, ~34k API doc entries scraped from
     devdocs.io).
  2. Filters by the union of libraries appearing in the run's tasks
     (e.g. for a run that hits numpy, pandas, and random, only those
     libs' docs are loaded — not all 13).
  3. Ingests the filtered corpus once at run start via the new
     ``corpus_for_run`` hook (see ``BenchmarkAdapter`` /
     ``deviations.md`` platform section).
  4. Sets ``reset_memory_per_task = False`` (instance attribute, not
     class attribute) so the corpus survives across tasks.
  5. Builds a per-task retrieval query that includes the ``libs``
     hint — BigCodeBench uniquely provides the libraries used by
     each task, which CodeRAG-Bench's own evaluation lacks.

When the flag is off (the default), all memory hooks remain no-op
and the leaderboard cell reads as the no-memory baseline. See
``deviations.md`` B7 (no-corpus default) and B12 (CodeRAG-Bench
corpus mode).

Faithful port of the inspect_evals scorer and prompt construction:
  https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/bigcodebench/bigcodebench.py

Paper: Zhuo et al., 2024 — https://arxiv.org/abs/2406.15877
Upstream repo: https://github.com/bigcode-project/bigcodebench

Dataset
-------
HuggingFace: ``bigcode/bigcodebench``. The dataset exposes one split per
version tag (``v0.1.0_hf``, ``v0.1.1``, ``v0.1.2``, ``v0.1.3``,
``v0.1.4``); each has 1140 tasks. We default to ``v0.1.2`` pinned to
revision ``b74c0d0bf70d2c0bc459be537895cca163007f1a`` to match the
inspect_evals reference. Override via ``params["version"]``.

Schema (relevant fields, all strings unless noted):
  ``task_id`` (e.g. ``"BigCodeBench/0"``), ``complete_prompt``,
  ``instruct_prompt``, ``code_prompt`` (bare signature only; used for
  calibration), ``canonical_solution``, ``test`` (Python source defining
  ``TestCases(unittest.TestCase)``), ``entry_point`` (always
  ``"task_func"`` in this dataset), ``libs``, ``doc_struct``.

Note on ``libs``: the HF dataset stores ``libs`` as a **stringified**
Python list (e.g. ``"['random', 'itertools']"``), not as a JSON
``list[str]`` and not as a comma-separated string. The adapter parses
this back to ``list[str]`` via ``ast.literal_eval`` with a
comma-split fallback for robustness.

Variants
--------
BigCodeBench has two prompt modes:
  - ``complete``: function signature + docstring to be continued. For chat
    models we prefix with the inspect_evals INSTRUCTION_PROMPT (verbatim).
  - ``instruct``: natural-language instruction with embedded signature.
    Used standalone (no instruction prefix).

The execution pipeline is identical for both.

Calibrated mode
---------------
Upstream's ``bigcodebench.evaluate`` defaults to **calibrated**
evaluation — it prepends ``code_prompt + "\\n    pass\\n"`` to the
candidate solution before exec so chat models that don't echo the
function signature still produce a runnable ``task_func``. The real
solution then redefines ``task_func``, last definition wins.

CrossBench's adapter exposes a ``calibrated`` parameter (default
``True``, matching upstream). Disabling it reproduces the legacy
inspect_evals-style behavior. See ``deviations.md`` entry B1.

Code extraction (``_find_code``)
--------------------------------
Byte-for-byte port of inspect_evals' ``find_code``. Two fence patterns
tried in order (```python first, then plain ```); first match wins; fall
back to raw completion if neither matches.

Execution
---------
Upstream ``untrusted_check`` loads the combined ``code + "\\n" + test``
into a fresh module, pulls out ``TestCases``, and runs it via
``unittest.TestLoader().loadTestsFromTestCase(...)``. inspect_evals
achieves the same effect with one less layer: append the standard
``if __name__ == '__main__': unittest.main()`` tail so that
``python -c <combined>`` actually runs the tests. We port that form
byte-identically::

    answer + "\\n" + test + "\\n" + "\\nif __name__ == '__main__':\\n    unittest.main()"

The concrete concatenation inspect_evals uses is
``answer + "\\n" + test + "\\n" + _UNITTEST_TAIL`` where ``_UNITTEST_TAIL``
begins with a leading ``"\\n"`` — producing a blank line between the test
class and the ``if __name__`` guard. Preserved byte-for-byte.

Run inside the ``cross-bench-bigcodebench:latest`` Docker sandbox (must
have the 139 BigCodeBench libs pre-installed). Non-zero exit or timeout
= failure.

Metrics
-------
Per task: ``pass`` (bool) and ``pass@1`` (0/1). The engine averages
``pass@1`` across tasks. ``metrics_schema()`` declares the canonical
column set so summary.json columns are stable across runs and across
memory systems.

Memory integration (default vs corpus-mode)
-------------------------------------------
**Default: no-op.** Each BigCodeBench task is self-contained —
function signature + docstring + tests, no cross-file context, no
per-task supplemental data. Cross-task memory accumulation would
violate upstream's per-task-independent evaluation contract; ingesting
canonical solutions would be direct contamination. So in the default
configuration the adapter inherits the no-op
``ingest_for_memory`` / ``format_memory_prompt`` /
``get_retrieval_query`` from :class:`BenchmarkAdapter`. Running
``-b bigcodebench`` with a memory profile produces numerically
identical results to the no-memory baseline.

**Opt-in: CodeRAG-Bench library-docs corpus.** When
``params["use_lib_docs_corpus"] = True`` (typically via the
``bigcodebench_lib_docs`` profile mixin), the adapter overrides
``corpus_for_run`` to ingest the CodeRAG-Bench
``library-documentation`` corpus once at run start (see deviation
B12). ``reset_memory_per_task`` is set to ``False`` so the corpus
survives across tasks. ``get_retrieval_query`` is overridden to
include the per-task ``libs`` hint as the retrieval signal — a small
upgrade over CodeRAG-Bench's own evaluation, which has no libs field.

See ``deviations.md`` B7 (default no-op rationale) and B12
(CodeRAG-Bench corpus integration).

Simplifications vs. upstream ``bigcodebench.evaluate``:
  - Single sample per task. Upstream reports ``pass@{1,5,10}`` via the
    HumanEval unbiased estimator; with ``n=1`` these reduce to pass@1.
  - No ``reliability_guard`` / resource limits
    (``max_as_limit``/``max_data_limit``/``max_stack_limit``). The Docker
    sandbox provides OS-level isolation instead.
  - Calibrated mode is now a configurable parameter (default ``True``,
    matching upstream). See ``deviations.md`` B1.
  - No tree-sitter ``sanitize.py`` post-processing. We use the
    inspect_evals fenced-code extraction instead.
"""

from __future__ import annotations

import ast
import re
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from cross_bench.benchmarks.base import (
    MEMORY_TOKEN_SCHEMA_KEYS,
    BenchmarkAdapter,
)
from cross_bench.benchmarks.code_rag_corpora import load_corpus
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import CorpusItem, Task, TaskGranularity

if TYPE_CHECKING:
    pass

DATASET_PATH = "bigcode/bigcodebench"
# Pinned upstream revision — matches inspect_evals for reproducibility.
BIGCODEBENCH_DATASET_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
DEFAULT_VERSION = "v0.1.2"
# Calibration injection — upstream ``evaluate.py:305-306``. The string
# is byte-faithful: ``code_prompt + "\n    pass\n"`` with one leading
# newline between the import block and the ``pass`` line of the stub.
_CALIBRATION_TAIL = "\n    pass\n"

# Instruction prefix for the ``complete`` variant. Byte-identical to
# inspect_evals INSTRUCTION_PROMPT (note the trailing ``\n``).
_INSTRUCTION_PROMPT = (
    "Read the following function signature and docstring, and fully "
    "implement the function described. Make sure to include the import "
    "statements and use the same variable names as stated in the header.\n"
)

# Appended after generated code + test so ``python -c`` actually runs the
# unittest TestCases class. Matches inspect_evals verify() exactly — a
# leading "\n" is intentional (inspect_evals builds with a triple-quoted
# string that starts with a newline, so the final concat has a blank
# line between test and the ``if __name__`` guard).
_UNITTEST_TAIL = "\nif __name__ == '__main__':\n    unittest.main()"


def _parse_libs(raw: Any) -> list[str]:
    """Parse the dataset's ``libs`` field into ``list[str]``.

    The HF dataset stores ``libs`` as a **stringified Python list**:
    ``"['random', 'itertools']"``. Not a JSON list. Not a
    comma-separated string. Not a list. ``ast.literal_eval`` is the
    safe parser; on any failure we fall back to a comma-split (in case
    a future mirror normalizes the field) then to an empty list.
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw:
        return []
    # Try ast.literal_eval first — handles "['a', 'b']" / "['a']" / "[]".
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    # Fallback for a hypothetical comma-separated mirror format.
    return [s.strip() for s in raw.split(",") if s.strip()]


@register_adapter
class BigCodeBenchAdapter(BenchmarkAdapter):
    name = "bigcodebench"
    granularity = TaskGranularity.FUNCTION

    def __init__(self) -> None:
        # ``reset_memory_per_task`` is dialed at instance init based on
        # whether the corpus is opted into for this run. The class
        # default (True) keeps legacy behavior; we flip to False inside
        # ``corpus_for_run`` once we know the user asked for the
        # corpus.
        self._corpus_active: bool = False

    @property
    def reset_memory_per_task(self) -> bool:  # type: ignore[override]
        # When the corpus is loaded, suppress per-task reset so it
        # survives. Otherwise inherit the class default (True).
        return not self._corpus_active

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        variant = params.get("variant", "complete")
        if variant not in ("complete", "instruct"):
            raise ValueError(f"variant must be 'complete' or 'instruct', got {variant!r}")

        version = params.get("version", DEFAULT_VERSION)
        # Calibrated mode — defaults to True (matches upstream's
        # ``bigcodebench.evaluate`` default). Disable for legacy
        # inspect_evals-style behavior.
        calibrated = bool(params.get("calibrated", True))
        ds = load_dataset(
            DATASET_PATH,
            split=version,
            revision=BIGCODEBENCH_DATASET_REVISION,
        )
        limit = params.get("limit")
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            if variant == "complete":
                prompt = _INSTRUCTION_PROMPT + row["complete_prompt"]
            else:
                prompt = row["instruct_prompt"]
            yield Task(
                task_id=row["task_id"],
                benchmark=self.name,
                granularity=self.granularity,
                prompt=prompt,
                metadata={
                    "variant": variant,
                    "version": version,
                    "calibrated": calibrated,
                    "entry_point": row["entry_point"],
                    "libs": _parse_libs(row.get("libs")),
                    "canonical_solution": row["canonical_solution"],
                    "complete_prompt": row["complete_prompt"],
                    "code_prompt": row.get("code_prompt", ""),
                },
                ground_truth=row["test"],
            )

    # ------------------------------------------------------------------
    # Memory integration — opt-in CodeRAG-Bench library-docs corpus.
    # Default: all hooks return no-op / None (see B7 in deviations.md).
    # ------------------------------------------------------------------

    def corpus_for_run(
        self,
        params: dict[str, Any] | None = None,
        tasks: list[Task] | None = None,
    ) -> AsyncIterator[CorpusItem] | None:
        """Stream the CodeRAG-Bench library-documentation corpus.

        Active only when ``params["use_lib_docs_corpus"] = True``.
        Filters the 34k-doc corpus down to the union of libraries
        actually used by tasks in this run (typically 5-30 libs out
        of the 13 library prefixes the corpus covers).

        Returns ``None`` (the default) when the flag is off — the
        engine then takes the legacy per-task ingestion path.
        """
        params = params or {}
        if not params.get("use_lib_docs_corpus"):
            return None

        # Compute the lib-union from the materialized task list.
        # Empty union → no filter → load everything, which is
        # rarely what you want; we treat that as "no corpus" for
        # safety (logged inside corpus_for_run flow).
        libs: set[str] = set()
        if tasks:
            for t in tasks:
                for lib in (t.metadata or {}).get("libs", []) or []:
                    if isinstance(lib, str) and lib.strip():
                        libs.add(lib.strip())

        if not libs:
            return None

        self._corpus_active = True
        return load_corpus("library-documentation", libraries=libs)

    def get_retrieval_query(self, task: Task) -> str | None:
        """Per-task retrieval query when the corpus is active.

        Concatenates ``complete_prompt`` (signature + docstring) with
        the libs hint. The libs hint is unique to BigCodeBench —
        CodeRAG-Bench's own evaluation doesn't have it — so the
        memory-mode column is slightly better positioned than the
        paper's published RAG numbers for analogous tasks.

        Returns ``None`` when the corpus isn't active so the engine
        falls back to the legacy ``query == prompt`` semantics.
        """
        if not self._corpus_active:
            return None
        meta = task.metadata or {}
        complete_prompt = meta.get("complete_prompt", "") or ""
        libs = meta.get("libs") or []
        libs_str = " ".join(libs) if libs else ""
        if libs_str:
            return f"{complete_prompt}\nlibraries: {libs_str}"
        return complete_prompt or None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        meta = task.metadata or {}
        answer = _find_code(generated_output)

        # Calibrated mode (upstream default) — prepend the bare
        # signature + ``pass`` stub so chat models that don't echo the
        # signature still produce a runnable ``task_func``. The real
        # solution redefines it; last definition wins. Upstream
        # ``evaluate.py:305-306`` byte-for-byte.
        if meta.get("calibrated", True):
            code_prompt = meta.get("code_prompt", "") or ""
            if code_prompt:
                answer = code_prompt + _CALIBRATION_TAIL + answer

        test_code = task.ground_truth
        combined = answer + "\n" + test_code + "\n" + _UNITTEST_TAIL

        if sandbox is None:
            sandbox = await sandbox_mgr.create(self.get_sandbox_image(task), task)

        timeout = int(meta.get("_timeout_seconds", 30))
        result = await sandbox_mgr.execute(
            sandbox,
            f"python3 -c {_shell_quote(combined)}",
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
        return "cross-bench-bigcodebench:latest"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "variant": "complete",
            "version": DEFAULT_VERSION,
            "calibrated": True,
            # Corpus-mode toggle — opt-in. Stack with the
            # ``bigcodebench_lib_docs`` profile mixin to enable.
            "use_lib_docs_corpus": False,
        }

    def metrics_schema(self) -> list[str]:
        """Stable column set for BigCodeBench summary.json.

        Headlines (``pass_rate`` / ``avg_pass@1``) are upstream's
        single metric. Latency keys are populated by
        ``execute_single_shot``. Memory-related keys are declared (and
        stay ``null``) because BigCodeBench inherits the default
        no-op memory hooks — the column is reported as "not measured"
        rather than silently disappearing across runs that mix
        memory-enabled and memory-disabled rows.
        """
        return [
            # Grading
            "pass_rate", "accuracy",
            "avg_pass@1",
            # Latency (set by execute_single_shot). When the
            # CodeRAG-Bench corpus is active,
            # ``avg_search_latency_ms`` reflects per-task retrieval
            # over the corpus; otherwise it stays null
            # ("not measured") per B7's no-op default.
            "avg_search_latency_ms",
            "avg_inference_latency_ms",
            "avg_total_latency_ms",
            # Cost / scale
            "total_tokens_in", "total_tokens_out",
            "avg_tokens_per_task",
            "total_tasks",
            # Memory-system internal token cost (separate from bench LLM)
            *MEMORY_TOKEN_SCHEMA_KEYS,
            # Run health
            "passed", "failed", "errors", "timeouts",
        ]


def _find_code(completion: str) -> str:
    """Byte-for-byte port of inspect_evals ``find_code``.

    Tries ```python...``` first, then plain ```...```. First match from
    the concatenated list wins. Falls back to the raw completion when no
    fenced block is found.
    """
    pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
    pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern_1.findall(completion) + pattern_2.findall(completion)
    extracted_answer = matches[0] if matches else completion
    return str(extracted_answer)


def _shell_quote(s: str) -> str:
    """POSIX single-quote escaping: ``'…'\\''…'``."""
    return "'" + s.replace("'", "'\\''") + "'"
