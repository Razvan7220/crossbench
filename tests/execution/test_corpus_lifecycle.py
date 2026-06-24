"""Engine tests for the run-scoped corpus ingestion lifecycle.

The ``corpus_for_run`` hook on ``BenchmarkAdapter`` (default ``None``)
gives benchmarks like BigCodeBench an opt-in path to ingest a shared
corpus once at run start instead of per-task. Combined with
``reset_memory_per_task = False`` the corpus survives across the whole
task loop.

These tests verify the engine's wiring without touching real
middlewares — a recording double captures every memory call so we can
assert the order of ``post_response`` (corpus + per-task) /
``pre_prompt`` / ``reset`` events.

Critical regression invariant: when an adapter does NOT override
``corpus_for_run`` (returns ``None``), the engine must take the legacy
per-task path with ``memory.reset()`` between tasks. This is what
RepoBench, CrossCodeEval, and every other existing benchmark depend
on; breaking it would silently leak per-task context across tasks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.execution.engine import RunEngine
from cross_bench.models import (
    CorpusItem,
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
)
from cross_bench.systems.base import LLMDriver, MemoryMiddleware, PromptStrategy


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM(LLMDriver):
    async def complete(
        self, prompt: str, config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return "ok", StepRecord(
            step_index=0, action="completion", input_text=prompt, output_text="ok",
        )

    async def chat(
        self, messages: list[dict[str, str]], config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return "ok", StepRecord(
            step_index=0, action="chat", input_text=str(messages), output_text="ok",
        )


class _PassthroughStrategy(PromptStrategy):
    def format(
        self,
        task: Task,
        benchmark_prompt: str,
        memory_context: str | None = None,
    ) -> str | list[dict[str, str]]:
        return benchmark_prompt


class _RecordingMemory(MemoryMiddleware):
    """Records the timeline of every memory call so we can assert on
    ordering and counts. ``events`` is the audit trail: each entry is
    a 2-tuple ``(event_kind, payload)`` where event_kind is one of
    ``"post"``, ``"pre"``, ``"reset"``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        self.events.append(("pre", {"prompt": prompt, "context": dict(context)}))
        return prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        self.events.append(("post", {"response": response, "context": dict(context)}))

    async def reset(self) -> None:
        self.events.append(("reset", None))


class _NoOpSandboxManager:
    async def create(self, image: str, task: Task) -> Any:
        return None

    async def destroy(self, sandbox: Any) -> None:
        pass

    async def execute(self, sandbox: Any, cmd: str, timeout: int = 60) -> Any:
        raise AssertionError("No sandbox exec expected in these tests")


class _FakeArtifactStore:
    async def store_run_config(self, config: Any) -> None:
        pass

    async def store_task_result(self, run_id: str, result: Any) -> None:
        pass

    async def store_run_summary(self, result: Any) -> None:
        pass


class _FakeResultRepo:
    async def create_run(self, config: Any) -> None:
        pass

    async def save_task_result(self, result: Any) -> None:
        pass

    async def complete_run(
        self, run_id: str, metrics: Any, completed_at: Any
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _LegacyAdapter(BenchmarkAdapter):
    """Default contract — no corpus, default per-task reset.

    This is what every benchmark that predates the corpus hook looks
    like. Asserting that the engine's behavior on this adapter is
    UNCHANGED is the most important regression-safety test in this
    file."""

    name = "legacy_stub"
    granularity = TaskGranularity.FUNCTION
    requires_sandbox = False

    def __init__(self, n_tasks: int = 2) -> None:
        self._n_tasks = n_tasks

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        for i in range(self._n_tasks):
            yield Task(
                task_id=f"legacy_{i}",
                benchmark=self.name,
                granularity=self.granularity,
                prompt=f"task {i}",
                metadata={},
                ground_truth="ok",
            )

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: Any,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        return {"pass": True}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""


class _CorpusAdapter(BenchmarkAdapter):
    """Opts into the corpus hook with a deterministic 3-document corpus.

    Sets ``reset_memory_per_task = False`` (instance attribute, like
    BigCodeBench does at runtime) so the corpus survives across
    tasks."""

    name = "corpus_stub"
    granularity = TaskGranularity.FUNCTION
    requires_sandbox = False
    reset_memory_per_task = False

    def __init__(self, n_tasks: int = 2, corpus_size: int = 3) -> None:
        self._n_tasks = n_tasks
        self._corpus_size = corpus_size

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        for i in range(self._n_tasks):
            yield Task(
                task_id=f"corpus_task_{i}",
                benchmark=self.name,
                granularity=self.granularity,
                prompt=f"task {i}",
                metadata={},
                ground_truth="ok",
            )

    def corpus_for_run(
        self,
        params: dict[str, Any] | None = None,
        tasks: list[Task] | None = None,
    ) -> AsyncIterator[CorpusItem] | None:
        # Capture how many tasks the engine handed in so we can assert
        # the engine populates the kwarg.
        self._tasks_seen_at_corpus_call = list(tasks or [])
        return _make_corpus(self._corpus_size)

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: Any,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        return {"pass": True}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""


async def _make_corpus(n: int) -> AsyncIterator[CorpusItem]:
    for i in range(n):
        yield CorpusItem(
            content=f"doc {i}",
            source_id=f"corpus:{i}",
            metadata={"library": "stub"},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _engine() -> RunEngine:
    return RunEngine(
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
        result_repo=_FakeResultRepo(),  # type: ignore[arg-type]
        sandbox_mgr=_NoOpSandboxManager(),  # type: ignore[arg-type]
    )


def _config() -> RunConfig:
    return RunConfig(
        benchmark="stub",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.SINGLE_SHOT,
    )


# ---------------------------------------------------------------------------
# Critical regression invariant: legacy adapters keep per-task reset
# ---------------------------------------------------------------------------


class TestLegacyPathUnchanged:
    """Adapters that don't override ``corpus_for_run`` MUST keep the
    legacy per-task ingestion + reset pattern. This test guards every
    pre-existing benchmark (RepoBench, CrossCodeEval, LoCoMo, …)
    against silent contamination from the new code path."""

    @pytest.mark.asyncio
    async def test_no_corpus_ingest_when_corpus_for_run_returns_none(self):
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_LegacyAdapter(n_tasks=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        # No corpus reset / ingest events should appear before the
        # first task. The first ``post`` event (if any) belongs to a
        # task's per-LLM-call ingestion, never to a corpus item.
        # Specifically, no post event with source_id starting "corpus:".
        for kind, payload in memory.events:
            if kind == "post":
                ctx = payload["context"]
                assert not str(ctx.get("source_id", "")).startswith("corpus:")

    @pytest.mark.asyncio
    async def test_reset_fires_between_tasks_for_legacy_adapter(self):
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_LegacyAdapter(n_tasks=3),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        reset_count = sum(1 for k, _ in memory.events if k == "reset")
        # 3 tasks × 1 reset per task = 3 resets. The legacy default
        # reset_memory_per_task=True must be honored.
        assert reset_count == 3

    @pytest.mark.asyncio
    async def test_legacy_path_works_without_memory(self):
        """memory=None must short-circuit the corpus_for_run check
        (no AttributeError, no calls to the adapter's hook)."""
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_LegacyAdapter(n_tasks=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=None,
        )
        # No assertion needed — the test passes if no exception fires.


# ---------------------------------------------------------------------------
# Corpus path: ingest before tasks, suppress reset, preserve order
# ---------------------------------------------------------------------------


class TestCorpusLifecycle:
    @pytest.mark.asyncio
    async def test_corpus_ingest_runs_before_first_task(self):
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_CorpusAdapter(n_tasks=2, corpus_size=3),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        # Find the first reset (corpus-mode does memory.reset() once
        # at run start). Then the next 3 events should all be
        # ``post`` with corpus source_ids.
        reset_idx = next(
            i for i, (k, _) in enumerate(memory.events) if k == "reset"
        )
        assert reset_idx == 0  # run-start reset is the very first event
        for j in range(1, 4):
            kind, payload = memory.events[j]
            assert kind == "post"
            assert payload["context"]["source_id"] == f"corpus:{j - 1}"

    @pytest.mark.asyncio
    async def test_corpus_survives_across_tasks_no_intermediate_reset(self):
        """``reset_memory_per_task = False`` ⇒ no reset between tasks.
        The corpus ingested at run start stays in memory for every
        task that follows."""
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_CorpusAdapter(n_tasks=3, corpus_size=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        reset_count = sum(1 for k, _ in memory.events if k == "reset")
        # Exactly 1 reset (the corpus pre-ingest reset). No per-task
        # resets fire because reset_memory_per_task = False.
        assert reset_count == 1

    @pytest.mark.asyncio
    async def test_corpus_for_run_receives_materialized_tasks(self):
        adapter = _CorpusAdapter(n_tasks=4, corpus_size=2)
        await _engine()._run_code_gen(
            config=_config(),
            adapter=adapter,
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=_RecordingMemory(),
        )
        # Engine must hand the materialized task list to corpus_for_run
        # so the adapter can derive run-scoped filters (e.g.
        # BigCodeBench uses this to filter the library-doc corpus by
        # the union of libs across the task set).
        seen = adapter._tasks_seen_at_corpus_call
        assert len(seen) == 4
        assert {t.task_id for t in seen} == {
            f"corpus_task_{i}" for i in range(4)
        }

    @pytest.mark.asyncio
    async def test_per_task_pre_prompt_still_fires_on_top_of_corpus(self):
        """Per-task ``pre_prompt`` (the retrieval call) is still
        invoked once per task, layered on top of the run-scoped
        corpus."""
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_CorpusAdapter(n_tasks=3, corpus_size=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        pre_count = sum(1 for k, _ in memory.events if k == "pre")
        # 3 tasks → 3 pre_prompt calls.
        assert pre_count == 3

    @pytest.mark.asyncio
    async def test_corpus_returns_none_means_legacy_path(self):
        """An adapter that overrides ``corpus_for_run`` but returns
        ``None`` (e.g. corpus opt-in flag is off) must take the
        legacy path. ``reset_memory_per_task`` should still default
        to True for the run."""

        class _OptionalCorpus(_CorpusAdapter):
            reset_memory_per_task = True  # corpus is OFF for this run

            def corpus_for_run(self, params=None, tasks=None):
                return None  # explicit opt-out

        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_OptionalCorpus(n_tasks=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        # No corpus events → 2 per-task resets (legacy path).
        reset_count = sum(1 for k, _ in memory.events if k == "reset")
        assert reset_count == 2

    @pytest.mark.asyncio
    async def test_corpus_iter_exception_does_not_crash_run(self):
        """If corpus_for_run raises, the run continues with the
        legacy path. Failing to ingest is a recoverable degradation
        — running tasks with an empty memory is better than aborting
        the whole evaluation."""

        class _FailingCorpus(_CorpusAdapter):
            def corpus_for_run(self, params=None, tasks=None):
                raise RuntimeError("HF dataset unreachable")

        memory = _RecordingMemory()
        # Should NOT raise.
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_FailingCorpus(n_tasks=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        # No corpus events ingested.
        post_corpus = [
            e for e in memory.events
            if e[0] == "post"
            and str(e[1]["context"].get("source_id", "")).startswith("corpus:")
        ]
        assert post_corpus == []

    @pytest.mark.asyncio
    async def test_corpus_item_metadata_flows_to_post_response(self):
        memory = _RecordingMemory()
        await _engine()._run_code_gen(
            config=_config(),
            adapter=_CorpusAdapter(n_tasks=1, corpus_size=2),
            llm=_StubLLM(),
            strategy=_PassthroughStrategy(),
            memory=memory,
        )
        post_corpus = [
            e[1] for e in memory.events
            if e[0] == "post"
            and str(e[1]["context"].get("source_id", "")).startswith("corpus:")
        ]
        # ``library`` was set on every CorpusItem; it must surface in
        # post_response context for vendor-side filters / debugging.
        assert all(p["context"]["library"] == "stub" for p in post_corpus)
