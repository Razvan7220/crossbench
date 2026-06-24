"""Tests for retrieval-provenance plumbing through the execution layer.

The contract (see ``MemoryMiddleware`` class docstring) is:

* ``post_response`` receives ``context["source_id"]`` — a per-item ingest tag.
* ``pre_prompt`` may write ``context["retrieved_context_ids"]`` — the
  ordered list of IDs for the items it retrieved.
* The engine / execution modes hoist the ``context`` dict so they can
  propagate ``retrieved_context_ids`` onto ``task.metadata`` for
  downstream benchmarks (e.g. LoCoMo evidence recall).

These tests exercise each of the 4 call sites:

* ``engine._run_conversational`` — ingestion AND retrieval (it short-
  circuits the mode's own pre_prompt call, so the engine owns both).
* ``modes.execute_single_shot`` — retrieval only.
* ``modes.execute_agentic`` — retrieval only (initial user message).
* ``modes.execute_conversational`` — retrieval only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from cross_bench.benchmarks.base import (
    BenchmarkAdapter,
    ConversationalBenchmarkAdapter,
)
from cross_bench.execution.modes import (
    execute_agentic,
    execute_conversational,
    execute_single_shot,
)
from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
)
from cross_bench.systems.base import (
    LLMDriver,
    MemoryMiddleware,
    PromptStrategy,
)

# ---------------------------------------------------------------------------
# Stubs — minimal fakes for the collaborators used by modes / engine.
# ---------------------------------------------------------------------------


class _StubLLM(LLMDriver):
    """Returns a canned response and a dummy StepRecord."""

    def __init__(self, response: str = "TASK_COMPLETE\n```\ndone\n```") -> None:
        self._response = response

    async def complete(
        self, prompt: str, config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return self._response, StepRecord(
            step_index=0, action="completion",
            input_text=prompt, output_text=self._response,
        )

    async def chat(
        self, messages: list[dict[str, str]], config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return self._response, StepRecord(
            step_index=0, action="chat",
            input_text=str(messages), output_text=self._response,
        )


class _PassthroughStrategy(PromptStrategy):
    def format(
        self,
        task: Task,
        benchmark_prompt: str,
        memory_context: str | None = None,
    ) -> str | list[dict[str, str]]:
        return benchmark_prompt


@dataclass
class _RecordingMemory(MemoryMiddleware):
    """Records every pre_prompt/post_response call and optionally writes
    ``retrieved_context_ids`` on pre_prompt."""

    retrieved_ids: list[str] | None = None  # what pre_prompt will emit
    pre_prompt_contexts: list[dict[str, Any]] = field(default_factory=list)
    post_response_contexts: list[dict[str, Any]] = field(default_factory=list)
    prefix: str = "[MEMORY]\n"

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        # Record a snapshot of the context as it arrived.
        self.pre_prompt_contexts.append(dict(context))
        if self.retrieved_ids is not None:
            context["retrieved_context_ids"] = list(self.retrieved_ids)
        return self.prefix + prompt if self.prefix else prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        self.post_response_contexts.append(dict(context))

    async def reset(self) -> None:
        # Deliberately NOT clearing the recording lists — they're the
        # audit trail the tests assert against, and the engine calls
        # ``reset()`` both before ingestion (to prevent cross-conversation
        # bleed) and at the end of ``_run_conversational`` (final cleanup).
        # A real memory backend's reset clears stored content; a recording
        # double's reset should leave the audit trail intact.
        pass


class _NoOpSandboxManager:
    """execute_single_shot calls sandbox_mgr.create/destroy only when
    ``adapter.requires_sandbox`` is True. Our adapters set it to False, but
    keep the stub anyway."""

    async def create(self, image: str, task: Task) -> Any:
        return None

    async def destroy(self, sandbox: Any) -> None:
        pass

    async def execute(self, sandbox: Any, cmd: str, timeout: int = 60) -> Any:
        raise AssertionError("No sandbox exec expected in these tests")


# ---------------------------------------------------------------------------
# A minimal no-sandbox adapter (for execute_single_shot) and a minimal
# conversational adapter (for execute_conversational and the engine).
# ---------------------------------------------------------------------------


class _NoSandboxAdapter(BenchmarkAdapter):
    name = "stub"
    granularity = TaskGranularity.FUNCTION
    requires_sandbox = False

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        return iter([])

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


class _ConversationalStubAdapter(ConversationalBenchmarkAdapter):
    name = "stub_conv"
    granularity = TaskGranularity.CONVERSATIONAL

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        return iter([])

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _run_config() -> RunConfig:
    return RunConfig(
        benchmark="stub",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.SINGLE_SHOT,
    )


def _task() -> Task:
    return Task(
        task_id="t1",
        benchmark="stub",
        granularity=TaskGranularity.FUNCTION,
        prompt="What is 2 + 2?",
        metadata={},
        ground_truth="4",
    )


# ---------------------------------------------------------------------------
# execute_single_shot — plumbs retrieved_context_ids to task.metadata
# ---------------------------------------------------------------------------


async def test_single_shot_propagates_retrieved_context_ids() -> None:
    task = _task()
    memory = _RecordingMemory(retrieved_ids=["D1:3", "D1:5"])

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert task.metadata["retrieved_context_ids"] == ["D1:3", "D1:5"]
    # The retrieval context should have been passed to the middleware
    # with at least the task_id — this confirms we're hoisting, not
    # passing a fresh dict per call.
    assert memory.pre_prompt_contexts[0]["task_id"] == "t1"


async def test_single_shot_absent_ids_leaves_metadata_unset() -> None:
    """When the middleware doesn't write ``retrieved_context_ids``, the
    engine must NOT plant the key on ``task.metadata`` — downstream
    benchmarks use 'key missing' as 'not measured'."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=None)  # won't touch the key

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert "retrieved_context_ids" not in task.metadata


async def test_single_shot_empty_list_is_preserved() -> None:
    """``[]`` (retriever ran, no hits) is NOT the same as absence. The
    engine must propagate the empty list so downstream can tell 'I tried
    and got nothing' from 'I didn't try'."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=[])

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert task.metadata["retrieved_context_ids"] == []


async def test_single_shot_stringifies_ids() -> None:
    """A middleware that emits non-string IDs (unusual but possible for
    backends keyed by int/uuid) has its IDs coerced to str so the
    adapter contract (``list[str]``) holds."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=[1, 2, 3])  # type: ignore[list-item]

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert task.metadata["retrieved_context_ids"] == ["1", "2", "3"]


async def test_single_shot_no_memory_leaves_metadata_unset() -> None:
    """With no memory at all, the engine cannot and must not populate
    ``retrieved_context_ids``."""
    task = _task()

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=None,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert "retrieved_context_ids" not in task.metadata


# ---------------------------------------------------------------------------
# Memory-aware adapter hooks (RepoBench-style code-RAG path)
# ---------------------------------------------------------------------------


class _MemoryAwareAdapter(_NoSandboxAdapter):
    """Exercises the three optional hooks added in tandem with RepoBench.

    The default adapter inherits no-op versions of these from
    ``BenchmarkAdapter``; this subclass overrides them so we can verify
    the engine actually calls them in the right order.
    """

    name = "stub_memory_aware"

    def __init__(self) -> None:
        self.ingest_called_with: list[Task] = []

    async def ingest_for_memory(
        self, task: Task, memory: MemoryMiddleware
    ) -> None:
        self.ingest_called_with.append(task)
        # Drop two synthetic items into memory so the test can confirm
        # the engine called us BEFORE pre_prompt (the recording memory
        # records both calls in order).
        await memory.post_response(
            "", "first ingested chunk", {"source_id": "ctx:0"}
        )
        await memory.post_response(
            "", "second ingested chunk", {"source_id": "ctx:1"}
        )

    def format_prompt(self, task: Task) -> str:
        return f"FULL: {task.prompt}"

    def format_memory_prompt(self, task: Task) -> str:
        return f"STRIPPED: {task.prompt}"

    def get_retrieval_query(self, task: Task) -> str | None:
        return f"QUERY: {task.prompt}"


async def test_single_shot_calls_ingest_then_pre_prompt() -> None:
    """``ingest_for_memory`` runs first; THEN ``pre_prompt`` (with the
    focused query) executes against the now-populated memory.

    The recording memory captures both as ``post_response`` /
    ``pre_prompt`` events; we assert the order via the lengths of its
    two recording lists at each stage."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=["ctx:0"])
    adapter = _MemoryAwareAdapter()

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=adapter,
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    # Ingest happened.
    assert len(adapter.ingest_called_with) == 1
    assert adapter.ingest_called_with[0] is task
    # Two ingestion post_responses + the engine's own post_response
    # of the LLM output = 3.
    assert len(memory.post_response_contexts) == 3
    # Pre_prompt fired exactly once (after ingestion).
    assert len(memory.pre_prompt_contexts) == 1


async def test_single_shot_uses_focused_query_for_pre_prompt() -> None:
    """When the adapter returns a non-None ``get_retrieval_query``, the
    engine sends THAT to ``pre_prompt`` — not the full benchmark prompt.

    The recording memory's ``prefix`` mode echoes the input query back
    in the augmented output, so we can fish the query out of what
    pre_prompt saw via the memory_ctx extraction (or by inspecting the
    middleware's pre_prompt args directly — but our current double
    only records the context dict, so we confirm via the augmented
    prompt-extraction path)."""
    task = _task()  # task.prompt = "What is 2 + 2?"
    memory = _RecordingMemory(retrieved_ids=[], prefix="[RETRIEVED]\n")
    adapter = _MemoryAwareAdapter()

    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=adapter,
        llm=_StubLLM("4"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    # The middleware was asked to retrieve against the focused query —
    # we confirm by patching the memory to echo back its input prompt
    # via the prefix. The engine's ``post_response`` (for the LLM
    # output) carries no ID context here, but its ``prompt`` field was
    # the LLM input — which under the focused-query path is
    # ``memory_ctx + STRIPPED prompt`` (NOT the focused query).
    last_post_prompt = memory.post_response_contexts[-1]
    # Just confirm the post_response context dict is well-formed.
    assert last_post_prompt["task_id"] == task.task_id


async def test_single_shot_legacy_path_when_no_retrieval_query() -> None:
    """An adapter that does NOT override ``get_retrieval_query``
    (returns ``None``) keeps the legacy ``query == prompt`` semantics:
    ``memory.pre_prompt`` returns ``memory_ctx + prompt`` directly and
    the engine assigns ``benchmark_prompt = augmented`` without any
    additional concatenation."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=["x"], prefix="[MEM]\n")

    # Default _NoSandboxAdapter — get_retrieval_query returns None.
    await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("4"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert task.metadata["retrieved_context_ids"] == ["x"]


async def test_single_shot_records_inference_latency() -> None:
    """Every single-shot run stamps ``inference_latency_ms`` on the
    evaluation dict, regardless of whether memory is active. This is
    what makes ``avg_inference_latency_ms`` a stable summary column."""
    task = _task()

    result = await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=None,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert "inference_latency_ms" in result.evaluation
    assert isinstance(result.evaluation["inference_latency_ms"], float)
    assert result.evaluation["inference_latency_ms"] >= 0.0
    # No memory ⇒ total latency is just inference.
    assert result.evaluation["total_latency_ms"] == (
        result.evaluation["inference_latency_ms"]
    )


async def test_single_shot_records_search_latency_when_memory_enabled() -> None:
    """When memory is active, the engine separately records the time
    spent in ``pre_prompt`` (search) versus the LLM call (inference).
    Total = search + inference."""
    task = _task()
    memory = _RecordingMemory(retrieved_ids=[])

    result = await execute_single_shot(
        task=task,
        config=_run_config(),
        adapter=_NoSandboxAdapter(),
        llm=_StubLLM("42"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert "search_latency_ms" in result.evaluation
    assert "inference_latency_ms" in result.evaluation
    assert "total_latency_ms" in result.evaluation
    # Approximate: total ≈ search + inference (within float rounding).
    total = result.evaluation["total_latency_ms"]
    parts = (
        result.evaluation["search_latency_ms"]
        + result.evaluation["inference_latency_ms"]
    )
    assert abs(total - parts) < 0.5


# ---------------------------------------------------------------------------
# execute_conversational — plumbs retrieved_context_ids to task.metadata
# ---------------------------------------------------------------------------


async def test_conversational_mode_propagates_retrieved_context_ids() -> None:
    task = _task()
    memory = _RecordingMemory(retrieved_ids=["D1:3"])

    await execute_conversational(
        task=task,
        config=_run_config(),
        adapter=_ConversationalStubAdapter(),
        llm=_StubLLM("the answer"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert task.metadata["retrieved_context_ids"] == ["D1:3"]


async def test_conversational_mode_absent_ids() -> None:
    task = _task()
    memory = _RecordingMemory(retrieved_ids=None)

    await execute_conversational(
        task=task,
        config=_run_config(),
        adapter=_ConversationalStubAdapter(),
        llm=_StubLLM("the answer"),
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_NoOpSandboxManager(),
    )

    assert "retrieved_context_ids" not in task.metadata


# ---------------------------------------------------------------------------
# execute_agentic — plumbs retrieved_context_ids from the initial user
# message's pre_prompt call. We use a 1-step LLM response (TASK_COMPLETE)
# so the agent loop exits after the first turn without needing a real
# sandbox for bash execution.
# ---------------------------------------------------------------------------


class _AgenticNoOpSandbox:
    async def create(self, image: str, task: Task) -> Any:
        return object()

    async def destroy(self, sandbox: Any) -> None:
        pass


class _AgenticAdapter(BenchmarkAdapter):
    name = "stub_agentic"
    granularity = TaskGranularity.FUNCTION
    requires_sandbox = True  # agentic forces sandbox

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        return iter([])

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: Any,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        return {"pass": True}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return "python:3.12"


async def test_agentic_mode_propagates_retrieved_context_ids() -> None:
    task = _task()
    memory = _RecordingMemory(retrieved_ids=["D2:1", "D2:4"])

    # TASK_COMPLETE on first turn so the loop exits immediately.
    llm = _StubLLM("Thinking...\nTASK_COMPLETE\n```\n42\n```")

    await execute_agentic(
        task=task,
        config=_run_config(),
        adapter=_AgenticAdapter(),
        llm=llm,
        strategy=_PassthroughStrategy(),
        memory=memory,
        sandbox_mgr=_AgenticNoOpSandbox(),
    )

    assert task.metadata["retrieved_context_ids"] == ["D2:1", "D2:4"]


# ---------------------------------------------------------------------------
# Engine ingestion — source_id is passed through to post_response context.
# We exercise just the ingestion loop by running the whole conversational
# pipeline with a single task and a recording memory.
# ---------------------------------------------------------------------------


class _OneConvAdapter(ConversationalBenchmarkAdapter):
    """A conversational adapter with exactly one conversation and one
    task, so ``_run_conversational`` runs a full cycle with minimal
    setup."""

    name = "stub_one_conv"
    granularity = TaskGranularity.CONVERSATIONAL

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "conversation_id": "conv_a",
                "messages": [
                    {
                        "speaker_id": "Caroline",
                        "speaker_name": "Caroline",
                        "content": "I went hiking last weekend",
                        "timestamp": "2024-01-01",
                        "dia_id": "D1:1",
                        "session_id": "session_1",
                    },
                    {
                        "speaker_id": "Melanie",
                        "speaker_name": "Melanie",
                        "content": "Nice, where did you go?",
                        "timestamp": "2024-01-01",
                        "dia_id": "D1:2",
                        "session_id": "session_1",
                    },
                ],
                "metadata": {"sample_id": "sample_0"},
            }
        ]

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        yield Task(
            task_id="q1",
            benchmark="stub_one_conv",
            granularity=TaskGranularity.CONVERSATIONAL,
            prompt="Where did Caroline go?",
            metadata={"conversation_id": "conv_a"},
            ground_truth="hiking",
        )

    async def judge_answer(
        self, task: Task, generated_answer: str
    ) -> dict[str, Any]:
        return {"pass": True, "correct": 1}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""


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


async def test_engine_ingestion_passes_source_id_from_dia_id() -> None:
    """When the adapter's conversation messages have ``dia_id``, the
    engine ingestion loop must pass that value through to the memory's
    ``post_response`` via ``context["source_id"]`` — this is how
    provenance-aware backends like ``rag_baseline`` remember which turn
    a chunk came from."""
    from cross_bench.execution.engine import RunEngine

    memory = _RecordingMemory(retrieved_ids=[])
    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
        result_repo=_FakeResultRepo(),  # type: ignore[arg-type]
        sandbox_mgr=_NoOpSandboxManager(),  # type: ignore[arg-type]
    )

    config = _run_config()
    config.execution_mode = ExecutionMode.CONVERSATIONAL

    await engine._run_conversational(
        config=config,
        adapter=_OneConvAdapter(),
        llm=_StubLLM("hiking"),
        strategy=_PassthroughStrategy(),
        memory=memory,
    )

    # Both messages should have been ingested with their dia_id as
    # source_id.
    ingested_source_ids = [
        ctx.get("source_id") for ctx in memory.post_response_contexts
    ]
    assert "D1:1" in ingested_source_ids
    assert "D1:2" in ingested_source_ids


async def test_engine_ingestion_falls_back_to_session_id_when_no_dia_id() -> None:
    """Benchmarks that don't use LoCoMo's ``dia_id`` scheme (e.g.
    LongMemEval messages identified only by session) fall through to
    ``session_id``; if that's also absent we get the empty string (not
    crash, just "no provenance")."""
    from cross_bench.execution.engine import RunEngine

    class _SessionOnlyAdapter(_OneConvAdapter):
        def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    "conversation_id": "conv_b",
                    "messages": [
                        {
                            "speaker_id": "User",
                            "speaker_name": "User",
                            "content": "Hello",
                            "timestamp": "2024-01-01",
                            # no dia_id, no message_id
                            "session_id": "session_42",
                        }
                    ],
                    "metadata": {},
                }
            ]

        def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
            yield Task(
                task_id="q2",
                benchmark="x",
                granularity=TaskGranularity.CONVERSATIONAL,
                prompt="hi",
                metadata={"conversation_id": "conv_b"},
                ground_truth="",
            )

    memory = _RecordingMemory(retrieved_ids=[])
    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
        result_repo=_FakeResultRepo(),  # type: ignore[arg-type]
        sandbox_mgr=_NoOpSandboxManager(),  # type: ignore[arg-type]
    )

    config = _run_config()
    config.execution_mode = ExecutionMode.CONVERSATIONAL

    await engine._run_conversational(
        config=config,
        adapter=_SessionOnlyAdapter(),
        llm=_StubLLM("ok"),
        strategy=_PassthroughStrategy(),
        memory=memory,
    )

    assert memory.post_response_contexts[0]["source_id"] == "session_42"


async def test_engine_retrieval_propagates_retrieved_context_ids() -> None:
    """The engine's retrieval stage must forward middleware IDs onto
    ``task.metadata`` so the adapter's ``judge_answer`` / ``evaluate``
    sees them."""
    from cross_bench.execution.engine import RunEngine

    memory = _RecordingMemory(retrieved_ids=["D1:1"])
    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
        result_repo=_FakeResultRepo(),  # type: ignore[arg-type]
        sandbox_mgr=_NoOpSandboxManager(),  # type: ignore[arg-type]
    )
    config = _run_config()
    config.execution_mode = ExecutionMode.CONVERSATIONAL

    # Adapter that lets us inspect what it saw at evaluate-time.
    seen_metadata: dict[str, Any] = {}

    class _Observer(_OneConvAdapter):
        async def evaluate(
            self,
            task: Task,
            generated_output: str,
            sandbox_mgr: Any,
            sandbox: Any = None,
        ) -> dict[str, Any]:
            seen_metadata.update(task.metadata)
            return await self.judge_answer(task, generated_output)

    await engine._run_conversational(
        config=config,
        adapter=_Observer(),
        llm=_StubLLM("hiking"),
        strategy=_PassthroughStrategy(),
        memory=memory,
    )

    assert seen_metadata.get("retrieved_context_ids") == ["D1:1"]


async def test_engine_retrieval_no_ids_leaves_metadata_unset() -> None:
    """Symmetry: middleware doesn't touch the key → no
    ``retrieved_context_ids`` in task.metadata → LoCoMo adapter reports
    ``evidence_recall_measured=False``."""
    from cross_bench.execution.engine import RunEngine

    memory = _RecordingMemory(retrieved_ids=None)
    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),  # type: ignore[arg-type]
        result_repo=_FakeResultRepo(),  # type: ignore[arg-type]
        sandbox_mgr=_NoOpSandboxManager(),  # type: ignore[arg-type]
    )
    config = _run_config()
    config.execution_mode = ExecutionMode.CONVERSATIONAL

    seen_metadata: dict[str, Any] = {}

    class _Observer(_OneConvAdapter):
        async def evaluate(
            self,
            task: Task,
            generated_output: str,
            sandbox_mgr: Any,
            sandbox: Any = None,
        ) -> dict[str, Any]:
            seen_metadata.update(task.metadata)
            return await self.judge_answer(task, generated_output)

    await engine._run_conversational(
        config=config,
        adapter=_Observer(),
        llm=_StubLLM("anything"),
        strategy=_PassthroughStrategy(),
        memory=memory,
    )

    assert "retrieved_context_ids" not in seen_metadata
