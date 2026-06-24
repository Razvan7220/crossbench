"""Core domain models for the evaluation platform."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskGranularity(Enum):
    """Classification of benchmark task scope."""

    FUNCTION = "function"
    CONTEST = "contest"
    CROSS_FILE = "cross_file"
    REPO = "repo"
    TERMINAL = "terminal"
    CONVERSATIONAL = "conversational"


class ExecutionMode(Enum):
    """How the system under test interacts with the benchmark."""

    SINGLE_SHOT = "single_shot"
    AGENTIC = "agentic"
    CONVERSATIONAL = "conversational"
    #: Mini-SWE-agent scaffold (pinned upstream commit). The canonical
    #: scaffold for SWE-bench leaderboard submissions; see
    #: ``execution/scaffolds/mini_swe_agent.py`` and deviations.md S1.
    MINI_SWE_AGENT = "mini_swe_agent"
    #: Scale AI's SEAL scaffold — their customized SWE-agent at
    #: ``scaleapi/SWE-agent@scale-customizations``, the canonical
    #: scaffold for SWE-bench Pro leaderboard submissions. Wraps
    #: ``sweagent run-batch`` as a subprocess per task and consumes
    #: the resulting ``.pred`` patch. See
    #: ``execution/scaffolds/seal.py`` and deviations.md S23.
    SEAL = "seal"
    #: Apply ``task.ground_truth`` through the benchmark harness with no
    #: LLM call. Used for SWE-bench Pro harness sanity checks
    #: (``configs/benchmarks/swebench_pro_gold.toml``).
    GOLD_EVAL = "gold_eval"


class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Task: a single unit of work from a benchmark
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    benchmark: str
    granularity: TaskGranularity
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    context_files: list[dict[str, str]] | None = None
    ground_truth: Any | None = None


# ---------------------------------------------------------------------------
# System configuration: what we're evaluating
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    provider: str  # "openai", "anthropic", "generic_openai"
    model: str
    temperature: float = 0.0
    max_tokens: int = 16384
    context_window: int = 128_000  # prompt budget; gpt-4o=128k, gpt-4.1=1M, claude=200k
    top_p: float = 1.0
    # Best-effort determinism hint. Both OpenAI and Anthropic document `seed`
    # as a hint, not a contract — same seed usually but not always produces
    # the same output. Pair with multiple reruns when reporting cell values.
    # ``None`` means "don't send the parameter" (preserves pre-2026-05-25
    # call shape exactly).
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    provider: str  # "mem0v1_api", "mem0v1_embedded", "mem0v2_api", "mem0v2_embedded", "zep_api", "zep_embedded", "letta_api", "memos_embedded", "evermemos_api", "evermemos_embedded", "hindsight_api", "hindsight_embedded", "mastra_api", "mastra_embedded", "rag", "graphiti_embedded"
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    provider: str  # "openai", "sentence_transformer"
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptStrategyConfig:
    type: str  # "zero_shot", "few_shot", "chain_of_thought", "retrieval_augmented"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    llm: LLMConfig
    prompt_strategy: PromptStrategyConfig
    memory: MemoryConfig | None = None
    embedding: EmbeddingConfig | None = None


# ---------------------------------------------------------------------------
# Run configuration: the full spec for one evaluation
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    benchmark: str
    system: SystemConfig
    execution_mode: ExecutionMode = ExecutionMode.SINGLE_SHOT
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_agent_steps: int = 20
    timeout_seconds: int = 300
    sandbox_image: str = ""
    benchmark_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Step and result records
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    step_index: int
    action: str  # "completion", "chat", "tool_call", "memory_query", "memory_store", "bash"
    input_text: str
    output_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    run_id: str
    status: TaskStatus
    steps: list[StepRecord]
    final_output: str
    evaluation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CorpusItem:
    """One document in a benchmark's run-scoped retrieval corpus.

    Returned by :meth:`BenchmarkAdapter.corpus_for_run` (default
    ``None`` ⇒ legacy per-task ingestion). When an adapter returns a
    corpus iterator, the engine drains it once before the task loop,
    posts each item to the memory store via
    :meth:`MemoryMiddleware.post_response`, and **suppresses
    per-task reset** so the corpus survives the whole run.

    Used today by ``BigCodeBenchAdapter`` to ingest the CodeRAG-Bench
    ``library-documentation`` corpus (Wang et al. 2024,
    arXiv:2406.14497). The natural ``MemoryMiddleware`` contract — one
    ``post_response`` per item, ``source_id`` for provenance — is
    preserved; this dataclass is just the typed payload.

    Fields:
      ``content``    — the document body, ingested verbatim into the
                       memory store. Mem0 with ``infer=True`` will
                       summarise it; the RAG baseline keeps it
                       byte-for-byte.
      ``source_id``  — stable per-document ID, surfaced back via
                       ``retrieved_context_ids`` so retrieval recall
                       can be measured against gold (e.g.
                       ``"lib:numpy.argsort"`` for CodeRAG-Bench's
                       library-documentation entries).
      ``metadata``   — vendor-side filtering hints (``library``,
                       ``version``, …). Middlewares that support
                       metadata filters can use these; ones that
                       don't ignore them.
    """

    content: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    task_results: list[TaskResult]
    # ``float | None`` (instead of ``float``) so unmeasured-but-declared
    # metrics surface as JSON ``null``. The benchmark adapter's
    # ``metrics_schema`` declares the canonical metric set; any key the
    # adapter expects but none of the tasks produced gets ``None`` here
    # so summary.json columns stay consistent across runs and memory
    # systems.
    aggregate_metrics: dict[str, float | None]
    started_at: datetime
    completed_at: datetime
    artifact_urls: dict[str, str] = field(default_factory=dict)
    status: RunStatus = RunStatus.COMPLETED
