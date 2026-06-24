"""Abstract benchmark adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, Iterator

from cross_bench.models import CorpusItem, Task, TaskGranularity

if TYPE_CHECKING:
    from cross_bench.execution.sandbox import Sandbox, SandboxManager
    from cross_bench.systems.base import MemoryMiddleware


# Memory-system internal-token columns the engine populates on every
# run, regardless of which middleware is active. Declared here so
# every benchmark's ``metrics_schema()`` can include them in one line
# and the leaderboard column set stays stable across memory systems.
# See deviations.md "Platform / cross-cutting" entry on internal-token
# accounting for the contract.
MEMORY_TOKEN_SCHEMA_KEYS: tuple[str, ...] = (
    "total_memory_tokens_in",
    "total_memory_tokens_out",
    "total_memory_llm_calls",
    "total_memory_io_tokens_in",
    "total_memory_io_tokens_out",
    "avg_memory_tokens_per_task",
    "avg_memory_io_tokens_per_task",
)


class BenchmarkAdapter(ABC):
    """Base class all benchmark adapters must implement.

    Adapters are responsible for:
    1. Loading tasks from the benchmark dataset.
    2. Formatting prompts for the system under test.
    3. Evaluating generated output against ground truth.

    Memory-aware adapters (cross-file code completion, repo-level
    benchmarks) additionally override the optional triple
    :meth:`ingest_for_memory` / :meth:`format_memory_prompt` /
    :meth:`get_retrieval_query` to move per-task context out of the
    prompt and into the memory store. The engine treats those hooks as
    no-ops by default, so existing adapters keep their no-memory
    behavior.
    """

    name: str
    granularity: TaskGranularity

    # If True, evaluation uses LLM judge instead of sandbox execution.
    # Override in subclasses for conversational / long-context benchmarks.
    requires_sandbox: bool = True

    # If True (default), the engine calls ``memory.reset()`` after every
    # task. Adapters that return a non-None ``corpus_for_run`` should
    # set this to ``False`` so the run-scoped corpus survives across
    # tasks. Per-task ingestion (``ingest_for_memory``) still fires
    # normally — it adds task-specific context on top of the corpus.
    reset_memory_per_task: bool = True

    @abstractmethod
    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        """Yield Task objects from the dataset.

        Args:
            params: Benchmark-specific parameters (split, filters, limits, etc.).
        """

    @abstractmethod
    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: SandboxManager,
        sandbox: Sandbox | None = None,
    ) -> dict[str, Any]:
        """Evaluate generated output against ground truth.

        Returns a dict of metric_name -> value (e.g. {"pass": True, "pass@1": 1.0}).
        """

    @abstractmethod
    def get_sandbox_image(self, task: Task | None = None) -> str:
        """Return the Docker image tag for this benchmark's execution sandbox.

        ``task`` is optional: most benchmarks share a single image across
        every task and ignore the argument. Benchmarks whose image varies
        per task (SWE-bench, Terminal-Bench) MUST inspect ``task`` and
        return the per-instance image — the engine pre-creates the
        sandbox using this method, so returning a generic image here
        bypasses the per-instance harness silently.
        """

    def format_prompt(self, task: Task) -> str:
        """Default prompt formatting — just returns the task prompt.

        This is the **no-memory baseline** prompt for code-gen
        benchmarks: any context the dataset bundles with the task is
        already inside ``task.prompt`` (e.g. RepoBench's cross-file
        snippets, CrossCodeEval's pre-retrieved CFC text).

        Override for benchmarks needing special formatting (e.g. adding
        context files). Memory-aware adapters should also override
        :meth:`format_memory_prompt` to return the prompt with
        memory-bound context stripped.
        """
        return task.prompt

    # ------------------------------------------------------------------
    # Memory integration — optional hooks for adapters that move
    # task-bundled context into a memory store at runtime.
    # ------------------------------------------------------------------

    def corpus_for_run(
        self,
        params: dict[str, Any] | None = None,
        tasks: list["Task"] | None = None,
    ) -> AsyncIterator[CorpusItem] | None:
        """Run-scoped corpus to ingest once before the task loop.

        Default: ``None`` — the engine takes the legacy per-task
        ingestion path (``ingest_for_memory`` per task,
        ``memory.reset()`` between tasks).

        When an override returns an async iterator of :class:`CorpusItem`,
        the engine:

        1. Calls ``memory.reset()`` once at run start.
        2. Drains the iterator, calling
           ``memory.post_response("", item.content, ctx)`` per item
           with ``ctx["source_id"] = item.source_id`` plus any
           ``item.metadata`` keys.
        3. Suppresses ``memory.reset()`` between tasks (controlled by
           :attr:`reset_memory_per_task`, which the adapter must set
           to ``False`` for the corpus to survive).
        4. Continues with per-task ``ingest_for_memory`` /
           ``pre_prompt`` / ``post_response`` as usual — those still
           run, layering task-specific context on top of the corpus.

        Used today by ``BigCodeBenchAdapter`` to load the
        CodeRAG-Bench ``library-documentation`` corpus (Wang et al.,
        arXiv:2406.14497) when ``params["use_lib_docs_corpus"]`` is
        set. Future repo-level benchmarks (SWE-Bench-with-memory,
        RepoEval-corpus) can use the same hook.

        Args:
            params: resolved benchmark params dict (same shape as
                :meth:`load_tasks` receives).
            tasks: the materialized task list for this run, in case
                the corpus loader needs to derive run-scoped filters
                from the actual tasks (e.g. BigCodeBench filters the
                library-documentation corpus by the union of ``libs``
                appearing in the run's tasks).
        """
        return None

    async def ingest_for_memory(
        self, task: "Task", memory: "MemoryMiddleware"
    ) -> None:
        """Ingest task-bundled context into memory before retrieval.

        Default: no-op. Override for benchmarks that bundle structured
        context with each task (RepoBench cross-file snippets, future
        repo-level code-RAG tasks). The engine calls this once per task
        immediately before :meth:`MemoryMiddleware.pre_prompt`, and
        immediately after :meth:`MemoryMiddleware.reset` (so the store
        is fresh).

        Implementations call ``memory.post_response("", content, ctx)``
        per chunk; ``ctx["source_id"]`` is the per-chunk provenance tag
        the engine forwards back as ``retrieved_context_ids`` so the
        evaluator can compute retrieval recall against
        dataset-supplied evidence indices.
        """
        return None

    def resolve_image_sources(
        self,
        images: "Iterable[str]",
        params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Map each bare local image tag to its pullable source URL.

        Most adapters return their image names already in pullable form
        (e.g. ``cross-bench-function:latest`` resolves locally if pre-
        loaded, or pulls from Docker Hub's default library namespace).
        For those, the default identity map (``{img: img for img in images}``)
        is correct.

        SWE-bench-style adapters whose ``get_sandbox_image()`` returns a
        bare name that does NOT exist on any default registry MUST
        override this to construct a pullable URL. The sandbox manager's
        :meth:`SandboxManager.prefetch_images` honors the returned map:
        for each ``(local_name, pullable_url)`` pair where ``local_name
        != pullable_url`` it pulls the URL and then re-tags the result
        as ``local_name`` so the sandbox launcher's
        ``containers.run(image=local_name, ...)`` finds it.

        Used by SWE-bench to translate ``sweb.eval.x86_64.<id>:latest``
        (the adapter-internal form, matching upstream's
        ``make_test_spec`` convention) into either:

        - ``docker.io/swebench/sweb.eval.x86_64.<id_1776>:latest`` —
          Princeton's published images, with the ``_1776_`` separator
          quirk (see ``deviations.md`` S4 / ``scripts/
          prestage_swebench_image.py`` / ``workflow.md`` "Local SWE-bench
          smoke");
        - ``<acct>.dkr.ecr.<region>.amazonaws.com/swebench:<id>`` —
          our private mirror, populated by the
          ``cross-bench-dev-mirror-swebench-images`` CodeBuild project.

        Args:
            images: bare local image tags (the return values of
                :meth:`get_sandbox_image`).
            params: resolved benchmark params dict — typically read for
                an ``image_source`` knob (e.g. ``"dockerhub_1776"`` /
                ``"ecr_mirror"`` / ``"local_build"``).

        Returns:
            ``{local_name: pullable_url}``. The keys are the same image
            names the adapter returns from ``get_sandbox_image``; the
            sandbox launcher continues to look them up by that local
            name. Identity-map entries are valid (and skipped by the
            prefetch retag pass).
        """
        return {img: img for img in images}

    async def teardown_task(self, task: "Task") -> None:
        """Per-task cleanup after the task finishes (success or error).

        Runs after the executor returns and task results are persisted,
        before :meth:`MemoryMiddleware.reset` when per-task reset is
        enabled. Default: no-op. RepoBench uses this to remove an on-disk
        synthetic workspace created at ingest time.
        """
        return None

    def format_memory_prompt(self, task: "Task") -> str:
        """LLM prompt to use when memory is enabled.

        Default: same as :meth:`format_prompt`. Override when the
        no-memory baseline prompt embeds context that — under memory
        mode — has been moved to the memory store and would
        double-count if also kept in the prompt (RepoBench's cross-file
        snippets, e.g.).
        """
        return self.format_prompt(task)

    def get_retrieval_query(self, task: "Task") -> str | None:
        """Retrieval query for :meth:`MemoryMiddleware.pre_prompt`.

        Default: ``None`` — the engine falls back to the LLM prompt
        itself as the query.

        Override when a focused query (RepoCoder: imports + last 20
        lines of the cropped code, https://arxiv.org/abs/2303.12570)
        retrieves better than embedding the entire prompt.
        Returning ``None`` preserves the legacy "query == prompt"
        behavior that all benchmarks shipped with originally.
        """
        return None

    def get_default_params(self) -> dict[str, Any]:
        """Default benchmark parameters, overridable per benchmark."""
        return {}

    def metrics_schema(self) -> list[str]:
        """Declared metric keys this benchmark publishes on every run.

        The engine's aggregator pre-seeds summary.json with these keys so
        the column set is **stable across memory systems and across
        runs** — a switch from ``mem0v2_embedded`` to ``evermemos_embedded``
        does NOT add or drop columns. Metrics that no task produced
        surface as ``None`` (JSON ``null``), which is honest "not
        measured" rather than the silent omission of a key.

        Default: empty (legacy benchmarks rely on auto-discovery and
        accept schema drift). Override in conversational benchmarks
        that need leaderboard-grade consistency.
        """
        return []


class ConversationalBenchmarkAdapter(BenchmarkAdapter):
    """Base class for long-context / conversational memory benchmarks.

    These benchmarks follow the EverMemOS evaluation pipeline:
      1. Ingest conversations into the memory system (add stage)
      2. For each question, search the memory system for relevant context
      3. Generate an answer using the LLM with retrieved context
      4. Judge the answer quality (LLM-as-judge or exact match)

    Unlike code-gen benchmarks, these do NOT use Docker sandboxes.
    """

    requires_sandbox: bool = False

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Load conversation data for ingestion into memory.

        Returns a list of conversation dicts, each with:
          - conversation_id: str
          - messages: list[dict] with keys: speaker_id, speaker_name, content, timestamp
          - metadata: dict (optional)
        """
        return []

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: SandboxManager,
        sandbox: Sandbox | None = None,
    ) -> dict[str, Any]:
        """Judge the generated answer against the golden answer.

        Default: uses LLM judge. Override for exact-match or hybrid evaluation.
        """
        return await self.judge_answer(task, generated_output)

    async def judge_answer(
        self,
        task: Task,
        generated_answer: str,
    ) -> dict[str, Any]:
        """LLM-as-judge evaluation. Override for custom evaluation logic."""
        golden = task.ground_truth or ""
        # Simple string match fallback — real benchmarks override this
        correct = generated_answer.strip().lower() == str(golden).strip().lower()
        return {"pass": correct, "correct": int(correct)}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        """Conversational benchmarks don't use sandboxes."""
        return ""
