"""Run engine: orchestrates a full benchmark evaluation run."""

from __future__ import annotations

import asyncio
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from cross_bench.artifact_store import ArtifactStore
from cross_bench.benchmarks import get_adapter
from cross_bench.benchmarks.base import ConversationalBenchmarkAdapter
from cross_bench.evaluation.memscore import (
    compute_latency_stats,
    compute_memscore,
    memscore_to_dict,
)
from cross_bench.evaluation.provider_prompts import format_answer_prompt
from cross_bench.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    judge_relevance,
    RetrievalMetrics,
)
from cross_bench.execution.modes import (
    execute_agentic,
    execute_conversational,
    execute_gold_eval,
    execute_single_shot,
)
from cross_bench.execution.scaffolds.mini_swe_agent import (
    execute_mini_swe_agent,
)
from cross_bench.execution.scaffolds.seal import execute_seal
from cross_bench.execution.sandbox import SandboxManager
from cross_bench.notifications import (
    _derive_outcome as _derive_outcome_from_result,
    notify_task_done,
)
from cross_bench.models import (
    ExecutionMode,
    RunConfig,
    RunResult,
    RunStatus,
    TaskResult,
    TaskStatus,
)
from cross_bench.results.repository import ResultRepository
from cross_bench.systems.registry import (
    get_llm_driver,
    get_memory_middleware,
    get_prompt_strategy,
)

logger = structlog.get_logger()


_MEMORY_TOKEN_STAT_KEYS: tuple[str, ...] = (
    "tokens_in",
    "tokens_out",
    "llm_calls",
    "io_tokens_in",
    "io_tokens_out",
)


def _snapshot_memory_stats(memory: Any) -> dict[str, int]:
    """Read the middleware's cumulative internal-token counters.

    Returns the five-key dict declared by
    :meth:`cross_bench.systems.base.MemoryMiddleware.get_internal_token_stats`,
    or zeros when no middleware is active / the method is missing. The
    engine takes one snapshot before each task's memory operations and
    one after, then stamps the deltas on the task's evaluation dict
    (see :func:`_stamp_memory_delta`).
    """
    if memory is None:
        return {k: 0 for k in _MEMORY_TOKEN_STAT_KEYS}
    getter = getattr(memory, "get_internal_token_stats", None)
    if getter is None:
        return {k: 0 for k in _MEMORY_TOKEN_STAT_KEYS}
    try:
        stats = getter()
    except Exception:
        return {k: 0 for k in _MEMORY_TOKEN_STAT_KEYS}
    if not isinstance(stats, dict):
        return {k: 0 for k in _MEMORY_TOKEN_STAT_KEYS}
    return {k: int(stats.get(k, 0) or 0) for k in _MEMORY_TOKEN_STAT_KEYS}


def _stamp_memory_delta(
    evaluation: dict[str, Any],
    before: dict[str, int],
    after: dict[str, int],
    *,
    additive: bool = False,
) -> None:
    """Stamp ``after - before`` deltas onto a task's evaluation dict.

    Keys land as ``memory_tokens_in``, ``memory_tokens_out``,
    ``memory_llm_calls``, ``memory_io_tokens_in``,
    ``memory_io_tokens_out``. With ``additive=True``, existing values
    on the dict are added to (used for the conversational-flow first
    task that needs to roll up its conversation's ingestion delta on
    top of its own search delta).
    """
    for key in _MEMORY_TOKEN_STAT_KEYS:
        delta = after.get(key, 0) - before.get(key, 0)
        if delta < 0:
            # Counters reset mid-task (e.g. middleware bug) — clamp at
            # zero rather than emit nonsense negative tokens.
            delta = 0
        eval_key = f"memory_{key}"
        if additive and eval_key in evaluation and isinstance(
            evaluation[eval_key], (int, float)
        ):
            evaluation[eval_key] = int(evaluation[eval_key]) + delta
        else:
            evaluation[eval_key] = delta


def _count_tokens(text: str) -> int:
    """Token count under a single, model-agnostic encoder (``cl100k_base``).

    We deliberately do NOT use the bench LLM's native tokenizer. The metric
    we report — retrieval / scaffolding token counts — is a property of the
    *content* the memory system surfaced, not of the specific model that
    consumed it. Using a different encoder per model would silently make the
    same chunk count as 100 tokens for Claude and 95 for GPT-4o, and the
    leaderboard column wouldn't mean one thing across rows.

    ``cl100k_base`` (GPT-4 / GPT-3.5-Turbo) is the natural standard: it's
    already what every non-OpenAI model in CrossBench falls back to, every
    OpenAI model up through GPT-4-Turbo uses it natively, and Anthropic /
    most open-weight tokenizers track it within ~5% on English text. The
    only models whose native count differs notably are GPT-4o family
    (``o200k_base``) and they're still tightly correlated with ``cl100k_base``
    on the same input.

    tiktoken caches encoders internally, so calls in the hot loop pay no
    per-call init cost.
    """
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


class RunEngine:
    """Orchestrates a complete benchmark evaluation run.

    Supports two evaluation paradigms:

    **Code-generation benchmarks** (HumanEval, MBPP, SWE-Bench, …):
      For each task: format → (memory augment) → LLM call → sandbox execute → evaluate

    **Conversational benchmarks** (LoCoMo, LongMemEval, ConvoMem, …):
      1. Ingest conversations into the memory system
      2. Wait for indexing to complete (polling phase)
      3. For each question: search memory → build context → LLM answer → judge
      4. Compute MemScore, retrieval metrics, per-category accuracy
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        result_repo: ResultRepository,
        sandbox_mgr: SandboxManager | None = None,
    ):
        self._artifacts = artifact_store
        self._results = result_repo
        self._sandbox_mgr = sandbox_mgr or SandboxManager()

    async def execute_run(self, config: RunConfig) -> RunResult:
        started_at = datetime.now(timezone.utc)
        logger.info("run.start", run_id=config.run_id, benchmark=config.benchmark)

        await self._artifacts.store_run_config(config)
        await self._results.create_run(config)

        # Resolve components
        adapter = get_adapter(config.benchmark)
        llm = get_llm_driver(config.system.llm)
        strategy = get_prompt_strategy(config.system.prompt_strategy)
        memory = None
        if config.system.memory and config.system.memory.enabled:
            memory = get_memory_middleware(config.system.memory)

        # Dispatch to the appropriate pipeline
        if isinstance(adapter, ConversationalBenchmarkAdapter):
            task_results = await self._run_conversational(
                config, adapter, llm, strategy, memory,
            )
        else:
            task_results = await self._run_code_gen(
                config, adapter, llm, strategy, memory,
            )

        # Aggregate. Pass the adapter so the aggregator can pre-seed
        # the summary with the benchmark's declared metric schema —
        # keeps summary.json columns stable across memory systems and
        # across runs.
        completed_at = datetime.now(timezone.utc)
        metrics = self._aggregate(task_results, adapter=adapter)

        run_result = RunResult(
            run_id=config.run_id,
            config=config,
            task_results=task_results,
            aggregate_metrics=metrics,
            started_at=started_at,
            completed_at=completed_at,
            status=RunStatus.COMPLETED,
        )

        await self._artifacts.store_run_summary(run_result)
        await self._results.complete_run(config.run_id, metrics, completed_at)

        logger.info("run.complete", run_id=config.run_id, metrics=metrics)
        return run_result

    # ------------------------------------------------------------------
    # Code-generation pipeline (original)
    # ------------------------------------------------------------------

    async def _run_code_gen(self, config, adapter, llm, strategy, memory):
        params = {**adapter.get_default_params(), **config.benchmark_params}
        tasks = list(adapter.load_tasks(params))
        logger.info("run.tasks_loaded", count=len(tasks))

        # Image pre-pull pre-flight (deviations.md S4). Walk every task
        # to collect the union of per-task images, dedupe, and pull in
        # parallel before the loop begins. Critical for SWE-bench where
        # the 500-instance Verified set ships 500 per-instance Docker
        # images — cold pulls would otherwise eat into each task's
        # timeout. For benchmarks with one shared image, this is a single
        # cheap pull. ``hasattr`` keeps duck-typed test fakes happy.
        if hasattr(self._sandbox_mgr, "prefetch_images"):
            try:
                images = {adapter.get_sandbox_image(t) for t in tasks}
                images.discard("")
                if images:
                    # Let the adapter translate bare local tags to
                    # pullable URLs (Docker Hub / ECR mirror / passthrough)
                    # before the prefetch. SWE-bench uses this to bridge
                    # its bare ``sweb.eval.x86_64.<id>`` tag to the
                    # actual registry source — see
                    # ``BenchmarkAdapter.resolve_image_sources`` and
                    # deviations.md S4.
                    if hasattr(adapter, "resolve_image_sources"):
                        source_map = adapter.resolve_image_sources(
                            images, params
                        )
                    else:
                        source_map = {img: img for img in images}
                    await self._sandbox_mgr.prefetch_images(source_map)
            except Exception as e:
                # Don't abort the run — per-task ``create()`` will surface
                # the real per-image failure if pulls still fail then.
                logger.warning("run.prefetch_images_failed", error=str(e))

        # Execution-mode dispatch. ``MINI_SWE_AGENT`` is SWE-bench's
        # paper-faithful scaffold (deviations.md S1); ``AGENTIC`` is the
        # generic bash-REPL loop; ``SINGLE_SHOT`` is the default for code-gen
        # adapters that emit a single completion (HumanEval, RepoBench, etc).
        if config.execution_mode == ExecutionMode.MINI_SWE_AGENT:
            executor = execute_mini_swe_agent
        elif config.execution_mode == ExecutionMode.SEAL:
            executor = execute_seal
        elif config.execution_mode == ExecutionMode.GOLD_EVAL:
            executor = execute_gold_eval
        elif config.execution_mode == ExecutionMode.AGENTIC:
            executor = execute_agentic
        else:
            executor = execute_single_shot

        # Run-scoped corpus ingestion (CodeRAG-Bench-style).
        #
        # An adapter that returns a non-None ``corpus_for_run`` opts
        # into a per-run shared memory corpus (e.g. BigCodeBench with
        # the CodeRAG-Bench library-documentation corpus). The engine
        # ingests once before the task loop and — controlled by the
        # adapter's ``reset_memory_per_task`` flag — suppresses
        # ``memory.reset()`` between tasks so the corpus survives.
        #
        # When ``corpus_for_run`` returns ``None`` (the default for
        # every existing benchmark) the engine takes the legacy
        # per-task path: reset between tasks, no run-scoped ingest.
        # The cost of opting out is exactly one ``hasattr`` check
        # plus one method call — no behavior change.
        corpus_iter = None
        if memory:
            try:
                corpus_iter = adapter.corpus_for_run(params, tasks=tasks)
            except Exception as e:
                logger.error("run.corpus_for_run_error", error=str(e))
                corpus_iter = None

        if memory and corpus_iter is not None:
            await memory.reset()
            ingest_t0 = time.monotonic()
            n_ingested = 0
            try:
                async for item in corpus_iter:
                    await memory.post_response(
                        "",
                        item.content,
                        {
                            "source_id": item.source_id,
                            **(item.metadata or {}),
                        },
                    )
                    n_ingested += 1
            except Exception as e:
                logger.error(
                    "run.corpus_ingest_error",
                    error=str(e),
                    items_so_far=n_ingested,
                )
            ingest_ms = (time.monotonic() - ingest_t0) * 1000
            logger.info(
                "run.corpus_ingested",
                run_id=config.run_id,
                items=n_ingested,
                elapsed_ms=round(ingest_ms, 1),
            )

        task_results: list[TaskResult] = []
        # ``outcomes`` mirrors ``task_results`` for the Slack progress bar:
        # one ``resolved`` / ``failed`` / ``errored`` / ``timeout`` string
        # per finished task in shard order.
        outcomes: list[str] = []
        for i, task in enumerate(tasks):
            logger.info(
                "task.start",
                run_id=config.run_id,
                task_id=task.task_id,
                progress=f"{i + 1}/{len(tasks)}",
            )
            memory_before_task = _snapshot_memory_stats(memory)
            try:
                result = await executor(
                    task=task,
                    config=config,
                    adapter=adapter,
                    llm=llm,
                    strategy=strategy,
                    memory=memory,
                    sandbox_mgr=self._sandbox_mgr,
                )
            except Exception as e:
                logger.error("task.error", task_id=task.task_id, error=str(e))
                result = TaskResult(
                    task_id=task.task_id,
                    run_id=config.run_id,
                    status=TaskStatus.ERROR,
                    steps=[],
                    final_output="",
                    error=str(e),
                )

            # Per-task memory delta. Captures ingest_for_memory +
            # pre_prompt + post_response combined since each happens
            # inside the executor.
            memory_after_task = _snapshot_memory_stats(memory)
            _stamp_memory_delta(
                result.evaluation,
                memory_before_task,
                memory_after_task,
            )

            task_results.append(result)
            outcomes.append(_derive_outcome_from_result(result))
            await self._artifacts.store_task_result(config.run_id, result)
            await self._results.save_task_result(result)
            notify_task_done(
                result, config, task=task,
                index=i + 1, total=len(tasks), outcomes=outcomes,
            )

            try:
                await adapter.teardown_task(task)
            except Exception as e:
                logger.warning(
                    "adapter.teardown_task_failed",
                    task_id=task.task_id,
                    error=str(e),
                )

            # Reset between tasks ONLY when the adapter doesn't have
            # a run-scoped corpus to preserve. ``reset_memory_per_task``
            # defaults to True so existing benchmarks (RepoBench,
            # CrossCodeEval, …) keep their per-task isolation.
            if memory and getattr(adapter, "reset_memory_per_task", True):
                await memory.reset()

            logger.info(
                "task.complete",
                task_id=task.task_id,
                status=result.status.value,
            )

        return task_results

    # ------------------------------------------------------------------
    # Conversational pipeline (LoCoMo / LongMemEval / ConvoMem)
    # ------------------------------------------------------------------

    async def _run_conversational(self, config, adapter, llm, strategy, memory):
        """Execute the 4-stage conversational memory evaluation pipeline.

        Stages (following EverMemOS + MemoryBench architecture):
          1. ADD — Ingest conversations into the memory system
          2. INDEX — Wait for memory system to finish indexing
          3. SEARCH + ANSWER — For each question, retrieve and answer
          4. EVALUATE — Judge answers, compute retrieval + MemScore metrics
        """
        params = {**adapter.get_default_params(), **config.benchmark_params}

        # Load all conversations and tasks upfront
        conversations = adapter.load_conversations(params)
        all_tasks = list(adapter.load_tasks(params))

        # Stratified sampling: if sample_per_type is set, sample N per category
        sample_per_type = params.get("sample_per_type")
        if sample_per_type:
            all_tasks = self._stratified_sample(all_tasks, sample_per_type)

        # Group tasks by conversation_id for per-conversation evaluation.
        # Each conversation is processed independently: ingest → index → answer → reset.
        # This prevents memory from one conversation bleeding into another.
        tasks_by_conv: dict[str, list] = {}
        for task in all_tasks:
            conv_id = task.metadata.get("conversation_id", "")
            tasks_by_conv.setdefault(conv_id, []).append(task)

        conv_by_id = {c["conversation_id"]: c for c in conversations}

        logger.info("run.tasks_loaded", count=len(all_tasks),
                     conversations=len(tasks_by_conv))

        indexing_wait = params.get("indexing_wait_seconds", 0)
        indexing_poll_interval = params.get("indexing_poll_interval", 5)
        total_messages_ingested = 0
        ingest_start = time.monotonic()

        task_results: list[TaskResult] = []
        search_latencies: list[float] = []
        inference_latencies: list[float] = []
        context_token_counts: list[int] = []
        retrieval_token_counts: list[int] = []
        scaffolding_token_counts: list[int] = []
        all_retrieval_metrics: list[RetrievalMetrics] = []
        judge_config = params.get("llm_judge", {})

        # Pre-compute the answer-prompt scaffolding token count. Scaffolding
        # is constant per (provider, prompt_style); tokenizing the template
        # with empty {context}/{question} substitutions gives the exact
        # boilerplate cost the retrieval contribution rides on top of.
        provider_for_scaffolding = ""
        if config.system.memory and config.system.memory.enabled:
            provider_for_scaffolding = config.system.memory.provider
        scaffolding_text = format_answer_prompt(
            provider=provider_for_scaffolding, question="", context="",
        )
        scaffolding_tokens_per_task = _count_tokens(scaffolding_text)

        # Process each conversation independently
        for conv_id, conv_tasks in tasks_by_conv.items():
            conv = conv_by_id.get(conv_id)

            # Stage 1: Ingest this conversation into memory
            conv_ingest_before: dict[str, int] = {
                k: 0 for k in _MEMORY_TOKEN_STAT_KEYS
            }
            conv_ingest_after: dict[str, int] = {
                k: 0 for k in _MEMORY_TOKEN_STAT_KEYS
            }
            if conv and memory:
                # Reset memory before each conversation to prevent cross-contamination
                await memory.reset()
                # Snapshot post-reset baseline so the delta over the
                # ingestion loop captures only the messages we just
                # streamed in (not anything that survived the reset).
                conv_ingest_before = _snapshot_memory_stats(memory)

                logger.info("run.conv_ingesting", conv_id=conv_id,
                            messages=len(conv.get("messages", [])))
                for msg in conv.get("messages", []):
                    speaker = msg.get("speaker_name", msg.get("speaker_id", "unknown"))
                    content = msg.get("content", "")
                    # Pass raw content — speaker attribution is handled via
                    # context["speaker_id"] and stored as speaker_name in the
                    # middleware. _format_messages() prepends "speaker:" when
                    # formatting for LLM prompts.
                    #
                    # ``source_id`` is the provenance tag: middlewares that
                    # preserve per-item IDs (e.g. ``rag_baseline``) tag each
                    # chunk with it so retrieval can surface a matching
                    # ``retrieved_context_ids`` list. For LoCoMo this is the
                    # upstream ``dia_id`` (e.g. ``"D1:3"``); when absent we
                    # fall back to the ``session_id`` (``"S1"``) or the empty
                    # string. See ``MemoryMiddleware`` class docstring.
                    source_id = (
                        msg.get("dia_id")
                        or msg.get("message_id")
                        or msg.get("session_id")
                        or ""
                    )
                    await memory.post_response("", content, {
                        "conversation_id": conv_id,
                        "timestamp": msg.get("timestamp", ""),
                        "speaker_id": speaker,
                        "source_id": str(source_id),
                    })
                    total_messages_ingested += 1

                # Flush any remaining accumulator
                if hasattr(memory, "_flush_accumulator") and hasattr(memory, "_accumulator"):
                    if memory._accumulator:
                        await memory._flush_accumulator(flush=True)

                # Stage 2: Indexing wait phase
                if indexing_wait > 0:
                    logger.info("run.indexing_wait", conv_id=conv_id,
                                wait_seconds=indexing_wait)
                    waited = 0
                    while waited < indexing_wait:
                        if hasattr(memory, "is_indexed") and callable(memory.is_indexed):
                            if await memory.is_indexed():
                                break
                        await asyncio.sleep(min(indexing_poll_interval,
                                                indexing_wait - waited))
                        waited += indexing_poll_interval

                # Snapshot the post-ingestion counters; the delta vs
                # ``conv_ingest_before`` is the per-conversation
                # ingestion cost. Stamped on the first task of this
                # conversation below so per-task aggregation rolls
                # ingestion into the run total exactly once.
                conv_ingest_after = _snapshot_memory_stats(memory)

            # Stage 3 + 4: Answer this conversation's questions while memory is loaded
            logger.info("run.conv_answering", conv_id=conv_id,
                        questions=len(conv_tasks))

            first_task_of_conv = True
            for task in conv_tasks:
                task_idx = len(task_results)
                logger.info(
                    "task.start",
                    run_id=config.run_id,
                    task_id=task.task_id,
                    progress=f"{task_idx + 1}/{len(all_tasks)}",
                )

                # Snapshot memory counters before this task's search so
                # the delta below captures only this task's
                # contribution. Ingestion delta is added to the first
                # task of the conv (see ``first_task_of_conv`` below).
                memory_before_task = _snapshot_memory_stats(memory)

                # Stage 3a: Search — retrieve relevant context from memory
                search_context = ""
                raw_search_results: list[str] = []
                if memory:
                    t0 = time.monotonic()
                    # ``retrieval_context`` is a mutable side-channel:
                    # middlewares MAY populate ``retrieved_context_ids``
                    # with the list of per-item provenance IDs for the
                    # retrieved chunks. See ``MemoryMiddleware`` class
                    # docstring. Hoisted to a named variable so we can
                    # inspect it after the call.
                    retrieval_context: dict[str, Any] = {"task_id": task.task_id}
                    augmented = await memory.pre_prompt(
                        task.prompt, retrieval_context
                    )
                    search_ms = (time.monotonic() - t0) * 1000
                    search_latencies.append(search_ms)

                    # Extract the memory context that was prepended
                    if augmented != task.prompt:
                        idx = augmented.find(task.prompt)
                        search_context = augmented[:idx].strip() if idx > 0 else ""
                        # Split context into individual results for retrieval metrics
                        raw_search_results = [
                            r.strip() for r in search_context.split("\n\n---\n\n")
                            if r.strip()
                        ]
                        if not raw_search_results and search_context:
                            raw_search_results = [
                                r.strip() for r in search_context.split("\n\n")
                                if r.strip()
                            ]

                    # Exact context-token count under the standard encoder.
                    # Feeds MemScore (accuracy / latency / context_tokens).
                    context_token_counts.append(_count_tokens(search_context))

                    task.metadata["search_context"] = search_context
                    task.metadata["search_latency_ms"] = search_ms

                    # Forward retrieval provenance to the adapter. We
                    # propagate ``None`` / missing as an explicit absence
                    # (distinct from an empty list which means "retriever
                    # ran, nothing matched"); benchmarks such as
                    # ``locomo_original`` use this distinction to report
                    # ``evidence_recall_measured`` honestly.
                    retrieved_ids = retrieval_context.get("retrieved_context_ids")
                    if isinstance(retrieved_ids, list):
                        task.metadata["retrieved_context_ids"] = [
                            str(x) for x in retrieved_ids
                        ]

                # Stage 3b: Compute retrieval metrics for this question
                if raw_search_results and task.ground_truth:
                    try:
                        relevance_labels = await judge_relevance(
                            question=task.prompt,
                            golden_answer=str(task.ground_truth),
                            search_results=raw_search_results,
                            llm_config=judge_config,
                        )
                        ret_metrics = compute_retrieval_metrics(
                            relevance_labels,
                            k=params.get("retrieval_k", 5),
                        )
                        all_retrieval_metrics.append(ret_metrics)
                    except Exception as e:
                        logger.debug("retrieval_metrics.error", error=str(e))

                inference_ms_for_task: float | None = None
                try:
                    # Stage 3c + 4: Answer + Evaluate
                    infer_t0 = time.monotonic()
                    result = await execute_conversational(
                        task=task,
                        config=config,
                        adapter=adapter,
                        llm=llm,
                        strategy=strategy,
                        memory=None,  # already searched above
                        sandbox_mgr=self._sandbox_mgr,
                    )
                    inference_ms_for_task = (time.monotonic() - infer_t0) * 1000
                    inference_latencies.append(inference_ms_for_task)
                except Exception as e:
                    logger.error("task.error", task_id=task.task_id, error=str(e))
                    result = TaskResult(
                        task_id=task.task_id,
                        run_id=config.run_id,
                        status=TaskStatus.ERROR,
                        steps=[],
                        final_output="",
                        error=str(e),
                    )

                # Attach timing metrics to evaluation. ``total_latency_ms``
                # is the wall-clock from the start of search to the end of
                # the last evaluator step — i.e. *what the user waited for*.
                # The benchmark adapter stamps ``llm_judge_latency_ms`` on
                # the eval dict when judging is enabled (see
                # ``locomo_original.judge_answer``); we sum it in here so
                # consumers don't have to.
                #
                # Per-task values come from THIS task only — never index
                # ``inference_latencies[-1]``, which leaks the previous
                # task's latency onto a failed task.
                search_ms_for_task = (
                    search_latencies[-1] if memory else 0.0
                )
                judge_ms_for_task = float(
                    result.evaluation.get("llm_judge_latency_ms") or 0.0
                )
                if memory:
                    result.evaluation["search_latency_ms"] = search_ms_for_task
                if inference_ms_for_task is not None:
                    result.evaluation["inference_latency_ms"] = (
                        inference_ms_for_task
                    )
                result.evaluation["total_latency_ms"] = round(
                    search_ms_for_task
                    + (inference_ms_for_task or 0.0)
                    + judge_ms_for_task,
                    1,
                )

                # Two complementary token quantities per task, exact under
                # the standard encoder (so cross-system rows are comparable):
                #   * retrieval_tokens — what the memory system surfaced,
                #     i.e. ``search_context + question``. Varies by
                #     middleware quality and is the input to MemScore-style
                #     comparisons.
                #   * scaffolding_tokens — answer-prompt template overhead
                #     around the retrieval (system instructions, FINAL
                #     ANSWER directive, etc.). Constant per (provider,
                #     style); reported so total ≈ retrieval + scaffolding
                #     can be reasoned about without re-tokenizing.
                retrieval_text = (
                    search_context + "\n" + task.prompt
                    if search_context else task.prompt
                )
                retrieval_tokens = _count_tokens(retrieval_text)
                retrieval_token_counts.append(retrieval_tokens)
                scaffolding_token_counts.append(scaffolding_tokens_per_task)
                result.evaluation["retrieval_tokens"] = retrieval_tokens
                result.evaluation["scaffolding_tokens"] = scaffolding_tokens_per_task

                # Snapshot post-task counters and stamp the per-task
                # memory delta. Tokens stamped under ``memory_*`` keys
                # land in ``_aggregate``'s generic numeric averager AND
                # the explicit ``total_memory_*`` rollup below.
                memory_after_task = _snapshot_memory_stats(memory)
                _stamp_memory_delta(
                    result.evaluation,
                    memory_before_task,
                    memory_after_task,
                )
                if first_task_of_conv and memory is not None:
                    # Roll the conversation's ingestion delta onto its
                    # first task. Sum-across-tasks now yields the run
                    # total without double-counting.
                    _stamp_memory_delta(
                        result.evaluation,
                        conv_ingest_before,
                        conv_ingest_after,
                        additive=True,
                    )
                first_task_of_conv = False

                task_results.append(result)
                await self._artifacts.store_task_result(config.run_id, result)
                await self._results.save_task_result(result)
                notify_task_done(
                    result, config, task=task,
                    index=len(task_results), total=len(all_tasks),
                    outcomes=[_derive_outcome_from_result(r) for r in task_results],
                )

                logger.info(
                    "task.complete",
                    task_id=task.task_id,
                    status=result.status.value,
                )

        ingest_elapsed_ms = (time.monotonic() - ingest_start) * 1000
        logger.info("run.ingestion_time_ms", elapsed_ms=round(ingest_elapsed_ms, 1),
                     messages=total_messages_ingested)

        # Final cleanup
        if memory:
            await memory.reset()

        # Attach extra metrics to task results for aggregation
        self._attach_conversational_metrics(
            task_results, search_latencies, inference_latencies,
            context_token_counts, retrieval_token_counts,
            scaffolding_token_counts,
            all_retrieval_metrics, ingest_elapsed_ms,
        )

        return task_results

    def _attach_conversational_metrics(
        self,
        task_results: list[TaskResult],
        search_latencies: list[float],
        inference_latencies: list[float],
        context_token_counts: list[int],
        retrieval_token_counts: list[int],
        scaffolding_token_counts: list[int],
        all_retrieval_metrics: list[RetrievalMetrics],
        ingest_elapsed_ms: float,
    ) -> None:
        """Attach MemScore, timing, and retrieval metrics to the first result for aggregation."""
        if not task_results:
            return

        # Compute MemScore
        passed = sum(1 for r in task_results if r.status == TaskStatus.PASS)
        total = len(task_results)
        accuracy = passed / total if total > 0 else 0.0
        avg_latency = sum(search_latencies) / len(search_latencies) if search_latencies else 0.0
        avg_tokens = sum(context_token_counts) / len(context_token_counts) if context_token_counts else 0.0

        ms = compute_memscore(accuracy, avg_latency, avg_tokens)
        ms.latency = compute_latency_stats(search_latencies)

        # Store in first result's evaluation for _aggregate to pick up
        memscore_data = memscore_to_dict(ms)
        task_results[0].evaluation["_memscore"] = memscore_data

        # Timing breakdown
        total_search_ms = sum(search_latencies)
        total_inference_ms = sum(inference_latencies)
        avg_inference = total_inference_ms / len(inference_latencies) if inference_latencies else 0.0
        avg_retrieval_tokens = (
            sum(retrieval_token_counts) / len(retrieval_token_counts)
            if retrieval_token_counts else 0.0
        )
        avg_scaffolding_tokens = (
            sum(scaffolding_token_counts) / len(scaffolding_token_counts)
            if scaffolding_token_counts else 0.0
        )

        timing = {
            "ingestion_time_ms": round(ingest_elapsed_ms, 1),
            "total_retrieval_time_ms": round(total_search_ms, 1),
            "avg_retrieval_time_ms": round(avg_latency, 1),
            "total_inference_time_ms": round(total_inference_ms, 1),
            "avg_inference_time_ms": round(avg_inference, 1),
            "avg_retrieval_tokens": round(avg_retrieval_tokens),
            "total_retrieval_tokens": sum(retrieval_token_counts),
            "avg_scaffolding_tokens": round(avg_scaffolding_tokens),
            "total_scaffolding_tokens": sum(scaffolding_token_counts),
        }
        task_results[0].evaluation["_timing"] = timing

        logger.info(
            "run.timing_summary",
            ingestion_ms=timing["ingestion_time_ms"],
            total_retrieval_ms=timing["total_retrieval_time_ms"],
            avg_retrieval_ms=timing["avg_retrieval_time_ms"],
            total_inference_ms=timing["total_inference_time_ms"],
            avg_inference_ms=timing["avg_inference_time_ms"],
            avg_retrieval_tokens=timing["avg_retrieval_tokens"],
            avg_scaffolding_tokens=timing["avg_scaffolding_tokens"],
        )

        # Retrieval metrics
        if all_retrieval_metrics:
            ret_agg = aggregate_retrieval_metrics(all_retrieval_metrics)
            task_results[0].evaluation["_retrieval_metrics"] = ret_agg

    def _stratified_sample(self, tasks, sample_per_type: int):
        """Sample N tasks per category for balanced evaluation."""
        import random

        by_category: dict[str, list] = {}
        for task in tasks:
            cat = task.metadata.get("category_name", task.metadata.get("category", "unknown"))
            by_category.setdefault(str(cat), []).append(task)

        sampled = []
        for cat, cat_tasks in by_category.items():
            n = min(sample_per_type, len(cat_tasks))
            sampled.extend(random.sample(cat_tasks, n))
        return sampled

    # ------------------------------------------------------------------
    # Metrics aggregation (unified for both pipelines)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        results: list[TaskResult],
        *,
        adapter: Any | None = None,
    ) -> dict[str, float | None]:
        """Aggregate per-task metrics into a stable summary dict.

        ``adapter`` is the benchmark adapter used to run; when it
        declares a ``metrics_schema()`` we pre-seed every declared key
        with ``None`` so unmeasured-but-declared metrics surface as
        explicit ``null`` rather than silently disappearing from the
        column set. That gives downstream consumers (dashboards,
        leaderboards) a stable schema across memory systems and runs.
        """
        total = len(results)
        if total == 0:
            return {}

        # Pre-seed declared schema with None ("not measured"). Anything
        # the rest of this function fills in overrides the seed — but
        # absence of measurement now shows up as ``null`` in JSON.
        metrics: dict[str, float | None] = {}
        if adapter is not None and hasattr(adapter, "metrics_schema"):
            try:
                for key in adapter.metrics_schema() or []:
                    metrics[key] = None
            except Exception:
                # A buggy schema must not break aggregation.
                pass

        passed = sum(1 for r in results if r.status == TaskStatus.PASS)
        errors = sum(1 for r in results if r.status == TaskStatus.ERROR)
        timeouts = sum(1 for r in results if r.status == TaskStatus.TIMEOUT)

        # Collect per-task numeric metrics
        all_eval_keys: set[str] = set()
        for r in results:
            all_eval_keys.update(
                k for k, v in r.evaluation.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            )

        # Overlay run-level metrics on top of the pre-seeded schema.
        metrics.update({
            "pass_rate": passed / total,
            "total_tasks": float(total),
            "passed": float(passed),
            "failed": float(total - passed - errors - timeouts),
            "errors": float(errors),
            "timeouts": float(timeouts),
        })

        # Also report "accuracy" as alias for conversational benchmarks
        metrics["accuracy"] = passed / total

        # Average numeric evaluation metrics
        for key in all_eval_keys:
            if key == "pass":
                continue
            values = [
                r.evaluation[key]
                for r in results
                if key in r.evaluation and isinstance(r.evaluation[key], (int, float))
            ]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)

        # Top-level pass-rate aliases for the two grading columns the
        # leaderboard cares about — surfaced explicitly so consumers
        # don't have to compute them from ``avg_*`` keys.
        # ``pass_rate_f1``     — fraction of tasks where Porter-stemmed
        #                        F1 cleared the threshold (paper-faithful).
        # ``pass_rate_llm_judge`` — fraction the LLM judge marked correct
        #                          (Mem0-paper "J" score). Only present
        #                          when ``judge.llm_judge.enabled = true``.
        f1_pass_values = [
            r.evaluation.get("f1_pass")
            for r in results
            if isinstance(r.evaluation.get("f1_pass"), (int, float))
        ]
        if f1_pass_values:
            metrics["pass_rate_f1"] = sum(f1_pass_values) / len(f1_pass_values)
        judge_pass_values = [
            r.evaluation.get("llm_judge_correct")
            for r in results
            if isinstance(r.evaluation.get("llm_judge_correct"), (int, float))
        ]
        if judge_pass_values:
            metrics["pass_rate_llm_judge"] = (
                sum(judge_pass_values) / len(judge_pass_values)
            )

        # Per-category accuracy (for LoCoMo / LongMemEval / ConvoMem)
        category_results: dict[str, list[bool]] = {}
        for r in results:
            cat = r.evaluation.get("category", "")
            if cat:
                cat_key = str(cat)
                category_results.setdefault(cat_key, []).append(r.status == TaskStatus.PASS)
        for cat, outcomes in category_results.items():
            metrics[f"accuracy_cat_{cat}"] = sum(outcomes) / len(outcomes)

        # Token usage and per-task latency
        tokens_in_per_task = [sum(s.tokens_in for s in r.steps) for r in results]
        tokens_out_per_task = [sum(s.tokens_out for s in r.steps) for r in results]
        tokens_total_per_task = [i + o for i, o in zip(tokens_in_per_task, tokens_out_per_task)]
        latency_per_task = [sum(s.latency_ms for s in r.steps) for r in results]
        total_tokens_in = sum(tokens_in_per_task)
        total_tokens_out = sum(tokens_out_per_task)
        total_latency = sum(latency_per_task)
        metrics["total_tokens_in"] = float(total_tokens_in)
        metrics["total_tokens_out"] = float(total_tokens_out)
        metrics["total_latency_ms"] = total_latency
        if total > 0:
            metrics["avg_latency_ms"] = total_latency / total
            metrics["avg_tokens_per_task"] = (
                total_tokens_in + total_tokens_out
            ) / total
            metrics["p50_latency_ms"] = _percentile(latency_per_task, 50)
            metrics["p95_latency_ms"] = _percentile(latency_per_task, 95)
            metrics["p50_tokens_per_task"] = _percentile(tokens_total_per_task, 50)
            metrics["p95_tokens_per_task"] = _percentile(tokens_total_per_task, 95)

        # Memory-system internal tokens (separate from the bench-LLM
        # totals above). Per-task deltas were stamped on each result's
        # evaluation dict as ``memory_tokens_in / _out / _llm_calls /
        # _io_tokens_in / _io_tokens_out``. The generic numeric
        # averager above already produces ``avg_memory_*``; we add
        # explicit ``total_memory_*`` columns mirroring the bench-LLM
        # surface so leaderboard consumers don't have to derive them.
        # See deviations.md "Platform / cross-cutting" entry on
        # internal-token accounting for the full contract.
        mem_tokens_in_per_task = [
            int(r.evaluation.get("memory_tokens_in", 0) or 0)
            for r in results
        ]
        mem_tokens_out_per_task = [
            int(r.evaluation.get("memory_tokens_out", 0) or 0)
            for r in results
        ]
        mem_io_in_per_task = [
            int(r.evaluation.get("memory_io_tokens_in", 0) or 0)
            for r in results
        ]
        mem_io_out_per_task = [
            int(r.evaluation.get("memory_io_tokens_out", 0) or 0)
            for r in results
        ]
        mem_calls_per_task = [
            int(r.evaluation.get("memory_llm_calls", 0) or 0)
            for r in results
        ]
        total_mem_tokens_in = sum(mem_tokens_in_per_task)
        total_mem_tokens_out = sum(mem_tokens_out_per_task)
        total_mem_io_in = sum(mem_io_in_per_task)
        total_mem_io_out = sum(mem_io_out_per_task)
        total_mem_calls = sum(mem_calls_per_task)
        metrics["total_memory_tokens_in"] = float(total_mem_tokens_in)
        metrics["total_memory_tokens_out"] = float(total_mem_tokens_out)
        metrics["total_memory_llm_calls"] = float(total_mem_calls)
        metrics["total_memory_io_tokens_in"] = float(total_mem_io_in)
        metrics["total_memory_io_tokens_out"] = float(total_mem_io_out)
        if total > 0:
            metrics["avg_memory_tokens_per_task"] = (
                total_mem_tokens_in + total_mem_tokens_out
            ) / total
            metrics["avg_memory_io_tokens_per_task"] = (
                total_mem_io_in + total_mem_io_out
            ) / total

        # MemScore / retrieval / timing carriers — conversational runs attach
        # one shard-level metrics dict to ``task_results[0].evaluation`` per
        # shard. A single-shard run has exactly one carrier; merged runs (see
        # :mod:`cross_bench.sharding`) have one per shard. Average each numeric
        # field across all carriers so parent summaries reflect the full run
        # rather than just the first shard's slice.
        _merge_shard_carrier(results, "_memscore", metrics)
        _merge_shard_carrier(results, "_retrieval_metrics", metrics)
        _merge_shard_carrier(results, "_timing", metrics)

        return metrics


def _merge_shard_carrier(
    results: list[TaskResult], key: str, metrics: dict[str, float]
) -> None:
    """Average per-shard carrier dicts (``_memscore`` etc.) into ``metrics``.

    Each shard attaches one carrier dict to ``task_results[0].evaluation[key]``
    holding numeric metrics for that shard. We collect every carrier seen in
    ``results`` and emit the mean of each numeric field so a merged run
    reports a true full-run number instead of the first shard's slice.
    Non-numeric fields (e.g. nested ``latency`` dicts) take the first carrier
    verbatim — averaging structured payloads would lose meaning.
    """
    carriers = [r.evaluation[key] for r in results if isinstance(r.evaluation.get(key), dict)]
    if not carriers:
        return
    all_keys: set[str] = set()
    for c in carriers:
        all_keys.update(c.keys())
    for k in all_keys:
        vals = [c[k] for c in carriers if isinstance(c.get(k), (int, float))]
        if vals:
            metrics[k] = sum(vals) / len(vals)
            continue
        # Non-numeric field — keep the first non-null structured value so
        # nested payloads (latency stats, etc.) survive the merge.
        for c in carriers:
            if k in c and c[k] is not None:
                metrics[k] = c[k]
                break


def _percentile(data: list[float], p: float) -> float:
    """Inclusive-method percentile (matches numpy.percentile default).

    Handles the edge cases where ``statistics.quantiles`` raises:
    empty input returns 0.0; a single value is its own percentile.
    """
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    cuts = statistics.quantiles(data, n=100, method="inclusive")
    return cuts[int(p) - 1]
