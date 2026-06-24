"""Execution modes: single-shot completion, agentic multi-step, and conversational."""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.execution.sandbox import Sandbox, SandboxManager
from cross_bench.models import RunConfig, StepRecord, Task, TaskResult, TaskStatus
from cross_bench.systems.base import LLMDriver, MemoryMiddleware, PromptStrategy

logger = structlog.get_logger()


def _memory_side_context(task: Task, **extra: Any) -> dict[str, Any]:
    """Context dict for memory middleware (recall / store provenance)."""
    ctx: dict[str, Any] = {"task_id": task.task_id, **extra}
    cw = (task.metadata or {}).get("code_workspace")
    if cw:
        ctx["code_workspace"] = str(cw)
    return ctx


async def execute_single_shot(
    task: Task,
    config: RunConfig,
    adapter: BenchmarkAdapter,
    llm: LLMDriver,
    strategy: PromptStrategy,
    memory: MemoryMiddleware | None,
    sandbox_mgr: SandboxManager,
) -> TaskResult:
    """Run a single-shot completion for one task.

    Memory-aware adapters (RepoBench, future code-RAG benchmarks) get
    three optional hooks:

    1. :meth:`BenchmarkAdapter.ingest_for_memory` runs first, populating
       the memory store with task-bundled context (e.g. RepoBench's
       cross-file snippets).
    2. :meth:`BenchmarkAdapter.format_memory_prompt` provides the LLM
       prompt without that bundled context (since it's now in memory
       and will be retrieved). Defaults to ``format_prompt`` when not
       overridden.
    3. :meth:`BenchmarkAdapter.get_retrieval_query` provides a focused
       query (RepoCoder: imports + last 20 lines of the cropped code).
       When the adapter returns ``None`` we fall back to the legacy
       behavior where ``query == prompt`` and ``pre_prompt`` returns
       ``memory_ctx + prompt`` directly.
    """
    if memory:
        await adapter.ingest_for_memory(task, memory)
        benchmark_prompt = adapter.format_memory_prompt(task)
        retrieval_query = adapter.get_retrieval_query(task)
    else:
        benchmark_prompt = adapter.format_prompt(task)
        retrieval_query = None

    # Memory pre-prompt augmentation
    memory_ctx = None
    if memory:
        # Hoist the retrieval context dict so we can read back the
        # optional ``retrieved_context_ids`` provenance list. See
        # ``MemoryMiddleware`` docstring.
        retrieval_context: dict[str, Any] = _memory_side_context(task)
        search_t0 = time.monotonic()
        if retrieval_query is None:
            # Legacy path: middleware sees the full prompt, returns
            # ``memory_ctx + prompt``. Existing adapters rely on this.
            augmented = await memory.pre_prompt(
                benchmark_prompt, retrieval_context
            )
            if augmented != benchmark_prompt:
                memory_ctx = augmented[
                    : augmented.index(benchmark_prompt)
                ].strip()
                benchmark_prompt = augmented
        else:
            # Focused-query path: search with a small adapter-supplied
            # query, extract the prepended memory block, prepend it
            # manually onto the (typically much larger) benchmark
            # prompt. Keeps the LLM input as ``memory_ctx + full_file``
            # while the retriever only sees the focused query.
            augmented = await memory.pre_prompt(
                retrieval_query, retrieval_context
            )
            if augmented != retrieval_query:
                idx = augmented.find(retrieval_query)
                if idx > 0:
                    memory_ctx = augmented[:idx].strip()
                else:
                    # Middleware didn't echo the query verbatim
                    # (truncated, transformed, …). Treat the entire
                    # augmented string as the memory context.
                    memory_ctx = augmented.strip()
                if memory_ctx:
                    benchmark_prompt = f"{memory_ctx}\n\n{benchmark_prompt}"
        search_ms = (time.monotonic() - search_t0) * 1000
        task.metadata["search_latency_ms"] = round(search_ms, 1)

        retrieved_ids = retrieval_context.get("retrieved_context_ids")
        if isinstance(retrieved_ids, list):
            task.metadata["retrieved_context_ids"] = [str(x) for x in retrieved_ids]

    # Format via prompt strategy
    formatted = strategy.format(task, benchmark_prompt, memory_ctx)

    # Call LLM
    inference_t0 = time.monotonic()
    if isinstance(formatted, list):
        output, step = await llm.chat(formatted, {})
    else:
        output, step = await llm.complete(formatted, {})
    inference_ms = (time.monotonic() - inference_t0) * 1000

    step.step_index = 0

    # Memory post-response
    if memory:
        await memory.post_response(benchmark_prompt, output, _memory_side_context(task))

    # Evaluate
    if adapter.requires_sandbox:
        # Pass the task so per-instance benchmarks (SWE-bench,
        # Terminal-Bench) can return their task-specific image. For
        # benchmarks with a single image, the task arg is ignored.
        sandbox_image = config.sandbox_image or adapter.get_sandbox_image(task)
        sandbox = await sandbox_mgr.create(sandbox_image, task)
        try:
            evaluation = await adapter.evaluate(task, output, sandbox_mgr, sandbox)
        except Exception as e:
            logger.error("evaluation.error", task_id=task.task_id, error=str(e))
            evaluation = {"pass": False, "error": str(e)}
        finally:
            await sandbox_mgr.destroy(sandbox)
    else:
        try:
            evaluation = await adapter.evaluate(task, output, sandbox_mgr, sandbox=None)
        except Exception as e:
            logger.error("evaluation.error", task_id=task.task_id, error=str(e))
            evaluation = {"pass": False, "error": str(e)}

    # Stamp latency keys on the evaluation dict so the aggregator picks
    # them up via the standard ``avg_<key>`` autoload. ``search_latency``
    # is only meaningful when memory is active; inference is always
    # recorded so non-memory rows can be compared on raw cost.
    evaluation["inference_latency_ms"] = round(inference_ms, 1)
    if memory:
        search_ms = task.metadata.get("search_latency_ms", 0.0)
        evaluation["search_latency_ms"] = float(search_ms)
        evaluation["total_latency_ms"] = round(
            float(search_ms) + inference_ms, 1
        )
    else:
        evaluation["total_latency_ms"] = round(inference_ms, 1)

    status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL
    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=[step],
        final_output=output,
        evaluation=evaluation,
    )


async def execute_gold_eval(
    task: Task,
    config: RunConfig,
    adapter: BenchmarkAdapter,
    llm: LLMDriver,
    strategy: PromptStrategy,
    memory: MemoryMiddleware | None,
    sandbox_mgr: SandboxManager,
) -> TaskResult:
    """Evaluate the dataset gold patch without calling the LLM.

    Harness sanity check: feeds ``task.ground_truth`` through the same
    ``adapter.evaluate`` path a scaffold would use after submission.
    """
    _ = (llm, strategy, memory)  # unused — gold eval skips inference and memory

    patch = task.ground_truth
    if patch is None or not str(patch).strip():
        # Latency columns explicit-zero on the missing-patch early-return
        # too — keeps the summary.json schema stable across every
        # gold_eval outcome (EE sec.3). The main path below also stamps
        # these via the shared trailer; both branches must emit them or
        # the column count flaps based on outcome distribution. Small
        # severity (real Verified / Pro rows always carry a gold
        # ``patch`` field) but cheap to fix and exactly the schema-flap
        # the SEAL submodules_missing branch already guards against.
        evaluation: dict[str, Any] = {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": False,
            "error": "missing_gold_patch",
            "scaffold": "gold_eval",
            "search_latency_ms": 0.0,
            "inference_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        return TaskResult(
            task_id=task.task_id,
            run_id=config.run_id,
            status=TaskStatus.FAIL,
            steps=[],
            final_output="",
            evaluation=evaluation,
        )

    patch_text = str(patch)

    if adapter.requires_sandbox:
        sandbox_image = config.sandbox_image or adapter.get_sandbox_image(task)
        create_kwargs: dict[str, Any] = {}
        if (task.metadata or {}).get("variant") in {"pro"}:
            create_kwargs["entrypoint"] = "/bin/bash"
        sandbox = await sandbox_mgr.create(sandbox_image, task, **create_kwargs)
        try:
            evaluation = await adapter.evaluate(
                task, patch_text, sandbox_mgr, sandbox
            )
        except Exception as e:
            logger.error("evaluation.error", task_id=task.task_id, error=str(e))
            evaluation = {"pass": False, "error": str(e)}
        finally:
            await sandbox_mgr.destroy(sandbox)
    else:
        try:
            evaluation = await adapter.evaluate(
                task, patch_text, sandbox_mgr, sandbox=None
            )
        except Exception as e:
            logger.error("evaluation.error", task_id=task.task_id, error=str(e))
            evaluation = {"pass": False, "error": str(e)}

    evaluation.setdefault("scaffold", "gold_eval")
    # All three latency columns explicit-zero — gold_eval skips both
    # retrieval and inference by design (just applies the dataset's
    # ``patch`` field through the harness), so 0.0 across the board is
    # the honest value. Without ``search_latency_ms`` the summary
    # schema would drop the column for gold_eval rows, breaking
    # cross-scaffold comparability (EE sec.3).
    evaluation["search_latency_ms"] = 0.0
    evaluation["inference_latency_ms"] = 0.0
    evaluation["total_latency_ms"] = 0.0

    status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL
    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=[],
        final_output=patch_text,
        evaluation=evaluation,
    )


async def execute_agentic(
    task: Task,
    config: RunConfig,
    adapter: BenchmarkAdapter,
    llm: LLMDriver,
    strategy: PromptStrategy,
    memory: MemoryMiddleware | None,
    sandbox_mgr: SandboxManager,
) -> TaskResult:
    """Run an agentic multi-step loop for one task."""
    sandbox = await sandbox_mgr.create(adapter.get_sandbox_image(task), task)
    steps: list[StepRecord] = []
    history: list[dict[str, str]] = []
    final_output = ""

    try:
        # Initial system + task prompt
        benchmark_prompt = adapter.format_prompt(task)
        system_msg = (
            "You are a coding agent. You can execute bash commands by wrapping them in "
            "```bash\\n<command>\\n```. When you have completed the task, output "
            "TASK_COMPLETE followed by your final solution in a code block."
        )
        history.append({"role": "system", "content": system_msg})

        if memory:
            # Hoist the retrieval context dict so we can read back the
            # optional ``retrieved_context_ids`` provenance list. See
            # ``MemoryMiddleware`` docstring.
            retrieval_context: dict[str, Any] = _memory_side_context(task)
            benchmark_prompt = await memory.pre_prompt(
                benchmark_prompt, retrieval_context
            )
            retrieved_ids = retrieval_context.get("retrieved_context_ids")
            if isinstance(retrieved_ids, list):
                task.metadata["retrieved_context_ids"] = [
                    str(x) for x in retrieved_ids
                ]
        history.append({"role": "user", "content": benchmark_prompt})

        for step_idx in range(config.max_agent_steps):
            response, step = await llm.chat(history, {})
            step.step_index = step_idx
            steps.append(step)
            history.append({"role": "assistant", "content": response})

            # Check for task completion
            if "TASK_COMPLETE" in response:
                final_output = _extract_code_block(response) or response
                break

            # Extract and execute bash commands
            bash_commands = _extract_bash_commands(response)
            if bash_commands:
                for cmd in bash_commands:
                    exec_result = await sandbox_mgr.execute(sandbox, cmd, timeout=config.timeout_seconds)
                    observation = (
                        f"Exit code: {exec_result.exit_code}\n"
                        f"stdout:\n{exec_result.stdout[:3000]}\n"
                        f"stderr:\n{exec_result.stderr[:1000]}"
                    )
                    history.append({"role": "user", "content": f"Command output:\n{observation}"})

                    # Record the bash step
                    steps.append(StepRecord(
                        step_index=step_idx,
                        action="bash",
                        input_text=cmd,
                        output_text=observation[:2000],
                        metadata={"exit_code": exec_result.exit_code},
                    ))
            else:
                # No commands and no completion — nudge the agent
                history.append({
                    "role": "user",
                    "content": "Please continue working on the task. Use bash commands or submit your solution.",
                })

            if memory:
                await memory.post_response(
                    str(history[-2:]),
                    response,
                    _memory_side_context(task, step=step_idx),
                )

        if not final_output:
            # Agent didn't explicitly complete — use last response
            final_output = _extract_code_block(response) or response

        # Evaluate
        evaluation = await adapter.evaluate(task, final_output, sandbox_mgr, sandbox)
        status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL

    except Exception as e:
        logger.error("agentic.error", task_id=task.task_id, error=str(e))
        evaluation = {"pass": False, "error": str(e)}
        status = TaskStatus.ERROR
    finally:
        await sandbox_mgr.destroy(sandbox)

    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=steps,
        final_output=final_output,
        evaluation=evaluation,
    )


def _extract_bash_commands(text: str) -> list[str]:
    """Extract bash commands from markdown code blocks."""
    pattern = r"```bash\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def _extract_code_block(text: str) -> str:
    """Extract the last code block from the response."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Conversational mode — for long-context memory benchmarks
# ---------------------------------------------------------------------------

async def execute_conversational(
    task: Task,
    config: RunConfig,
    adapter: BenchmarkAdapter,
    llm: LLMDriver,
    strategy: PromptStrategy,
    memory: MemoryMiddleware | None,
    sandbox_mgr: SandboxManager,
) -> TaskResult:
    """Answer a question using memory-retrieved context (no sandbox).

    This mode is for conversational / long-context benchmarks (LoCoMo,
    LongMemEval, ConvoMem) where evaluation uses LLM-as-judge instead
    of code execution.

    Uses provider-specific prompt templates when a memory provider is configured.
    """
    from cross_bench.evaluation.provider_prompts import format_answer_prompt

    context = task.metadata.get("search_context", "")
    question = task.prompt

    # Use provider-specific prompt if memory provider is known
    provider = ""
    if config.system.memory and config.system.memory.enabled:
        provider = config.system.memory.provider

    answer_prompt = format_answer_prompt(
        provider=provider,
        question=question,
        context=context,
    )

    # Augment with memory middleware if present
    if memory:
        # Hoist the retrieval context dict so we can read back the
        # optional ``retrieved_context_ids`` provenance list. See
        # ``MemoryMiddleware`` docstring.
        retrieval_context: dict[str, Any] = _memory_side_context(task)
        answer_prompt = await memory.pre_prompt(answer_prompt, retrieval_context)
        retrieved_ids = retrieval_context.get("retrieved_context_ids")
        if isinstance(retrieved_ids, list):
            task.metadata["retrieved_context_ids"] = [str(x) for x in retrieved_ids]

    # Format via prompt strategy
    formatted = strategy.format(task, answer_prompt, None)

    # Call LLM
    if isinstance(formatted, list):
        output, step = await llm.chat(formatted, {})
    else:
        output, step = await llm.complete(formatted, {})

    step.step_index = 0

    # Extract FINAL ANSWER from CoT responses (matches EverOS stage4_response.py)
    # Providers like evermemos use Chain-of-Thought prompting with a
    # "FINAL ANSWER:" marker; the judge should only see the concise answer.
    if "FINAL ANSWER:" in output:
        parts = output.split("FINAL ANSWER:")
        if len(parts) > 1:
            output = parts[-1].strip()

    if memory:
        await memory.post_response(answer_prompt, output, _memory_side_context(task))

    # Evaluate — conversational adapters use judge_answer, no sandbox needed
    evaluation = await adapter.evaluate(task, output, sandbox_mgr, sandbox=None)
    status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL

    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=[step],
        final_output=output,
        evaluation=evaluation,
    )
