"""Shared test fixtures."""

from __future__ import annotations

import pytest

from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
    TaskResult,
    TaskStatus,
)


@pytest.fixture
def sample_task():
    return Task(
        task_id="test_001",
        benchmark="humaneval",
        granularity=TaskGranularity.FUNCTION,
        prompt='def add(a, b):\n    """Return the sum of a and b."""\n',
        metadata={"entry_point": "add"},
        ground_truth='assert add(1, 2) == 3\nassert add(0, 0) == 0\n',
    )


@pytest.fixture
def sample_run_config():
    return RunConfig(
        benchmark="humaneval",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.SINGLE_SHOT,
    )


@pytest.fixture
def sample_task_result():
    return TaskResult(
        task_id="test_001",
        run_id="test-run-id",
        status=TaskStatus.PASS,
        steps=[
            StepRecord(
                step_index=0,
                action="completion",
                input_text="def add(a, b):",
                output_text="    return a + b",
                tokens_in=10,
                tokens_out=5,
                latency_ms=100.0,
            )
        ],
        final_output="    return a + b",
        evaluation={"pass": True},
    )
