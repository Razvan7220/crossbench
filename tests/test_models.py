"""Test core domain models."""

from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    RunResult,
    RunStatus,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
    TaskResult,
    TaskStatus,
)


def test_task_creation():
    task = Task(
        task_id="HE/1",
        benchmark="humaneval",
        granularity=TaskGranularity.FUNCTION,
        prompt="def foo():",
    )
    assert task.task_id == "HE/1"
    assert task.granularity == TaskGranularity.FUNCTION


def test_run_config_defaults():
    config = RunConfig(
        benchmark="humaneval",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
    )
    assert config.execution_mode == ExecutionMode.SINGLE_SHOT
    assert config.max_agent_steps == 20
    assert config.timeout_seconds == 300
    assert config.run_id  # UUID generated


def test_step_record():
    step = StepRecord(step_index=0, action="completion", input_text="in", output_text="out")
    assert step.tokens_in == 0
    assert step.latency_ms == 0.0


def test_task_status_enum():
    assert TaskStatus.PASS.value == "pass"
    assert TaskStatus.FAIL.value == "fail"
    assert TaskStatus.ERROR.value == "error"
    assert TaskStatus.TIMEOUT.value == "timeout"
