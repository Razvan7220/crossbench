"""Test prompt strategy formatting."""

from cross_bench.models import PromptStrategyConfig, Task, TaskGranularity
from cross_bench.systems.prompting.chain_of_thought import ChainOfThoughtStrategy
from cross_bench.systems.prompting.few_shot import FewShotStrategy
from cross_bench.systems.prompting.zero_shot import ZeroShotStrategy


def _make_task(prompt="def add(a, b):"):
    return Task(
        task_id="t1",
        benchmark="test",
        granularity=TaskGranularity.FUNCTION,
        prompt=prompt,
    )


def test_zero_shot_messages():
    """Without a system prompt, ZeroShotStrategy returns a single user message."""
    strategy = ZeroShotStrategy(PromptStrategyConfig(type="zero_shot"))
    result = strategy.format(_make_task(), "def add(a, b):")
    assert isinstance(result, list)
    assert result == [{"role": "user", "content": "def add(a, b):"}]


def test_zero_shot_with_system_prompt():
    strategy = ZeroShotStrategy(
        PromptStrategyConfig(type="zero_shot", params={"system_prompt": "You are a coder."})
    )
    result = strategy.format(_make_task(), "def add(a, b):")
    assert isinstance(result, list)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a coder."


def test_few_shot_includes_examples():
    examples = [
        {"input": "def inc(x):", "output": "    return x + 1"},
        {"input": "def dec(x):", "output": "    return x - 1"},
    ]
    strategy = FewShotStrategy(
        PromptStrategyConfig(type="few_shot", params={"examples": examples, "k": 2})
    )
    result = strategy.format(_make_task(), "def add(a, b):")
    assert "Example 1" in result
    assert "Example 2" in result
    assert "def add(a, b):" in result


def test_chain_of_thought():
    strategy = ChainOfThoughtStrategy(PromptStrategyConfig(type="chain_of_thought"))
    result = strategy.format(_make_task(), "def add(a, b):")
    assert "step by step" in result.lower()


def test_memory_context_included():
    strategy = ZeroShotStrategy(PromptStrategyConfig(type="zero_shot"))
    result = strategy.format(_make_task(), "def add(a, b):", memory_context="Previous: add adds numbers")
    assert isinstance(result, list)
    assert "Previous: add adds numbers" in result[-1]["content"]
    assert "def add(a, b):" in result[-1]["content"]
