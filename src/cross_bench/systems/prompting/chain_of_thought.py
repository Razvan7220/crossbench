"""Chain-of-thought prompting: instruct the model to reason step by step."""

from __future__ import annotations

from cross_bench.models import PromptStrategyConfig, Task
from cross_bench.systems.base import PromptStrategy
from cross_bench.systems.registry import register_prompt_strategy

_DEFAULT_COT_INSTRUCTION = (
    "Think through this step by step before writing your solution. "
    "Show your reasoning, then provide the final code."
)


@register_prompt_strategy("chain_of_thought")
class ChainOfThoughtStrategy(PromptStrategy):
    def __init__(self, config: PromptStrategyConfig):
        self._instruction = config.params.get("instruction", _DEFAULT_COT_INSTRUCTION)
        self._system_prompt = config.params.get("system_prompt", "")

    def format(self, task: Task, benchmark_prompt: str, memory_context: str | None = None):
        parts = []
        if memory_context:
            parts.append(memory_context)
        parts.append(benchmark_prompt)
        parts.append("")
        parts.append(self._instruction)

        user_content = "\n".join(parts)
        if self._system_prompt:
            return [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ]
        return user_content
