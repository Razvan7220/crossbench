"""Zero-shot prompting: task prompt with no examples."""

from __future__ import annotations

from cross_bench.models import PromptStrategyConfig, Task
from cross_bench.systems.base import PromptStrategy
from cross_bench.systems.registry import register_prompt_strategy


@register_prompt_strategy("zero_shot")
class ZeroShotStrategy(PromptStrategy):
    def __init__(self, config: PromptStrategyConfig):
        self._system_prompt = config.params.get("system_prompt", "")

    def format(self, task: Task, benchmark_prompt: str, memory_context: str | None = None):
        parts = []
        if memory_context:
            parts.append(memory_context)
        parts.append(benchmark_prompt)

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages
