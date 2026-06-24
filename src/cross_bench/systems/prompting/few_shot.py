"""Few-shot prompting: include k examples before the task."""

from __future__ import annotations

from cross_bench.models import PromptStrategyConfig, Task
from cross_bench.systems.base import PromptStrategy
from cross_bench.systems.registry import register_prompt_strategy


@register_prompt_strategy("few_shot")
class FewShotStrategy(PromptStrategy):
    def __init__(self, config: PromptStrategyConfig):
        self._examples: list[dict[str, str]] = config.params.get("examples", [])
        self._k = config.params.get("k", 3)
        self._system_prompt = config.params.get("system_prompt", "")

    def format(self, task: Task, benchmark_prompt: str, memory_context: str | None = None):
        examples = self._examples[:self._k]
        parts = []
        if memory_context:
            parts.append(memory_context)

        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Input:\n{ex.get('input', '')}")
            parts.append(f"Output:\n{ex.get('output', '')}")
            parts.append("")

        parts.append("Now complete the following:")
        parts.append(benchmark_prompt)

        user_content = "\n".join(parts)
        if self._system_prompt:
            return [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ]
        return user_content
