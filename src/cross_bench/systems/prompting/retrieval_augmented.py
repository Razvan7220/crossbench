"""Retrieval-augmented prompting: embed and retrieve relevant context."""

from __future__ import annotations

from cross_bench.models import PromptStrategyConfig, Task
from cross_bench.systems.base import PromptStrategy
from cross_bench.systems.registry import register_prompt_strategy


@register_prompt_strategy("retrieval_augmented")
class RetrievalAugmentedStrategy(PromptStrategy):
    """Uses pre-computed context (from memory middleware or embedding search)
    to augment the prompt with relevant code snippets or documentation."""

    def __init__(self, config: PromptStrategyConfig):
        self._system_prompt = config.params.get(
            "system_prompt",
            "Use the provided context to inform your solution.",
        )
        self._context_header = config.params.get("context_header", "Relevant context:")

    def format(self, task: Task, benchmark_prompt: str, memory_context: str | None = None):
        parts = []

        # Include cross-file context from the task itself
        if task.context_files:
            parts.append(self._context_header)
            for cf in task.context_files:
                parts.append(f"--- {cf.get('path', 'file')} ---")
                parts.append(cf.get("content", ""))
            parts.append("")

        # Include memory-retrieved context
        if memory_context:
            parts.append("Additional context from memory:")
            parts.append(memory_context)
            parts.append("")

        parts.append(benchmark_prompt)
        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
