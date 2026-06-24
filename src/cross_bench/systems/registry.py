"""System component discovery and instantiation."""

from __future__ import annotations

from typing import Any

from cross_bench.models import EmbeddingConfig, LLMConfig, MemoryConfig, PromptStrategyConfig
from cross_bench.systems.base import (
    EmbeddingDriver,
    LLMDriver,
    MemoryMiddleware,
    PromptStrategy,
)

# ---------------------------------------------------------------------------
# LLM drivers
# ---------------------------------------------------------------------------

_LLM_DRIVERS: dict[str, type[LLMDriver]] = {}


def register_llm(name: str):
    def decorator(cls: type[LLMDriver]):
        _LLM_DRIVERS[name] = cls
        return cls
    return decorator


def get_llm_driver(config: LLMConfig) -> LLMDriver:
    _ensure_llm_loaded()
    if config.provider not in _LLM_DRIVERS:
        raise KeyError(f"Unknown LLM provider '{config.provider}'. Available: {sorted(_LLM_DRIVERS)}")
    return _LLM_DRIVERS[config.provider](config)


# ---------------------------------------------------------------------------
# Memory middleware
# ---------------------------------------------------------------------------

_MEMORY_DRIVERS: dict[str, type[MemoryMiddleware]] = {}


def register_memory(name: str):
    def decorator(cls: type[MemoryMiddleware]):
        _MEMORY_DRIVERS[name] = cls
        return cls
    return decorator


def get_memory_middleware(config: MemoryConfig) -> MemoryMiddleware:
    _ensure_memory_loaded()
    if config.provider not in _MEMORY_DRIVERS:
        raise KeyError(f"Unknown memory provider '{config.provider}'. Available: {sorted(_MEMORY_DRIVERS)}")
    return _MEMORY_DRIVERS[config.provider](config)


# ---------------------------------------------------------------------------
# Embedding drivers
# ---------------------------------------------------------------------------

_EMBEDDING_DRIVERS: dict[str, type[EmbeddingDriver]] = {}


def register_embedding(name: str):
    def decorator(cls: type[EmbeddingDriver]):
        _EMBEDDING_DRIVERS[name] = cls
        return cls
    return decorator


def get_embedding_driver(config: EmbeddingConfig) -> EmbeddingDriver:
    _ensure_embedding_loaded()
    if config.provider not in _EMBEDDING_DRIVERS:
        raise KeyError(f"Unknown embedding provider '{config.provider}'.")
    return _EMBEDDING_DRIVERS[config.provider](config)


# ---------------------------------------------------------------------------
# Prompt strategies
# ---------------------------------------------------------------------------

_PROMPT_STRATEGIES: dict[str, type[PromptStrategy]] = {}


def register_prompt_strategy(name: str):
    def decorator(cls: type[PromptStrategy]):
        _PROMPT_STRATEGIES[name] = cls
        return cls
    return decorator


def get_prompt_strategy(config: PromptStrategyConfig) -> PromptStrategy:
    _ensure_prompt_loaded()
    if config.type not in _PROMPT_STRATEGIES:
        raise KeyError(f"Unknown prompt strategy '{config.type}'.")
    return _PROMPT_STRATEGIES[config.type](config)


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------

def _ensure_llm_loaded():
    if not _LLM_DRIVERS:
        import cross_bench.systems.llm.anthropic_driver  # noqa: F401
        import cross_bench.systems.llm.anthropic_bedrock_driver  # noqa: F401
        import cross_bench.systems.llm.generic_openai_driver  # noqa: F401
        import cross_bench.systems.llm.openai_driver  # noqa: F401


def _ensure_memory_loaded():
    if not _MEMORY_DRIVERS:
        import cross_bench.systems.memory.evermemos_embedded  # noqa: F401
        import cross_bench.systems.memory.evermemos_middleware  # noqa: F401
        import cross_bench.systems.memory.graphiti_embedded  # noqa: F401
        import cross_bench.systems.memory.hindsight_embedded  # noqa: F401
        import cross_bench.systems.memory.hindsight_middleware  # noqa: F401
        import cross_bench.systems.memory.letta_middleware  # noqa: F401
        import cross_bench.systems.memory.mastra_embedded  # noqa: F401
        import cross_bench.systems.memory.mastra_middleware  # noqa: F401
        import cross_bench.systems.memory.mem0v1_middleware  # noqa: F401
        import cross_bench.systems.memory.mem0v2_middleware  # noqa: F401
        import cross_bench.systems.memory.memos_middleware  # noqa: F401
        import cross_bench.systems.memory.rag_baseline  # noqa: F401
        import cross_bench.systems.memory.zep_middleware  # noqa: F401


def _ensure_embedding_loaded():
    if not _EMBEDDING_DRIVERS:
        import cross_bench.systems.embedding.ollama_embedding  # noqa: F401
        import cross_bench.systems.embedding.openai_embedding  # noqa: F401
        import cross_bench.systems.embedding.sentence_transformer  # noqa: F401


def _ensure_prompt_loaded():
    if not _PROMPT_STRATEGIES:
        import cross_bench.systems.prompting.chain_of_thought  # noqa: F401
        import cross_bench.systems.prompting.few_shot  # noqa: F401
        import cross_bench.systems.prompting.retrieval_augmented  # noqa: F401
        import cross_bench.systems.prompting.zero_shot  # noqa: F401
