"""Monkey-patch MemoryOS (``memos`` package) LLM classes for token capture.

Sibling of ``_mem0_token_hook.py``: same ContextVar-based accountant
pattern, applied to the upstream LLM classes used by
``SimpleStructMemReader`` and the cube-side ``Searcher`` pipeline.

The MemoryOS package isn't always installed in CI / development
environments. The hook tries several plausible upstream paths and
silently logs at debug if a target is missing. Even when no class
patches successfully, boundary tokenization in
``memos_middleware.py`` still fires — the upstream-wrap column just
reports zero for that call site.

Known target candidates (subject to upstream renames; report a debug
log line when a path is gone):
  - ``memos.llms.openai.OpenAILLM`` — synchronous OpenAI completion.
  - ``memos.llms.litellm.LiteLLMOpenAI`` — LiteLLM-mediated calls.
  - ``memos.llms.azure_openai.AzureOpenAILLM`` — Azure variant.
  - ``memos.llms.ollama.OllamaLLM`` — local Ollama.
  - ``memos.llms.huggingface.HuggingFaceLLM`` — HF transformers.

Each is expected to expose a ``generate`` (or ``generate_response``,
or ``__call__``) method returning the response string. We try each
known method name; the first one we find is wrapped.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_active_accountant: contextvars.ContextVar[
    Optional[Callable[[str, str], None]]
] = contextvars.ContextVar("memos_token_accountant", default=None)


def set_active_accountant(
    cb: Callable[[str, str], None],
) -> contextvars.Token:
    return _active_accountant.set(cb)


def reset_active_accountant(token: contextvars.Token) -> None:
    _active_accountant.reset(token)


async def run_in_executor_with_accountant(
    loop: Any,
    cb: Callable[[str, str], None],
    fn: Callable[[], Any],
) -> Any:
    """Run blocking ``fn`` in ``loop``'s default executor with ``cb``
    installed as the active accountant ON THE WORKER THREAD.

    Sibling of ``_mem0_token_hook.run_in_executor_with_accountant`` —
    see that docstring for the full rationale. In short: ContextVars set
    on the event-loop thread are invisible inside ``run_in_executor``
    worker threads, so the accountant must be set on the worker thread
    where MemoryOS's synchronous ``MemReader.get_memory`` /
    ``text_mem.search`` (and their in-thread LLM calls) actually run.
    Setting it before ``run_in_executor`` recorded zero internal-LLM
    tokens (the same bug fixed for mem0v2 on 2026-06-15).
    """

    def _runner() -> Any:
        token = _active_accountant.set(cb)
        try:
            return fn()
        finally:
            _active_accountant.reset(token)

    return await loop.run_in_executor(None, _runner)


_CANDIDATE_TARGETS: list[tuple[str, str]] = [
    ("memos.llms.openai", "OpenAILLM"),
    ("memos.llms.litellm", "LiteLLMOpenAI"),
    ("memos.llms.litellm", "LiteLLM"),
    ("memos.llms.azure_openai", "AzureOpenAILLM"),
    ("memos.llms.ollama", "OllamaLLM"),
    ("memos.llms.huggingface", "HuggingFaceLLM"),
    ("memos.llms.anthropic", "AnthropicLLM"),
]

_CANDIDATE_METHODS: tuple[str, ...] = ("generate", "generate_response", "chat")


def install_memos_token_hook() -> bool:
    """Patch known memos LLM classes idempotently. Returns True if at
    least one class was patched (or was already patched)."""
    patched_any = False
    for module_name, class_name in _CANDIDATE_TARGETS:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except Exception as e:
            logger.debug("memos token hook: skip %s (%s)", module_name, e)
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        if getattr(cls, "_cb_token_hook_installed", False):
            patched_any = True
            continue
        target_name: str | None = None
        original: Callable[..., Any] | None = None
        for method in _CANDIDATE_METHODS:
            candidate = getattr(cls, method, None)
            if candidate is not None and callable(candidate):
                target_name = method
                original = candidate
                break
        if original is None or target_name is None:
            logger.debug(
                "memos token hook: %s.%s exposes no known method",
                module_name, class_name,
            )
            continue

        def _make_wrapper(
            __original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped(self, *args: Any, **kwargs: Any) -> Any:
                result = __original(self, *args, **kwargs)
                cb = _active_accountant.get()
                if cb is not None:
                    # Best-effort prompt assembly — memos LLM calls
                    # take either a list of message dicts or a single
                    # string. Tokenize whichever shape we see.
                    prompt_parts: list[str] = []
                    for arg in args:
                        if isinstance(arg, str):
                            prompt_parts.append(arg)
                        elif isinstance(arg, list):
                            for m in arg:
                                if isinstance(m, dict):
                                    prompt_parts.append(
                                        m.get("content", "") or ""
                                    )
                                elif isinstance(m, str):
                                    prompt_parts.append(m)
                    for kv in kwargs.values():
                        if isinstance(kv, str):
                            prompt_parts.append(kv)
                        elif isinstance(kv, list):
                            for m in kv:
                                if isinstance(m, dict):
                                    prompt_parts.append(
                                        m.get("content", "") or ""
                                    )
                    prompt_text = "\n".join(p for p in prompt_parts if p)
                    response_text = (
                        result if isinstance(result, str) else ""
                    )
                    try:
                        cb(prompt_text, response_text)
                    except Exception as e:
                        logger.debug(
                            "memos token hook callback failed: %s", e
                        )
                return result
            return wrapped

        try:
            setattr(cls, target_name, _make_wrapper(original))
            cls._cb_token_hook_installed = True
            patched_any = True
        except Exception as e:
            logger.debug(
                "memos token hook: failed to patch %s.%s.%s: %s",
                module_name, class_name, target_name, e,
            )

    return patched_any
