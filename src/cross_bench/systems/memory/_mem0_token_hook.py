"""Monkey-patch mem0's LLM classes to record internal token usage.

mem0's ``LLMBase.generate_response`` returns only the response content
string — the underlying provider response's ``usage`` is dropped at
that layer, so even when ``OPENAI`` returns ``response.usage`` we lose
it the moment mem0 unwraps the content. To still surface mem0's
*internal* LLM cost (fact extraction, dedup, update logic) we wrap
``generate_response`` to tokenize the input messages and the output
content under ``cl100k_base`` — the same encoder used everywhere else
in CrossBench (see ``execution/engine.py::_count_tokens``).

Each patched call reads an active accountant from a ``ContextVar`` and
invokes it with ``(prompt_text, response_text)``. The ``ContextVar``
keys the accountant per logical call so concurrent ``Memory`` instances
in the same process don't cross-contaminate counts. **The ContextVar
does NOT propagate into ``loop.run_in_executor`` worker threads** (a var
set on the submitting thread reads as its default inside the executor),
so middlewares MUST set the accountant on the worker thread itself —
use :func:`run_in_executor_with_accountant`, which sets/resets it inside
the thunk. Setting it before ``run_in_executor`` (as the middlewares did
until 2026-06-15) silently recorded zero internal tokens while boundary
tokenization still worked.

The patch is idempotent and best-effort: if mem0 renames a class or
moves it between modules, we log at debug and skip that class —
boundary-tokenization on the middleware side (``_record_io``) still
fires regardless, so cells with the hook silently degraded look like
cells where the upstream-wrap returns zero.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_active_accountant: contextvars.ContextVar[
    Optional[Callable[[str, str], None]]
] = contextvars.ContextVar("mem0_token_accountant", default=None)


def set_active_accountant(
    cb: Callable[[str, str], None],
) -> contextvars.Token:
    """Install a callback that the patched ``generate_response`` invokes
    on every internal mem0 LLM call. Returns a token; pass it to
    :func:`reset_active_accountant` to restore the prior value."""
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

    ContextVars set on the calling (event-loop) thread are NOT visible
    inside ``run_in_executor`` worker threads — a ``ContextVar`` set in
    the submitting thread reads as its default inside the executor. So
    setting the accountant *before* ``run_in_executor`` (as the
    middlewares did until 2026-06-15) left mem0's patched
    ``generate_response`` — which runs on the worker thread, in-line with
    the synchronous ``Memory.add`` / ``Memory.search`` call — reading
    ``None`` and recording zero internal-LLM tokens, while boundary (IO)
    tokenization still worked (that was the all-zero
    ``total_memory_tokens_*`` / ``total_memory_llm_calls`` signature on
    the 2026-06-15 ``mem0v2-round2-*`` LoCoMo runs). Setting the
    accountant inside the thunk, on the same worker thread, restores the
    capture. Set→reset are balanced on that worker thread, so reused
    pool threads don't leak the accountant between calls.
    """

    def _runner() -> Any:
        token = _active_accountant.set(cb)
        try:
            return fn()
        finally:
            _active_accountant.reset(token)

    return await loop.run_in_executor(None, _runner)


_PATCH_ATTEMPTED = False


def install_mem0_token_hook() -> bool:
    """Patch mem0's known LLM classes to record tokens via the active
    accountant.

    Idempotent (per-process; the patch records on the class object).
    Returns True when at least one class was successfully patched (or
    was already patched on a prior call), False when no targets were
    importable. Always safe to call.
    """
    global _PATCH_ATTEMPTED
    patched_any = False

    candidates: list[tuple[str, str]] = [
        ("mem0.llms.openai", "OpenAILLM"),
        ("mem0.llms.openai_structured", "OpenAIStructuredLLM"),
        ("mem0.llms.anthropic", "AnthropicLLM"),
        ("mem0.llms.ollama", "OllamaLLM"),
        ("mem0.llms.aws_bedrock", "AWSBedrockLLM"),
        ("mem0.llms.azure_openai", "AzureOpenAILLM"),
        ("mem0.llms.litellm", "LiteLLM"),
    ]

    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except Exception as e:
            logger.debug("mem0 token hook: skip %s (%s)", module_name, e)
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            logger.debug(
                "mem0 token hook: class %s missing in %s", class_name, module_name
            )
            continue
        if getattr(cls, "_cb_token_hook_installed", False):
            patched_any = True
            continue
        original = getattr(cls, "generate_response", None)
        if original is None or not callable(original):
            logger.debug(
                "mem0 token hook: %s.%s has no generate_response",
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
                    messages = args[0] if args else kwargs.get("messages", [])
                    if isinstance(messages, list):
                        prompt_text = "\n".join(
                            m.get("content", "")
                            for m in messages
                            if isinstance(m, dict)
                        )
                    else:
                        prompt_text = ""
                    response_text = (
                        result if isinstance(result, str) else ""
                    )
                    try:
                        cb(prompt_text, response_text)
                    except Exception as e:
                        logger.debug(
                            "mem0 token hook callback failed: %s", e
                        )
                return result
            return wrapped

        try:
            cls.generate_response = _make_wrapper(original)
            cls._cb_token_hook_installed = True
            patched_any = True
        except Exception as e:
            logger.debug(
                "mem0 token hook: failed to patch %s.%s: %s",
                module_name, class_name, e,
            )

    _PATCH_ATTEMPTED = True
    return patched_any
