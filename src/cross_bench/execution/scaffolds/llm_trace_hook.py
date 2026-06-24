"""
LLM I/O trace hook — dumps the FULL prompt + FULL response of every
litellm.completion call to stderr (prefixed ``[llm-trace]`` so the
SEAL live-tee forwards it) and optionally to a JSONL sidecar file.

Why this exists
---------------
When sweagent exits with ``NoPrediction`` and the trajectory metrics
say ``~12 tokens out per step``, the operator needs to see EXACTLY
what the model is emitting. The step-counter shows when calls fire,
but nothing tells us what the model actually said. This hook closes
that gap.

Usage
-----
Set ``CROSSBENCH_LLM_TRACE_ENABLED=1`` in the subprocess env. The
SEAL scaffold reads ``[benchmark] llm_trace_enabled = true`` from
the active benchmark/profile TOML and sets the env var when active.

Output shape
------------
Each litellm.completion call produces two stderr blocks:

    [llm-trace] ════════════════════════════════════════════════════
    [llm-trace] step=N >>> REQUEST model=... n_messages=M
    [llm-trace] >>> message[0] role=system chars=...
    [llm-trace] <full content, multi-line, each line prefixed [llm-trace]>
    [llm-trace] >>> message[1] role=user chars=...
    [llm-trace] <full content>
    ...

    [llm-trace] ════════════════════════════════════════════════════
    [llm-trace] step=N <<< RESPONSE chars=... tokens_in=... tokens_out=... finish_reason=...
    [llm-trace] <full content, multi-line, each line prefixed [llm-trace]>

Each block also lands in ``CROSSBENCH_LLM_TRACE_FILE`` (one JSON
object per line: ``{"step": N, "direction": "request"|"response", ...}``).

Live-tee
--------
``[llm-trace]`` is in seal.py's ``_LIVE_TEE_PREFIXES``, so lines
stream to the operator's terminal in real time. With a 12-step
trajectory generating ~4 KB of stderr per step, the operator sees
~50 KB of dumped LLM I/O for a typical SEAL Pro task — manageable
in a terminal pager and easy to grep.

Recursion guard
---------------
log_pre_api_call / log_success_event don't make litellm.completion
calls, so no recursion guard is needed. The hook is purely passive —
read inputs, print, return.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

ENV_ENABLED = "CROSSBENCH_LLM_TRACE_ENABLED"
ENV_FILE = "CROSSBENCH_LLM_TRACE_FILE"
ENV_MAX_CONTENT_CHARS = "CROSSBENCH_LLM_TRACE_MAX_CONTENT_CHARS"
ENV_STDERR_DISABLED = "CROSSBENCH_LLM_TRACE_STDERR_DISABLED"

DEFAULT_MAX_CONTENT_CHARS = 80_000


def _enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "0").strip() == "1"


def _stderr_disabled() -> bool:
    """Skip the noisy per-step stderr dump; keep the registered-banner +
    JSONL sidecar. Useful for long ablation runs where you want the
    full trajectory in a file but not flooding CloudWatch/terminal.
    """
    return os.environ.get(ENV_STDERR_DISABLED, "0").strip() == "1"


def _get_int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


#: Module-level counter for step labelling. Mirrors the step-counter
#: convention.
_trace_counter: list[int] = [0]


def _reset_counter() -> None:
    _trace_counter[0] = 0


def _emit_block(prefix_line: str, body: str) -> None:
    """Print ``prefix_line`` then ``body`` with every line prefixed.

    No-op when ``CROSSBENCH_LLM_TRACE_STDERR_DISABLED=1`` — the JSONL
    sidecar still gets the full record. Useful for keeping the
    operator's terminal / CloudWatch quiet during long ablation runs.
    """
    if _stderr_disabled():
        return
    sep = "[llm-trace] " + "═" * 60
    print(sep, file=sys.stderr, flush=True)
    print(prefix_line, file=sys.stderr, flush=True)
    if not body:
        return
    for line in body.splitlines():
        print(f"[llm-trace] {line}", file=sys.stderr, flush=True)


def _truncate(s: str, max_chars: int) -> tuple[str, bool]:
    if len(s) <= max_chars:
        return s, False
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2 :]
    return (
        f"{head}\n[llm-trace] ... <{len(s) - max_chars} chars truncated> ...\n{tail}",
        True,
    )


def _append_to_file(record: dict[str, Any]) -> None:
    path = os.environ.get(ENV_FILE, "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        print(
            f"[llm-trace] file_write_failed path={path} error={e}",
            file=sys.stderr, flush=True,
        )


def _stringify_content(content: Any) -> str:
    """LiteLLM messages can have str OR list-of-content-blocks
    (Anthropic-style)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    parts.append(
                        f"[tool_use name={block.get('name', '?')} "
                        f"input={json.dumps(block.get('input'), default=str)[:500]}]"
                    )
                elif block_type == "tool_result":
                    parts.append(
                        f"[tool_result tool_use_id={block.get('tool_use_id', '?')} "
                        f"content={_stringify_content(block.get('content'))[:500]}]"
                    )
                else:
                    parts.append(json.dumps(block, default=str)[:500])
            else:
                parts.append(str(block)[:500])
        return "\n".join(parts)
    return str(content)


def register_llm_trace_hook() -> bool:
    """Attach the ``LLMTrace`` ``CustomLogger`` to ``litellm.callbacks``."""
    if not _enabled():
        return False
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError:
        return False

    for cb in getattr(litellm, "callbacks", []):
        if cb.__class__.__name__ == "LLMTrace":
            return True

    max_chars = _get_int_env(ENV_MAX_CONTENT_CHARS, DEFAULT_MAX_CONTENT_CHARS)

    class LLMTrace(CustomLogger):  # type: ignore[misc]
        def log_pre_api_call(  # type: ignore[override]
            self, model: str, messages: list, kwargs: dict
        ) -> None:
            if not _enabled() or not isinstance(messages, list):
                return
            step = _trace_counter[0]  # NOT incremented yet — paired with success_event
            n_msg = len(messages)
            req_record: dict[str, Any] = {
                "step": step,
                "direction": "request",
                "model": model,
                "n_messages": n_msg,
                "timestamp": time.time(),
                "messages": [],
            }
            _emit_block(
                f"[llm-trace] step={step} >>> REQUEST model={model} n_messages={n_msg}",
                "",
            )
            for idx, m in enumerate(messages):
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "?")
                content_str = _stringify_content(m.get("content"))
                truncated_content, did_truncate = _truncate(content_str, max_chars)
                req_record["messages"].append(
                    {
                        "idx": idx,
                        "role": role,
                        "chars": len(content_str),
                        "truncated": did_truncate,
                        "content": content_str,
                    }
                )
                _emit_block(
                    f"[llm-trace] >>> message[{idx}] role={role} chars={len(content_str)}"
                    + (" (TRUNCATED in stderr)" if did_truncate else ""),
                    truncated_content,
                )
            _append_to_file(req_record)

        def log_success_event(  # type: ignore[override]
            self,
            kwargs: dict,
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            if not _enabled():
                return
            step = _trace_counter[0]
            _trace_counter[0] = step + 1  # advance AFTER printing both halves
            # Extract content + meta from the response
            content_str = ""
            finish_reason = ""
            tokens_in = 0
            tokens_out = 0
            try:
                choices = getattr(response_obj, "choices", None) or response_obj.get("choices", [])
                if choices:
                    first = choices[0]
                    msg = getattr(first, "message", None) or first.get("message", {})
                    raw_content = (
                        getattr(msg, "content", None)
                        if msg is not None
                        else None
                    )
                    if raw_content is None and isinstance(msg, dict):
                        raw_content = msg.get("content")
                    content_str = _stringify_content(raw_content)
                    # Tool calls (Anthropic / OpenAI bedrock-style)
                    tool_calls = (
                        getattr(msg, "tool_calls", None)
                        if msg is not None
                        else None
                    )
                    if tool_calls is None and isinstance(msg, dict):
                        tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            tc_dict = (
                                tc if isinstance(tc, dict)
                                else getattr(tc, "model_dump", lambda: {})()
                            )
                            content_str += (
                                f"\n[tool_call name={tc_dict.get('function', {}).get('name', '?')} "
                                f"args={json.dumps(tc_dict.get('function', {}).get('arguments'), default=str)[:500]}]"
                            )
                    finish_reason = (
                        getattr(first, "finish_reason", None)
                        if first is not None
                        else None
                    ) or (first.get("finish_reason") if isinstance(first, dict) else "")
                usage = getattr(response_obj, "usage", None) or response_obj.get("usage", {})
                if usage is not None:
                    tokens_in = (
                        getattr(usage, "prompt_tokens", None)
                        or (usage.get("prompt_tokens") if isinstance(usage, dict) else 0)
                        or 0
                    )
                    tokens_out = (
                        getattr(usage, "completion_tokens", None)
                        or (usage.get("completion_tokens") if isinstance(usage, dict) else 0)
                        or 0
                    )
            except Exception as e:
                print(
                    f"[llm-trace] response_parse_error step={step} "
                    f"error_type={type(e).__name__} error={e}",
                    file=sys.stderr, flush=True,
                )

            truncated_content, did_truncate = _truncate(content_str, max_chars)
            _append_to_file(
                {
                    "step": step,
                    "direction": "response",
                    "chars": len(content_str),
                    "truncated": did_truncate,
                    "finish_reason": finish_reason,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "timestamp": time.time(),
                    "content": content_str,
                }
            )
            _emit_block(
                f"[llm-trace] step={step} <<< RESPONSE chars={len(content_str)} "
                f"tokens_in={tokens_in} tokens_out={tokens_out} "
                f"finish_reason={finish_reason!r}"
                + (" (TRUNCATED in stderr)" if did_truncate else ""),
                truncated_content,
            )

        def log_failure_event(  # type: ignore[override]
            self,
            kwargs: dict,
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            """Surface Bedrock / OpenAI / litellm errors loudly.

            Sweagent's outer loop catches model exceptions and may
            terminate the agent without a clear root-cause log. This
            handler dumps the exception's class, message, and (when
            present) the HTTP status / response body so the operator
            knows whether it's context-overflow, throttling, auth,
            content-filter, etc. Always emits to stderr — independent
            of ``CROSSBENCH_LLM_TRACE_STDERR_DISABLED`` (which gates
            the per-step REQUEST/RESPONSE noise, not actual errors).
            """
            if not _enabled():
                return
            step = _trace_counter[0]
            _trace_counter[0] = step + 1
            exc = kwargs.get("exception") if isinstance(kwargs, dict) else None
            exc_type = type(exc).__name__ if exc is not None else "UnknownException"
            exc_msg = str(exc) if exc is not None else "(no exception object on kwargs)"
            # Bedrock / OpenAI errors often carry .response or .status_code
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            response_body = ""
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    response_body = (
                        response.text if hasattr(response, "text") else str(response)
                    )[:2000]
                except Exception:
                    response_body = ""
            # Always emit — bypass the stderr-disabled gate (errors
            # are signal, not noise).
            sep = "[llm-trace] " + "✗" * 60
            print(sep, file=sys.stderr, flush=True)
            print(
                f"[llm-trace] step={step} <<< ERROR exception={exc_type} "
                f"status_code={status_code}",
                file=sys.stderr, flush=True,
            )
            for line in exc_msg.splitlines():
                print(f"[llm-trace] {line}", file=sys.stderr, flush=True)
            if response_body:
                print(f"[llm-trace] --- response body (first 2000 chars) ---", file=sys.stderr, flush=True)
                for line in response_body.splitlines():
                    print(f"[llm-trace] {line}", file=sys.stderr, flush=True)
            print(sep, file=sys.stderr, flush=True)
            # JSONL record so post-processing can grep for errors
            _append_to_file({
                "step": step,
                "direction": "error",
                "exception_type": exc_type,
                "status_code": status_code,
                "message": exc_msg,
                "response_body": response_body,
                "timestamp": time.time(),
            })

    litellm.callbacks.append(LLMTrace())
    print(
        f"[llm-trace] registered ENV_ENABLED=1 "
        f"max_content_chars={max_chars} "
        f"file={os.environ.get(ENV_FILE, '<unset>')!r}",
        file=sys.stderr, flush=True,
    )
    return True


SITECUSTOMIZE_REGISTER_LINE = """\
try:
    from cross_bench.execution.scaffolds.llm_trace_hook import (
        register_llm_trace_hook,
    )
    register_llm_trace_hook()
except Exception:
    pass
"""
