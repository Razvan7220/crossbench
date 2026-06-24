"""Per-call mem0 memory injection for the SEAL (SWE-agent) scaffold.

Strategy
--------
SEAL runs SWE-agent as an opaque subprocess (``python -m sweagent run-batch``);
the agent loop calls ``litellm.completion(...)`` once per step
(``third_party/swe_agent/sweagent/agent/models.py::LiteLLMModel._single_query``).
A CrossBench-side ``MemoryMiddleware.pre_prompt`` hook cannot reach into
that subprocess. Instead, this module registers a LiteLLM ``CustomLogger``
**inside** the subprocess so every step's outgoing messages flow through
mem0 ``Memory.search()`` (recall) and every successful response flows
through ``Memory.add()`` (store).

Architectural notes:

* mem0 is a Python library, not a daemon. The hook ``import mem0``
  directly and instantiates ``mem0.Memory(config=...)`` lazily inside
  the subprocess. No HTTP, no timeouts to tune.
* mem0 state is **per-task** by construction: the sweagent subprocess
  restarts between SEAL tasks, so each task gets a fresh in-process
  Qdrant + SQLite history. No cross-task contamination.
* mem0's ``Memory.add(infer=...)`` flag is load-bearing for SWE-bench
  code-RAG (see deviations.md M9 / C13). The orchestrator-side
  ``mem0_no_infer`` profile sets ``infer=False``; this hook forwards
  the flag via ``CROSSBENCH_MEM0_INFER``.

Activation contract
-------------------
This module is a no-op unless ``CROSSBENCH_MEM0_USER_ID`` is set in the
subprocess env. ``seal.py::execute_seal`` exports the variables below
**only** when a ``Mem0V2EmbeddedMiddleware`` is stacked on the run.

``CROSSBENCH_MEM0_USER_ID``
    Per-run user_id. mem0v2's ``_user_id`` is set once in ``__init__``
    and is NOT reassigned by ``Mem0V2EmbeddedMiddleware.reset()``.
    Per-task isolation comes
    from the sweagent subprocess restart between tasks (fresh
    ``mem0.Memory()`` per subprocess), not from id rotation.
    Required — acts as the primary activation gate.
``CROSSBENCH_MEM0_INFER``
    ``"1"`` to enable mem0's LLM fact-extractor (default), ``"0"`` to
    store ingested chunks verbatim. **Always set ``"0"`` for SWE-bench
    code-RAG columns** per deviations.md M9 + C13.
``CROSSBENCH_MEM0_TOP_K``
    Max memories per recall. Defaults to 20 (matches the mem0
    middleware's ``retrieval_limit`` default).
``CROSSBENCH_MEM0_LLM_PROVIDER`` / ``_LLM_MODEL`` / ``_LLM_BASE_URL``
    Internal-LLM config for mem0's extractor. Auto-followed from the
    bench LLM via ``_inherit_bench_llm_into_memory`` (config.py).
``CROSSBENCH_MEM0_EMB_PROVIDER`` / ``_EMB_MODEL``
    Embedder config for mem0's vector store. Auto-followed from the
    bench embedder.
``CROSSBENCH_MEM0_FIRST_STEP_ONLY``
    When ``"1"``, only the **first** ``litellm.completion`` call in the
    subprocess fires recall; subsequent calls skip. Mirrors the
    ``memory_first_step_only`` benchmark mixin used by mini-SWE-agent.
``CROSSBENCH_MEM0_DISABLE_STORE``
    When ``"1"``, skips ``.add(...)`` on success. Recall stays on.
``CROSSBENCH_MEM0_METRICS_FILE``
    When set to a writable path, every recall + store appends one JSON
    line. ``seal.py::execute_seal`` reads + aggregates onto the
    evaluation dict (``mem0_recall_calls`` / ``mem0_recall_memories`` /
    ``mem0_recall_latency_ms`` / ``mem0_store_calls`` /
    ``mem0_store_tokens_estimate``). The engine then averages those
    across tasks into ``summary.json``.
``CROSSBENCH_MEM0_STATUS_FILE``
    When set, the sitecustomize bootstrap appends an import/register
    status line ("import_local_ok" / "register_ok" / etc.) so the
    launcher can diagnose silent registration failures.

Bootstrap
---------
``execute_seal`` writes a tiny ``sitecustomize.py`` to the per-task tempdir
that runs ``register_mem0_hook()`` from this module. Prepending the
tempdir to ``PYTHONPATH`` lets Python's site-init pick it up automatically
when the sweagent subprocess starts. The hook then attaches to every
``litellm.completion`` call inside that subprocess.

Failure modes
-------------
Every fallible call (mem0 import, Memory instantiation, search, add,
metric IO) is wrapped in ``try/except Exception`` and logged to stderr.
A failure degrades to "no injection this step" rather than aborting the
agent loop.

License notice: mem0 is Apache 2.0 (Copyright Mem0). This module
integrates mem0 as a runtime dependency only; no source is vendored.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# Env-var names — public so seal.py / tests can reference them by constant
# ---------------------------------------------------------------------------

ENV_USER_ID = "CROSSBENCH_MEM0_USER_ID"
ENV_INFER = "CROSSBENCH_MEM0_INFER"
ENV_TOP_K = "CROSSBENCH_MEM0_TOP_K"
ENV_LLM_PROVIDER = "CROSSBENCH_MEM0_LLM_PROVIDER"
ENV_LLM_MODEL = "CROSSBENCH_MEM0_LLM_MODEL"
ENV_LLM_BASE_URL = "CROSSBENCH_MEM0_LLM_BASE_URL"
ENV_EMB_PROVIDER = "CROSSBENCH_MEM0_EMB_PROVIDER"
ENV_EMB_MODEL = "CROSSBENCH_MEM0_EMB_MODEL"
ENV_FIRST_STEP_ONLY = "CROSSBENCH_MEM0_FIRST_STEP_ONLY"
ENV_DISABLE_STORE = "CROSSBENCH_MEM0_DISABLE_STORE"
ENV_METRICS_FILE = "CROSSBENCH_MEM0_METRICS_FILE"
ENV_STATUS_FILE = "CROSSBENCH_MEM0_STATUS_FILE"

DEFAULT_TOP_K = 20


# ---------------------------------------------------------------------------
# Module-level singletons (subprocess-scoped)
# ---------------------------------------------------------------------------

# One Memory instance per subprocess (= per task). Lazy-init on first call.
_memory_instance: Any = None
# Tracks litellm.completion call index for FIRST_STEP_ONLY gating.
_recall_counter = [0]
# Set once, by the first successful Memory() construction, so we can
# log "instance ready" exactly once per subprocess. Idempotent guard.
_init_logged = [False]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hook_enabled() -> bool:
    """Primary activation gate: ENV_USER_ID must be set + non-empty."""
    return bool(os.environ.get(ENV_USER_ID, "").strip())


def _log(msg: str) -> None:
    """Loud-but-cheap stderr log line, prefixed for grep-ability."""
    try:
        print(f"[mem0-hook] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _append_metric(record: dict[str, Any]) -> None:
    """Append one JSON line to ENV_METRICS_FILE. Silent no-op on failure."""
    path = os.environ.get(ENV_METRICS_FILE, "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _build_mem0_config() -> dict[str, Any]:
    """Compose the ``mem0.Memory.from_config`` config dict from env vars.

    Returns ``{}`` (empty dict) to signal "use mem0 defaults" — the caller
    then instantiates ``Memory()`` instead of ``Memory.from_config(...)``.

    Mirrors ``mem0v2_middleware._build_embedded_config`` byte-for-byte
    for the keys it cares about: ``llm.provider``, ``llm.config.model``,
    ``llm.config.openai_base_url``, ``llm.config.api_key``,
    ``embedder.provider``, ``embedder.config.model``,
    ``embedder.config.openai_base_url``. This keeps the in-subprocess
    mem0 instance numerically equivalent to the parent-side one when
    given the same inputs.
    """
    cfg: dict[str, Any] = {}
    llm_provider = os.environ.get(ENV_LLM_PROVIDER, "").strip()
    llm_model = os.environ.get(ENV_LLM_MODEL, "").strip()
    llm_base = os.environ.get(ENV_LLM_BASE_URL, "").strip()
    emb_provider = os.environ.get(ENV_EMB_PROVIDER, "").strip()
    emb_model = os.environ.get(ENV_EMB_MODEL, "").strip()

    if llm_provider or llm_model:
        llm_cfg: dict[str, Any] = {}
        if llm_model:
            llm_cfg["model"] = llm_model
        if llm_base:
            llm_cfg["openai_base_url"] = llm_base
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            llm_cfg["api_key"] = api_key
        cfg["llm"] = {
            "provider": llm_provider or "openai",
            "config": llm_cfg,
        }
    if emb_provider or emb_model:
        emb_cfg: dict[str, Any] = {}
        if emb_model:
            emb_cfg["model"] = emb_model
        cfg["embedder"] = {
            "provider": emb_provider or "openai",
            "config": emb_cfg,
        }
    return cfg


def _get_memory() -> Any:
    """Lazy-instantiate mem0.Memory on first call. Returns ``None`` on failure."""
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance
    try:
        from mem0 import Memory  # type: ignore
    except Exception as e:
        _log(f"mem0 import failed: {type(e).__name__}: {e}")
        return None
    cfg = _build_mem0_config()
    try:
        if cfg:
            _memory_instance = Memory.from_config(config_dict=cfg)
        else:
            _memory_instance = Memory()
    except Exception as e:
        _log(f"Memory() construction failed: {type(e).__name__}: {e}")
        return None
    if not _init_logged[0]:
        _init_logged[0] = True
        _log(
            f"Memory instance ready (config_keys={list(cfg.keys())}, "
            f"user_id={os.environ.get(ENV_USER_ID, '?')})"
        )
    return _memory_instance


def _top_k() -> int:
    try:
        return int(os.environ.get(ENV_TOP_K, str(DEFAULT_TOP_K)))
    except ValueError:
        return DEFAULT_TOP_K


def _infer_flag() -> bool:
    """Read CROSSBENCH_MEM0_INFER. Defaults to True (mem0's own default)."""
    raw = os.environ.get(ENV_INFER, "1").strip()
    return raw != "0"


def _flatten_content(content: Any) -> str:
    """Flatten str / list-of-content-parts shapes to a single string.

    LiteLLM passes ``content`` as either a plain string or a list of
    multimodal parts. mem0's search/add wants a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


def _extract_response_text(response_obj: Any) -> str:
    """Pull the assistant text from a LiteLLM ``ModelResponse``.

    Returns ``""`` if no extractable text is found.
    """
    try:
        choices = getattr(response_obj, "choices", None)
        if choices is None and isinstance(response_obj, dict):
            choices = response_obj.get("choices")
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content:
            return content
        # Tool-call fallback: JSON-encode for storage so identifiers survive.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        if tool_calls:
            try:
                serialized = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None) or (
                        tc.get("function") if isinstance(tc, dict) else None
                    )
                    serialized.append({
                        "id": getattr(tc, "id", None) or (
                            tc.get("id") if isinstance(tc, dict) else None
                        ),
                        "function": {
                            "name": (
                                getattr(fn, "name", None)
                                if fn is not None and not isinstance(fn, dict)
                                else (fn or {}).get("name")
                            ),
                            "arguments": (
                                getattr(fn, "arguments", None)
                                if fn is not None and not isinstance(fn, dict)
                                else (fn or {}).get("arguments")
                            ),
                        },
                    })
                return json.dumps(serialized)
            except Exception:
                return ""
        return ""
    except Exception:
        return ""


def _recall_query(messages: list[dict[str, Any]]) -> str:
    """Pick the recall query from the live message list.

    Strategy: prefer the latest assistant message (reflects agent's
    current focus / next action). Fall back to the latest user message
    if there's no assistant content yet (step 0). Truncate to 2000
    chars — long codeblocks make embedding queries noisy.

    Note: a thin turn like ``"Let me check"`` produces a short query.
    If mem0 recall suffers from thin queries, consider joining the last
    N assistant messages instead of just the latest.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = _flatten_content(msg.get("content"))
            if text and text.strip():
                return text[:2000]
            break
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _flatten_content(msg.get("content"))
            if text and text.strip():
                return text[:2000]
            break
    return ""


def _format_memories_block(hits: list[dict[str, Any]]) -> str:
    """Render recalled mem0 memories as a prepend block for the agent prompt.

    Each ``hit`` is a mem0 search-result dict shaped roughly:
    ``{"memory": str, "score": float, "metadata": {"source_id": ...}}``.
    With ``infer=False`` the ``memory`` field is the verbatim chunk
    we stored. With ``infer=True`` it's the LLM-extracted NL summary.
    Both are useful as recall context.
    """
    if not hits:
        return ""
    lines = ["<recalled_episodes>"]
    for h in hits:
        body = h.get("memory") if isinstance(h, dict) else None
        if not body or not isinstance(body, str):
            continue
        meta = h.get("metadata") or {}
        src = (meta.get("source_id") if isinstance(meta, dict) else None) or ""
        prefix = f"[{src}] " if src else "- "
        lines.append(f"{prefix}{body}")
    lines.append("</recalled_episodes>")
    return "\n".join(lines)


def _render_episode(command: str, observation: str) -> str:
    """Render one trajectory step as a mem0-ingestable episode string.

    **Byte-identical to ``mini_swe_agent._render_episode``** (S18 Option E
    episode format). The two paths MUST emit the same shape so that a
    mem0v2 cell on ``-b swebench_pro_seal`` is directly comparable to
    its ``-b swebench_pro`` (mini-SWE-agent) counterpart — that
    comparability is the whole point of having a SEAL mem0 hook.

    Locked by ``test_render_episode_byte_equal_to_mini_swe_agent``: if
    either side drifts, CI fails immediately. Do NOT change this format
    without updating the mini-SWE-agent side in the same commit AND
    bumping the lock test.
    """
    return (
        f"Action:\n```\n{command}\n```\n\n"
        f"Observation:\n{observation}"
    )


def _prepend_to_last_user(
    messages: list[dict[str, Any]], block: str
) -> bool:
    """Mutate ``messages`` in place: prepend ``block`` to last user message.

    Returns True if mutation happened. The in-place mutation lets the
    LiteLLM caller's live list see the change before the HTTP request
    fires.
    """
    if not block:
        return False
    for msg in reversed(messages):
        if msg.get("role") == "user":
            existing = msg.get("content")
            if isinstance(existing, str):
                msg["content"] = block + "\n\n" + existing
            elif isinstance(existing, list):
                # Multimodal content: prepend a text part at position 0.
                msg["content"] = [{"type": "text", "text": block}] + existing
            else:
                msg["content"] = block
            return True
    return False


# ---------------------------------------------------------------------------
# LiteLLM CustomLogger
# ---------------------------------------------------------------------------


def _build_logger_class() -> Any:
    """Return a subclass of ``litellm.integrations.custom_logger.CustomLogger``.

    Constructed at registration time (not at module import) so the
    failure mode is graceful: if litellm isn't available in this
    subprocess, ``register_mem0_hook`` logs and returns False instead
    of raising.
    """
    from litellm.integrations.custom_logger import CustomLogger  # type: ignore

    class Mem0Injector(CustomLogger):  # type: ignore[misc, valid-type]
        """LiteLLM CustomLogger that splices mem0 memory into every call."""

        def log_pre_api_call(  # type: ignore[override]
            self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
        ) -> None:
            if not _hook_enabled():
                return
            if not isinstance(messages, list) or not messages:
                return
            # FIRST_STEP_ONLY gate.
            first_step_only = (
                os.environ.get(ENV_FIRST_STEP_ONLY, "0").strip() == "1"
            )
            call_index = _recall_counter[0]
            _recall_counter[0] = call_index + 1
            if first_step_only and call_index > 0:
                return

            query = _recall_query(messages)
            if not query or not query.strip():
                _log(f"step{call_index} skipped — empty recall query")
                return

            mem = _get_memory()
            if mem is None:
                return

            user_id = os.environ.get(ENV_USER_ID, "").strip()
            top_k = _top_k()

            t0 = time.time()
            try:
                # Mem0 v2.0.0 contract: user_id goes in filters, not top-level.
                result = mem.search(
                    query=query[:500],
                    filters={"user_id": user_id},
                    top_k=top_k,
                )
            except Exception as e:
                _log(f"search() failed: {type(e).__name__}: {e}")
                _append_metric({
                    "event": "recall",
                    "call_index": call_index,
                    "error": f"{type(e).__name__}: {e}",
                    "tokens": 0,
                    "latency_ms": (time.time() - t0) * 1000,
                    "memories": 0,
                })
                return
            latency_ms = (time.time() - t0) * 1000

            hits = (
                result.get("results", [])
                if isinstance(result, dict)
                else (result or [])
            )
            block = _format_memories_block(hits)
            if block:
                _prepend_to_last_user(messages, block)
            _append_metric({
                "event": "recall",
                "call_index": call_index,
                "tokens": len(block) // 4,  # cheap chars/4 proxy
                "latency_ms": latency_ms,
                "memories": len(hits),
                "injection_chars": len(block),
                **({"injection_text": block} if call_index == 0 and block else {}),
            })

        def log_success_event(  # type: ignore[override]
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: float,
            end_time: float,
        ) -> None:
            if not _hook_enabled():
                return
            if os.environ.get(ENV_DISABLE_STORE, "0").strip() == "1":
                return
            messages = kwargs.get("messages") if isinstance(kwargs, dict) else None
            if not isinstance(messages, list) or not messages:
                return

            mem = _get_memory()
            if mem is None:
                return

            # The store payload is the (last user message, assistant response)
            # pair — i.e. one episode. mem0's Option-E SWE-bench pattern.
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = _flatten_content(m.get("content"))
                    break
            assistant = _extract_response_text(response_obj)
            if not last_user.strip() or not assistant.strip():
                # Empty content — no useful episode to store.
                return

            user_id = os.environ.get(ENV_USER_ID, "").strip()
            infer = _infer_flag()
            call_index = _recall_counter[0] - 1  # most recent recall

            # Episode shape mirrors mini-SWE-agent S18 Option E
            # byte-for-byte via the shared ``_render_episode``
            # helper (locked by a parallel byte-equality test).
            # 8000-char truncation per side: episodes that exceed
            # the model's context don't help recall anyway
            # (see deviations.md S25 §3).
            episode_text = _render_episode(
                last_user[:8000], assistant[:8000]
            )

            t0 = time.time()
            try:
                mem.add(
                    [{"role": "user", "content": episode_text}],
                    user_id=user_id,
                    metadata={
                        "source_id": f"step:{call_index}",
                        "step": call_index,
                    },
                    infer=infer,
                )
            except Exception as e:
                _log(f"add() failed: {type(e).__name__}: {e}")
                _append_metric({
                    "event": "store",
                    "call_index": call_index,
                    "error": f"{type(e).__name__}: {e}",
                    "tokens_estimate": 0,
                    "latency_ms": (time.time() - t0) * 1000,
                })
                return
            latency_ms = (time.time() - t0) * 1000

            tokens_estimate = (len(last_user) + len(assistant)) // 4
            _append_metric({
                "event": "store",
                "call_index": call_index,
                "tokens_estimate": tokens_estimate,
                "latency_ms": latency_ms,
                "infer": infer,
            })

    return Mem0Injector


# ---------------------------------------------------------------------------
# Public entrypoint — called by sitecustomize.py inside the subprocess
# ---------------------------------------------------------------------------


def register_mem0_hook() -> bool:
    """Register the mem0 CustomLogger with litellm. Idempotent.

    Returns True on success, False on graceful failure (litellm absent,
    mem0 absent, or env contract not satisfied).
    """
    if not _hook_enabled():
        # Env contract not satisfied — silent no-op so the sweagent
        # subprocess starts cleanly even on non-mem0 cells.
        return False
    try:
        import litellm  # type: ignore
    except Exception as e:
        _log(f"litellm import failed: {type(e).__name__}: {e}")
        return False
    try:
        injector_cls = _build_logger_class()
    except Exception as e:
        _log(f"CustomLogger class build failed: {type(e).__name__}: {e}")
        return False
    # Idempotency: don't double-register on hot-reload.
    callbacks = list(getattr(litellm, "callbacks", None) or [])
    for cb in callbacks:
        if type(cb).__name__ == "Mem0Injector":
            _log("Mem0Injector already registered (idempotent skip)")
            return True
    callbacks.append(injector_cls())
    litellm.callbacks = callbacks
    _log(
        f"registered Mem0Injector "
        f"(user_id={os.environ.get(ENV_USER_ID, '?')}, "
        f"top_k={_top_k()}, "
        f"infer={_infer_flag()}, "
        f"first_step_only={os.environ.get(ENV_FIRST_STEP_ONLY, '0')}, "
        f"disable_store={os.environ.get(ENV_DISABLE_STORE, '0')})"
    )
    return True


# ---------------------------------------------------------------------------
# sitecustomize.py stub generator (used by seal.py)
# ---------------------------------------------------------------------------


SITECUSTOMIZE_STUB_MEM0 = '''
# --- mem0 LiteLLM hook (CrossBench SEAL scaffold) ---
# Activates only when CROSSBENCH_MEM0_USER_ID is set.
#
# Defensive ``_status`` definition: ``seal.py`` appends THIS block onto
# the sitecustomize.py the step-counter writer produced first.
# ``step_counter_hook.SITECUSTOMIZE_STUB`` does NOT define ``_status``,
# so the appended stanza would hit a NameError on every status call
# without this guard. Redefining when already present is safe — our
# ``_status_mem0`` writes to CROSSBENCH_MEM0_STATUS_FILE.
if "_status" not in dir() or "_status_mem0" not in dir():
    import os as _os_mem0
    def _status_mem0(msg):
        p = _os_mem0.environ.get("CROSSBENCH_MEM0_STATUS_FILE", "").strip()
        if not p:
            return
        try:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(msg + "\\n")
        except Exception:
            pass
try:
    import mem0_litellm_hook as _mem0_local
    if _mem0_local.register_mem0_hook():
        _status_mem0("mem0_local_ok")
    else:
        _status_mem0("mem0_disabled (CROSSBENCH_MEM0_USER_ID not set)")
except Exception as _mem0_e1:
    try:
        from cross_bench.execution.scaffolds.mem0_litellm_hook import (
            register_mem0_hook as _mem0_reg,
        )
        if _mem0_reg():
            _status_mem0("mem0_pkg_ok")
        else:
            _status_mem0("mem0_disabled (CROSSBENCH_MEM0_USER_ID not set)")
    except Exception as _mem0_e2:
        _status_mem0(f"mem0_failed: {type(_mem0_e2).__name__}: {_mem0_e2}")
'''
