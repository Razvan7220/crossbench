"""Locks the internal/boundary token-recording wiring across middlewares.

Companion to the 2026-06-15 ContextVar-across-executor fix
(``_mem0_token_hook`` / ``_memos_token_hook``) and the API-variant
``_record_io`` wiring landed alongside it.

Two layers:

* **Behavioural** — construct each owned *embedded* middleware
  (hindsight / graphiti / mastra), mock its LLM client to return a
  response carrying a ``usage`` object, invoke the LLM helper once, and
  assert ``get_internal_token_stats()`` actually incremented. This is
  the empirical confirmation that ``_record_internal_llm`` is reached on
  the real call path (evermemos was already confirmed end-to-end on a
  live LoCoMo run).

* **AST lock** — guards that every *boundary recorder* (delegated +
  API-variant middlewares) calls ``_record_io`` and every *internal
  recorder* (owned embedded re-implementations) calls
  ``_record_internal_llm``, so a future refactor can't silently drop the
  instrumentation. Dep-free, mirrors ``test_reset_counter_contract.py``.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from cross_bench.models import MemoryConfig

MEMORY_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "cross_bench" / "systems" / "memory"
)


def _fake_response(prompt_tokens: int = 11, completion_tokens: int = 7,
                   content: str = "fake-response"):
    """An object shaped like an OpenAI chat completion with usage."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Behavioural — embedded middlewares actually record internal-LLM tokens
# ---------------------------------------------------------------------------


def test_hindsight_embedded_llm_records_internal_tokens(monkeypatch):
    from cross_bench.systems.memory.hindsight_embedded import (
        HindsightEmbeddedMiddleware,
    )

    mw = HindsightEmbeddedMiddleware(
        MemoryConfig(provider="hindsight_embedded", params={})
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr(mw, "_get_client", lambda: client)

    out = asyncio.run(mw._llm("user prompt", system="sys"))
    assert out == "fake-response"
    stats = mw.get_internal_token_stats()
    assert stats["llm_calls"] == 1
    assert stats["tokens_in"] == 11
    assert stats["tokens_out"] == 7


def test_graphiti_embedded_llm_records_internal_tokens(monkeypatch):
    from cross_bench.systems.memory.graphiti_embedded import (
        GraphitiEmbeddedMiddleware,
    )

    mw = GraphitiEmbeddedMiddleware(
        MemoryConfig(provider="graphiti_embedded", params={})
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr(mw, "_get_client", lambda: client)

    out = asyncio.run(mw._llm("user prompt", system="sys"))
    assert out == "fake-response"
    stats = mw.get_internal_token_stats()
    assert stats["llm_calls"] == 1
    assert stats["tokens_in"] == 11
    assert stats["tokens_out"] == 7


def test_mastra_embedded_llm_call_records_internal_tokens(monkeypatch):
    import openai

    from cross_bench.systems.memory.mastra_embedded import (
        MastraEmbeddedMiddleware,
    )

    mw = MastraEmbeddedMiddleware(
        MemoryConfig(provider="mastra_embedded", params={})
    )
    client = MagicMock()
    # mastra's _llm_call is synchronous and constructs OpenAI(...) inline.
    client.chat.completions.create = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)

    out = mw._llm_call("sys", "user")
    assert out == "fake-response"
    stats = mw.get_internal_token_stats()
    assert stats["llm_calls"] == 1
    assert stats["tokens_in"] == 11
    assert stats["tokens_out"] == 7


# ---------------------------------------------------------------------------
# AST lock — the recording wiring is present per middleware category
# ---------------------------------------------------------------------------

# Delegated (opaque upstream package) + API variants (remote server):
# record the CrossBench↔middleware boundary via _record_io.
_BOUNDARY_RECORDERS = {
    "mem0v1_middleware.py",
    "mem0v2_middleware.py",
    "memos_middleware.py",
    "evermemos_middleware.py",
    "hindsight_middleware.py",
    "mastra_middleware.py",
    "zep_middleware.py",
    "letta_middleware.py",
}

# Owned in-process re-implementations: run their own LLM and record exact
# internal usage via _record_internal_llm.
_INTERNAL_RECORDERS = {
    "evermemos_embedded.py",
    "hindsight_embedded.py",
    "graphiti_embedded.py",
    "mastra_embedded.py",
}


def _file_calls_method(path: Path, method_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
        ):
            return True
    return False


def test_named_recorder_files_exist():
    """Guard against a rename making the locks below pass vacuously."""
    for name in _BOUNDARY_RECORDERS | _INTERNAL_RECORDERS:
        assert (MEMORY_DIR / name).is_file(), f"missing middleware file: {name}"


def test_boundary_recorders_call_record_io():
    missing = [
        name
        for name in sorted(_BOUNDARY_RECORDERS)
        if not _file_calls_method(MEMORY_DIR / name, "_record_io")
    ]
    assert not missing, (
        "boundary middlewares missing self._record_io() — the "
        "CrossBench↔middleware seam is unmeasured: " + ", ".join(missing)
    )


def test_internal_recorders_call_record_internal_llm():
    missing = [
        name
        for name in sorted(_INTERNAL_RECORDERS)
        if not _file_calls_method(MEMORY_DIR / name, "_record_internal_llm")
    ]
    assert not missing, (
        "embedded middlewares missing self._record_internal_llm() — "
        "internal-LLM cost is unmeasured: " + ", ".join(missing)
    )
