"""Every memory middleware's ``reset()`` must call ``_reset_internal_counters()``.

The ``MemoryMiddleware`` contract (``systems/base.py``) requires
subclasses' ``reset`` to zero the internal-token counters. Engine-side
per-task aggregation is snapshot-delta based
(``engine._snapshot_memory_stats`` / ``_stamp_memory_delta``), so a
missing call does NOT corrupt the summary columns — but the contract
keeps the raw counters conversation-scoped for any consumer reading
``get_internal_token_stats`` directly, and this lock keeps the base-class
docstring and the implementations from drifting apart again (the
2026-06-11 inspection found 7 of 16 implementations missing the call).

AST-based so it needs none of the middlewares' optional deps installed —
same lint-test pattern as
``test_config.py::test_builtin_profiles_must_not_set_scaffold_run_keys``.
"""

from __future__ import annotations

import ast
from pathlib import Path

MEMORY_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "cross_bench" / "systems" / "memory"
)


def _calls_reset_internal_counters(fn: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_reset_internal_counters":
                return True
    return False


def test_every_middleware_reset_calls_reset_internal_counters():
    assert MEMORY_DIR.is_dir(), f"middleware dir missing: {MEMORY_DIR}"

    offenders: list[str] = []
    checked = 0
    for path in sorted(MEMORY_DIR.glob("*.py")):
        if path.name.startswith("_"):
            # __init__.py + the _mem0/_memos token-hook helpers — no
            # middleware classes live there.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for item in cls.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "reset":
                    checked += 1
                    if not _calls_reset_internal_counters(item):
                        offenders.append(f"{path.name}::{cls.name}.reset")

    # Sanity floor so a glob/path regression can't make this pass
    # vacuously (16 middleware reset() implementations as of 2026-06.)
    assert checked >= 14, f"only found {checked} reset() implementations under {MEMORY_DIR}"

    assert not offenders, (
        "MemoryMiddleware.reset implementations missing "
        "self._reset_internal_counters() (see the internal-LLM token "
        "accounting contract in systems/base.py): " + "; ".join(offenders)
    )
