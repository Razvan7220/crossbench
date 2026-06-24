"""Lock the evermemos_embedded prompts to upstream EverCore byte-for-byte.

The embedded variant of the EverMemOS middleware is a re-implementation of
the EverCore algorithm in-process. Its faithfulness depends on the prompt
templates matching upstream byte-for-byte. This test fails the moment a
prompt drifts.

Audit baseline: ``methods/EverCore/`` at commit ``facd7b69`` of
``EverMind-AI/EverOS`` (2026-04-27). See deviations.md entry E.E3.

Two run modes:
1. **Live diff** (preferred): if the upstream EverCore checkout is on disk
   at the path discoverable via the ``EVERCORE_PATH`` env var (or the
   default ``../EverOS/methods/EverCore``), each test re-imports the
   upstream prompt module and compares byte-for-byte.
2. **Frozen hash**: if no checkout is available, fall back to comparing
   against SHA-256 hashes captured here. Updating the upstream baseline
   means re-running ``make update-evermemos-prompt-hashes`` (TODO) or
   editing ``EXPECTED_SHA256`` below by hand AFTER reading the upstream
   prompt diff.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


# Frozen hashes of upstream prompts at audit-baseline commit facd7b69.
# These are the SHA-256 of the *Python string value* (post-import), not
# of the source file. Update only after reading and accepting the
# upstream diff.
EXPECTED_SHA256 = {
    "BATCH_BOUNDARY_DETECTION_PROMPT": (
        "48494559d4d5e51e6671aaec9d598a2bf567fdf6ec26b2ea2acbd3ba39cb41be"
    ),
    "DEFAULT_CUSTOM_INSTRUCTIONS": (
        "69646cb6d8f8716898045628106903ef6227884893a3a8005caa01db504905a6"
    ),
    "EPISODE_GENERATION_PROMPT": (
        "f65511e574b9940e2d2cf3c420ba55b4830d6351f0be14b2525b32734a7c1d2f"
    ),
    "FORESIGHT_GENERATION_PROMPT": (
        "e4ad0774b30c9ae2990417119a14d00ecf5ad4e8a17805b568c5ae06018811f6"
    ),
    "ATOMIC_FACT_PROMPT": (
        "c2429c21facc72d545e9b237d9b0f9019e7cb26c2eab10615b87f20f1be0f300"
    ),
    "SUFFICIENCY_CHECK_PROMPT": (
        "09100d7fecfb04f64764ed4ac72be74e6d357630733afc7e833f63949e148a4c"
    ),
    "MULTI_QUERY_GENERATION_PROMPT": (
        "d88e89287bd1e142a2a51be2cf8913d96c9ac7dcf1e11aeb48b0b1dfc5fdf2e6"
    ),
    "REFINED_QUERY_PROMPT": (
        "ea6adf6812ed3648f70ce8ac85e325b1b9679540666604a700b2e0eef7e6afc1"
    ),
}


# Map our middleware constant → (upstream relative path, upstream constant name)
UPSTREAM_PROMPTS = {
    "BATCH_BOUNDARY_DETECTION_PROMPT": (
        "src/memory_layer/prompts/en/conv_prompts.py",
        "CONV_BATCH_BOUNDARY_DETECTION_PROMPT",
    ),
    "DEFAULT_CUSTOM_INSTRUCTIONS": (
        "src/memory_layer/prompts/en/episode_mem_prompts.py",
        "DEFAULT_CUSTOM_INSTRUCTIONS",
    ),
    "EPISODE_GENERATION_PROMPT": (
        "src/memory_layer/prompts/en/episode_mem_prompts.py",
        "EPISODE_GENERATION_PROMPT",
    ),
    "FORESIGHT_GENERATION_PROMPT": (
        "src/memory_layer/prompts/en/foresight_prompts.py",
        "FORESIGHT_GENERATION_PROMPT",
    ),
    "ATOMIC_FACT_PROMPT": (
        "src/memory_layer/prompts/en/atomic_fact_prompts.py",
        "ATOMIC_FACT_PROMPT",
    ),
    "SUFFICIENCY_CHECK_PROMPT": (
        "evaluation/src/adapters/evermemos/prompts/sufficiency_check_prompts.py",
        "SUFFICIENCY_CHECK_PROMPT",
    ),
    "MULTI_QUERY_GENERATION_PROMPT": (
        "evaluation/src/adapters/evermemos/prompts/multi_query_prompts.py",
        "MULTI_QUERY_GENERATION_PROMPT",
    ),
    "REFINED_QUERY_PROMPT": (
        "evaluation/src/adapters/evermemos/prompts/refined_query_prompts.py",
        "REFINED_QUERY_PROMPT",
    ),
}


def _evercore_root() -> Path | None:
    raw = os.environ.get("EVERCORE_PATH")
    if raw:
        p = Path(raw).expanduser()
        return p if p.exists() else None
    # Default colocation: ../EverOS/methods/EverCore relative to this repo.
    candidate = Path(__file__).resolve().parents[2] / ".." / "EverOS" / "methods" / "EverCore"
    candidate = candidate.resolve()
    return candidate if candidate.exists() else None


def _load_upstream(rel_path: str, name: str) -> str:
    root = _evercore_root()
    if root is None:
        pytest.skip("EverCore checkout not available; falling back to frozen hashes")
    full = root / rel_path
    spec = importlib.util.spec_from_file_location("up", full)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load upstream module at {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


@pytest.fixture(scope="module")
def ours():
    from cross_bench.systems.memory import evermemos_embedded
    return evermemos_embedded


@pytest.mark.parametrize("name", list(UPSTREAM_PROMPTS.keys()))
def test_prompt_byte_exact_against_upstream(name, ours):
    """If the EverCore checkout is on disk, our prompt must match it byte-for-byte."""
    rel_path, upstream_name = UPSTREAM_PROMPTS[name]
    upstream = _load_upstream(rel_path, upstream_name)  # may pytest.skip
    actual = getattr(ours, name)
    assert actual == upstream, (
        f"{name} drifted from upstream {upstream_name} at "
        f"methods/EverCore/{rel_path}; lengths: ours={len(actual)} "
        f"theirs={len(upstream)}"
    )


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_SHA256.items()))
def test_prompt_frozen_hash(name, expected, ours):
    """Frozen-hash regression: prompt SHA-256 must match audit-baseline.

    This runs even when EverCore isn't on disk. If you intentionally
    update an upstream prompt, also update EXPECTED_SHA256 in this file.
    """
    actual = getattr(ours, name).encode("utf-8")
    digest = hashlib.sha256(actual).hexdigest()
    assert digest == expected, (
        f"{name} hash drifted from audit baseline. "
        f"Got {digest}, expected {expected}. "
        "If the change was intentional, update EXPECTED_SHA256 in "
        "tests/systems/test_evermemos_prompts.py and document the "
        "delta in deviations.md (E.E3)."
    )
