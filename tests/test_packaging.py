"""Lock the orchestrator Docker image's pip-install line against pyproject extras.

Why this exists
---------------
Benchmark adapters and memory middlewares routinely import their third-party
packages **lazily** (inside ``evaluate()`` / ``pre_prompt()`` / ``post_response``)
so that a user who only runs ``humaneval`` doesn't pay an ``evalplus`` install
cost. Lazy imports mean a missing dependency stays invisible until a task
actually fires.

In the dev environment we mitigate this by installing the relevant extras
locally; in CI we run ``pytest`` against a Python env that may or may not have
every extra. Neither catches the case that bit us on the first SWE-bench AWS
smoke deploy (``swebench-mem0v2-noinfer-smoke-20260520-165152``): the
``swebench>=4.1`` package was declared as the ``benchmark-swebench`` extra in
``pyproject.toml`` but the Dockerfile's ``pip install`` line didn't include
``benchmark-swebench``. 10/10 tasks errored with
``ModuleNotFoundError: No module named 'swebench'`` after an 8-second container
run — there is no environment where this gap was visible BEFORE production.

These tests close that loop by parsing both files at test time and asserting
that every declared (non-dev) optional-dependency extra is installed by the
Dockerfile, transitively expanding umbrella extras
(``cross-bench[memory-mem0,...]``-style references).

When the test fails
-------------------
1. Read the assertion message. It lists the extras the Dockerfile is missing.
2. Either add the extra to the appropriate Dockerfile umbrella
   (``memory`` / ``benchmark``) in ``pyproject.toml``, OR add it directly to
   the Dockerfile pip-install line, OR — if the extra genuinely shouldn't be
   in the orchestrator image — add it to :data:`SKIP_EXTRAS` below with a
   one-line rationale comment.
3. Re-run the test; it should pass.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


# Extras NOT expected to be installed in the orchestrator image. Every entry
# must have a one-line rationale — silent skip-listing is exactly the failure
# mode this test is meant to prevent.
SKIP_EXTRAS: dict[str, str] = {
    "dev": "test-only deps (pytest, ruff, mypy) — not needed in the running image",
    "infra": "Pulumi runs outside the orchestrator (on the deployer's laptop / CI)",
    "all": "pyproject meta-umbrella that pulls in the dev / infra extras above",
    "memory-mem0v1": (
        "mem0ai v1 and v2 share the import name ``mem0`` and have "
        "incompatible search() signatures — pip resolves to one or the other, "
        "never both. The default image installs memory-mem0 (=v2); a v1 image "
        "is a separate build, see deviations.md V1.M14."
    ),
}


# Regex that locates the orchestrator's extras-install line — i.e. the
# ``pip install ... ".[...]"`` invocation that installs all optional
# dependencies. The Dockerfile may have OTHER ``pip install`` lines (e.g.
# the ``--no-deps .`` overlay that installs cross-bench itself after the
# real source has been copied); we ignore those by requiring the matched
# argument to include the ``.[<extras>]`` syntax.
#
# Arbitrary flags are tolerated between ``pip install`` and the spec —
# we currently use ``-c constraints.txt`` and previously ``--no-cache-dir``;
# both should match. The ``[^\n]*?`` is non-greedy and same-line so the
# ``--no-deps .`` overlay (different line, no ``.[<extras>]``) isn't
# misclassified as the extras-install line.
_PIP_INSTALL_RE = re.compile(
    r'pip install[^\n]*?"\.\[([^\]]+)\]"',
)

# Pattern for ``cross-bench[a,b,c]`` umbrella references inside an extra's
# dependency list.
_UMBRELLA_RE = re.compile(r"cross-bench\[([^\]]+)\]")


def _repo_root() -> Path:
    """Walk up from this file until we hit ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate pyproject.toml above tests/")


def _load_extras() -> dict[str, list[str]]:
    """``{extra_name: list_of_requirement_strings}`` from pyproject.toml."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        pp = tomllib.load(f)
    return pp["project"]["optional-dependencies"]


def _dockerfile_extras() -> set[str]:
    """The extras passed to ``pip install`` in the orchestrator Dockerfile."""
    dockerfile = (_repo_root() / "Dockerfile").read_text(encoding="utf-8")
    matches = _PIP_INSTALL_RE.findall(dockerfile)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one ``pip install --no-cache-dir \".[...]\"`` "
            f"line in Dockerfile; found {len(matches)}. Update "
            f"_PIP_INSTALL_RE if the install structure changed."
        )
    return {part.strip() for part in matches[0].split(",")}


def _expand_umbrellas(
    seed: set[str],
    extras: dict[str, list[str]],
) -> set[str]:
    """Walk umbrella references transitively.

    If the Dockerfile installs the ``memory`` umbrella, and ``memory`` is
    declared as ``cross-bench[memory-mem0,memory-zep,...]``, then
    ``memory-mem0`` etc. are also installed. We follow these references to
    a fixpoint so the leaf check below sees every reachable extra.
    """
    reachable = set(seed)
    queue = list(seed)
    while queue:
        e = queue.pop()
        for dep in extras.get(e, []):
            for sub in _UMBRELLA_RE.findall(dep):
                for s in sub.split(","):
                    s = s.strip()
                    if s and s not in reachable:
                        reachable.add(s)
                        queue.append(s)
    return reachable


def test_dockerfile_pip_install_line_is_findable():
    """Sanity check — if the pip-install structure changed, this test fails
    LOUDLY so the dep-coverage test below doesn't silently pass on garbage."""
    # ``_dockerfile_extras`` raises if the regex doesn't match exactly once.
    extras = _dockerfile_extras()
    assert extras, "Dockerfile pip-install line resolved to an empty extras set"


def test_dockerfile_installs_all_declared_extras():
    """Every non-dev pyproject optional-dependency must be installed by the image.

    Catches the class of bug where an adapter or middleware adds a new
    third-party dependency (via a new ``benchmark-foo`` or ``memory-bar``
    extra) but the Dockerfile umbrella isn't updated to pick it up. Without
    this lock the gap is invisible until the task is dispatched on AWS and
    fails for that specific benchmark — see the test docstring for the
    historical incident.
    """
    extras = _load_extras()
    declared = set(extras.keys())

    installed = _dockerfile_extras()
    reachable = _expand_umbrellas(installed, extras)

    expected = declared - SKIP_EXTRAS.keys()
    missing = expected - reachable

    if missing:
        details = "\n".join(f"  - {m}" for m in sorted(missing))
        raise AssertionError(
            f"\nThe orchestrator Dockerfile doesn't install these "
            f"pyproject extras:\n{details}\n\n"
            f"Either:\n"
            f"  1. Add them to an umbrella (``memory`` or ``benchmark``) in "
            f"pyproject.toml so the Dockerfile picks them up automatically.\n"
            f"  2. Add them directly to the Dockerfile's pip-install line.\n"
            f"  3. If the extra genuinely shouldn't be in the orchestrator "
            f"image, add it to SKIP_EXTRAS in this test with a one-line "
            f"rationale comment.\n\n"
            f"Dockerfile currently installs (after umbrella expansion): "
            f"{sorted(reachable)}\n"
            f"pyproject declares: {sorted(declared)}\n"
        )


def test_no_undeclared_skip_extras():
    """Every entry in :data:`SKIP_EXTRAS` must correspond to a real pyproject
    extra. If someone renames an extra (e.g. ``dev`` → ``dev-tools``) the
    skip-listing should fail loudly rather than silently mask a missing entry."""
    declared = set(_load_extras().keys())
    bogus = set(SKIP_EXTRAS.keys()) - declared
    assert not bogus, (
        f"SKIP_EXTRAS references extras that no longer exist in pyproject: "
        f"{sorted(bogus)}"
    )


@pytest.mark.parametrize("extra,rationale", sorted(SKIP_EXTRAS.items()))
def test_skip_extras_have_rationale(extra: str, rationale: str):
    """Every skip-listed extra must have a non-empty rationale comment.

    The string is part of the SKIP_EXTRAS dict literal; this test makes sure
    no one drops a one-word placeholder there and forgets to explain why.
    """
    assert rationale and len(rationale.strip()) >= 20, (
        f"SKIP_EXTRAS[{extra!r}] rationale is too short — write at least one "
        f"clear sentence explaining why this extra is NOT in the orchestrator "
        f"image. Current: {rationale!r}"
    )
