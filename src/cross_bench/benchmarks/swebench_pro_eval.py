"""SWE-bench Pro eval-harness port.

Pro is Scale AI's benchmark (``ScaleAI/SWE-bench_Pro`` on HuggingFace,
731 tasks across 11 repos and 4 languages). Its grading pipeline is
**not** the Princeton ``swebench`` PyPI package — it uses a custom
flow defined in ``scaleapi/SWE-bench_Pro-os/swe_bench_pro_eval.py``
with per-instance ``run_script.sh`` + ``parser.py`` shipped in the
upstream repo under ``run_scripts/<instance_id>/``.

This module is the CrossBench-side port: same algorithm, but running
inside :class:`SandboxManager` and consuming dataset fields from
``Task.metadata`` (set by ``SWEBenchAdapter.load_tasks`` when
``variant = "pro"``).

Pinned upstream:  https://github.com/scaleapi/SWE-bench_Pro-os
                  (``main`` HEAD as of 2026-05-25 audit)

Algorithm (one task)
--------------------
1. Build entryscript from the row:
   - ``git reset --hard <base_commit> && git checkout <base_commit>``
   - ``git apply -v /workspace/patch.diff``
   - run the row's ``before_repo_set_cmd`` (last line of the multi-line
     command per upstream's ``create_entryscript``)
   - ``bash /workspace/run_script.sh <selected_test_files_to_run,...>
     > /workspace/stdout.log 2> /workspace/stderr.log``
   - ``python /workspace/parser.py /workspace/stdout.log
     /workspace/stderr.log /workspace/output.json``
2. Write ``patch.diff``, ``run_script.sh``, ``parser.py``,
   ``entryscript.sh`` into the sandbox under ``/workspace/``.
3. Execute the entryscript. Read ``/workspace/output.json``.
4. ``resolved`` = ``{t.name for t in output.tests if status == PASSED} ⊇
   (fail_to_pass ∪ pass_to_pass)`` (subset semantic; matches upstream
   ``main()``'s accuracy computation).

Where the per-instance scripts come from
----------------------------------------
Resolution order:

1. **Vendored git submodule** at ``third_party/swebench_pro/`` (preferred).
   Initialise via ``git submodule update --init --recursive``. This is
   the path the SEAL scaffold (deviations.md S23) also reads from, so
   running ``-b swebench_pro_seal`` and ``-b swebench_pro`` shares one
   pinned upstream snapshot.
2. **Override env var** ``$CROSS_BENCH_PRO_SCRIPTS_DIR``. Honored when
   set, regardless of submodule state — useful for pre-staging a
   different snapshot in the orchestrator image.
3. **Runtime ``git clone``** to ``~/.cache/cross_bench/swebench_pro_scripts``
   (legacy path; kept as a fallback so existing deployments without the
   submodule don't break). On first call, clones ``main`` of the upstream
   repo; subsequent calls reuse the clone.

See deviations.md S21 for the documented Pro divergence vs Verified.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cross_bench.execution.sandbox import Sandbox, SandboxManager

#: Upstream Pro repo URL. Pin a commit via ``CROSS_BENCH_PRO_SCRIPTS_REF``
#: when reproducibility matters; default ``main`` tracks the latest set.
_PRO_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
_PRO_REPO_REF = os.environ.get("CROSS_BENCH_PRO_SCRIPTS_REF", "main")

#: Default cache root for the cloned scripts. Operators can override via
#: ``CROSS_BENCH_PRO_SCRIPTS_DIR`` to point at a pre-staged directory
#: (e.g. a Docker volume baked into the orchestrator image).
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "cross_bench" / "swebench_pro_scripts"

#: Vendored submodule path (relative to repo root). Preferred over the
#: runtime-clone fallback when present and non-empty.
_SUBMODULE_PATH = Path("third_party") / "swebench_pro"

_PRO_EVAL_TIMEOUT_DEFAULT = 1800  # mirrors upstream Modal sandbox timeout=60*60 ceiling
_PRO_WORKSPACE_DIR = "/workspace"


def _find_repo_root() -> Path | None:
    """Walk up looking for the ``.gitmodules`` marker.

    Returns ``None`` when CrossBench has been installed outside its
    source tree (e.g. ``pip install cross-bench``) — the caller then
    falls back to the runtime-clone path.

    When ``CROSS_BENCH_REPO_ROOT`` is set it is **authoritative**:
    we return either the override path (when it has a `.gitmodules` or
    the submodule dir) or ``None`` (so the caller falls through to the
    runtime-clone fallback). The override never falls through to the
    walk-up — that's what makes the env var a useful test hook.
    """
    env_override = os.environ.get("CROSS_BENCH_REPO_ROOT")
    if env_override:
        p = Path(env_override).resolve()
        if (p / ".gitmodules").exists() or (p / _SUBMODULE_PATH).exists():
            return p
        return None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".gitmodules").exists():
            return parent
    return None


def _submodule_run_scripts() -> Path | None:
    """Return the run_scripts/ dir under the submodule, if initialised."""
    root = _find_repo_root()
    if root is None:
        return None
    candidate = root / _SUBMODULE_PATH / "run_scripts"
    try:
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    except OSError:
        return None
    return None


def _scripts_root() -> Path:
    """Resolve the runtime-clone fallback root (env override or cache dir).

    Used only when neither the submodule nor an env override surfaces a
    usable scripts dir. See ``_load_instance_scripts`` for the full
    resolution order.
    """
    return Path(os.environ.get("CROSS_BENCH_PRO_SCRIPTS_DIR", str(_DEFAULT_CACHE_ROOT)))


_clone_lock = asyncio.Lock()


async def _ensure_pro_scripts_cloned() -> Path:
    """Clone (or reuse) the upstream Pro repo and return the ``run_scripts/`` root.

    Async lock prevents concurrent task evals from racing to clone the same
    repo. Read-only after clone, so subsequent callers short-circuit on the
    ``.git`` existence check.
    """
    root = _scripts_root()
    run_scripts = root / "run_scripts"
    if (root / ".git").exists() and run_scripts.exists():
        return run_scripts

    async with _clone_lock:
        # Re-check inside the lock — another coro may have cloned while we waited.
        if (root / ".git").exists() and run_scripts.exists():
            return run_scripts
        root.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        def _do_clone() -> None:
            if root.exists():
                # Partial clone from a prior crash — start fresh.
                import shutil
                shutil.rmtree(root, ignore_errors=True)
            subprocess.run(
                [
                    "git", "clone", "--depth=1",
                    "--branch", _PRO_REPO_REF,
                    _PRO_REPO_URL, str(root),
                ],
                check=True, capture_output=True, timeout=300,
            )

        await loop.run_in_executor(None, _do_clone)
        if not run_scripts.exists():
            raise FileNotFoundError(
                f"Pro repo cloned but {run_scripts} missing — upstream layout changed?"
            )
        return run_scripts


def _strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch.

    Byte-faithful copy of upstream ``strip_binary_hunks`` in
    ``swe_bench_pro_eval.py``. Pro graders can't apply binary diffs
    inside the sandbox; dropping them avoids ``git apply`` failure on
    an otherwise-valid patch.
    """
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _parse_selected_test_files(raw: Any) -> list[str]:
    """Parse the HF row's ``selected_test_files_to_run`` field.

    The field arrives as a stringified Python list (``"['t1', 't2']"``)
    from the HuggingFace dataset. Upstream uses ``eval(...)``; we use
    ``ast.literal_eval`` for safety. Empty/invalid → empty list.
    """
    import ast
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw or not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def _parse_test_id_set(raw: Any) -> set[str]:
    """Parse the HF row's ``fail_to_pass`` / ``pass_to_pass`` field into a set."""
    return set(_parse_selected_test_files(raw))


def _create_entryscript(metadata: dict[str, Any]) -> str:
    """Build the bash entryscript executed inside the per-instance image.

    Byte-faithful to upstream ``create_entryscript`` in
    ``swe_bench_pro_eval.py``, except we *omit* the ENV-export block:
    upstream parses the per-instance Dockerfile to re-export its ENVs,
    but those ENVs are already set inside the running container by the
    image itself. The export block is dead code in a non-Modal sandbox.
    """
    before_cmd_raw = str(metadata.get("before_repo_set_cmd", "")).strip()
    # Upstream takes the LAST line of the multi-line command — same here.
    before_cmd = before_cmd_raw.split("\n")[-1] if before_cmd_raw else ""
    test_files = ",".join(_parse_selected_test_files(metadata.get("selected_test_files_to_run", "")))
    base_commit = str(metadata.get("base_commit", ""))

    # ``set +e`` after the test command so a non-zero exit from
    # run_script.sh doesn't kill the parser run (the parser is what
    # we actually grade off).
    return (
        "#!/bin/bash\n"
        "set -e\n"
        "cd /app\n"
        f"git reset --hard {base_commit}\n"
        f"git checkout {base_commit}\n"
        "git apply -v /workspace/patch.diff\n"
        f"{before_cmd}\n"
        "set +e\n"
        f"bash /workspace/run_script.sh {test_files} > /workspace/stdout.log 2> /workspace/stderr.log\n"
        "RUN_RC=$?\n"
        "python3 /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json\n"
        "PARSER_RC=$?\n"
        "echo \"RUN_RC=$RUN_RC PARSER_RC=$PARSER_RC\"\n"
        "exit 0\n"
    )


async def _resolve_run_scripts_root() -> Path:
    """Pick the right run_scripts/ root, honoring the documented order.

    1. Vendored submodule under ``third_party/swebench_pro/run_scripts/``
       (preferred — matches what the SEAL scaffold reads).
    2. ``CROSS_BENCH_PRO_SCRIPTS_DIR`` env override, if set.
    3. Runtime ``git clone`` to the default cache root (legacy).
    """
    submodule_root = _submodule_run_scripts()
    if submodule_root is not None:
        return submodule_root
    if os.environ.get("CROSS_BENCH_PRO_SCRIPTS_DIR"):
        env_root = _scripts_root() / "run_scripts"
        if env_root.exists():
            return env_root
        # Env var pointed at an empty dir — fall through to clone the
        # default upstream into it so the next call short-circuits.
    return await _ensure_pro_scripts_cloned()


async def _load_instance_scripts(instance_id: str) -> tuple[str, str]:
    """Load ``run_script.sh`` and ``parser.py`` for one Pro instance.

    Returns ``(run_script, parser_script)``. Raises ``FileNotFoundError``
    if the instance isn't covered by the upstream ``run_scripts/`` tree.
    """
    run_scripts_root = await _resolve_run_scripts_root()
    inst_dir = run_scripts_root / instance_id
    run_path = inst_dir / "run_script.sh"
    parser_path = inst_dir / "parser.py"
    if not run_path.exists() or not parser_path.exists():
        raise FileNotFoundError(
            f"Pro scripts missing for {instance_id!r} under {run_scripts_root}. "
            "Either the instance isn't in the upstream run_scripts/ tree "
            "(check ScaleAI/SWE-bench_Pro coverage), the submodule is stale "
            "(`git submodule update --remote third_party/swebench_pro`), "
            "or the cache is stale (unset CROSS_BENCH_PRO_SCRIPTS_DIR)."
        )
    return run_path.read_text(), parser_path.read_text()


async def evaluate_pro_task(
    *,
    instance_id: str,
    metadata: dict[str, Any],
    patch: str,
    sandbox_mgr: "SandboxManager",
    sandbox: "Sandbox",
    timeout_seconds: int = _PRO_EVAL_TIMEOUT_DEFAULT,
) -> dict[str, Any]:
    """Run the Pro eval harness for one task inside the sandbox.

    Caller is responsible for creating the sandbox from the Pro-tagged
    image (``jefzda/sweap-images:<dockerhub_tag>``) and tearing it down.
    Workspace path inside the container is ``/workspace`` (matches
    upstream).

    Returns the same dict shape as Verified's ``evaluate`` —
    ``pass``, ``resolved``, ``patch_successfully_applied``, ``f2p_rate``,
    ``p2p_rate``, ``tests_status``, ``exit_code``, ``stdout``,
    ``stderr`` — so the engine's per-task JSON consumers don't care
    which variant produced the row.
    """
    result_stub: dict[str, Any] = {
        "pass": False,
        "resolved": False,
        "patch_successfully_applied": False,
    }

    if not patch or not patch.strip():
        result_stub["error"] = "empty_prediction"
        return result_stub

    f2p = _parse_test_id_set(metadata.get("fail_to_pass", ""))
    p2p = _parse_test_id_set(metadata.get("pass_to_pass", ""))
    if not (f2p | p2p):
        # Every published Pro row ships at least one fail_to_pass test,
        # so an empty union means this row's test metadata is missing or
        # garbled. Fail loudly BEFORE any run-scripts clone or sandbox
        # work: a silent ``resolved=False`` reads as model failure and a
        # vacuous subset check would read as success — neither is honest
        # about a row we cannot grade. See deviations.md S27.
        result_stub["error"] = "no_required_tests_parsed"
        return result_stub

    # Load per-instance test scripts (cached repo clone).
    try:
        run_script, parser_script = await _load_instance_scripts(instance_id)
    except FileNotFoundError as e:
        result_stub["error"] = f"scripts_missing: {e}"
        return result_stub

    cleaned_patch = _strip_binary_hunks(patch)
    entryscript = _create_entryscript(metadata)

    # Stage workspace files inside the sandbox.
    await sandbox_mgr.execute(sandbox, f"mkdir -p {_PRO_WORKSPACE_DIR}", timeout=30)
    await sandbox_mgr.write_file(sandbox, f"{_PRO_WORKSPACE_DIR}/patch.diff", cleaned_patch)
    await sandbox_mgr.write_file(sandbox, f"{_PRO_WORKSPACE_DIR}/run_script.sh", run_script)
    await sandbox_mgr.write_file(sandbox, f"{_PRO_WORKSPACE_DIR}/parser.py", parser_script)
    await sandbox_mgr.write_file(sandbox, f"{_PRO_WORKSPACE_DIR}/entryscript.sh", entryscript)

    # Execute the entryscript. ``entryscript.sh`` itself exits 0 even on
    # test-run failure — the truth is in /workspace/output.json — so a
    # non-zero return from the bash invocation here means git apply
    # failed or the parser itself crashed.
    exec_result = await sandbox_mgr.execute(
        sandbox,
        f"bash {_PRO_WORKSPACE_DIR}/entryscript.sh",
        timeout=timeout_seconds,
    )

    # Capture log files for diagnosis (best-effort).
    async def _read_sandbox_file(path: str) -> str:
        r = await sandbox_mgr.execute(
            sandbox,
            f"cat {path} 2>/dev/null || true",
            timeout=30,
        )
        return r.stdout or ""

    stdout_log = await _read_sandbox_file(f"{_PRO_WORKSPACE_DIR}/stdout.log")
    stderr_log = await _read_sandbox_file(f"{_PRO_WORKSPACE_DIR}/stderr.log")
    # If entryscript exited non-zero before reaching the parser, git apply
    # likely failed. The canonical signal is an empty output.json.
    output_json_text = await _read_sandbox_file(f"{_PRO_WORKSPACE_DIR}/output.json")

    if not output_json_text:
        # No output.json means the parser never ran. Most likely cause:
        # ``git apply`` failed inside the entryscript's ``set -e`` window.
        return {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": False,
            "error": "no_parser_output",
            "exit_code": exec_result.exit_code,
            "stdout": stdout_log or exec_result.stdout,
            "stderr": stderr_log or exec_result.stderr,
        }

    try:
        output = json.loads(output_json_text)
    except json.JSONDecodeError as e:
        return {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": True,
            "error": f"parser_output_invalid_json: {e}",
            "exit_code": exec_result.exit_code,
            "stdout": stdout_log,
            "stderr": stderr_log,
        }

    # Parser output schema:  {"tests": [{"name": str, "status": str}, ...]}
    tests = output.get("tests", []) if isinstance(output, dict) else []
    passed = {str(t.get("name", "")) for t in tests if str(t.get("status", "")).upper() == "PASSED"}

    required = f2p | p2p  # non-empty — guarded at the top of this function

    # Mirrors upstream ``main()``'s accuracy logic: ``resolved`` iff
    # every required test appears in ``passed``.
    resolved = required.issubset(passed)

    f2p_rate = (len(f2p & passed) / len(f2p)) if f2p else 1.0
    p2p_rate = (len(p2p & passed) / len(p2p)) if p2p else 1.0

    tests_status = {
        "FAIL_TO_PASS": {
            "success": sorted(f2p & passed),
            "failure": sorted(f2p - passed),
        },
        "PASS_TO_PASS": {
            "success": sorted(p2p & passed),
            "failure": sorted(p2p - passed),
        },
    }

    return {
        "pass": resolved,
        "resolved": resolved,
        "patch_successfully_applied": True,
        "f2p_rate": f2p_rate,
        "p2p_rate": p2p_rate,
        "tests_status": tests_status,
        "exit_code": exec_result.exit_code,
        "stdout": stdout_log,
        "stderr": stderr_log,
    }
