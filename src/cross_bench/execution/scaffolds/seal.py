"""SEAL (Scale AI's SWE-agent) scaffold for SWE-bench Pro.

Runs the canonical scaffold that produces Scale AI's public SWE-bench Pro
leaderboard numbers (https://labs.scale.com/leaderboard/swe_bench_pro_public).
The scaffold itself lives in two pinned git submodules:

* ``third_party/swe_agent`` — ``scaleapi/SWE-agent`` (branch
  ``scale-customizations``). The agent loop + LiteLLM model layer +
  SweRex sandbox/deployment layer + all prompts and tool definitions.
* ``third_party/swebench_pro`` — ``scaleapi/SWE-bench_Pro-os`` (branch
  ``main``). The per-instance ``run_scripts/`` and ``swe_bench_pro_eval.py``
  harness. CrossBench's :mod:`swebench_pro_eval` already ports the
  harness; this scaffold complements it by producing the patches that
  the harness grades.

Strategy: subprocess wrap, not Python port
------------------------------------------
Mini-SWE-agent (``mini_swe_agent.py``) is a ~150-LOC text-mode loop —
small enough to port directly so we can reuse CrossBench's ``LLMDriver``
and ``SandboxManager``. SWE-agent is the other end of that spectrum:
its own LiteLLM-based model layer, its own SweRex container manager
(with Modal / Docker / local / podman deployments), its own batch
runner and trajectory writer. Porting that is months of work and
introduces drift the moment Scale ships a new ``scale-customizations``
revision.

Instead, this scaffold treats SWE-agent as the opaque "produces patches"
tool that the upstream leaderboard already uses:

1. Initialise the submodules (``git submodule update --init --recursive``)
   and install the ``seal`` extra (``pip install -e .[seal]`` from the
   repo root). The extra is declared in ``pyproject.toml`` and resolves
   to the vendored submodule at ``third_party/swe_agent`` via a
   ``file:./`` URL — the canonical install path. Bake both into the
   orchestrator Dockerfile for AWS dispatches.
2. Build a single-instance file
   (``$WORKSPACE/<task_id>.json``) shaped per
   ``sweagent.run.batch_instances.SimpleBatchInstance`` —
   ``{instance_id, image_name, problem_statement, repo_name, base_commit}``.
3. Invoke ``python -m sweagent run-batch --config <pinned config>
   --instances.type=file --instances.path=<tempfile>
   --output_dir <tempdir> --agent.model.name <model>`` as a subprocess.
4. Harvest the patch from ``<output_dir>/<run_id>/<instance_id>/<instance_id>.pred``.
5. Hand the patch to ``adapter.evaluate(task, patch, sandbox_mgr,
   sandbox=None)`` — the Pro adapter routes through
   :mod:`swebench_pro_eval`, which creates a fresh sandbox from the
   Pro per-instance image and grades via the upstream
   ``run_scripts/<instance_id>/`` flow.

LLM credentials flow through environment variables (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``OPENAI_BASE_URL`` if set). The leaderboard
numbers come from SWE-agent using its own LiteLLM stack; we deliberately
do NOT proxy the LLM call through CrossBench's ``LLMDriver`` because
that would silently drift from the upstream column.

Sandbox creation is delegated to SWE-agent's own SweRex layer
(default ``DockerDeployment``). CrossBench's ``SandboxManager`` is only
involved in step 5 (grading) — the agent loop runs in SWE-agent's
own container, not ours.

Pinned upstream (verified at audit time):
    SWE-agent      = scaleapi/SWE-agent@scale-customizations
    Pro harness    = scaleapi/SWE-bench_Pro-os@main

See ``deviations.md`` § By benchmark → SWE-bench (entry S23).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from cross_bench.execution.scaffolds import (
    llm_trace_hook,
    mem0_litellm_hook,
    step_counter_hook,
)
from cross_bench.models import (
    RunConfig,
    StepRecord,
    Task,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from cross_bench.benchmarks.base import BenchmarkAdapter
    from cross_bench.execution.sandbox import SandboxManager
    from cross_bench.systems.base import LLMDriver, MemoryMiddleware, PromptStrategy

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Upstream pins (audit trail). See module docstring.
# ---------------------------------------------------------------------------

#: Repository URL recorded for diagnostics. The actual submodule URL lives
#: in ``.gitmodules`` and is what ``git submodule`` reads.
UPSTREAM_SWE_AGENT_REPO = "github.com/scaleapi/SWE-agent"
UPSTREAM_SWE_AGENT_BRANCH = "scale-customizations"
UPSTREAM_SWE_AGENT_COMMIT = "1ac158175178c9c40ed903fb59cc7aa3c87c75ae"

UPSTREAM_SWEBENCH_PRO_REPO = "github.com/scaleapi/SWE-bench_Pro-os"
UPSTREAM_SWEBENCH_PRO_BRANCH = "main"
UPSTREAM_SWEBENCH_PRO_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"


# ---------------------------------------------------------------------------
# Submodule discovery
# ---------------------------------------------------------------------------

#: Repository-relative submodule paths.
SWE_AGENT_SUBMODULE_PATH = Path("third_party") / "swe_agent"
SWEBENCH_PRO_SUBMODULE_PATH = Path("third_party") / "swebench_pro"

#: Default sweagent config under the submodule. ``config/tool_use.yaml`` is
#: the one Scale's leaderboard runs reference in ``sweagent_wrapper_configs/
#: example_config.yaml`` at the pinned SHA.
DEFAULT_SWE_AGENT_CONFIG = "config/tool_use.yaml"

#: Default LiteLLM model name. Matches Scale's reference Pro config
#: (anthropic/claude-3-7-sonnet-20250219). Can be overridden via the
#: ``model_name`` benchmark param.
DEFAULT_MODEL_NAME = "anthropic/claude-3-7-sonnet-20250219"

#: Per-instance cost limit (USD). 0 disables the cap (matches Scale's
#: example_config.yaml).
DEFAULT_PER_INSTANCE_COST_LIMIT = 0.0

#: Per-instance API call limit. Caps the agent's step count via the
#: model layer. Matches Scale's example_config.yaml (3 calls — that's
#: a very aggressive setting; the leaderboard uses higher).
DEFAULT_PER_INSTANCE_CALL_LIMIT = 0

#: Subprocess timeout for a single ``sweagent run-batch`` invocation
#: (one instance). Mirrors the upstream Modal sandbox timeout ceiling.
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 3600

#: Default number of *re*-attempts for a ``NoPrediction`` whose cause is a
#: TRANSIENT swerex infra failure (deviations.md S23 sub #9). 1 re-attempt
#: (= 2 total) recovers the bulk of the flaky tail (a residual leaked-port
#: container, an OOM, a Bedrock throttle window) at modest extra wall-clock.
#: Deterministic failures (``build_failed`` — e.g. the Alpine/musl base
#: images that can't run swerex's glibc standalone Python) are NEVER retried;
#: they'd fail identically and just burn ~4 s each. Override per-run via the
#: ``seal_max_retries`` benchmark param.
DEFAULT_SEAL_MAX_RETRIES = 1

#: ``swerex_infra_error`` tags that are worth retrying. These are the
#: transient host-level failures a fresh attempt (with the sub #8 orphan
#: sweep run again first) can clear. ``build_failed`` is deterministic and
#: deliberately absent; ``timeout`` is gated behind ``seal_retry_on_timeout``
#: (default off) because a retry doubles the worst-case wall-clock.
_RETRIABLE_INFRA_TAGS = frozenset({"port_in_use", "runtime_not_alive"})


def _resolve_prompt_caching_overlay() -> Path:
    """Path of the CrossBench-owned prompt-caching config overlay.

    ``configs/seal/cache_control.yaml`` enables SWE-agent's
    ``CacheControlHistoryProcessor`` (Anthropic / Bedrock prompt
    caching) when passed as a second ``--config`` after the pinned
    ``tool_use.yaml``. See deviations.md S26 for why this is an
    opt-in deviation from Scale's exact scaffold config.

    Resolved via ``cross_bench.config.CONFIGS_DIR`` (NOT
    ``_repo_root()``) so the same path logic that finds profile /
    benchmark TOMLs finds the overlay — local dev resolves relative
    to the source tree, Docker resolves via ``BENCH_CONFIGS_DIR``
    (``/app/configs``). Imported lazily to keep this module free of
    a top-level dependency on the config loader.
    """
    from cross_bench.config import CONFIGS_DIR

    return Path(CONFIGS_DIR) / "seal" / "cache_control.yaml"


def _resolve_str_replace_nudge_overlay() -> Path:
    """Path of the CrossBench-owned tool-selection-nudge config overlay.

    ``configs/seal/str_replace_nudge.yaml`` REPLACES Scale's one-line
    system template with an extended one that biases the model toward
    ``str_replace_editor str_replace`` for source edits + adds explicit
    "REMOVE the line, don't flip the value" guidance for attribute/
    default fixes. Opt-in via the ``seal_str_replace_nudge`` benchmark
    param. Cells produced with this overlay are NOT byte-comparable to
    Scale's exact config; every evaluation row carries
    ``str_replace_nudge_enabled`` so consumers can bucket.

    Same resolution semantics as :func:`_resolve_prompt_caching_overlay`.
    """
    from cross_bench.config import CONFIGS_DIR

    return Path(CONFIGS_DIR) / "seal" / "str_replace_nudge.yaml"


def _repo_root() -> Path:
    """Locate the CrossBench repository root.

    Resolution order:

    1. ``CROSS_BENCH_REPO_ROOT`` env var — **authoritative when set**.
       If the override points at a non-repo dir, we still return that
       path; the caller's ``_submodule_initialised`` check then surfaces
       the missing-submodules error against that path. This makes the
       env var a clean test hook (set it, get exactly that root).
    2. Walk up from this file looking for ``.gitmodules`` or
       ``third_party/swe_agent``.
    3. Final fallback — ``Path.cwd()`` so the caller sees the path it
       tried.
    """
    env_override = os.environ.get("CROSS_BENCH_REPO_ROOT")
    if env_override:
        return Path(env_override).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".gitmodules").exists():
            return parent
        if (parent / SWE_AGENT_SUBMODULE_PATH).exists():
            return parent
    # Final fallback — let the caller see the path it tried.
    return Path.cwd()


def _submodule_initialised(path: Path) -> bool:
    """Return True if the submodule directory exists and is non-empty.

    Empty directory means ``git submodule update --init`` was never run;
    that's an operator-facing setup error we surface clearly rather than
    failing deep in subprocess land.
    """
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _assert_submodules_ready() -> tuple[Path, Path]:
    """Verify both submodules are initialised; return their absolute paths."""
    root = _repo_root()
    swe_agent = (root / SWE_AGENT_SUBMODULE_PATH).resolve()
    pro_scripts = (root / SWEBENCH_PRO_SUBMODULE_PATH).resolve()
    missing: list[str] = []
    if not _submodule_initialised(swe_agent):
        missing.append(str(SWE_AGENT_SUBMODULE_PATH))
    if not _submodule_initialised(pro_scripts):
        missing.append(str(SWEBENCH_PRO_SUBMODULE_PATH))
    if missing:
        raise RuntimeError(
            "SEAL scaffold requires git submodules. Missing: "
            f"{', '.join(missing)}. Run "
            "`git submodule update --init --recursive` from the repo root."
        )
    return swe_agent, pro_scripts


# ---------------------------------------------------------------------------
# Pro task → SWE-agent SimpleBatchInstance shape
# ---------------------------------------------------------------------------


def _build_instance_record(task: Task, adapter: "BenchmarkAdapter") -> dict[str, Any]:
    """Construct one ``SimpleBatchInstance`` dict for SWE-agent.

    Pro images come pre-loaded with the repo at ``/app`` (per the
    upstream Pro Dockerfiles), so ``repo_name`` is set to ``"app"`` —
    SWE-agent's ``PreExistingRepoConfig`` then knows to ``cd /app``
    before issuing commands. ``base_commit`` is the upstream
    ``base_commit`` field from the HF row (so SWE-agent can reset
    the worktree on agent-loop start).

    For Verified / Lite / Full SWE-bench tasks (non-Pro), the SEAL
    scaffold falls back to ``repo_name="testbed"`` — same as upstream
    SWE-agent's ``SimpleBatchInstance.from_swe_bench`` helper. The SEAL
    scaffold itself is benchmark-agnostic; the Pro-specific contract is
    the image-name format and the ``/app`` repo location.
    """
    metadata = task.metadata or {}
    instance_id = metadata.get("instance_id") or task.task_id
    image_name = adapter.get_sandbox_image(task)
    problem_statement = adapter.format_prompt(task)
    base_commit = str(metadata.get("base_commit", "HEAD"))
    variant = metadata.get("variant", "")
    if variant == "pro":
        # Pro images mount the repo at /app per the Scale Dockerfiles.
        repo_name = "app"
    else:
        # SWE-bench (Verified/Lite/Full) images mount at /testbed.
        repo_name = "testbed"
    return {
        "instance_id": str(instance_id),
        "image_name": str(image_name),
        "problem_statement": str(problem_statement),
        "repo_name": repo_name,
        "base_commit": base_commit,
    }


def _write_instances_file(
    instance: dict[str, Any], workspace: Path
) -> Path:
    """Persist a single-instance JSON file SWE-agent can load.

    SWE-agent's ``InstancesFromFile`` accepts JSON or YAML. JSON keeps
    the test surface narrower (no YAML-quoting edge cases) and is what
    ``sweagent.run.common.load_file`` happily reads.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / f"{instance['instance_id']}.json"
    path.write_text(json.dumps([instance]), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


def _build_sweagent_command(
    *,
    instances_path: Path,
    output_dir: Path,
    config_path: str,
    model_name: str,
    per_instance_cost_limit: float,
    per_instance_call_limit: int,
    deployment_type: str = "docker",
    extra_args: list[str] | None = None,
    variant: str = "",
    prompt_caching_config: str = "",
    str_replace_nudge_config: str = "",
) -> list[str]:
    """Compose the argv for ``python -m sweagent run-batch ...``.

    Kept narrowly testable: deterministic argument order, no
    environment-dependent branches. The ``deployment_type`` defaults to
    ``docker`` so local runs and AWS ECS dispatches both work without
    Modal credentials.

    Pro-specific GLIBC fix
    ----------------------
    SWE-agent's ``swerex.DockerDeployment`` defaults
    ``python_standalone_dir = "/root"`` (set in
    ``sweagent/run/batch_instances.py::SimpleBatchInstance.to_full_batch_instance``).
    That triggers a per-task layered Docker build that COPYs an
    independently-built Python 3.11 binary into the instance's image
    and runs ``--version`` on it.

    The pre-built Python 3.11 binary requires **GLIBC 2.32+**; SWE-bench
    Pro images are based on Ubuntu Focal (GLIBC 2.31) and the layered
    build fails on step 12 with::

        /root/python3.11/bin/python3:
          /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found

    Symptom downstream: ``sweagent run-batch`` exits 0 (subprocess
    didn't crash) but writes **no .pred file**, so every task lands
    with ``exit_status = "NoPrediction"``. Observed empirically on the
    2026-05-26 ``seal-haiku45-bedrock-20260526-201622`` run:
    20/20 errored after ~17 s each.

    The fix per upstream's batch_instances.py:137 comment ("you can
    disable this by setting ``python_standalone_dir`` to empty string"):
    inject ``--instances.deployment.python_standalone_dir=`` (empty
    value after the equals sign) for Pro variants so swerex skips the
    standalone Python install and uses the Pro image's own Python 3
    (Focal ships Python 3.8, sufficient for swerex's per-step helpers
    at the pinned SHA).

    Verified-variant SWE-bench images don't carry the same GLIBC
    constraint (Verified base images are modern enough), so we only
    inject for ``variant == "pro"`` to avoid changing behaviour for
    leaderboard-comparable Verified cells.
    """
    cmd = [
        "python",
        "-m",
        "sweagent",
        "run-batch",
        "--config",
        config_path,
    ]
    # Optional prompt-caching overlay (deviations.md S26). MUST come
    # after the base config — sweagent merges ``--config`` files in
    # order (later wins, ``merge_nested_dicts`` recursive merge), so
    # the overlay ADDS ``agent.history_processors`` without touching
    # the pinned config's templates / tools. And BEFORE ``extra_args``
    # so ``seal_extra_args`` keeps final say over everything.
    if prompt_caching_config:
        cmd.extend(["--config", prompt_caching_config])
    # Tool-selection / mechanism-of-fix nudge overlay (deviations.md).
    # Stacked AFTER caching so its system_template override wins if
    # both are enabled (caching doesn't touch system_template).
    if str_replace_nudge_config:
        cmd.extend(["--config", str_replace_nudge_config])
    cmd += [
        "--instances.type=file",
        f"--instances.path={instances_path}",
        f"--instances.deployment.type={deployment_type}",
        # Leave python_standalone_dir at swerex's default (``/root``) so the
        # multi-stage build runs and produces a standalone Python 3.11 inside
        # the Pro task image. The orchestrator Dockerfile patches swerex's
        # ``glibc_dockerfile`` to use ``python:3.11-slim-bookworm`` (GLIBC 2.36)
        # instead of the now-Trixie ``python:3.11-slim`` (GLIBC 2.38), so the
        # produced binaries link cleanly against the Pro images' libc. Passing
        # an empty string here disables the build but lands swerex on a fragile
        # ``pip install pipx && pipx run swe-rex`` fallback inside the Pro
        # container — that path produces ``returncode=0`` with no ``.pred``
        # in practice. Don't reintroduce the empty-string override.
        "--output_dir",
        str(output_dir),
        "--num_workers",
        "1",
        f"--agent.model.name={model_name}",
        f"--agent.model.per_instance_cost_limit={per_instance_cost_limit}",
        f"--agent.model.per_instance_call_limit={per_instance_call_limit}",
    ]
    # NOTE: We deliberately do NOT pass ``--instances.deployment.python_standalone_dir=``
    # (empty value) or ``--instances.deployment.docker_args=[...]`` from this
    # scaffold any more. The full Pro fix lives entirely in the orchestrator
    # Dockerfile's "Patch swerex docker.py" RUN step, which sed-rewrites the
    # installed swerex ``docker.py`` to (1) pin the GLIBC builder base to
    # bullseye, (2) treat empty ``python_standalone_dir`` as falsy, (3) force
    # the public PyPI mirror on swerex's stage-2 pip install, (4) clear the
    # Pro images' inherited ``/bin/bash`` ENTRYPOINT in ``docker run``, and
    # (5) bundle libssl/libcrypto into the standalone Python lib dir. See
    # deviations.md S23 for the audit trail. Earlier scaffold-side workarounds
    # (round 1 empty-standalone-dir, round 3 entrypoint flag) were obsoleted
    # by the Dockerfile patch and removed during the swe-seal→main merge.
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _build_subprocess_env(
    swe_agent_root: Path,
    *,
    extra_pythonpath: list[Path] | None = None,
) -> dict[str, str]:
    """Compose the env dict for the sweagent subprocess.

    Inherits the parent process env (so ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` / ``OPENAI_BASE_URL`` flow through), then
    prepends the submodule path to ``PYTHONPATH`` so ``python -m sweagent``
    resolves to the vendored copy even if a different ``sweagent`` is
    installed globally.

    ``extra_pythonpath`` (e.g. the per-task sitecustomize tempdir for the
    LiteLLM hooks) is prepended **before** the submodule so Python's
    site-init picks up our ``sitecustomize.py`` first.
    """
    env = dict(os.environ)
    sep = ";" if os.name == "nt" else ":"
    path_parts: list[str] = []
    if extra_pythonpath:
        path_parts.extend(str(p) for p in extra_pythonpath)
    path_parts.append(str(swe_agent_root))
    existing_pp = env.get("PYTHONPATH", "")
    if existing_pp:
        path_parts.append(existing_pp)
    env["PYTHONPATH"] = sep.join(path_parts)
    return env


def _resolve_mem0_hook_config(
    memory: "MemoryMiddleware | None",
    *,
    benchmark_params: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Return the mem0 LiteLLM-hook env contract when memory is a mem0v2 middleware.

    Detection is duck-typed on the mem0-specific attrs ``_user_id`` AND
    ``_memory`` (the in-process ``mem0.Memory`` instance) AND
    ``_retrieval_limit``. Returns ``None`` for any other memory backend.

    Bench-LLM auto-follow flows through automatically: the mem0
    middleware's ``_config.params`` carries ``llm_provider`` /
    ``llm_model`` / ``llm_base_url`` (set by
    ``_inherit_bench_llm_into_memory`` in ``config.py``) — we forward
    those values into the subprocess env so the in-subprocess mem0
    instance uses the same internal LLM as the parent-side one.

    Forwards the ``memory_first_step_only`` / ``memory_retrieve_only``
    benchmark mixins as the corresponding mem0 hook env vars.
    """
    if memory is None:
        return None
    # Duck-type: mem0v2 middleware exposes ``_memory`` (mem0.Memory),
    # ``_user_id``, ``_retrieval_limit``.
    if not hasattr(memory, "_memory") or not hasattr(memory, "_user_id"):
        return None
    user_id = str(getattr(memory, "_user_id", "")).strip()
    if not user_id:
        return None
    env: dict[str, str] = {
        mem0_litellm_hook.ENV_USER_ID: user_id,
        mem0_litellm_hook.ENV_TOP_K: str(
            getattr(memory, "_retrieval_limit", mem0_litellm_hook.DEFAULT_TOP_K)
        ),
        mem0_litellm_hook.ENV_INFER: "1" if getattr(memory, "_infer", True) else "0",
    }
    # Forward auto-followed bench LLM into mem0's internal config so the
    # in-subprocess mem0.Memory uses the same extractor model as the
    # parent. Values live on memory._config.params (set by
    # _inherit_bench_llm_into_memory).
    params = getattr(getattr(memory, "_config", None), "params", None) or {}
    for src_key, env_key in [
        ("llm_provider", mem0_litellm_hook.ENV_LLM_PROVIDER),
        ("llm_model", mem0_litellm_hook.ENV_LLM_MODEL),
        ("llm_base_url", mem0_litellm_hook.ENV_LLM_BASE_URL),
        ("embedder_provider", mem0_litellm_hook.ENV_EMB_PROVIDER),
        ("embedder_model", mem0_litellm_hook.ENV_EMB_MODEL),
    ]:
        val = params.get(src_key)
        if val:
            env[env_key] = str(val)
    # Forward the schedule mixins (memory_first_step_only /
    # memory_retrieve_only) — these come from the benchmark TOML, so
    # the caller passes them explicitly.
    bp = benchmark_params or {}
    if bool(bp.get("memory_first_step_only", False)):
        env[mem0_litellm_hook.ENV_FIRST_STEP_ONLY] = "1"
        env[mem0_litellm_hook.ENV_DISABLE_STORE] = "1"
    elif bool(bp.get("memory_retrieve_only", False)):
        env[mem0_litellm_hook.ENV_DISABLE_STORE] = "1"
    return env


def _cleanup_orphan_swerex_containers() -> dict[str, Any]:
    """Remove leaked swerex sibling containers from prior SEAL tasks.

    **The bug this exists to defend against (deviations.md S23 sub #8).**
    When a SEAL subprocess is killed by our outer
    ``seal_subprocess_timeout_seconds``, Python's process-tree termination
    DOES NOT cascade to docker containers spawned by sweagent via the
    mounted ``docker.sock``. The swerex sibling container stays alive on
    the EC2 host's docker daemon, and (because of sub #6's
    ``--network=container:<orch>`` mode) holds port 8000 inside the
    orchestrator's network namespace. The NEXT SEAL task then can't bind
    ``0.0.0.0:8000`` in that namespace and fails with
    ``[Errno 98] address already in use``.

    Empirical evidence: the ``seal-haiku45-bedrock-port-drop-20260527-153500``
    run hit this exactly — task 9 timed out leaving its swerex container
    alive; tasks 10-20 all hit ``Errno 98`` at swerex-remote startup, then
    sat through the 60-min outer timeout. That's 12 wasted tasks (~12
    EC2-task-hours) attributable to a single leaked container.

    Mitigation: before every SEAL invocation, sweep any container whose
    name starts with ``jefzdasweap-images`` (swerex's sanitized name
    prefix for SWE-bench Pro containers — see
    ``DockerDeployment._get_container_name``). Best-effort; any failure
    here is logged and swallowed.

    Caveats:

    - This scaffold is single-process per orchestrator container today.
      Adding ``--parallel`` SEAL would need to scope cleanup per task
      (e.g. by task-id label on the docker run, set via the BatchInstance
      ``docker_args``) so concurrent tasks don't kill each other.
    - The cleanup only catches Pro images; Verified SEAL (not wired today)
      would need its own filter.

    Returns a status dict for logging / tests:
    ``{"swept": int, "ids": list[str], "rm_returncode": int | None,
       "error": str | None}``.
    """
    status: dict[str, Any] = {"swept": 0, "ids": [], "rm_returncode": None, "error": None}
    try:
        ps = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=jefzdasweap-images"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if ps.returncode != 0:
            status["error"] = f"docker ps rc={ps.returncode}: {(ps.stderr or '').strip()[:200]}"
            return status
        ids = [i.strip() for i in (ps.stdout or "").splitlines() if i.strip()]
        if not ids:
            return status
        rm = subprocess.run(
            ["docker", "rm", "-f", *ids],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        status["swept"] = len(ids)
        status["ids"] = ids
        status["rm_returncode"] = rm.returncode
        logger.info(
            "seal.cleanup.orphan_swerex_containers",
            count=len(ids),
            rm_returncode=rm.returncode,
            sample_ids=ids[:5],
        )
    except FileNotFoundError:
        status["error"] = "docker CLI not on PATH (local test env without docker)"
    except subprocess.TimeoutExpired:
        status["error"] = "docker ps/rm timed out"
        logger.warning("seal.cleanup.orphan_swerex_timed_out")
    except Exception as e:  # noqa: BLE001 — defensive; never fatal
        status["error"] = f"{type(e).__name__}: {e}"
        logger.warning("seal.cleanup_orphan_swerex_failed", error=str(e))
    return status


def _rmi_pro_image(image: str) -> dict[str, Any]:
    """Remove a SWE-bench Pro Docker image post-task to reclaim disk.

    SWE-bench Pro images (``jefzda/sweap-images:<tag>``) are 3-5 GB each
    and unique per instance. ``SandboxManager.prefetch_images``
    pre-pulls all shard images before the task loop starts, so a 30-task
    Pro shard accumulates ~120 GB on the host's EBS volume. On a
    c5.xlarge with the default root volume size, two of 25 shards on a
    731-instance fan-out tipped past disk capacity (``no space left on
    device``).

    This helper is invoked from ``execute_seal`` after ``adapter.evaluate``
    returns for Pro tasks. It removes the just-finished instance's
    image so subsequent tasks on the same shard see freed disk. The
    next task's image was already pre-pulled by
    ``SandboxManager.prefetch_images``, so no extra re-pull cost.

    Best-effort: docker CLI failures, missing image, or container-
    still-using-image errors are logged and swallowed. The task itself
    has already produced its evaluation; this is pure disk hygiene.

    Caveats:

    - Gated on Pro variant only -- Verified/Lite images can be shared
      across instances (``sweb.eval.x86_64.<id>``) and removing one mid-
      run is more likely to cost a re-pull on a subsequent task in the
      same shard.
    - Uses ``-f`` to force-remove even when a stopped container still
      references the image (swerex containers cleaned up by
      ``_cleanup_orphan_swerex_containers`` may not have fully detached
      yet on tight loops).

    Returns ``{"image": str, "rmi_returncode": int | None, "error":
    str | None}`` for logging / tests.
    """
    status: dict[str, Any] = {"image": image, "rmi_returncode": None, "error": None}
    if not image:
        status["error"] = "empty image name"
        return status
    try:
        rmi = subprocess.run(
            ["docker", "rmi", "-f", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        status["rmi_returncode"] = rmi.returncode
        if rmi.returncode == 0:
            logger.info("seal.cleanup.pro_image_removed", image=image)
        else:
            # rc=1 usually means "image not found" -- already gone, fine.
            stderr_tail = (rmi.stderr or "").strip()[-200:]
            logger.info(
                "seal.cleanup.pro_image_rmi_nonzero",
                image=image,
                rc=rmi.returncode,
                stderr=stderr_tail,
            )
            status["error"] = stderr_tail
    except FileNotFoundError:
        status["error"] = "docker CLI not on PATH"
    except subprocess.TimeoutExpired:
        status["error"] = "docker rmi timed out"
        logger.warning("seal.cleanup.pro_image_rmi_timed_out", image=image)
    except Exception as e:  # noqa: BLE001 - defensive; never fatal
        status["error"] = f"{type(e).__name__}: {e}"
        logger.warning("seal.cleanup.pro_image_rmi_failed", image=image, error=str(e))
    return status


def _format_stdout_for_step(
    text: str, *, head: int = 16000, tail: int = 16000
) -> str:
    """Truncate a long sweagent stdout for storage as ``Step.output_text``.

    A 60-min agent loop can produce >100KB of stdout. Storing all of it
    in the per-task JSON bloats S3 / Postgres rows. Keeping only the
    head (the original ``[:16000]`` slice) hid the *most* useful part:
    failure modes typically surface near the END of stdout (errors, last
    LLM response, swerex stop() output). The
    ``seal-haiku45-bedrock-port-drop-20260527-153500`` post-mortem
    couldn't diagnose task 9 because the relevant tail was discarded.

    Retain both head and tail with an explicit sentinel in the middle so
    operators inspecting the per-task JSON see how much was dropped.
    Defaults give 32KB total (head + tail) — bounded but useful.
    """
    if not text:
        return ""
    if len(text) <= head + tail:
        return text
    dropped = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n[... {dropped} chars truncated by SEAL scaffold ...]\n\n"
        + text[-tail:]
    )


#: Lines whose stripped form starts with any of these prefixes get
#: forwarded live to the parent's stderr while the sweagent subprocess
#: runs. Everything else is still captured into the returned buffers
#: (so post-task aggregation and trajectory inspection work) but does
#: NOT flood the operator's terminal / CloudWatch with sweagent's
#: per-step chatter, Docker build progress, swerex DEBUG, etc. — that
#: noise is exactly what 4412f45 reverted.
_LIVE_TEE_PREFIXES: tuple[str, ...] = (
    "[step-counter]",
    "[llm-trace]",
)


def _should_live_forward(line: str) -> bool:
    """Return True iff ``line`` should be tee'd live to the parent stderr."""
    s = line.lstrip()
    return any(s.startswith(p) for p in _LIVE_TEE_PREFIXES)


async def _invoke_sweagent(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, str, str]:
    """Run ``sweagent run-batch`` and return ``(returncode, stdout, stderr)``.

    Wrapped in a thread executor so the engine's asyncio loop isn't
    blocked. Test code stubs this function directly rather than
    monkey-patching ``subprocess.run`` so the boundary stays clean.

    Stdout / stderr are streamed line-by-line: lines matching
    ``_LIVE_TEE_PREFIXES`` are forwarded to the parent's stderr in real
    time (so operators see the ``[step-counter]`` / ``[llm-trace]`` lines
    during the run, not after). All lines are still captured and returned in
    full for post-task aggregation.
    """
    import sys as _sys
    import threading

    loop = asyncio.get_event_loop()

    def _run() -> tuple[int, str, str]:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            return (
                127,
                "",
                f"[seal] sweagent invocation failed: {e}. "
                "Is `sweagent` installed in this Python env? "
                "Run `pip install -e .[seal]` from the repo root.",
            )

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []

        def _drain(pipe: Any, buf: list[str]) -> None:
            for line in iter(pipe.readline, ""):
                buf.append(line)
                if _should_live_forward(line):
                    print(line, end="", file=_sys.stderr, flush=True)
            pipe.close()

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            return (
                124,
                "".join(stdout_buf),
                "".join(stderr_buf) + f"\n[seal] sweagent timed out after {timeout_seconds}s",
            )

        t_out.join(timeout=5)
        t_err.join(timeout=5)
        return proc.returncode, "".join(stdout_buf), "".join(stderr_buf)

    return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# Patch harvesting
# ---------------------------------------------------------------------------


def _find_pred_file(output_dir: Path, instance_id: str) -> Path | None:
    """Locate the ``.pred`` file SWE-agent dropped for one instance.

    SWE-agent writes to ``<output_dir>/<run_id>/<instance_id>/<instance_id>.pred``
    where ``<run_id>`` is autogenerated. Glob across the run directory.
    """
    if not output_dir.exists():
        return None
    # Pattern: <run_id>/<instance_id>/<instance_id>.pred
    for candidate in output_dir.glob(f"*/{instance_id}/{instance_id}.pred"):
        if candidate.is_file():
            return candidate
    # Fallback: any .pred under any subdir matching the instance_id.
    for candidate in output_dir.rglob(f"{instance_id}.pred"):
        if candidate.is_file():
            return candidate
    return None


def _extract_traj_summary(output_dir: Path, instance_id: str) -> dict[str, float | int]:
    """Read sweagent's ``.traj`` and extract per-task model_stats + step count.

    SWE-agent writes ``<output_dir>/<run_id>/<instance_id>/<instance_id>.traj``
    as a single pretty-printed JSON document via
    ``DefaultAgent.save_trajectory`` (``json.dumps(data, indent=2)`` in
    ``third_party/swe_agent/sweagent/agent/agents.py``). Shape:

        {
          "trajectory": [{"role": "assistant", ...}, ...],
          "info": {"model_stats": {"tokens_sent": int,
                                    "tokens_received": int,
                                    "api_calls": int,
                                    "instance_cost": float}, ...}
        }

    Returns ``{}`` when the file is missing / unparseable so callers can
    blindly ``evaluation.setdefault`` over the result without guards.

    Returned keys (any subset, based on what's actually present):

    * ``steps_taken`` — count of ``role == "assistant"`` events in
      ``trajectory``. The agent-loop step count.
    * ``total_tokens_in`` — agent LiteLLM tokens sent. Maps to
      ``tokens_sent`` upstream.
    * ``total_tokens_out`` — agent LiteLLM tokens received.
    * ``api_calls`` — number of LiteLLM completions during the task.
    * ``instance_cost_usd`` — upstream's ``instance_cost`` (USD).

    Note on tokens: these are the SEAL agent's bench-LLM tokens.
    """
    if not output_dir.exists():
        return {}
    traj_path: Path | None = None
    for candidate in output_dir.glob(f"*/{instance_id}/{instance_id}.traj"):
        if candidate.is_file():
            traj_path = candidate
            break
    if traj_path is None:
        for candidate in output_dir.rglob(f"{instance_id}.traj"):
            if candidate.is_file():
                traj_path = candidate
                break
    if traj_path is None:
        return {}
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return {}

    out: dict[str, float | int] = {}
    traj = data.get("trajectory") if isinstance(data, dict) else None
    if isinstance(traj, list):
        steps = sum(
            1 for ev in traj
            if isinstance(ev, dict) and ev.get("role") == "assistant"
        )
        if steps > 0:
            out["steps_taken"] = steps

    info = data.get("info") if isinstance(data, dict) else None
    stats = info.get("model_stats") if isinstance(info, dict) else None
    if isinstance(stats, dict):
        ts = stats.get("tokens_sent")
        tr = stats.get("tokens_received")
        ac = stats.get("api_calls")
        ic = stats.get("instance_cost")
        if isinstance(ts, (int, float)):
            out["total_tokens_in"] = int(ts)
        if isinstance(tr, (int, float)):
            out["total_tokens_out"] = int(tr)
        if isinstance(ac, (int, float)):
            out["api_calls"] = int(ac)
        if isinstance(ic, (int, float)):
            out["instance_cost_usd"] = float(ic)
    return out


def _persist_sidecars_to_s3(
    *,
    output_dir: Path,
    instance_id: str,
    task_id: str,
    run_id: str,
    benchmark: str,
    llm_trace_root: Path,
) -> None:
    """Upload sweagent's ``.traj`` and the LiteLLM I/O JSONL to S3.

    Both files persist diagnostic information that the standard
    ``ArtifactStore.store_task_result`` doesn't carry (too large to
    inline on the evaluation dict): the full agent-tool-user dialog
    sweagent writes via ``DefaultAgent.save_trajectory`` and the
    per-LLM-call request/response bodies captured by the optional
    ``llm_trace_enabled`` flag.

    **Layout.** Both upload to
    ``s3://<BENCH_ARTIFACTS_BUCKET>/runs/<benchmark>/<run_id>/sidecars/<task_id>/{trajectory.traj,llm_trace.jsonl}``
    so they sit next to the standard ``tasks/<task_id>.json`` row for
    easy cross-reference.

    **Triggers.**
    - ``BENCH_ARTIFACTS_BACKEND=s3`` AND ``BENCH_ARTIFACTS_BUCKET=<bucket>``
      env vars set (matches the orchestrator task def's env).
    - ``.traj`` upload: always attempted when the file exists in
      ``output_dir`` (sweagent writes one per task by default).
    - ``llm_trace.jsonl`` upload: only when ``llm_trace_root`` contains
      a file for this task (operator enabled ``llm_trace_enabled``).

    **Failure semantics.** Best-effort, swallow-all. A failed upload
    logs a single ``seal.sidecar_upload_failed`` warning and lets the
    task's evaluation row go through normally. The diagnostic data is
    nice-to-have, not load-bearing for grading.
    """
    if os.environ.get("BENCH_ARTIFACTS_BACKEND") != "s3":
        return
    bucket = os.environ.get("BENCH_ARTIFACTS_BUCKET")
    if not bucket:
        return

    s3_prefix = f"runs/{benchmark}/{run_id}/sidecars/{task_id}"
    try:
        import boto3
        s3 = boto3.client("s3")
    except Exception as e:
        logger.warning("seal.sidecar_upload_failed", reason="boto3_init", error=str(e))
        return

    # 1. sweagent .traj — full {trajectory, history, info, replay_config, environment}.
    #    Path: <output_dir>/*/<instance_id>/<instance_id>.traj
    try:
        traj_paths = list(output_dir.glob(f"*/{instance_id}/{instance_id}.traj"))
        if not traj_paths:
            traj_paths = list(output_dir.rglob(f"{instance_id}.traj"))
        if traj_paths:
            traj_key = f"{s3_prefix}/trajectory.traj"
            s3.upload_file(str(traj_paths[0]), bucket, traj_key)
            logger.info("seal.sidecar_uploaded", kind="traj",
                        bucket=bucket, key=traj_key, task_id=task_id)
        else:
            logger.debug("seal.sidecar_traj_missing",
                         output_dir=str(output_dir), instance_id=instance_id)
    except Exception as e:
        logger.warning("seal.sidecar_upload_failed", kind="traj",
                       error=str(e), task_id=task_id)

    # 2. llm_trace.jsonl — one JSON line per LiteLLM completion call
    #    (request + response bodies). Written when ``llm_trace_enabled``
    #    was set in the benchmark TOML. Path is /tmp persistent root.
    try:
        trace_file = llm_trace_root / f"{task_id}.jsonl"
        if trace_file.exists():
            trace_key = f"{s3_prefix}/llm_trace.jsonl"
            s3.upload_file(str(trace_file), bucket, trace_key)
            logger.info("seal.sidecar_uploaded", kind="llm_trace",
                        bucket=bucket, key=trace_key, task_id=task_id,
                        size=trace_file.stat().st_size)
    except Exception as e:
        logger.warning("seal.sidecar_upload_failed", kind="llm_trace",
                       error=str(e), task_id=task_id)


def _aggregate_mem0_metrics(metrics_path: Path) -> dict[str, float | int]:
    """Read the per-call JSONL produced by :mod:`mem0_litellm_hook` and
    return a flat dict of totals + counts to stamp on ``evaluation``.

    Returns the empty dict when the file is missing / unreadable so the
    caller can blindly ``evaluation.update(...)`` without a guard.

    Aggregated fields land on ``evaluation``:

    * ``mem0_recall_calls`` — number of ``Memory.search()`` calls
    * ``mem0_recall_tokens`` — sum of (rendered injection chars)/4
    * ``mem0_recall_latency_ms`` — sum of per-call wall-clock
    * ``mem0_recall_memories`` — sum of ``len(results)`` returned
    * ``mem0_store_calls`` — number of ``Memory.add()`` calls
    * ``mem0_store_tokens_estimate`` — sum of (prompt+response chars)/4

    The engine's ``_aggregate`` step averages numeric ``evaluation``
    fields across tasks, so summary.json picks up ``avg_mem0_*`` for
    every key above without further plumbing.
    """
    if not metrics_path.exists():
        return {}
    totals: dict[str, float | int] = {
        "mem0_recall_calls": 0,
        "mem0_recall_tokens": 0,
        "mem0_recall_latency_ms": 0.0,
        "mem0_recall_memories": 0,
        "mem0_store_calls": 0,
        "mem0_store_tokens_estimate": 0,
    }
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            if event == "recall":
                totals["mem0_recall_calls"] = int(totals["mem0_recall_calls"]) + 1
                totals["mem0_recall_tokens"] = int(totals["mem0_recall_tokens"]) + int(
                    row.get("tokens") or 0
                )
                totals["mem0_recall_latency_ms"] = float(
                    totals["mem0_recall_latency_ms"]
                ) + float(row.get("latency_ms") or 0.0)
                totals["mem0_recall_memories"] = int(
                    totals["mem0_recall_memories"]
                ) + int(row.get("memories") or 0)
                if row.get("call_index") == 0 and row.get("injection_text"):
                    totals["mem0_recall_first_injection"] = row["injection_text"]
            elif event == "store":
                totals["mem0_store_calls"] = int(totals["mem0_store_calls"]) + 1
                totals["mem0_store_tokens_estimate"] = int(
                    totals["mem0_store_tokens_estimate"]
                ) + int(row.get("tokens_estimate") or 0)
    except OSError as e:
        logger.warning(
            "seal.mem0_metrics_read_failed", error=str(e), path=str(metrics_path)
        )
        return {}
    return totals


def _extract_patch_from_pred(pred_path: Path) -> str:
    """Read ``.pred`` and pull out the ``model_patch`` field.

    ``save_predictions`` in SWE-agent writes
    ``{"model_name_or_path": ..., "instance_id": ..., "model_patch": ...}``
    as JSON. ``model_patch`` can be ``None`` if the agent never
    submitted (cost-cap, call-cap, format error, etc.); treat that as
    an empty patch — the eval harness will surface it as
    ``patch_apply_failed``.
    """
    if not pred_path.exists():
        return ""
    raw = pred_path.read_text(encoding="utf-8")
    if not raw.strip():
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Plain-text .pred (some older sweagent paths write raw patches).
        return raw
    if isinstance(payload, dict):
        patch = payload.get("model_patch") or payload.get("patch") or ""
        return str(patch) if patch else ""
    return ""


# ---------------------------------------------------------------------------
# Swerex infra-failure classification (deviations.md S23 sub #9)
# ---------------------------------------------------------------------------


def _classify_swerex_failure(
    stdout: str, stderr: str, returncode: int
) -> str:
    """Classify a ``NoPrediction`` SEAL outcome by its swerex infra cause.

    The SEAL subprocess exits cleanly (returncode 0) and writes no ``.pred``
    when swerex can't stand up the per-task runtime — sweagent catches the
    swerex exception, logs it, and reports "Found 0 predictions". The
    cause is only visible in the captured stdout/stderr; this classifier
    turns it into a stable tag so (a) the retry loop knows whether a fresh
    attempt could help, and (b) the leaderboard can bucket infra losses
    out of the model's ``resolved`` denominator.

    Tags (returned in priority order; the four signatures don't co-occur
    in practice — a build failure never reaches the container-run stage,
    a port collision never reaches ``_build_image``):

    * ``"build_failed"`` — swerex's ``_build_image`` step
      (``docker build --build-arg BASE_IMAGE=<pro image> -``) returned
      non-zero. **Deterministic** — overwhelmingly the Alpine/musl Pro base
      images (``golang:*-alpine``: teleport / vuls / flipt) where the glibc
      standalone Python can't exec (``apt-get`` absent → exit 127). Not
      retriable; see the deferred musl-build fix in deviations.md S23.
    * ``"port_in_use"`` — swerex-remote couldn't bind ``0.0.0.0:8000``
      (``[Errno 98] address already in use``). **Transient** — a prior
      task's swerex sibling container leaked and still holds the port in
      the orchestrator's shared network namespace (sub #6/#8). A retry
      after the orphan sweep clears it.
    * ``"runtime_not_alive"`` — swerex started the container but the
      remote runtime never answered (``Container process terminated`` /
      ``Runtime did not start within timeout``) for a reason other than a
      port clash. **Transient-ish** — worth one retry.
    * ``"timeout"`` — our outer subprocess timeout fired (returncode 124),
      or sweagent reported ``timed out after``. Retried only when
      ``seal_retry_on_timeout`` is set (a retry doubles wall-clock).
    * ``""`` — no recognised swerex infra signature (e.g. the agent ran
      but produced no patch for a model-side reason, or an unrecognised
      crash). Not auto-retried.

    Pure function over the FULL (untruncated) stdout/stderr — call it with
    the buffers returned by :func:`_invoke_sweagent`, before
    :func:`_format_stdout_for_step` trims them for storage.
    """
    text = f"{stdout or ''}\n{stderr or ''}"
    # 1. Build-image failure (deterministic). The traceback frame
    #    ``in _build_image`` and the ``docker build`` CalledProcessError
    #    are both unambiguous; match either so a swerex refactor that
    #    keeps one marker still classifies.
    if "in _build_image" in text or "Command '['docker', 'build'" in text:
        return "build_failed"
    # 2. Port collision (transient — leaked sibling container).
    if "address already in use" in text or "Errno 98" in text:
        return "port_in_use"
    # 3. Runtime never came alive for some other reason (transient-ish).
    if (
        "Container process terminated" in text
        or "Runtime did not start within timeout" in text
    ):
        return "runtime_not_alive"
    # 4. Our outer timeout, or sweagent's own.
    if returncode == 124 or "timed out after" in text:
        return "timeout"
    return ""


# ---------------------------------------------------------------------------
# Execution mode entry point
# ---------------------------------------------------------------------------


async def execute_seal(
    task: Task,
    config: RunConfig,
    adapter: "BenchmarkAdapter",
    llm: "LLMDriver",
    strategy: "PromptStrategy",
    memory: "MemoryMiddleware | None",
    sandbox_mgr: "SandboxManager",
) -> TaskResult:
    """Run Scale's SEAL scaffold (SWE-agent at ``scale-customizations``)
    for one task, then grade the resulting patch via ``adapter.evaluate``.

    Notes on the parameter shape:

    * ``llm`` and ``strategy`` are **unused** — SWE-agent owns the model
      call. They're kept in the signature so the engine's dispatch
      table (one signature for every scaffold) doesn't need a special
      case. The argument-name silence is intentional; reviewers should
      not "fix" it by deleting the parameters.
    * ``memory`` is consulted **only** for the optional mem0 LiteLLM
      hook path. When a mem0v2 embedded middleware is stacked
      (duck-typed via ``_memory`` + ``_user_id`` attrs), we (a) call
      ``reset()`` for per-task isolation, (b) write a
      ``sitecustomize.py`` stub into the per-task tempdir that
      registers the mem0 hook inside the sweagent subprocess, (c)
      prepend that tempdir to ``PYTHONPATH`` and export the mem0 hook
      env vars to the subprocess. The hook fires before every
      ``litellm.completion`` call in SWE-agent's agent loop. Other
      memory middlewares are ignored — SWE-agent's loop is opaque to
      CrossBench's :class:`MemoryMiddleware` contract for any other
      backend, and memory-stack experimentation on other backends
      belongs on mini-swe-agent (the ``swebench_pro`` benchmark) per
      ``deviations.md`` S23. SEAL cells with mem0 injection are NOT
      comparable to Scale's public Pro leaderboard (which has no
      memory) — they are an internal CrossBench ablation column.
    """
    del llm, strategy  # unused — see docstring.

    start_time = time.time()
    steps: list[StepRecord] = []
    exit_status: str = ""
    error_text: str = ""
    submission: str = ""
    sweagent_returncode: int = -1
    sweagent_stdout: str = ""
    sweagent_stderr: str = ""
    # Time spent inside the sweagent subprocess. Initialised here so the
    # submodule-missing / no-invocation paths also emit 0 — keeps the
    # ``inference_latency_ms`` column stable across all SEAL rows
    # regardless of which branch produced them (EE sec.3).
    inference_seconds: float = 0.0

    try:
        swe_agent_root, _pro_root = _assert_submodules_ready()
    except RuntimeError as e:
        return TaskResult(
            task_id=task.task_id,
            run_id=config.run_id,
            status=TaskStatus.ERROR,
            steps=[],
            final_output="",
            evaluation={
                "pass": False,
                "resolved": False,
                "patch_successfully_applied": False,
                "scaffold": "seal",
                "scaffold_swe_agent_commit": UPSTREAM_SWE_AGENT_COMMIT,
                "scaffold_pro_commit": UPSTREAM_SWEBENCH_PRO_COMMIT,
                "error": f"submodules_missing: {e}",
                # Latency split — all zero on the no-invocation path
                # so the column shape is stable across SEAL rows
                # regardless of whether sweagent ran. EE sec.3.
                "search_latency_ms": 0.0,
                "inference_latency_ms": 0.0,
                "total_latency_ms": round((time.time() - start_time) * 1000, 1),
                "mem0_hook_enabled": False,
                "prompt_caching_enabled": False,
                "str_replace_nudge_enabled": False,
            },
        )

    # Resolve scaffold params from the benchmark TOML.
    bench_params = dict(config.benchmark_params or {})
    config_path = str(bench_params.get("seal_config_path", DEFAULT_SWE_AGENT_CONFIG))
    model_name = str(bench_params.get("seal_model_name", DEFAULT_MODEL_NAME))
    per_instance_cost_limit = float(
        bench_params.get("seal_per_instance_cost_limit", DEFAULT_PER_INSTANCE_COST_LIMIT)
    )
    per_instance_call_limit = int(
        bench_params.get("seal_per_instance_call_limit", DEFAULT_PER_INSTANCE_CALL_LIMIT)
    )
    deployment_type = str(bench_params.get("seal_deployment_type", "docker"))
    extra_args_raw = bench_params.get("seal_extra_args") or []
    extra_args = [str(a) for a in extra_args_raw] if isinstance(extra_args_raw, list) else []
    # Opt-in Anthropic/Bedrock prompt caching (deviations.md S26).
    # ``prompt_caching_config`` stays "" unless the operator opted in
    # AND the overlay file is actually present — the row flag
    # ``prompt_caching_enabled`` is derived from the same string so it
    # reflects what the subprocess really received, never the intent.
    prompt_caching_config = ""
    if bench_params.get("seal_prompt_caching"):
        overlay = _resolve_prompt_caching_overlay()
        if overlay.is_file():
            prompt_caching_config = str(overlay)
        else:
            # Loud but non-fatal: the run is still valid, just uncached
            # (and the row flag stays False, so cells remain honestly
            # bucketed). A typo'd deploy shows up as this error line +
            # CloudWatch CacheReadInputTokenCount staying at zero.
            logger.error(
                "seal.prompt_caching_overlay_missing",
                task_id=task.task_id,
                expected_path=str(overlay),
            )
    # Opt-in tool-selection nudge — see configs/seal/str_replace_nudge.yaml
    # docstring + deviations.md. Same wiring shape as prompt_caching:
    # path stays "" unless opted-in AND the file exists, the row flag
    # ``str_replace_nudge_enabled`` is derived from the same string so it
    # reflects what the subprocess actually received.
    str_replace_nudge_config = ""
    if bench_params.get("seal_str_replace_nudge"):
        overlay = _resolve_str_replace_nudge_overlay()
        if overlay.is_file():
            str_replace_nudge_config = str(overlay)
        else:
            logger.error(
                "seal.str_replace_nudge_overlay_missing",
                task_id=task.task_id,
                expected_path=str(overlay),
            )
    subprocess_timeout = int(
        bench_params.get(
            "seal_subprocess_timeout_seconds",
            config.timeout_seconds or DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
        )
    )

    instance = _build_instance_record(task, adapter)

    # bench_params_raw is read here so flags like ``llm_trace_enabled``
    # work on baseline / no-memory cells too.
    bench_params_raw = dict(config.benchmark_params or {})
    # Optional mem0 LiteLLM hook: only fires when ``memory`` is a mem0v2
    # embedded middleware (duck-typed). Non-mem0 backends and baseline /
    # no-memory cells leave ``mem0_env`` None and SEAL runs without
    # injection.
    mem0_env = _resolve_mem0_hook_config(memory, benchmark_params=bench_params_raw)
    if mem0_env is not None and memory is not None:
        # Per-task isolation contract.
        #
        # Mem0V2EmbeddedMiddleware.reset() calls
        # ``delete_all(user_id=self._user_id)`` to wipe the parent-side
        # vector store for that user, but it does NOT reassign
        # ``self._user_id`` — the id is set once in ``__init__`` from
        # ``config.params.get("user_id", "bench_default")`` and stays
        # constant for the life of the middleware instance. So per-task
        # isolation here is NOT provided by id rotation. It IS provided by
        # the sweagent SUBPROCESS restart between tasks — each task gets a
        # fresh in-process ``mem0.Memory()`` in its own subprocess, so
        # cross-task replay can't happen by construction. The reset() call
        # here primarily wipes any orchestrator-side state the parent-side
        # mem0 might have accumulated (e.g. from a prior run on the same
        # process).
        try:
            await memory.reset()
        except Exception as e:
            logger.warning(
                "seal.mem0_reset_failed",
                task_id=task.task_id,
                error=str(e),
            )

    # Default-init the trajectory summary so the post-``with`` eval-stamp
    # block sees a defined name even if the harvest line never ran
    # (subprocess raised, .traj missing, etc.). Populated inside the with
    # block once the subprocess returns.
    traj_summary: dict[str, float | int] = {}

    # Use a per-task tempdir so concurrent fan-outs (CrossBench's
    # ``--parallel N``) don't trample each other's output_dir.
    with tempfile.TemporaryDirectory(prefix=f"seal-{task.task_id}-") as workdir:
        workdir_path = Path(workdir)
        instances_path = _write_instances_file(instance, workdir_path / "instances")
        output_dir = workdir_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Drop a sitecustomize.py stub into a dedicated subdir and prepend
        # that subdir to PYTHONPATH so Python's site-init picks it up when
        # the sweagent subprocess starts. The always-on step counter is
        # written first; the mem0 hook (when active) is appended on top of
        # the same sitecustomize.py.
        extra_pythonpath: list[Path] = []
        # mem0 metrics file — only written when the mem0 hook is active.
        # The mem0-branch below assigns these explicitly; this default
        # ensures the post-subprocess aggregation block can read them
        # unconditionally and get an empty dict back on non-mem0 cells.
        mem0_metrics_path: Path | None = None
        mem0_status_path: Path | None = None
        # Always-on step counter — fires for every cell (including
        # baseline / no_memory). Writes one stderr line per
        # litellm.completion to CloudWatch and persists the final
        # count to ``step_count_path`` so the parent can harvest it
        # after the subprocess exits.
        site_dir = workdir_path / "sitecustomize_d"
        step_counter_hook.write_sitecustomize_stub(str(site_dir))
        extra_pythonpath.append(site_dir)
        step_count_path: Path | None = workdir_path / "step_count.txt"
        # The step-counter env var is set in the subprocess env later.
        if mem0_env is not None:
            # Mem0 hook active — copy the mem0 module next to the
            # sitecustomize stub. The stub appends the mem0 registration
            # block (SITECUSTOMIZE_STUB_MEM0) onto the existing
            # sitecustomize.py written by step_counter_hook's writer.
            mem0_metrics_path = workdir_path / "mem0_metrics.jsonl"
            mem0_env[mem0_litellm_hook.ENV_METRICS_FILE] = str(mem0_metrics_path)
            mem0_status_path = workdir_path / "mem0_status.txt"
            mem0_env[mem0_litellm_hook.ENV_STATUS_FILE] = str(mem0_status_path)
            # Side-load: copy the mem0 hook module next to sitecustomize.py
            # and append a registration stanza to sitecustomize.py.
            try:
                import shutil
                source = Path(mem0_litellm_hook.__file__)
                shutil.copy2(source, site_dir / "mem0_litellm_hook.py")
                sitecustomize_py = site_dir / "sitecustomize.py"
                with open(sitecustomize_py, "a", encoding="utf-8") as fh:
                    fh.write(mem0_litellm_hook.SITECUSTOMIZE_STUB_MEM0)
            except Exception as e:
                logger.warning(
                    "seal.mem0_sitecustomize_write_failed",
                    task_id=task.task_id,
                    error=str(e),
                )
            logger.info(
                "seal.mem0_hook_enabled",
                task_id=task.task_id,
                user_id=mem0_env.get(mem0_litellm_hook.ENV_USER_ID),
                top_k=mem0_env.get(mem0_litellm_hook.ENV_TOP_K),
                infer=mem0_env.get(mem0_litellm_hook.ENV_INFER),
                first_step_only=mem0_env.get(
                    mem0_litellm_hook.ENV_FIRST_STEP_ONLY, "0"
                ),
                disable_store=mem0_env.get(
                    mem0_litellm_hook.ENV_DISABLE_STORE, "0"
                ),
                metrics_file=str(mem0_metrics_path),
                status_file=str(mem0_status_path),
            )
            # Loud warning when ``infer=True`` on a code-RAG benchmark.
            # Per deviations.md M9 + C13, mem0's default LLM extractor
            # synonymizes identifiers and the SWE-bench grader (literal
            # patch match) collapses. The fix is to stack
            # ``mem0_no_infer``, which sets ``_infer=False``. Without
            # this warning operators just see degraded numbers and
            # blame the model rather than the missing mixin.
            if mem0_env.get(mem0_litellm_hook.ENV_INFER) == "1":
                logger.warning(
                    "seal.mem0_infer_true_on_code_rag",
                    task_id=task.task_id,
                    hint=(
                        "mem0 is running with infer=True on SWE-bench. "
                        "Add '-p mem0_no_infer' to the profile stack "
                        "or expect identifier loss in the recall stream. "
                        "See deviations.md M9 + C13."
                    ),
                )

        cmd = _build_sweagent_command(
            instances_path=instances_path,
            output_dir=output_dir,
            config_path=config_path,
            model_name=model_name,
            per_instance_cost_limit=per_instance_cost_limit,
            per_instance_call_limit=per_instance_call_limit,
            deployment_type=deployment_type,
            extra_args=extra_args,
            variant=str((task.metadata or {}).get("variant", "")),
            prompt_caching_config=prompt_caching_config,
            str_replace_nudge_config=str_replace_nudge_config,
        )
        env = _build_subprocess_env(
            swe_agent_root,
            extra_pythonpath=extra_pythonpath,
        )
        # Overlay mem0 env when the mem0 hook is active.
        if mem0_env is not None:
            env.update(mem0_env)
        # Tell the step counter where to persist the final call count.
        # The sitecustomize-loaded counter writes after every step, so
        # even a crashed subprocess leaves the count behind for us.
        if step_count_path is not None:
            env[step_counter_hook.ENV_STEP_COUNT_FILE] = str(step_count_path)

        # Optional LLM I/O trace — opt-in via [benchmark]
        # ``llm_trace_enabled = true`` in any TOML. Independent of
        # memory; works on baseline cells too. Dumps every
        # litellm.completion's full input + output to stderr (prefixed
        # ``[llm-trace]`` so the live-tee surfaces it).
        if bench_params_raw.get("llm_trace_enabled"):
            env[llm_trace_hook.ENV_ENABLED] = "1"
            # Per-task JSONL sidecar — one JSON line per request +
            # response (full body, never truncated). Written to a
            # PERSISTENT host path (NOT the workdir tempdir, which
            # auto-deletes when this ``with`` block exits) so the
            # operator can post-process the trajectory after the
            # task finishes.
            trace_root = Path("/tmp/cross-bench-llm-traces") / config.run_id
            trace_root.mkdir(parents=True, exist_ok=True)
            trace_file = trace_root / f"{task.task_id}.jsonl"
            env[llm_trace_hook.ENV_FILE] = str(trace_file)
            # Suppress per-step stderr dump when ``llm_trace_stderr =
            # false`` in the benchmark TOML. JSONL still gets every
            # record; useful for long runs where you want the file
            # but not the terminal flood.
            if bench_params_raw.get("llm_trace_stderr") is False:
                env[llm_trace_hook.ENV_STDERR_DISABLED] = "1"
            max_chars = bench_params_raw.get("llm_trace_max_content_chars")
            if max_chars is not None:
                env[llm_trace_hook.ENV_MAX_CONTENT_CHARS] = str(int(max_chars))
            logger.info(
                "seal.llm_trace_enabled",
                task_id=task.task_id,
                trace_file=str(trace_file),
                stderr_disabled=env.get(llm_trace_hook.ENV_STDERR_DISABLED) == "1",
            )

        # Retry policy (deviations.md S23 sub #9). A ``NoPrediction`` whose
        # cause is a TRANSIENT swerex infra failure (leaked-port container,
        # runtime-not-alive) is retried up to ``seal_max_retries`` times.
        # The sub #8 orphan sweep runs again before every attempt so a
        # container leaked by the previous attempt can't poison the next.
        # Deterministic ``build_failed`` (the Alpine/musl Pro base images
        # that can't run swerex's glibc standalone Python) is never retried
        # — it'd fail identically and just burn ~4 s.
        max_retries = int(bench_params.get("seal_max_retries", DEFAULT_SEAL_MAX_RETRIES))
        retry_on_timeout = bool(bench_params.get("seal_retry_on_timeout", False))

        attempts = 0
        infra_error = ""
        pred_path: Path | None = None
        cleanup_status: dict[str, Any] = {"swept": 0, "error": None}
        while True:
            attempts += 1
            # Sub #8: sweep swerex sibling containers leaked by a prior SEAL
            # task OR by this task's previous attempt. Runs before EVERY
            # attempt. Without it, a task that timed out leaks port 8000 in
            # the orchestrator's network namespace and the next attempt
            # fails with ``Errno 98 address already in use`` at swerex-remote
            # startup. Best-effort. See ``_cleanup_orphan_swerex_containers``.
            cleanup_status = _cleanup_orphan_swerex_containers()

            logger.info(
                "seal.invoke",
                task_id=task.task_id,
                attempt=attempts,
                max_attempts=max_retries + 1,
                model=model_name,
                deployment=deployment_type,
                config_path=config_path,
                instances_path=str(instances_path),
                output_dir=str(output_dir),
                timeout_seconds=subprocess_timeout,
                cleanup_swept=cleanup_status["swept"],
                prompt_caching=bool(prompt_caching_config),
                str_replace_nudge=bool(str_replace_nudge_config),
            )

            # Accumulate the sweagent subprocess wallclock across attempts
            # so ``inference_latency_ms`` reflects total time spent in
            # ``sweagent run-batch`` (EE sec.3). ``search`` stays 0 (SEAL
            # has no memory step); ``total`` is the whole execute_seal.
            inference_t0 = time.monotonic()
            (
                sweagent_returncode,
                sweagent_stdout,
                sweagent_stderr,
            ) = await _invoke_sweagent(
                cmd,
                cwd=swe_agent_root,
                env=env,
                timeout_seconds=subprocess_timeout,
            )
            inference_seconds += time.monotonic() - inference_t0

            pred_path = _find_pred_file(output_dir, instance["instance_id"])
            # Classify on the FULL (untruncated) buffers — before
            # _format_stdout_for_step trims them for storage.
            infra_error = _classify_swerex_failure(
                sweagent_stdout, sweagent_stderr, sweagent_returncode
            )
            if pred_path is not None:
                break  # a .pred exists (Submitted or Empty) — stop retrying.
            # No .pred → NoPrediction. Retry only transient infra causes.
            retriable = infra_error in _RETRIABLE_INFRA_TAGS or (
                infra_error == "timeout" and retry_on_timeout
            )
            if attempts <= max_retries and retriable:
                logger.warning(
                    "seal.retry_noprediction",
                    task_id=task.task_id,
                    attempt=attempts,
                    next_attempt=attempts + 1,
                    swerex_infra_error=infra_error,
                    returncode=sweagent_returncode,
                )
                continue
            break

        # A ``.pred`` (Submitted or Empty) means swerex stood up fine —
        # ``swerex_infra_error`` only describes a NoPrediction cause, so
        # clear any signature picked up from incidental stdout noise.
        if pred_path is not None:
            infra_error = ""

        # Harvest agent-LLM model_stats from the .traj file written by
        # sweagent's DefaultAgent.save_trajectory. Reads tokens_sent /
        # tokens_received / api_calls / instance_cost from a single JSON
        # document — see ``_extract_traj_summary`` docstring. Must happen
        # inside the ``with`` block (output_dir vanishes after).
        traj_summary = _extract_traj_summary(output_dir, instance["instance_id"])

        # Optional sidecar persistence: when running under the S3
        # artifact backend (``BENCH_ARTIFACTS_BACKEND=s3``), copy
        # sweagent's full ``.traj`` AND the LiteLLM I/O JSONL trace
        # (when ``llm_trace_enabled`` was set) to S3 under
        # ``runs/<benchmark>/<run_id>/sidecars/<task_id>/``. Both
        # files would otherwise vanish with the workdir tempdir
        # (.traj) or live forever under ``/tmp`` on the ECS host
        # (llm_trace.jsonl). Persisting them enables byte-for-byte
        # comparison against Scale's published trajectories
        # (s3://scaleapi-results/swe-bench-pro/.../traj/<id>/<id>.traj).
        # Best-effort: upload failures are logged and swallowed —
        # the agent's evaluation row still goes through normally.
        _persist_sidecars_to_s3(
            output_dir=output_dir,
            instance_id=instance["instance_id"],
            task_id=task.task_id,
            run_id=config.run_id,
            benchmark=getattr(config, "benchmark", "swebench_pro_seal"),
            llm_trace_root=Path("/tmp/cross-bench-llm-traces") / config.run_id,
        )

        steps.append(
            StepRecord(
                step_index=0,
                action="sweagent_run_batch",
                input_text=" ".join(cmd),
                # Sub #8: keep both head AND tail of sweagent stdout — the
                # original ``[:16000]`` slice discarded the failure mode
                # (which surfaces near the END of long agent loops).
                # See ``_format_stdout_for_step`` docstring.
                output_text=_format_stdout_for_step(sweagent_stdout or ""),
                # Populating tokens_in / tokens_out / latency_ms here makes
                # the engine's existing aggregator (engine.py::_aggregate)
                # surface ``total_tokens_in``, ``total_tokens_out``,
                # ``avg_tokens_per_task``, and the latency percentiles in
                # ``summary.json`` automatically. Without these the SEAL
                # column reads as 0 tokens — the row that motivated
                # commit ``572b6cd`` to hide the columns in Slack.
                tokens_in=int(traj_summary.get("total_tokens_in", 0)),
                tokens_out=int(traj_summary.get("total_tokens_out", 0)),
                latency_ms=round(inference_seconds * 1000, 1),
                metadata={
                    "returncode": sweagent_returncode,
                    "stderr_tail": (sweagent_stderr or "")[-4000:],
                    "cleanup_swept": cleanup_status["swept"],
                    "cleanup_error": cleanup_status.get("error"),
                    "attempts": attempts,
                    "swerex_infra_error": infra_error or None,
                },
            )
        )

        if pred_path is None:
            exit_status = "NoPrediction"
            error_text = (
                "sweagent ran but no .pred file was produced. "
                f"returncode={sweagent_returncode}. "
                f"swerex_infra_error={infra_error or 'none'}. "
                f"attempts={attempts}. "
                f"stderr_tail={(sweagent_stderr or '')[-400:]}"
            )
            submission = ""
        else:
            submission = _extract_patch_from_pred(pred_path)
            if not submission.strip():
                exit_status = "EmptyPrediction"
                error_text = f"pred file {pred_path} contained an empty model_patch"
            else:
                exit_status = "Submitted"

        # Harvest mem0-hook metrics from the JSONL the hook appended
        # inside the subprocess. Must happen INSIDE the
        # ``TemporaryDirectory`` context — the dir vanishes when the
        # ``with`` block exits. Empty dict on non-mem0 cells
        # (mem0_metrics_path is None) — downstream stamping is a safe no-op.
        mem0_metrics: dict[str, float | int] = {}
        if mem0_metrics_path is not None:
            mem0_metrics = _aggregate_mem0_metrics(mem0_metrics_path)

        # Always-on LLM-call count, populated from the step-counter
        # hook regardless of which hook (if any) was active. The hook
        # writes the count to step_count_path after every step; we read
        # the final value here.
        llm_step_count: int = 0
        if step_count_path is not None and step_count_path.exists():
            try:
                llm_step_count = int(step_count_path.read_text().strip() or "0")
            except (OSError, ValueError):
                llm_step_count = 0
        # End-of-task summary log (one row per task, searchable in
        # CloudWatch via ``seal.step_count``). Works for every cell — the
        # source is the always-on step counter.
        logger.info(
            "seal.step_count",
            task_id=task.task_id,
            llm_calls=llm_step_count,
        )

    # Grade the patch via the adapter — for Pro this routes through
    # ``swebench_pro_eval.evaluate_pro_task``; for non-Pro variants it
    # routes through the Princeton ``swebench`` harness path. Either
    # way the adapter creates its own sandbox; we pass ``sandbox=None``.
    if submission.strip():
        try:
            evaluation = await adapter.evaluate(
                task, submission, sandbox_mgr, sandbox=None
            )
        except Exception as e:
            logger.error(
                "seal.evaluate_failed", task_id=task.task_id, error=str(e)
            )
            evaluation = {
                "pass": False,
                "resolved": False,
                "patch_successfully_applied": False,
                "error": f"evaluate_failed: {e}",
            }
    else:
        evaluation = {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": False,
            "error": exit_status or "no_submission",
        }
        if error_text:
            evaluation["error_text"] = error_text

    # Post-task Pro image cleanup: SandboxManager.prefetch_images
    # pre-pulls ~30 Pro images per shard (each 3-5 GB) BEFORE the task
    # loop, so the host accumulates ~120 GB and 2/25 shards on a
    # 731-instance fan-out have tipped past the c5.xlarge root EBS
    # capacity ("no space left on device"). Each completed Pro task's
    # image is uniquely tagged and never reused by another instance, so
    # removing it now reclaims 3-5 GB before the next task on this
    # shard runs. Gated on the Pro variant so Verified/Lite (where
    # images are shared across instances) keep their pre-pulled image
    # cache. Pure best-effort -- never affects the evaluation row.
    if str((task.metadata or {}).get("variant", "")) == "pro":
        try:
            _rmi_pro_image(str(adapter.get_sandbox_image(task)))
        except Exception as e:  # noqa: BLE001 - defensive; never fatal
            logger.warning(
                "seal.cleanup.pro_image_rmi_outer_failed",
                task_id=task.task_id, error=str(e),
            )

    # Scaffold provenance — both submodule SHAs land on every row so the
    # leaderboard can attribute numbers to a specific upstream snapshot.
    evaluation.setdefault("scaffold", "seal")
    evaluation.setdefault("scaffold_swe_agent_commit", UPSTREAM_SWE_AGENT_COMMIT)
    evaluation.setdefault("scaffold_pro_commit", UPSTREAM_SWEBENCH_PRO_COMMIT)
    evaluation.setdefault("scaffold_model", model_name)
    evaluation.setdefault("exit_status", exit_status)
    evaluation.setdefault("sweagent_returncode", sweagent_returncode)
    # ``traj_summary`` was harvested inside the ``with`` block above; it
    # carries steps_taken + total_tokens_in/out (already on the StepRecord
    # so the engine surfaces them in summary.json) + api_calls +
    # instance_cost_usd (no StepRecord field; stamped on evaluation here).
    # ``steps_taken`` priority order:
    # 1. ``traj_summary["steps_taken"]`` — direct .traj extraction (currently
    #    always None at the pinned sweagent commit; assistant-role schema
    #    mismatch — separate bug to fix)
    # 2. ``llm_step_count`` — the always-on step counter; works for every
    #    cell, captures true per-LLM-call step count
    # 3. ``len(steps)`` — fallback, always 1 for SEAL (one outer call to
    #    sweagent), only useful when both above are absent
    traj_steps_val = traj_summary.get("steps_taken")
    if traj_steps_val is not None:
        evaluation.setdefault("steps_taken", traj_steps_val)
    elif llm_step_count > 0:
        evaluation.setdefault("steps_taken", llm_step_count)
    else:
        evaluation.setdefault("steps_taken", len(steps))
    # Always also stamp the raw step counter value as its own field so
    # downstream consumers can distinguish the broken-extractor fallback
    # from the real measurement.
    evaluation.setdefault("llm_step_count", llm_step_count)
    # Also stamp tokens on ``evaluation`` per-task so the per-task Slack
    # bullets in ``_post_shard_slack`` (which reads ``evaluation`` via
    # ``_fetch_shard_tasks``) render them. The engine independently sums
    # them from ``r.steps[].tokens_in/out`` for the run-level summary;
    # both numbers come from the same source so they agree by construction.
    if "total_tokens_in" in traj_summary:
        evaluation.setdefault("total_tokens_in", traj_summary["total_tokens_in"])
    if "total_tokens_out" in traj_summary:
        evaluation.setdefault("total_tokens_out", traj_summary["total_tokens_out"])
    if "api_calls" in traj_summary:
        evaluation.setdefault("api_calls", traj_summary["api_calls"])
    if "instance_cost_usd" in traj_summary:
        evaluation.setdefault("instance_cost_usd", traj_summary["instance_cost_usd"])
    evaluation.setdefault("wallclock_seconds", round(time.time() - start_time, 3))
    # Latency split (EE sec.3) — three columns matching the schema mini-
    # SWE-agent / single-shot also produce.
    #   * ``search_latency_ms``  — constant 0 by design. SEAL doesn't
    #     wire per-step memory retrieval (the SWE-agent loop is opaque
    #     to CrossBench's MemoryMiddleware); see deviations.md S23.
    #   * ``inference_latency_ms`` — wallclock of the ``sweagent
    #     run-batch`` subprocess (which internally includes its own
    #     LiteLLM calls + the agent loop's step budget).
    #   * ``total_latency_ms`` — entire ``execute_seal`` duration
    #     including setup, sweagent invocation, .pred harvest, and the
    #     adapter.evaluate grading pass. Not equal to
    #     ``search + inference`` by design — captures everything the
    #     operator waited for.
    evaluation.setdefault("search_latency_ms", 0.0)
    evaluation.setdefault(
        "inference_latency_ms", round(inference_seconds * 1000, 1)
    )
    evaluation.setdefault(
        "total_latency_ms", round((time.time() - start_time) * 1000, 1)
    )
    # Prompt-caching provenance (deviations.md S26). True iff the
    # cache_control overlay was actually wired into the sweagent argv —
    # leaderboard consumers use it to separate cached cells from the
    # Scale-config-exact cells (caching changes request payloads and
    # throttling/latency dynamics, not prompts).
    evaluation.setdefault("prompt_caching_enabled", bool(prompt_caching_config))
    # Same shape for the tool-selection nudge overlay (deviations.md).
    # Reflects whether the system-template-override overlay was actually
    # wired into sweagent's --config chain. Operators bucket cells via
    # this flag — the overlay changes the system template, so cells
    # with it are NOT byte-comparable to Scale's reference config.
    evaluation.setdefault("str_replace_nudge_enabled", bool(str_replace_nudge_config))
    # Honest infra-failure accounting (deviations.md S23 sub #9). When
    # sweagent produced no patch because swerex couldn't stand up the
    # runtime, ``swerex_infra_error`` carries the classified cause
    # (``build_failed`` / ``port_in_use`` / ``runtime_not_alive`` /
    # ``timeout``) and is ``None`` otherwise. Leaderboard consumers bucket
    # ``build_failed`` rows OUT of the model's ``resolved`` denominator —
    # they're un-attemptable infra (the Alpine/musl base images), not a
    # model miss. ``seal_attempts`` records how many tries it took (1 =
    # first-try; >1 = a transient failure that a retry recovered or
    # exhausted). Both are stable columns on every SEAL row.
    evaluation.setdefault("swerex_infra_error", infra_error or None)
    evaluation.setdefault("seal_attempts", attempts)
    # Mem0 hook provenance. Operators bucket SEAL rows into no-memory vs
    # mem0-injected via this flag.
    evaluation.setdefault("mem0_hook_enabled", mem0_env is not None)
    # Stamp mem0-hook metric totals. Empty dict on non-mem0 cells, so
    # this is a safe no-op there. The engine's ``_aggregate`` averages
    # these into ``summary.json`` as ``avg_mem0_*`` automatically.
    for k, v in mem0_metrics.items():
        evaluation.setdefault(k, v)

    status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL
    if exit_status == "NoPrediction":
        status = TaskStatus.ERROR

    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=steps,
        final_output=submission,
        evaluation=evaluation,
    )
