"""Tests for the SEAL scaffold (deviations.md S22).

The SEAL scaffold wraps Scale AI's customized SWE-agent (the canonical
scaffold for the SWE-bench Pro public leaderboard) as a subprocess.
Tests live in five tiers, each addressing a different drift surface:

1. **Submodule pins** — both ``third_party/swe_agent`` and
   ``third_party/swebench_pro`` SHAs are locked. Updating either
   submodule requires updating the pin AND deviations.md S22 — that's
   the audit trail.

2. **Submodule presence** — when the submodules are absent (clean
   clone without ``--recurse-submodules``), the scaffold returns a
   loud ``submodules_missing`` error instead of failing mysteriously
   deep inside subprocess land.

3. **Argv composition** — ``_build_sweagent_command`` is the only
   non-trivial code path before subprocess invocation. Verify it
   produces the exact argv shape upstream documents, including the
   ``--instances.deployment.type=docker`` defaulting and the
   ``--agent.model.*`` flag plumbing.

4. **Instance file shape** — what we hand to SWE-agent via
   ``InstancesFromFile``. The shape must match
   ``SimpleBatchInstance.model_validate`` at the pinned SWE-agent SHA.

5. **End-to-end loop with stubs** — the full ``execute_seal`` path
   with subprocess + adapter both faked. Verifies: happy path
   (patch produced and graded), no-pred-file path, empty-patch path,
   adapter-evaluate-failure path, scaffold metadata stamping.

Smoke recipe (not in this file — operator-run):
   cross-bench run -b swebench_pro_seal -p haiku45 \\
       --task-id <pro-id> --db-url none --artifacts-dir ./artifacts
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cross_bench.execution.scaffolds import seal as seal_mod
from cross_bench.execution.scaffolds.seal import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SWE_AGENT_CONFIG,
    SWE_AGENT_SUBMODULE_PATH,
    SWEBENCH_PRO_SUBMODULE_PATH,
    UPSTREAM_SWE_AGENT_COMMIT,
    UPSTREAM_SWEBENCH_PRO_COMMIT,
    _build_instance_record,
    _build_sweagent_command,
    _cleanup_orphan_swerex_containers,
    _extract_patch_from_pred,
    _find_pred_file,
    _format_stdout_for_step,
    _submodule_initialised,
    _write_instances_file,
    execute_seal,
)
from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    SystemConfig,
    Task,
    TaskGranularity,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# 1. Submodule SHA pins — change requires deviations.md S22 update.
# ---------------------------------------------------------------------------


def test_swe_agent_submodule_pinned_sha():
    """The SWE-agent submodule SHA the scaffold targets.

    Updating the submodule (``git submodule update --remote
    third_party/swe_agent``) requires updating this constant AND a
    deviations.md S22 note describing what changed in the upstream
    snapshot. Drift here is exactly what the frozen-pin pattern is
    designed to catch.
    """
    assert UPSTREAM_SWE_AGENT_COMMIT == (
        "1ac158175178c9c40ed903fb59cc7aa3c87c75ae"
    )


def test_swebench_pro_submodule_pinned_sha():
    """The SWE-bench_Pro-os submodule SHA."""
    assert UPSTREAM_SWEBENCH_PRO_COMMIT == (
        "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
    )


def test_default_swe_agent_config_path():
    """``config/tool_use.yaml`` is the file Scale's example references."""
    assert DEFAULT_SWE_AGENT_CONFIG == "config/tool_use.yaml"


def test_default_model_name_matches_scale_example_config():
    """Matches the model in scaleapi/SWE-agent sweagent_wrapper_configs/example_config.yaml."""
    assert DEFAULT_MODEL_NAME == "anthropic/claude-3-7-sonnet-20250219"


def test_submodule_relative_paths():
    """Locations the scaffold walks up the tree to find."""
    assert SWE_AGENT_SUBMODULE_PATH == Path("third_party") / "swe_agent"
    assert SWEBENCH_PRO_SUBMODULE_PATH == Path("third_party") / "swebench_pro"


# ---------------------------------------------------------------------------
# 2. Submodule presence checks
# ---------------------------------------------------------------------------


def test_submodule_initialised_false_for_missing_dir(tmp_path: Path):
    assert _submodule_initialised(tmp_path / "does-not-exist") is False


def test_submodule_initialised_false_for_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _submodule_initialised(empty) is False


def test_submodule_initialised_true_when_populated(tmp_path: Path):
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "marker").write_text("x")
    assert _submodule_initialised(populated) is True


# ---------------------------------------------------------------------------
# 3. Argv composition
# ---------------------------------------------------------------------------


def test_build_sweagent_command_minimal(tmp_path: Path):
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "instances.json",
        output_dir=tmp_path / "out",
        config_path="config/tool_use.yaml",
        model_name="anthropic/claude-3-7-sonnet-20250219",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
    )
    # First five args are the deterministic prefix.
    assert cmd[:5] == ["python", "-m", "sweagent", "run-batch", "--config"]
    # Config path is the 6th arg.
    assert cmd[5] == "config/tool_use.yaml"
    # Subsequent flags are present (order is contract).
    joined = " ".join(cmd)
    assert "--instances.type=file" in joined
    assert f"--instances.path={tmp_path / 'instances.json'}" in joined
    assert "--instances.deployment.type=docker" in joined  # docker default
    assert "--output_dir" in cmd
    assert str(tmp_path / "out") in cmd
    assert "--num_workers" in cmd
    assert "1" in cmd  # num_workers value
    assert "--agent.model.name=anthropic/claude-3-7-sonnet-20250219" in joined
    assert "--agent.model.per_instance_cost_limit=0.0" in joined
    assert "--agent.model.per_instance_call_limit=0" in joined


def test_build_sweagent_command_passes_deployment_type(tmp_path: Path):
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="gpt-4o",
        per_instance_cost_limit=2.5,
        per_instance_call_limit=10,
        deployment_type="modal",
    )
    assert "--instances.deployment.type=modal" in " ".join(cmd)


def test_build_sweagent_command_pro_does_not_inject_swerex_flags(tmp_path: Path):
    """SEAL scaffold no longer passes swerex-side workaround flags from
    the command-line for Pro variants. The full Pro fix lives in the
    orchestrator Dockerfile's swerex docker.py patch (5 substitutions:
    bullseye builder pin, truthy ``python_standalone_dir`` check, PyPI
    index override, ``--entrypoint=`` clear, libssl bundle). See
    deviations.md S23.

    Earlier scaffold-side rounds (commits 45e52a5 round-1
    ``--python_standalone_dir=``, 6f53fa4 round-3
    ``--docker_args=[\"--entrypoint\",\"\"]``) were obsoleted by the
    Dockerfile patch and removed during the swe-seal → main merge —
    this test pins that contract so a future contributor doesn't
    accidentally re-introduce the scaffold-side flags.
    """
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
        variant="pro",
    )
    # Round-1 flag must NOT be present: swerex's default
    # python_standalone_dir=/root is what the Dockerfile-patched
    # swerex expects (bullseye builder + libssl bundle compensate
    # for the GLIBC mismatch).
    assert not any('--instances.deployment.python_standalone_dir=' in arg for arg in cmd)
    # Round-3 flag must NOT be present: the Dockerfile patch
    # inserts ``--entrypoint=`` directly into swerex's docker-run
    # argv at the source-code level.
    assert not any('--instances.deployment.docker_args' in arg for arg in cmd)


def test_build_sweagent_command_non_pro_unchanged(tmp_path: Path):
    """Non-Pro variants get the same argv shape as Pro — neither path
    carries scaffold-side swerex workarounds anymore. The Dockerfile
    patch only triggers on Pro task images at runtime (bookworm /
    Focal libc) anyway.
    """
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="m",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
        variant="verified",
    )
    assert not any('--instances.deployment.python_standalone_dir=' in arg for arg in cmd)
    assert not any('--instances.deployment.docker_args' in arg for arg in cmd)


def test_build_sweagent_command_appends_extra_args(tmp_path: Path):
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="m",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
        extra_args=["--instances.deployment.runtime_timeout=7200", "--foo=bar"],
    )
    assert cmd[-2:] == [
        "--instances.deployment.runtime_timeout=7200",
        "--foo=bar",
    ]


# ---------------------------------------------------------------------------
# 4. Instance file shape — what SWE-agent's SimpleBatchInstance consumes.
# ---------------------------------------------------------------------------


class _StubProAdapter:
    """Minimal adapter that returns Pro-shaped image names + prompts."""

    name = "swebench"

    def __init__(self, evaluate_result: dict[str, Any] | None = None) -> None:
        self.evaluate_result = evaluate_result or {"pass": False}
        self.evaluate_calls: list[str] = []

    def get_sandbox_image(self, task: Task) -> str:
        return f"jefzda/sweap-images:{task.metadata.get('dockerhub_tag', 'x')}"

    def format_prompt(self, task: Task) -> str:
        return task.prompt

    async def evaluate(
        self, task: Task, generated_output: str, sandbox_mgr: Any, sandbox: Any = None
    ) -> dict[str, Any]:
        self.evaluate_calls.append(generated_output)
        return dict(self.evaluate_result)


def _pro_task() -> Task:
    return Task(
        task_id="instance_flipt-io__flipt-abc",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="Issue: foo bar baz",
        metadata={
            "instance_id": "instance_flipt-io__flipt-abc",
            "variant": "pro",
            "base_commit": "deadbeef",
            "dockerhub_tag": "instance_flipt-io__flipt-abc",
        },
    )


def _verified_task() -> Task:
    return Task(
        task_id="django__django-11133",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="Issue: HttpResponse doesn't handle memoryview objects",
        metadata={
            "instance_id": "django__django-11133",
            "variant": "verified",
            "base_commit": "abc123",
        },
    )


def test_build_instance_record_pro_uses_app_repo_name():
    """Pro images ship the repo at /app — SWE-agent's PreExistingRepoConfig
    must be told ``repo_name="app"`` so the agent cd's there."""
    rec = _build_instance_record(_pro_task(), _StubProAdapter())
    assert rec["instance_id"] == "instance_flipt-io__flipt-abc"
    assert rec["image_name"] == "jefzda/sweap-images:instance_flipt-io__flipt-abc"
    assert rec["repo_name"] == "app"
    assert rec["base_commit"] == "deadbeef"
    assert rec["problem_statement"] == "Issue: foo bar baz"


def test_build_instance_record_verified_uses_testbed_repo_name():
    """Verified / Lite / Full SWE-bench images use /testbed (upstream default)."""
    rec = _build_instance_record(_verified_task(), _StubProAdapter())
    assert rec["repo_name"] == "testbed"


def test_write_instances_file_emits_json_array(tmp_path: Path):
    """SWE-agent expects ``[{...}]`` shape; verify the JSON round-trips."""
    workspace = tmp_path / "ws"
    rec = {
        "instance_id": "x__y-1",
        "image_name": "img",
        "problem_statement": "P",
        "repo_name": "app",
        "base_commit": "c",
    }
    path = _write_instances_file(rec, workspace)
    assert path.name == "x__y-1.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == [rec]


# ---------------------------------------------------------------------------
# 5a. Patch harvesting from .pred files
# ---------------------------------------------------------------------------


def test_find_pred_file_finds_canonical_layout(tmp_path: Path):
    """sweagent writes <output_dir>/<run_id>/<instance_id>/<instance_id>.pred."""
    out = tmp_path / "out"
    inst = "django__django-11133"
    pred = out / "run-uuid" / inst / f"{inst}.pred"
    pred.parent.mkdir(parents=True)
    pred.write_text('{"model_patch": "diff"}')
    found = _find_pred_file(out, inst)
    assert found == pred


def test_find_pred_file_returns_none_when_missing(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    assert _find_pred_file(out, "missing") is None


def test_extract_patch_from_pred_canonical_json(tmp_path: Path):
    pred = tmp_path / "p.pred"
    pred.write_text(
        json.dumps(
            {
                "model_name_or_path": "claude",
                "instance_id": "x",
                "model_patch": "diff --git a/x b/x\n",
            }
        )
    )
    assert _extract_patch_from_pred(pred) == "diff --git a/x b/x\n"


def test_extract_patch_from_pred_null_patch_returns_empty(tmp_path: Path):
    """Agent never submitted (cost-cap, call-cap, format error) → None."""
    pred = tmp_path / "p.pred"
    pred.write_text(
        json.dumps(
            {
                "model_name_or_path": "claude",
                "instance_id": "x",
                "model_patch": None,
            }
        )
    )
    assert _extract_patch_from_pred(pred) == ""


def test_extract_patch_from_pred_plaintext_fallback(tmp_path: Path):
    """Some older sweagent paths write the raw patch as plain text."""
    pred = tmp_path / "p.pred"
    pred.write_text("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n")
    assert "diff --git" in _extract_patch_from_pred(pred)


def test_extract_patch_from_pred_missing_file_returns_empty(tmp_path: Path):
    assert _extract_patch_from_pred(tmp_path / "absent") == ""


# ---------------------------------------------------------------------------
# 5b. End-to-end loop with stubs
# ---------------------------------------------------------------------------


def _seal_config(**overrides: Any) -> RunConfig:
    params = {
        "seal_config_path": "config/tool_use.yaml",
        "seal_model_name": "anthropic/claude-3-7-sonnet-20250219",
        "seal_per_instance_cost_limit": 0.0,
        "seal_per_instance_call_limit": 0,
        "seal_deployment_type": "docker",
        "seal_subprocess_timeout_seconds": 60,
    }
    params.update(overrides)
    return RunConfig(
        run_id="seal-test",
        benchmark="swebench",
        execution_mode=ExecutionMode.SEAL,
        system=SystemConfig(
            llm=LLMConfig(provider="stub", model="stub"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
            memory=None,
        ),
        max_agent_steps=250,
        timeout_seconds=3600,
        benchmark_params=params,
    )


@dataclass
class _FakeSandboxManager:
    """The SEAL scaffold doesn't directly use the sandbox manager — it
    just hands it to ``adapter.evaluate``. So the fake doesn't need to
    do anything; it just exists to be passed through."""

    pass


@pytest.fixture
def submodules_present(tmp_path: Path, monkeypatch):
    """Pretend submodules are initialised at a fake repo root."""
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "third_party" / "swe_agent" / "sweagent").mkdir(parents=True)
    (fake_repo / "third_party" / "swe_agent" / "sweagent" / "__init__.py").write_text("")
    (fake_repo / "third_party" / "swebench_pro" / "run_scripts").mkdir(parents=True)
    (fake_repo / "third_party" / "swebench_pro" / "run_scripts" / ".keep").write_text("")
    (fake_repo / ".gitmodules").write_text("[submodule] # fake")
    monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(fake_repo))
    return fake_repo


@pytest.mark.asyncio
async def test_execute_seal_submodules_missing_returns_error(tmp_path, monkeypatch):
    """Surface a loud setup error rather than failing deep in subprocess."""
    fake_repo = tmp_path / "empty"
    fake_repo.mkdir()
    (fake_repo / ".gitmodules").write_text("[submodule]")
    monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(fake_repo))

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert result.status == TaskStatus.ERROR
    assert "submodules_missing" in result.evaluation["error"]
    assert result.evaluation["scaffold"] == "seal"
    # The pins land on the evaluation row even on the error path so the
    # leaderboard can attribute the failure to a specific submodule snapshot.
    assert result.evaluation["scaffold_swe_agent_commit"] == UPSTREAM_SWE_AGENT_COMMIT
    assert result.evaluation["scaffold_pro_commit"] == UPSTREAM_SWEBENCH_PRO_COMMIT


@pytest.mark.asyncio
async def test_execute_seal_happy_path(submodules_present, monkeypatch):
    """Subprocess produces a .pred file → patch flows to adapter.evaluate."""
    captured: dict[str, Any] = {}

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        # Capture the call shape for assertions.
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout_seconds
        # Find --output_dir in cmd and write a .pred there for the
        # harvest path to pick up.
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "run-1" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(
            json.dumps(
                {
                    "model_name_or_path": "claude",
                    "instance_id": inst_id,
                    "model_patch": "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n",
                }
            )
        )
        return 0, "sweagent ok", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)

    adapter = _StubProAdapter(
        evaluate_result={
            "pass": True,
            "resolved": True,
            "patch_successfully_applied": True,
        }
    )

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )

    # Patch landed in adapter.evaluate.
    assert len(adapter.evaluate_calls) == 1
    assert "diff --git a/x b/x" in adapter.evaluate_calls[0]
    # Adapter said pass=True → status is PASS.
    assert result.status == TaskStatus.PASS
    assert result.evaluation["pass"] is True
    # Scaffold metadata stamped.
    assert result.evaluation["scaffold"] == "seal"
    assert result.evaluation["scaffold_swe_agent_commit"] == UPSTREAM_SWE_AGENT_COMMIT
    assert result.evaluation["scaffold_pro_commit"] == UPSTREAM_SWEBENCH_PRO_COMMIT
    assert result.evaluation["exit_status"] == "Submitted"
    assert result.evaluation["sweagent_returncode"] == 0
    # Subprocess saw the expected argv & cwd.
    assert "sweagent" in captured["cmd"]
    assert "run-batch" in captured["cmd"]
    assert "third_party" in str(captured["cwd"])
    assert "swe_agent" in str(captured["cwd"])


@pytest.mark.asyncio
async def test_execute_seal_no_pred_file_returns_error(submodules_present, monkeypatch):
    """If sweagent runs but produces no .pred, we surface NoPrediction."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        return 1, "", "sweagent failed inside"

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)
    adapter = _StubProAdapter()
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert result.status == TaskStatus.ERROR
    assert result.evaluation["error"] == "NoPrediction"
    assert "sweagent failed inside" in result.evaluation.get("error_text", "")
    assert result.evaluation["sweagent_returncode"] == 1
    # adapter.evaluate NOT called when there's no patch to grade.
    assert adapter.evaluate_calls == []


@pytest.mark.asyncio
async def test_execute_seal_empty_patch_skips_evaluate(submodules_present, monkeypatch):
    """sweagent produced a .pred but model_patch is null → no evaluate call."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "r" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(json.dumps({"model_patch": None}))
        return 0, "", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)
    adapter = _StubProAdapter()
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert result.evaluation["error"] == "EmptyPrediction"
    assert adapter.evaluate_calls == []


@pytest.mark.asyncio
async def test_execute_seal_evaluate_failure_surfaces_clean(submodules_present, monkeypatch):
    """Adapter.evaluate raising shouldn't take down the run."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "r" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(json.dumps({"model_patch": "diff content"}))
        return 0, "", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)

    class _FailingAdapter(_StubProAdapter):
        async def evaluate(self, *a, **kw):
            raise RuntimeError("docker fell over")

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_FailingAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    # Status flips to FAIL (we submitted a patch), but the error is recorded.
    assert result.status == TaskStatus.FAIL
    assert "docker fell over" in result.evaluation["error"]


@pytest.mark.asyncio
async def test_execute_seal_passes_model_param_through_to_argv(
    submodules_present, monkeypatch
):
    """Overriding ``seal_model_name`` flows to ``--agent.model.name=...``."""
    captured_cmd: list[str] = []

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        captured_cmd.extend(cmd)
        return 0, "", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)
    adapter = _StubProAdapter()
    cfg = _seal_config(seal_model_name="openai/gpt-4o-2024-08-06")
    await execute_seal(
        task=_pro_task(),
        config=cfg,
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    joined = " ".join(captured_cmd)
    assert "--agent.model.name=openai/gpt-4o-2024-08-06" in joined


@pytest.mark.asyncio
async def test_execute_seal_unused_llm_strategy_memory_args_accepted(
    submodules_present, monkeypatch
):
    """The signature accepts ``llm``/``strategy``/``memory`` for engine
    dispatch parity even though the scaffold doesn't use them."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        return 1, "", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)

    # Sentinel objects standing in for llm/strategy/memory. The scaffold
    # must NOT call any methods on them (or this would AttributeError).
    class _NotCallable:
        pass

    sentinel = _NotCallable()

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(),
        llm=sentinel,
        strategy=sentinel,
        memory=sentinel,
        sandbox_mgr=_FakeSandboxManager(),
    )
    # The sentinel attributes were never touched (no AttributeError).
    assert result.evaluation["scaffold"] == "seal"


# ---------------------------------------------------------------------------
# 6. Engine dispatch — verify the ExecutionMode enum routes correctly.
# ---------------------------------------------------------------------------


def test_execution_mode_seal_enum_value():
    """``execution_mode = "seal"`` in benchmark TOMLs must parse cleanly."""
    assert ExecutionMode("seal") == ExecutionMode.SEAL
    assert ExecutionMode.SEAL.value == "seal"


def test_engine_dispatches_execution_mode_seal():
    """The engine's dispatch table includes the SEAL executor."""
    # Read engine source rather than importing the closure — the dispatch
    # block is the contract.
    from importlib import import_module

    engine_mod = import_module("cross_bench.execution.engine")
    # Direct import success means the seal scaffold module is wired in.
    assert hasattr(engine_mod, "execute_seal")


# ---------------------------------------------------------------------------
# 7. Latency split (EE sec.3) — three columns on every SEAL row, regardless
#    of which branch produced it. Keeps the schema stable across happy /
#    error / no-invocation paths so summary.json column count doesn't
#    flap based on outcome distribution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_seal_latency_split_present_on_happy_path(
    submodules_present, monkeypatch
):
    """SEAL stamps ``search_latency_ms`` / ``inference_latency_ms`` /
    ``total_latency_ms`` on every successful row.

    ``search`` is constant zero (SEAL has no memory step by design).
    ``inference`` is the wallclock of the sweagent subprocess.
    ``total`` is the entire ``execute_seal`` duration including
    setup + sweagent + .pred harvest + adapter.evaluate."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        # Inject a non-trivial inference duration so inference != 0
        # without depending on real wall-clock noise.
        import asyncio as _aio
        await _aio.sleep(0)  # yield to keep test deterministic + cheap
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "r" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(json.dumps({"model_patch": "diff content"}))
        return 0, "", ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(evaluate_result={"pass": True}),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    ev = result.evaluation
    # All three columns present.
    assert "search_latency_ms" in ev
    assert "inference_latency_ms" in ev
    assert "total_latency_ms" in ev
    # SEAL has no memory step → search is constant zero.
    assert ev["search_latency_ms"] == 0.0
    # Inference & total are non-negative floats. Don't pin specific
    # values — wallclock varies — but verify the relationship:
    # inference (subprocess only) ≤ total (whole function).
    assert isinstance(ev["inference_latency_ms"], (int, float))
    assert isinstance(ev["total_latency_ms"], (int, float))
    assert ev["inference_latency_ms"] >= 0.0
    assert ev["total_latency_ms"] >= ev["inference_latency_ms"]


@pytest.mark.asyncio
async def test_execute_seal_latency_split_present_on_submodules_missing(
    tmp_path, monkeypatch
):
    """Even the early-return submodules-missing branch stamps the three
    latency columns. Schema stays stable across outcomes.
    """
    fake_repo = tmp_path / "empty"
    fake_repo.mkdir()
    (fake_repo / ".gitmodules").write_text("[submodule]")
    monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(fake_repo))

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    ev = result.evaluation
    assert ev["search_latency_ms"] == 0.0
    assert ev["inference_latency_ms"] == 0.0
    # total_latency_ms is the elapsed function time — should be small but
    # non-negative.
    assert ev["total_latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_execute_seal_latency_split_present_on_no_pred_path(
    submodules_present, monkeypatch
):
    """sweagent ran (so ``inference_latency_ms`` is measured) but no
    .pred was produced — all three columns still land."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        return 1, "", "sweagent failed inside"

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake_invoke)
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    ev = result.evaluation
    assert "search_latency_ms" in ev
    assert "inference_latency_ms" in ev
    assert "total_latency_ms" in ev
    assert ev["search_latency_ms"] == 0.0
    assert ev["inference_latency_ms"] >= 0.0  # measured around the failed call


# ---------------------------------------------------------------------------
# 8. S22 / S23 config-file contract — every benchmark / profile TOML the
#    deviations doc claims to ship must actually exist on disk. Reviewer
#    feedback (DV sec.1): "deviation notes describing nonexistent files
#    are exactly the silent-deviation pattern this file is meant to
#    prevent". This test pins the four S22 files + the swebench_pro_seal
#    TOML so a future contributor can't accidentally drop one without
#    failing CI.
# ---------------------------------------------------------------------------


def test_swebench_gold_eval_config_files_exist():
    """S22 documents four config files; all four must ship."""
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    configs = [
        "configs/benchmarks/swebench_gold.toml",
        "configs/benchmarks/swebench_pro_gold.toml",
        "configs/profiles/swebench_gold.toml",
        "configs/profiles/swebench_pro_gold.toml",
    ]
    missing = [c for c in configs if not (here / c).is_file()]
    assert not missing, (
        f"deviations.md S22 references these files but they don't ship: "
        f"{missing}. Either ship the missing TOMLs or amend S22."
    )


def test_swebench_pro_seal_config_file_exists():
    """S23 documents ``configs/benchmarks/swebench_pro_seal.toml``."""
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    assert (here / "configs/benchmarks/swebench_pro_seal.toml").is_file()


def test_swebench_gold_benchmark_toml_has_required_fields():
    """The shipped swebench_gold benchmark TOML routes through gold_eval."""
    import tomllib
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    data = tomllib.loads(
        (here / "configs/benchmarks/swebench_gold.toml").read_text(encoding="utf-8")
    )
    assert data["run"]["benchmark"] == "swebench"
    assert data["run"]["execution_mode"] == "gold_eval"
    assert data["benchmark"]["variant"] == "verified"


def test_swebench_pro_gold_profile_toml_has_memory_disabled():
    """Pro gold profile tags the run and disables memory (mirror swebench_gold)."""
    import tomllib
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    data = tomllib.loads(
        (here / "configs/profiles/swebench_pro_gold.toml").read_text(encoding="utf-8")
    )
    assert "swebench-pro" in data["run"]["tags"]
    assert data["system"]["memory"]["enabled"] is False


# ---------------------------------------------------------------------------
# 9. gold_eval latency split (EE sec.3) — same three-column contract as
#    SEAL, all constant zero (gold_eval skips both retrieval and inference
#    by design). Without ``search_latency_ms`` the summary schema would
#    drop the column for gold_eval rows, breaking cross-scaffold
#    comparability.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_gold_eval_stamps_search_latency_ms():
    """gold_eval emits all three latency columns even though it does no
    retrieval or inference. Locks the EE sec.3 fix."""
    from cross_bench.execution.modes import execute_gold_eval

    class _GoldStubAdapter:
        name = "swebench"
        requires_sandbox = False

        def get_sandbox_image(self, task):
            return "test:latest"

        async def evaluate(self, task, patch, sandbox_mgr, sandbox=None):
            return {"pass": True, "resolved": True}

    task = Task(
        task_id="django__django-11099",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="",
        metadata={"variant": "verified"},
        ground_truth="diff --git a/x b/x\n",
    )
    cfg = _seal_config()  # any valid RunConfig
    result = await execute_gold_eval(
        task=task, config=cfg,
        adapter=_GoldStubAdapter(),
        llm=None, strategy=None, memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    ev = result.evaluation
    # All three columns present and zero.
    assert ev["search_latency_ms"] == 0.0
    assert ev["inference_latency_ms"] == 0.0
    assert ev["total_latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_execute_gold_eval_stamps_latency_on_missing_gold_patch():
    """Reviewer carryover (EE sec.3): the ``missing_gold_patch``
    early-return path in ``execute_gold_eval`` must also stamp the
    three latency columns. Without it, rows where ``task.ground_truth``
    is empty/None come back with no latency columns, re-creating the
    column-flap this fix is meant to prevent.

    Mirrors the SEAL ``submodules_missing`` branch — which stamps the
    three keys on its early return. Consistency between the two
    scaffolds is the EE sec.3 invariant.
    """
    from cross_bench.execution.modes import execute_gold_eval

    class _NeverCalledAdapter:
        name = "swebench"
        requires_sandbox = False

        def get_sandbox_image(self, task):
            raise AssertionError("evaluate should not be reached")

        async def evaluate(self, task, patch, sandbox_mgr, sandbox=None):
            raise AssertionError("evaluate should not be reached")

    task = Task(
        task_id="missing_gold",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="",
        metadata={"variant": "verified"},
        ground_truth=None,  # triggers the missing_gold_patch branch
    )
    result = await execute_gold_eval(
        task=task, config=_seal_config(),
        adapter=_NeverCalledAdapter(),
        llm=None, strategy=None, memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    ev = result.evaluation
    assert ev["error"] == "missing_gold_patch"
    assert ev["scaffold"] == "gold_eval"
    # All three columns present even though we returned early.
    assert ev["search_latency_ms"] == 0.0
    assert ev["inference_latency_ms"] == 0.0
    assert ev["total_latency_ms"] == 0.0


def test_swebench_gold_in_builtin_profiles():
    """``swebench_gold`` + ``swebench_pro_gold`` are in BUILTIN_PROFILES
    so profile-discovery tooling (CLI completion, lint passes) sees them.

    Reviewer noted this is a pattern other swebench_* profiles share —
    fixing at least the two gold-eval profiles here in lockstep with
    them shipping (CF sec.1).
    """
    from cross_bench.config import BUILTIN_PROFILES
    assert "swebench_gold" in BUILTIN_PROFILES
    assert "swebench_pro_gold" in BUILTIN_PROFILES


# ---------------------------------------------------------------------------
# Sub #8: format_stdout_for_step (head + tail truncation)
# ---------------------------------------------------------------------------


def test_format_stdout_for_step_passes_short_text_through():
    """Stdout under the head+tail budget round-trips unchanged."""
    text = "line1\nline2\nline3\n"
    assert _format_stdout_for_step(text, head=16000, tail=16000) == text


def test_format_stdout_for_step_passes_empty_through():
    assert _format_stdout_for_step("") == ""
    assert _format_stdout_for_step(None or "") == ""  # explicit None-safety


def test_format_stdout_for_step_keeps_head_and_tail_when_over_budget():
    """Long stdout keeps head + sentinel + tail."""
    # 100KB of stdout, head=100, tail=100 → keeps first 100 + last 100 + sentinel
    head_marker = "HEAD_MARKER_FOR_TEST_"
    tail_marker = "TAIL_MARKER_FOR_TEST_"
    middle = "X" * 100_000
    text = head_marker + middle + tail_marker
    out = _format_stdout_for_step(text, head=len(head_marker) + 50, tail=len(tail_marker) + 50)
    # Head marker preserved
    assert head_marker in out
    # Tail marker preserved
    assert tail_marker in out
    # Truncation sentinel present (with byte count)
    assert "chars truncated by SEAL scaffold" in out
    # Reasonable bounded size (much smaller than input)
    assert len(out) < len(text) // 100


def test_format_stdout_for_step_sentinel_reports_dropped_byte_count():
    """The sentinel must include the number of bytes elided so operators
    inspecting per-task JSON know how much they're missing."""
    text = "a" * 1000 + "b" * 5000 + "c" * 1000
    out = _format_stdout_for_step(text, head=1000, tail=1000)
    # Dropped should equal 5000 (the middle "b" run)
    assert "5000 chars truncated" in out


def test_format_stdout_for_step_defaults_keep_32kb_total():
    """Default head=16000 + tail=16000 = 32KB total — bounded but useful."""
    text = "x" * 100_000
    out = _format_stdout_for_step(text)
    # Should be ~32KB + sentinel
    assert 32_000 < len(out) < 33_000


# ---------------------------------------------------------------------------
# Sub #8: cleanup_orphan_swerex_containers
# ---------------------------------------------------------------------------


def test_cleanup_no_orphans_returns_zero_swept(monkeypatch):
    """When docker ps returns no matching containers, swept=0 and no error."""

    class _FakeProc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return _FakeProc(stdout="")
        # docker rm should NOT be called when there are no orphans
        raise AssertionError(f"unexpected docker invocation: {cmd}")

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    status = _cleanup_orphan_swerex_containers()
    assert status["swept"] == 0
    assert status["ids"] == []
    assert status["error"] is None
    # Only docker ps was called — no rm
    assert len(calls) == 1
    assert calls[0][:2] == ["docker", "ps"]


def test_cleanup_sweeps_matching_containers(monkeypatch):
    """When docker ps returns IDs, sub #8 issues `docker rm -f <ids>`."""

    class _FakeProc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    rm_args: list[list[str]] = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "ps"]:
            return _FakeProc(stdout="abc123\ndef456\n")
        if cmd[:3] == ["docker", "rm", "-f"]:
            rm_args.append(cmd)
            return _FakeProc(returncode=0)
        raise AssertionError(f"unexpected: {cmd}")

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    status = _cleanup_orphan_swerex_containers()
    assert status["swept"] == 2
    assert status["ids"] == ["abc123", "def456"]
    assert status["rm_returncode"] == 0
    assert rm_args == [["docker", "rm", "-f", "abc123", "def456"]]


def test_cleanup_filter_targets_swerex_name_prefix(monkeypatch):
    """The docker ps filter must be ``name=jefzdasweap-images`` (swerex's
    sanitized prefix for SWE-bench Pro containers). If a future swerex
    revision changes the sanitization, this test fails loudly."""

    class _FakeProc:
        def __init__(self, stdout="", returncode=0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    seen_filters: list[str] = []

    def fake_run(cmd, **kw):
        for i, arg in enumerate(cmd):
            if arg == "--filter" and i + 1 < len(cmd):
                seen_filters.append(cmd[i + 1])
        return _FakeProc(stdout="")

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    _cleanup_orphan_swerex_containers()
    assert "name=jefzdasweap-images" in seen_filters


def test_cleanup_swallows_docker_missing(monkeypatch):
    """No docker CLI → cleanup records the error and returns; doesn't raise."""

    def fake_run(cmd, **kw):
        raise FileNotFoundError("docker not on PATH")

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    status = _cleanup_orphan_swerex_containers()
    assert status["swept"] == 0
    assert status["error"] is not None
    assert "docker CLI not on PATH" in status["error"]


def test_cleanup_swallows_timeout(monkeypatch):
    """Hung `docker ps` doesn't propagate — cleanup is best-effort."""

    def fake_run(cmd, **kw):
        raise seal_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=10)

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    status = _cleanup_orphan_swerex_containers()
    assert status["swept"] == 0
    assert status["error"] is not None
    assert "timed out" in status["error"]


def test_cleanup_swallows_docker_ps_failure(monkeypatch):
    """Non-zero rc on docker ps → record error, no rm."""

    class _FakeProc:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "ps"]:
            return _FakeProc(stderr="Cannot connect to the Docker daemon", returncode=1)
        raise AssertionError("docker rm should not be called when ps fails")

    monkeypatch.setattr(seal_mod.subprocess, "run", fake_run)
    status = _cleanup_orphan_swerex_containers()
    assert status["swept"] == 0
    assert status["error"] is not None
    assert "rc=1" in status["error"]


# ---------------------------------------------------------------------------
# 9. Prompt caching (deviations.md S26) — opt-in cache_control overlay.
# ---------------------------------------------------------------------------


def _make_pred_writing_invoke(captured: dict[str, Any]):
    """fake ``_invoke_sweagent`` that captures argv and writes a .pred."""

    async def fake_invoke(cmd, *, cwd, env, timeout_seconds):
        captured["cmd"] = cmd
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "run-1" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(
            json.dumps(
                {
                    "model_name_or_path": "claude",
                    "instance_id": inst_id,
                    "model_patch": "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n",
                }
            )
        )
        return 0, "sweagent ok", ""

    return fake_invoke


def test_prompt_caching_overlay_exists_and_parses():
    """The shipped overlay enables exactly the pinned SWE-agent's
    ``CacheControlHistoryProcessor`` and nothing else. ``agent`` must be
    the ONLY top-level key — any other key would risk overriding the
    pinned ``tool_use.yaml`` on the recursive ``merge_nested_dicts``
    merge.
    """
    yaml = pytest.importorskip("yaml")
    overlay = seal_mod._resolve_prompt_caching_overlay()
    assert overlay.is_file(), f"overlay missing at {overlay}"
    data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"agent"}
    assert set(data["agent"].keys()) == {"history_processors"}
    processors = data["agent"]["history_processors"]
    assert processors == [{"type": "cache_control", "last_n_messages": 2}]


def test_build_sweagent_command_includes_prompt_caching_overlay(tmp_path: Path):
    """Overlay --config comes AFTER the base config (later file wins on
    sweagent's merge) and BEFORE the instance args."""
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="m",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
        prompt_caching_config="/app/configs/seal/cache_control.yaml",
    )
    config_indices = [i for i, a in enumerate(cmd) if a == "--config"]
    assert len(config_indices) == 2
    base_idx, overlay_idx = config_indices
    assert cmd[base_idx + 1] == "config/tool_use.yaml"
    assert cmd[overlay_idx + 1] == "/app/configs/seal/cache_control.yaml"
    assert overlay_idx > base_idx
    assert overlay_idx < cmd.index("--instances.type=file")


def test_build_sweagent_command_omits_overlay_by_default(tmp_path: Path):
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="m",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
    )
    assert cmd.count("--config") == 1


def test_build_sweagent_command_extra_args_follow_overlay(tmp_path: Path):
    """``seal_extra_args`` keeps final say — appended after the overlay."""
    cmd = _build_sweagent_command(
        instances_path=tmp_path / "i.json",
        output_dir=tmp_path / "o",
        config_path="config/tool_use.yaml",
        model_name="m",
        per_instance_cost_limit=0.0,
        per_instance_call_limit=0,
        extra_args=["--agent.model.api_base=https://bedrock-runtime.us-east-1.amazonaws.com"],
        prompt_caching_config="/overlay.yaml",
    )
    overlay_idx = cmd.index("/overlay.yaml")
    extra_idx = cmd.index(
        "--agent.model.api_base=https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    assert extra_idx > overlay_idx


@pytest.mark.asyncio
async def test_execute_seal_prompt_caching_wires_overlay_and_stamps_flag(
    submodules_present, monkeypatch
):
    """``seal_prompt_caching = true`` → the real shipped overlay lands in
    argv and the row flag flips True. The resolver goes through
    ``CONFIGS_DIR`` (source tree), not the fake repo root, so this
    exercises the actual ``configs/seal/cache_control.yaml``."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", _make_pred_writing_invoke(captured))

    adapter = _StubProAdapter(
        evaluate_result={"pass": True, "resolved": True, "patch_successfully_applied": True}
    )
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_prompt_caching=True),
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )

    cmd = captured["cmd"]
    config_values = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--config"]
    assert len(config_values) == 2
    assert config_values[1].endswith("cache_control.yaml")
    assert Path(config_values[1]).is_file()
    assert result.evaluation["prompt_caching_enabled"] is True


@pytest.mark.asyncio
async def test_execute_seal_prompt_caching_default_off(submodules_present, monkeypatch):
    """Without the opt-in, argv carries only the pinned config and the
    row flag is False — the leaderboard-comparable cell is untouched."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", _make_pred_writing_invoke(captured))

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(
            evaluate_result={"pass": True, "resolved": True, "patch_successfully_applied": True}
        ),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )

    assert captured["cmd"].count("--config") == 1
    assert result.evaluation["prompt_caching_enabled"] is False


@pytest.mark.asyncio
async def test_execute_seal_prompt_caching_missing_overlay_degrades_honestly(
    submodules_present, monkeypatch, tmp_path: Path
):
    """Opt-in with a missing overlay file → run continues UNCACHED and the
    row flag stays False (flag reflects what the subprocess received,
    never the intent). Guards against a deploy that ships the code but
    not ``configs/seal/cache_control.yaml``."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", _make_pred_writing_invoke(captured))
    monkeypatch.setattr(
        seal_mod, "_resolve_prompt_caching_overlay", lambda: tmp_path / "absent.yaml"
    )

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_prompt_caching=True),
        adapter=_StubProAdapter(
            evaluate_result={"pass": True, "resolved": True, "patch_successfully_applied": True}
        ),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )

    assert captured["cmd"].count("--config") == 1
    assert result.evaluation["prompt_caching_enabled"] is False


@pytest.mark.asyncio
async def test_execute_seal_submodules_missing_stamps_prompt_caching_false(
    tmp_path, monkeypatch
):
    """Column-shape stability: the early-error path carries the flag too."""
    fake_repo = tmp_path / "empty"
    fake_repo.mkdir()
    (fake_repo / ".gitmodules").write_text("[submodule]")
    monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(fake_repo))

    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_prompt_caching=True),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert result.status == TaskStatus.ERROR
    assert result.evaluation["prompt_caching_enabled"] is False


def test_seal_prompt_caching_profile_toml():
    """The mixin flips exactly the one benchmark param and is listed in
    BUILTIN_PROFILES (gets the no-scaffold-keys CI validation)."""
    import tomllib

    here = Path(__file__).resolve().parents[2]
    data = tomllib.loads(
        (here / "configs/profiles/seal_prompt_caching.toml").read_text(encoding="utf-8")
    )
    assert data["benchmark"]["seal_prompt_caching"] is True
    assert "run" not in data or "execution_mode" not in data.get("run", {})

    from cross_bench.config import BUILTIN_PROFILES

    assert "seal_prompt_caching" in BUILTIN_PROFILES


# ---------------------------------------------------------------------------
# 10. Sub #9 — transient-NoPrediction retry + swerex infra-error bucketing
#     (deviations.md S23 sub #9).
# ---------------------------------------------------------------------------

# Representative swerex stdout signatures (mirror the real per-task JSONs
# from the pro-hf731-no_memory-r1-20260614 run + the port-drop run).
_BUILD_FAILED_STDOUT = (
    '🤠 ERROR    Traceback (most recent call last):\n'
    '  File ".../swerex/deployment/docker.py", line 240, in start\n'
    '      image_id = self._build_image()\n'
    '  File ".../swerex/deployment/docker.py", line 224, in _build_image\n'
    '      subprocess.check_output(\n'
    "subprocess.CalledProcessError: Command '['docker', 'build', '-q', "
    "'--build-arg', 'BASE_IMAGE=jefzda/sweap-images:gravitational.teleport-x', "
    "'-']' returned non-zero exit status 127.\n"
)
_PORT_IN_USE_STDOUT = (
    "🦖 INFO     Starting runtime at 58877\n"
    "RuntimeError: Container process terminated.\n"
    "stderr:\n"
    "ERROR:    [Errno 98] error while attempting to bind on address "
    "('0.0.0.0', 8000): address already in use\n"
)
_RUNTIME_NOT_ALIVE_STDOUT = (
    "🦖 INFO     Starting runtime at 8000\n"
    "swerex.exceptions.SwerexException: Runtime did not start within timeout\n"
)


def test_classify_swerex_failure_build_failed():
    assert seal_mod._classify_swerex_failure(_BUILD_FAILED_STDOUT, "", 0) == "build_failed"


def test_classify_swerex_failure_port_in_use_beats_runtime():
    # The port-collision stdout contains BOTH "Container process terminated"
    # and "address already in use"; the specific port tag must win.
    assert seal_mod._classify_swerex_failure(_PORT_IN_USE_STDOUT, "", 0) == "port_in_use"


def test_classify_swerex_failure_runtime_not_alive():
    assert (
        seal_mod._classify_swerex_failure(_RUNTIME_NOT_ALIVE_STDOUT, "", 0)
        == "runtime_not_alive"
    )


def test_classify_swerex_failure_timeout_returncode():
    assert seal_mod._classify_swerex_failure("agent ran a long time", "", 124) == "timeout"


def test_classify_swerex_failure_timeout_text():
    assert (
        seal_mod._classify_swerex_failure("", "[seal] sweagent timed out after 3600s", 1)
        == "timeout"
    )


def test_classify_swerex_failure_none_for_benign():
    assert seal_mod._classify_swerex_failure("sweagent finished, no patch", "", 0) == ""


def test_classify_swerex_failure_plain_nonzero_is_not_build_failed():
    # A generic "returned non-zero exit status 1" WITHOUT a docker-build /
    # _build_image marker must NOT be misclassified as build_failed.
    assert (
        seal_mod._classify_swerex_failure(
            "CalledProcessError: Command '['pytest']' returned non-zero exit status 1.",
            "",
            1,
        )
        == ""
    )


def test_retriable_infra_tags_excludes_build_failed_and_timeout():
    # Lock the retry policy: build_failed (deterministic) and timeout
    # (expensive) are NOT in the default retriable set.
    assert "build_failed" not in seal_mod._RETRIABLE_INFRA_TAGS
    assert "timeout" not in seal_mod._RETRIABLE_INFRA_TAGS
    assert seal_mod._RETRIABLE_INFRA_TAGS == frozenset(
        {"port_in_use", "runtime_not_alive"}
    )


def _make_sequenced_invoke(sequence: list[tuple[int, str, str, bool]]):
    """Build a fake ``_invoke_sweagent`` that returns ``sequence[i]`` on
    call ``i`` (clamped to the last entry). Each entry is
    ``(returncode, stdout, stderr, write_pred)``; when ``write_pred`` is
    True a valid (non-empty) .pred is written into that call's
    ``--output_dir`` so the harvest path finds a Submitted patch.
    Returns ``(fake, state)`` where ``state["n"]`` counts invocations.
    """
    state = {"n": 0}

    async def fake(cmd, *, cwd, env, timeout_seconds):
        i = state["n"]
        state["n"] += 1
        rc, out, err, write_pred = sequence[min(i, len(sequence) - 1)]
        if write_pred:
            idx = cmd.index("--output_dir")
            output_dir = Path(cmd[idx + 1])
            inst_id = "instance_flipt-io__flipt-abc"
            pred = output_dir / "r" / inst_id / f"{inst_id}.pred"
            pred.parent.mkdir(parents=True, exist_ok=True)
            pred.write_text(
                json.dumps({"model_patch": "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"})
            )
        return rc, out, err

    return fake, state


@pytest.mark.asyncio
async def test_execute_seal_retries_transient_noprediction(submodules_present, monkeypatch):
    """A leaked-port NoPrediction is retried; the retry recovers it."""
    fake, state = _make_sequenced_invoke(
        [
            (0, _PORT_IN_USE_STDOUT, "", False),  # attempt 1: port clash, no pred
            (0, "sweagent ok", "", True),          # attempt 2: succeeds
        ]
    )
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake)
    adapter = _StubProAdapter(
        evaluate_result={"pass": True, "resolved": True, "patch_successfully_applied": True}
    )
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),  # default seal_max_retries = 1
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert state["n"] == 2  # retried exactly once
    assert result.evaluation["seal_attempts"] == 2
    assert result.evaluation["exit_status"] == "Submitted"
    assert result.status == TaskStatus.PASS
    assert len(adapter.evaluate_calls) == 1
    # recovering attempt produced a pred → no residual infra error.
    assert result.evaluation["swerex_infra_error"] is None


@pytest.mark.asyncio
async def test_execute_seal_does_not_retry_build_failed(submodules_present, monkeypatch):
    """Deterministic Alpine/musl build failures are never retried."""
    fake, state = _make_sequenced_invoke([(0, _BUILD_FAILED_STDOUT, "", False)])
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake)
    adapter = _StubProAdapter()
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_max_retries=2),  # retries available but unused
        adapter=adapter,
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert state["n"] == 1  # NOT retried despite seal_max_retries=2
    assert result.evaluation["seal_attempts"] == 1
    assert result.evaluation["swerex_infra_error"] == "build_failed"
    assert result.evaluation["exit_status"] == "NoPrediction"
    assert result.status == TaskStatus.ERROR
    assert adapter.evaluate_calls == []


@pytest.mark.asyncio
async def test_execute_seal_retry_exhausted_transient(submodules_present, monkeypatch):
    """A persistently-transient failure is retried then given up on."""
    fake, state = _make_sequenced_invoke([(0, _PORT_IN_USE_STDOUT, "", False)])  # always fails
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake)
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_max_retries=1),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert state["n"] == 2  # 1 initial + 1 retry, then stop
    assert result.evaluation["seal_attempts"] == 2
    assert result.evaluation["swerex_infra_error"] == "port_in_use"
    assert result.evaluation["exit_status"] == "NoPrediction"
    assert result.status == TaskStatus.ERROR


@pytest.mark.asyncio
async def test_execute_seal_happy_path_stamps_infra_none_attempts_one(
    submodules_present, monkeypatch
):
    """First-try success → swerex_infra_error None, seal_attempts 1."""
    fake, state = _make_sequenced_invoke([(0, "ok", "", True)])
    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake)
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(),
        adapter=_StubProAdapter(evaluate_result={"pass": True}),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert state["n"] == 1
    assert result.evaluation["seal_attempts"] == 1
    assert result.evaluation["swerex_infra_error"] is None


@pytest.mark.asyncio
async def test_execute_seal_empty_prediction_not_retried(submodules_present, monkeypatch):
    """An EmptyPrediction (pred exists, patch null) is a model outcome, not
    infra — it must NOT be retried even if stdout carries transient noise,
    and swerex_infra_error must be None (swerex stood up fine)."""
    state = {"n": 0}

    async def fake(cmd, *, cwd, env, timeout_seconds):
        state["n"] += 1
        idx = cmd.index("--output_dir")
        output_dir = Path(cmd[idx + 1])
        inst_id = "instance_flipt-io__flipt-abc"
        pred = output_dir / "r" / inst_id / f"{inst_id}.pred"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text(json.dumps({"model_patch": None}))
        # Incidental transient-looking noise in stdout — must be ignored
        # because a pred WAS produced.
        return 0, _PORT_IN_USE_STDOUT, ""

    monkeypatch.setattr(seal_mod, "_invoke_sweagent", fake)
    result = await execute_seal(
        task=_pro_task(),
        config=_seal_config(seal_max_retries=3),
        adapter=_StubProAdapter(),
        llm=None,
        strategy=None,
        memory=None,
        sandbox_mgr=_FakeSandboxManager(),
    )
    assert state["n"] == 1  # pred exists → loop breaks, no retry
    assert result.evaluation["exit_status"] == "EmptyPrediction"
    assert result.evaluation["seal_attempts"] == 1
    assert result.evaluation["swerex_infra_error"] is None
