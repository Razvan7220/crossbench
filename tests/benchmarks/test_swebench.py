"""SWE-bench adapter tests — verifies the flow delegates correctly to swebench.

No Docker here. The tests stub :class:`SandboxManager` to record the exact
commands and file writes the adapter issues, and feed back canned stdout/stderr
so the grading path runs end-to-end. Where possible the assertions compare
against primitives imported from ``swebench.harness`` directly, so any upstream
change is caught.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

# Upstream ``swebench.harness`` imports the Unix-only ``resource`` module at
# package load time (via prepare_images.py). Stub it before the adapter does.
if "resource" not in sys.modules:
    try:
        import resource  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("resource")
        _stub.RLIMIT_NOFILE = 7
        _stub.getrlimit = lambda _r: (1024, 4096)
        _stub.setrlimit = lambda _r, _v: None
        sys.modules["resource"] = _stub

from cross_bench.benchmarks.swebench import (
    SWEBenchAdapter,
    _apply_patch_cascade,
    _section_rate,
    _task_to_instance,
)
from cross_bench.execution.sandbox import ExecutionResult, Sandbox
from cross_bench.models import Task, TaskGranularity

# ---------------------------------------------------------------------------
# Known-good instance: django__django-11133. Using the real FAIL_TO_PASS /
# PASS_TO_PASS values so the grading path exercises real parsing.
# ---------------------------------------------------------------------------
F2P_TESTS = [
    "test_memoryview_content (httpwrappers.tests.HttpResponseTests)",
]
P2P_TESTS = [
    "test_basic (httpwrappers.tests.HttpResponseTests)",
]


def _sample_task(
    *,
    patch_gt: str = "diff --git a/x b/x\n",
    test_patch: str = "diff --git a/tests b/tests\n",
) -> Task:
    return Task(
        task_id="django__django-11133",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="HttpResponse doesnt handle memoryview objects",
        metadata={
            "repo": "django/django",
            "base_commit": "879cc3d06ef6ba5e4a0d04a47e67c0f2bf2a84d2",
            "version": "3.0",
            "environment_setup_commit": "419a78300f7cd27611196e1e464d50fd0385ff27",
            "hints_text": "",
            "created_at": "2019-04-01T13:54:50Z",
            "difficulty": "<15 min fix",
            "test_patch": test_patch,
            "FAIL_TO_PASS": str(F2P_TESTS).replace("'", '"'),
            "PASS_TO_PASS": str(P2P_TESTS).replace("'", '"'),
        },
        ground_truth=patch_gt,
    )


# ---------------------------------------------------------------------------
# A fake SandboxManager that records every interaction and returns canned
# outputs driven by a queue. This lets us drive the adapter through the full
# happy / unhappy paths without any Docker at all.
# ---------------------------------------------------------------------------
@dataclass
class FakeSandboxManager:
    # tuples: (container_id, command, timeout) and (container_id, path, content)
    exec_responses: list[ExecutionResult] = field(default_factory=list)
    executed: list[tuple[str, str, int]] = field(default_factory=list)
    writes: list[tuple[str, str, str]] = field(default_factory=list)

    async def create(self, image: str, task: Any, **_: Any) -> Sandbox:
        return Sandbox(container_id=f"fake-{task.task_id}", image=image)

    async def execute(
        self, sandbox: Sandbox, command: str, timeout: int = 60
    ) -> ExecutionResult:
        self.executed.append((sandbox.container_id, command, timeout))
        if not self.exec_responses:
            raise AssertionError(f"Unexpected exec: {command}")
        return self.exec_responses.pop(0)

    async def write_file(self, sandbox: Sandbox, path: str, content: str) -> None:
        self.writes.append((sandbox.container_id, path, content))


def _ok() -> ExecutionResult:
    return ExecutionResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)


def _fail() -> ExecutionResult:
    return ExecutionResult(exit_code=1, stdout="", stderr="error", duration_ms=1.0)


# =============================================================================
# Registration + field preservation
# =============================================================================


def test_registered():
    from cross_bench.benchmarks import get_adapter
    a = get_adapter("swebench")
    assert isinstance(a, SWEBenchAdapter)
    assert a.name == "swebench"
    assert a.granularity == TaskGranularity.REPO


def test_default_params_variant_is_verified():
    """Default params: variant=verified, include_hints=False, retrieval=none.

    See deviations.md S2 (hints), S3 (retrieval), S4 (image_source),
    S12 (variant default).
    """
    assert SWEBenchAdapter().get_default_params() == {
        "variant": "verified",
        "include_hints": False,
        "retrieval": "none",
        "retrieval_bm25_token_budget": 27000,
        "image_source": "dockerhub_1776",
        "ecr_repo_uri": "",
        "pro_ecr_repo_uri": "",
    }


def test_format_prompt_omits_hints_when_include_hints_false():
    """Default WITHOUT-hints column: hints_text present in metadata but gate off."""
    task = _sample_task()
    task.metadata["hints_text"] = "Look at views.py:412"
    task.metadata["include_hints"] = False
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Hints:" not in rendered
    assert "views.py:412" not in rendered


def test_format_prompt_omits_hints_when_metadata_lacks_toggle():
    """Legacy task metadata without ``include_hints`` key defaults to off."""
    task = _sample_task()
    task.metadata["hints_text"] = "Look at views.py:412"
    # No include_hints key at all — defaults to falsy via .get().
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Hints:" not in rendered


def test_format_prompt_includes_hints_when_opted_in():
    """WITH-hints ceiling column: gate flipped, hints surface verbatim."""
    task = _sample_task()
    task.metadata["hints_text"] = "Look at views.py:412"
    task.metadata["include_hints"] = True
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Hints:" in rendered
    assert "views.py:412" in rendered


def test_format_prompt_no_hints_section_when_text_empty_even_if_toggled():
    """Empty hints_text → no Hints block, regardless of toggle (no header-without-body)."""
    task = _sample_task()
    task.metadata["hints_text"] = ""
    task.metadata["include_hints"] = True
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Hints:" not in rendered


def test_load_tasks_propagates_requires_network_for_verified_and_pro(monkeypatch):
    """``params.requires_network`` → task.metadata for every variant (S21 / BA sec.5)."""
    from cross_bench.benchmarks import swebench as swe

    verified_rows = _fake_swebench_rows()
    pro_rows = _fake_pro_rows()

    def _fake_load(ds_name, *args, **kwargs):
        if "SWE-bench_Pro" in ds_name:
            return _FakeHFDataset(pro_rows)
        return _FakeHFDataset(verified_rows)

    monkeypatch.setattr(swe, "load_dataset", _fake_load)

    verified_on = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "requires_network": True}
    ))
    assert verified_on[0].metadata["requires_network"] is True

    verified_off = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "requires_network": False}
    ))
    assert verified_off[0].metadata["requires_network"] is False

    pro_on = list(SWEBenchAdapter().load_tasks(
        {"variant": "pro", "requires_network": True}
    ))
    assert pro_on[0].metadata["requires_network"] is True


@pytest.mark.asyncio
async def test_requires_network_propagates_to_sandbox_bridge(monkeypatch):
    """Three-hop: load_tasks params → task.metadata → SandboxManager bridge."""
    from cross_bench.benchmarks import swebench as swe
    from cross_bench.execution.sandbox import SandboxManager

    rows = _fake_pro_rows()
    monkeypatch.setattr(
        swe, "load_dataset",
        lambda ds_name, *a, **k: _FakeHFDataset(rows),
    )

    run_kwargs: list[dict] = []

    class _Container:
        id = "cid"
        short_id = "cid"

    class _Containers:
        def run(self, **kwargs):
            run_kwargs.append(kwargs)
            return _Container()

    import sys

    client = type("Client", (), {"containers": _Containers()})()
    fake_docker = type("DockerModule", (), {})()
    fake_docker.from_env = lambda: client
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    tasks = list(SWEBenchAdapter().load_tasks(
        {"variant": "pro", "requires_network": True}
    ))
    task = tasks[0]
    assert task.metadata["requires_network"] is True

    mgr = SandboxManager()
    await mgr.create("jefzda/sweap-images:tag", task)
    assert run_kwargs[-1]["network_mode"] == "bridge"


def test_load_tasks_propagates_include_hints_into_task_metadata(monkeypatch):
    """``params.include_hints`` is stamped on every task so format_prompt can read it."""
    from cross_bench.benchmarks import swebench as swe

    fake_rows = _fake_swebench_rows()
    monkeypatch.setattr(swe, "load_dataset", _make_fake_load_dataset(fake_rows))

    tasks_off = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "include_hints": False}
    ))
    assert tasks_off[0].metadata["include_hints"] is False

    tasks_on = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "include_hints": True}
    ))
    assert tasks_on[0].metadata["include_hints"] is True


# =============================================================================
# Retrieval columns: none / bm25 / oracle (deviations.md S3)
# =============================================================================


class _FakeHFDataset:
    """Minimal stand-in for the HF ``Dataset`` shape we use: iteration,
    indexing, len, and ``select(range(...))`` returning another fake."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def select(self, indices):
        return _FakeHFDataset([self.rows[i] for i in indices])


def _fake_swebench_rows() -> list[dict]:
    return [
        {
            "instance_id": "django__django-1",
            "repo": "django/django",
            "base_commit": "abc",
            "problem_statement": "issue body",
            "patch": "diff …",
            "test_patch": "",
            "hints_text": "",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
            "version": "3.0",
            "environment_setup_commit": "",
            "created_at": "",
            "difficulty": "",
        }
    ]


def _make_fake_load_dataset(verified_rows, *, bm25_rows=None, oracle_rows=None):
    """Returns a stub ``load_dataset`` that dispatches by dataset name."""
    def _fake(ds_name, *args, **kwargs):
        if "Verified" in ds_name or "SWE-bench_Lite" in ds_name or ds_name.endswith("SWE-bench"):
            return _FakeHFDataset(verified_rows)
        if "bm25" in ds_name:
            return _FakeHFDataset(bm25_rows or [])
        if "oracle" in ds_name.lower():
            return _FakeHFDataset(oracle_rows or [])
        raise AssertionError(f"Unexpected dataset name: {ds_name}")
    return _fake


def test_default_params_include_retrieval_none():
    """``retrieval`` defaults to ``none`` to match the SWE-bench baseline column."""
    params = SWEBenchAdapter().get_default_params()
    assert params["retrieval"] == "none"
    assert params["retrieval_bm25_token_budget"] == 27000


def test_load_tasks_retrieval_none_skips_augmented_dataset(monkeypatch):
    """retrieval=none: no augmented dataset loaded; metadata.retrieval_context empty."""
    from cross_bench.benchmarks import swebench as swe

    calls = []

    def _track_load(ds_name, *a, **k):
        calls.append(ds_name)
        return _FakeHFDataset(_fake_swebench_rows())

    monkeypatch.setattr(swe, "load_dataset", _track_load)

    tasks = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "retrieval": "none"}
    ))
    # Only the base Verified dataset is loaded.
    assert len(calls) == 1
    assert "Verified" in calls[0]
    assert tasks[0].metadata["retrieval"] == "none"
    assert tasks[0].metadata["retrieval_context"] == ""


def test_load_tasks_retrieval_bm25_loads_bm25_dataset(monkeypatch):
    """retrieval=bm25 loads SWE-bench_bm25_27K (default budget) AND extracts code block."""
    from cross_bench.benchmarks import swebench as swe

    bm25_rows = [{
        "instance_id": "django__django-1",
        "text": (
            "You will be provided with a partial code base and an issue statement.\n"
            "<issue>\nissue body\n</issue>\n"
            "<code>\n[start of foo/bar.py]\nclass Foo:\n    pass\n[end of foo/bar.py]\n</code>\n"
            "I need you to solve this issue by generating a patch file.\n"
        ),
    }]
    monkeypatch.setattr(
        swe, "load_dataset",
        _make_fake_load_dataset(_fake_swebench_rows(), bm25_rows=bm25_rows),
    )

    tasks = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "retrieval": "bm25"}
    ))
    md = tasks[0].metadata
    assert md["retrieval"] == "bm25"
    assert md["retrieval_bm25_token_budget"] == 27000
    # Extracted block — the <code> contents only.
    assert "[start of foo/bar.py]" in md["retrieval_context"]
    assert "class Foo:" in md["retrieval_context"]
    # Wrappers / instructions outside <code> are NOT carried.
    assert "<issue>" not in md["retrieval_context"]
    assert "You will be provided" not in md["retrieval_context"]


def test_load_tasks_retrieval_oracle_loads_oracle_dataset(monkeypatch):
    """retrieval=oracle loads SWE-bench_oracle."""
    from cross_bench.benchmarks import swebench as swe

    oracle_rows = [{
        "instance_id": "django__django-1",
        "text": (
            "<code>\n[start of models.py]\n# gold modified file\n[end of models.py]\n</code>\n"
        ),
    }]
    monkeypatch.setattr(
        swe, "load_dataset",
        _make_fake_load_dataset(_fake_swebench_rows(), oracle_rows=oracle_rows),
    )

    tasks = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "retrieval": "oracle"}
    ))
    md = tasks[0].metadata
    assert md["retrieval"] == "oracle"
    assert "# gold modified file" in md["retrieval_context"]


def test_load_tasks_retrieval_bm25_budget_selects_correct_dataset(monkeypatch):
    """retrieval_bm25_token_budget routes to the matching ``SWE-bench_bm25_<N>K``."""
    from cross_bench.benchmarks import swebench as swe

    seen: list[str] = []

    def _stub(ds_name, *a, **k):
        seen.append(ds_name)
        if "bm25" in ds_name:
            return _FakeHFDataset([])  # empty retrieval map is fine for this test
        return _FakeHFDataset(_fake_swebench_rows())

    monkeypatch.setattr(swe, "load_dataset", _stub)

    list(SWEBenchAdapter().load_tasks({
        "variant": "verified",
        "retrieval": "bm25",
        "retrieval_bm25_token_budget": 40000,
    }))
    assert any("bm25_40K" in s for s in seen)


def test_load_tasks_invalid_retrieval_mode_raises(monkeypatch):
    """Unknown ``retrieval`` value is a hard error (no silent fallback)."""
    from cross_bench.benchmarks import swebench as swe

    monkeypatch.setattr(
        swe, "load_dataset",
        _make_fake_load_dataset(_fake_swebench_rows()),
    )
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        list(SWEBenchAdapter().load_tasks({
            "variant": "verified",
            "retrieval": "magic",
        }))


def test_load_tasks_invalid_bm25_budget_raises(monkeypatch):
    """Off-list BM25 token budget is a hard error (no silent fallback to 27K)."""
    from cross_bench.benchmarks import swebench as swe

    monkeypatch.setattr(
        swe, "load_dataset",
        _make_fake_load_dataset(_fake_swebench_rows()),
    )
    with pytest.raises(ValueError, match="retrieval_bm25_token_budget"):
        list(SWEBenchAdapter().load_tasks({
            "variant": "verified",
            "retrieval": "bm25",
            "retrieval_bm25_token_budget": 12345,
        }))


def test_load_tasks_retrieval_miss_yields_empty_context(monkeypatch):
    """Instance not in the augmented dataset → empty retrieval_context (honest miss)."""
    from cross_bench.benchmarks import swebench as swe

    # Verified row: django__django-1. BM25 dataset only covers a DIFFERENT id.
    monkeypatch.setattr(
        swe, "load_dataset",
        _make_fake_load_dataset(
            _fake_swebench_rows(),
            bm25_rows=[{
                "instance_id": "django__django-99",  # not in our Verified set
                "text": "<code>\nnoise\n</code>\n",
            }],
        ),
    )

    tasks = list(SWEBenchAdapter().load_tasks(
        {"variant": "verified", "retrieval": "bm25"}
    ))
    assert tasks[0].metadata["retrieval_context"] == ""


def test_format_prompt_injects_retrieval_block_when_present():
    """retrieval_context non-empty → <code> block appears in the rendered prompt."""
    task = _sample_task()
    task.metadata["retrieval"] = "bm25"
    task.metadata["retrieval_context"] = (
        "[start of foo.py]\nFOO_FILE_CONTENT\n[end of foo.py]"
    )
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Retrieved files (bm25):" in rendered
    assert "<code>" in rendered
    assert "FOO_FILE_CONTENT" in rendered
    assert "</code>" in rendered


def test_format_prompt_omits_retrieval_block_when_empty():
    """retrieval_context empty → no Retrieved-files section (no header-without-body)."""
    task = _sample_task()
    task.metadata["retrieval"] = "none"
    task.metadata["retrieval_context"] = ""
    rendered = SWEBenchAdapter().format_prompt(task)
    assert "Retrieved files" not in rendered
    assert "<code>" not in rendered


def test_extract_retrieval_block_handles_missing_block():
    """No ``<code>`` block in upstream text → empty extraction (no crash)."""
    from cross_bench.benchmarks.swebench import _extract_retrieval_block

    # Real published rows always have a <code> block, but our guard is the
    # safety net.
    assert _extract_retrieval_block("just prose, no block") == ""
    assert _extract_retrieval_block("") == ""


def test_extract_retrieval_block_extracts_multiline_content():
    """The block can span many lines incl. blank ones — re.DOTALL handles it."""
    from cross_bench.benchmarks.swebench import _extract_retrieval_block

    text = (
        "<code>\n"
        "line one\n"
        "\n"
        "line three\n"
        "</code>\n"
    )
    assert _extract_retrieval_block(text) == "line one\n\nline three"


def test_task_to_instance_round_trip_preserves_all_fields():
    """All 12 SWEbenchInstance fields must survive adapter → instance conversion."""
    t = _sample_task()
    inst = _task_to_instance(t)

    required_fields = {
        "instance_id", "repo", "base_commit", "patch", "test_patch",
        "problem_statement", "hints_text", "created_at", "version",
        "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
    }
    assert required_fields <= set(inst.keys()), (
        f"missing: {required_fields - set(inst.keys())}"
    )
    assert inst["instance_id"] == "django__django-11133"
    assert inst["repo"] == "django/django"
    assert inst["test_patch"] == "diff --git a/tests b/tests\n"
    assert inst["FAIL_TO_PASS"]
    assert inst["PASS_TO_PASS"]


# =============================================================================
# TestSpec delegation: image key + eval_script come from swebench
# =============================================================================


def test_instance_image_key_matches_upstream_testspec():
    """The image name must be exactly what swebench's TestSpec produces."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    inst = _task_to_instance(_sample_task())
    spec = make_test_spec(inst)
    assert spec.instance_image_key == "sweb.eval.x86_64.django__django-11133:latest"


def test_instance_image_key_lowercases_instance_id():
    """Upstream calls ``self.instance_id.lower()``; uppercase ids must flatten."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    t = _sample_task()
    t.task_id = "Django__Django-11133"
    spec = make_test_spec(_task_to_instance(t))
    assert spec.instance_image_key.endswith("django__django-11133:latest")


def test_eval_script_contains_markers_and_base_commit():
    """``spec.eval_script`` drives the whole test run; it must contain the
    markers the adapter splits on and the instance's base commit."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(_task_to_instance(_sample_task()))
    assert ">>>>> Start Test Output" in spec.eval_script
    assert ">>>>> End Test Output" in spec.eval_script
    assert "879cc3d06ef6ba5e4a0d04a47e67c0f2bf2a84d2" in spec.eval_script
    assert "/testbed" in spec.eval_script


# =============================================================================
# Patch-apply cascade
# =============================================================================


@pytest.mark.asyncio
async def test_patch_cascade_first_command_succeeds():
    mgr = FakeSandboxManager(exec_responses=[_ok()])
    sb = Sandbox(container_id="c", image="img")
    ok = await _apply_patch_cascade(mgr, sb, "diff …")
    assert ok is True
    assert len(mgr.executed) == 1
    assert mgr.executed[0][1].startswith("cd /testbed && git apply --verbose")


@pytest.mark.asyncio
async def test_patch_cascade_falls_through_to_patch_fuzz():
    mgr = FakeSandboxManager(exec_responses=[_fail(), _fail(), _ok()])
    sb = Sandbox(container_id="c", image="img")
    ok = await _apply_patch_cascade(mgr, sb, "diff …")
    assert ok is True
    cmds = [c for _, c, _ in mgr.executed]
    assert "git apply --verbose /" in cmds[0]
    assert "git apply --verbose --reject" in cmds[1]
    assert "patch --batch --fuzz=5 -p1 -i" in cmds[2]


@pytest.mark.asyncio
async def test_patch_cascade_all_fail_returns_false():
    mgr = FakeSandboxManager(exec_responses=[_fail(), _fail(), _fail()])
    sb = Sandbox(container_id="c", image="img")
    ok = await _apply_patch_cascade(mgr, sb, "diff …")
    assert ok is False
    assert len(mgr.executed) == 3


@pytest.mark.asyncio
async def test_patch_cascade_writes_patch_to_tmp():
    mgr = FakeSandboxManager(exec_responses=[_ok()])
    sb = Sandbox(container_id="c", image="img")
    await _apply_patch_cascade(mgr, sb, "PATCHBODY")
    assert len(mgr.writes) == 1
    _, path, content = mgr.writes[0]
    assert path.startswith("/tmp/")
    assert content == "PATCHBODY"


# =============================================================================
# Section-rate helper mirrors swebench.harness.grading.compute_{f2p,p2p}
# =============================================================================


def test_section_rate_empty_is_one():
    """Matches upstream: vacuous success for an empty F2P or P2P set."""
    assert _section_rate({"FAIL_TO_PASS": {"success": [], "failure": []}}, "FAIL_TO_PASS") == 1.0


def test_section_rate_partial():
    rep = {"PASS_TO_PASS": {"success": ["a", "b"], "failure": ["c", "d"]}}
    assert _section_rate(rep, "PASS_TO_PASS") == 0.5


def test_section_rate_all_pass():
    rep = {"FAIL_TO_PASS": {"success": ["a", "b", "c"], "failure": []}}
    assert _section_rate(rep, "FAIL_TO_PASS") == 1.0


# =============================================================================
# End-to-end evaluate() with stubbed sandbox — exercises the real grading path.
# =============================================================================


def _django_test_log(f2p_status: str, p2p_status: str) -> str:
    """Produce a Django test-log snippet the upstream parser will parse.

    The django parser emits lines like ``test_name (class) ... ok`` /
    ``test_name (class) ... FAIL``. We inject the canonical test names from
    ``F2P_TESTS`` / ``P2P_TESTS`` with the requested statuses.
    """
    def _line(name: str, status: str) -> str:
        return f"{name} ... {status}"

    return "\n".join([
        _line(F2P_TESTS[0], f2p_status),
        _line(P2P_TESTS[0], p2p_status),
    ])


def _eval_stdout(body: str) -> str:
    """Wrap a parser-ready log body in swebench's start/end markers."""
    return f">>>>> Start Test Output\n{body}\n>>>>> End Test Output\n"


@pytest.mark.asyncio
async def test_evaluate_empty_prediction_short_circuits():
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager()
    out = await adapter.evaluate(_sample_task(), "", mgr, None)
    assert out["pass"] is False
    assert out["resolved"] is False
    assert out["error"] == "empty_prediction"
    # No container interactions at all — short-circuit.
    assert mgr.executed == []
    assert mgr.writes == []


@pytest.mark.asyncio
async def test_evaluate_all_tests_pass_marks_resolved():
    """F2P passes + P2P passes → resolved=True, pass=True, f2p=p2p=1.0."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),  # git apply for prediction patch
        ExecutionResult(
            exit_code=0,
            stdout=_eval_stdout(_django_test_log("ok", "ok")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["pass"] is True
    assert out["resolved"] is True
    assert out["patch_successfully_applied"] is True
    assert out["f2p_rate"] == 1.0
    assert out["p2p_rate"] == 1.0
    assert out["exit_code"] == 0
    assert out["tests_status"]["FAIL_TO_PASS"]["success"] == F2P_TESTS
    assert out["tests_status"]["PASS_TO_PASS"]["success"] == P2P_TESTS


@pytest.mark.asyncio
async def test_evaluate_f2p_fails_not_resolved():
    """F2P still failing → resolved=False (and f2p_rate=0)."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=1,
            stdout=_eval_stdout(_django_test_log("FAIL", "ok")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["resolved"] is False
    assert out["pass"] is False
    assert out["f2p_rate"] == 0.0
    assert out["p2p_rate"] == 1.0
    assert out["tests_status"]["FAIL_TO_PASS"]["failure"] == F2P_TESTS


@pytest.mark.asyncio
async def test_evaluate_p2p_regresses_not_resolved():
    """F2P passes but P2P regressed → resolved=False."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=1,
            stdout=_eval_stdout(_django_test_log("ok", "FAIL")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["resolved"] is False
    assert out["f2p_rate"] == 1.0
    assert out["p2p_rate"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_patch_apply_failure_short_circuits():
    """All 3 apply commands fail → error=patch_apply_failed, no eval_script run."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[_fail(), _fail(), _fail()])
    out = await adapter.evaluate(_sample_task(), "broken diff", mgr, None)
    assert out["pass"] is False
    assert out["patch_successfully_applied"] is False
    assert out["error"] == "patch_apply_failed"
    # 3 attempts recorded, but no eval.sh execute.
    assert len(mgr.executed) == 3
    assert not any("eval.sh" in e[1] for e in mgr.executed)


@pytest.mark.asyncio
async def test_evaluate_missing_markers_reports_error():
    """If the eval script output has no markers, we surface a specific error."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=1,
            stdout="stuff that never printed markers",
            stderr="conda: command not found",
            duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["error"] == "missing_output_markers"
    assert out["patch_successfully_applied"] is True


@pytest.mark.asyncio
async def test_evaluate_writes_upstream_eval_script_to_container():
    """The script written to /eval.sh must be the exact ``spec.eval_script``."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    adapter = SWEBenchAdapter()
    task = _sample_task()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=0,
            stdout=_eval_stdout(_django_test_log("ok", "ok")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    await adapter.evaluate(task, "diff --git a/x b/x\n", mgr, None)

    expected = make_test_spec(_task_to_instance(task)).eval_script
    eval_writes = [w for w in mgr.writes if w[1] == "/eval.sh"]
    assert len(eval_writes) == 1
    assert eval_writes[0][2] == expected


@pytest.mark.asyncio
async def test_evaluate_sandbox_workdir_is_testbed():
    """Adapter must rewrite the sandbox work_dir to /testbed for SWE-bench."""
    adapter = SWEBenchAdapter()
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=0,
            stdout=_eval_stdout(_django_test_log("ok", "ok")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    sb = await mgr.create("img", _sample_task())
    await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, sb)
    assert sb.work_dir == "/testbed"


@pytest.mark.asyncio
async def test_evaluate_returns_full_stdout_no_truncation():
    """deviations.md S5: stdout/stderr must NOT be truncated in the eval dict."""
    adapter = SWEBenchAdapter()
    # Long stdout — bigger than the old 4000-byte cap so we'd see truncation.
    long_stdout = (
        _eval_stdout(_django_test_log("ok", "ok"))
        + "\n" + ("x" * 6000)
    )
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=0, stdout=long_stdout, stderr="", duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["stdout"] == long_stdout
    assert len(out["stdout"]) > 4000


@pytest.mark.asyncio
async def test_evaluate_returns_full_stderr_no_truncation():
    adapter = SWEBenchAdapter()
    long_stderr = "warning: " + ("y" * 8000)
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=0,
            stdout=_eval_stdout(_django_test_log("ok", "ok")),
            stderr=long_stderr,
            duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["stderr"] == long_stderr
    assert len(out["stderr"]) > 4000


@pytest.mark.asyncio
async def test_evaluate_missing_markers_path_preserves_full_logs():
    """The missing-markers error branch also returns full logs (no truncation)."""
    adapter = SWEBenchAdapter()
    long_stdout = "z" * 12000
    long_stderr = "w" * 12000
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=1, stdout=long_stdout, stderr=long_stderr, duration_ms=1.0,
        ),
    ])
    out = await adapter.evaluate(_sample_task(), "diff --git a/x b/x\n", mgr, None)
    assert out["error"] == "missing_output_markers"
    assert out["stdout"] == long_stdout
    assert out["stderr"] == long_stderr


@pytest.mark.asyncio
async def test_evaluate_honours_timeout_metadata_override():
    """``_timeout_seconds`` in task metadata overrides the 1800 s default."""
    adapter = SWEBenchAdapter()
    task = _sample_task()
    task.metadata["_timeout_seconds"] = 42
    mgr = FakeSandboxManager(exec_responses=[
        _ok(),
        ExecutionResult(
            exit_code=0,
            stdout=_eval_stdout(_django_test_log("ok", "ok")),
            stderr="",
            duration_ms=1.0,
        ),
    ])
    await adapter.evaluate(task, "diff --git a/x b/x\n", mgr, None)
    eval_exec = [e for e in mgr.executed if "eval.sh" in e[1]]
    assert len(eval_exec) == 1
    assert eval_exec[0][2] == 42


# =============================================================================
# Image-source resolution: dockerhub_1776 / ecr_mirror / local_build
# (deviations.md S4)
# =============================================================================


from cross_bench.benchmarks.swebench import (  # noqa: E402
    _dockerhub_url_for,
    _ecr_url_for,
    _split_bare_image,
)


class TestSplitBareImage:
    def test_x86_64_instance(self):
        arch, instance, tag = _split_bare_image(
            "sweb.eval.x86_64.astropy__astropy-12907:latest"
        )
        assert arch == "x86_64"
        assert instance == "astropy__astropy-12907"
        assert tag == "latest"

    def test_arm64_instance(self):
        arch, instance, tag = _split_bare_image(
            "sweb.eval.arm64.django__django-11133:v1.2"
        )
        assert arch == "arm64"
        assert instance == "django__django-11133"
        assert tag == "v1.2"

    def test_rejects_non_swebench_image(self):
        with pytest.raises(ValueError, match="Not a SWE-bench image tag"):
            _split_bare_image("cross-bench-function:latest")


class TestDockerHubUrl:
    def test_underscore_to_1776(self):
        """``__`` -> ``_1776_`` translation matches Princeton's publishing convention."""
        assert _dockerhub_url_for(
            "sweb.eval.x86_64.astropy__astropy-12907:latest"
        ) == (
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        )

    def test_preserves_arch_and_tag(self):
        assert _dockerhub_url_for(
            "sweb.eval.x86_64.django__django-11133:v2"
        ) == (
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11133:v2"
        )


class TestEcrUrl:
    def test_uses_instance_as_tag(self):
        """ECR mirror uses the bare ``__`` form as the tag (no separator swap)."""
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        assert _ecr_url_for(
            "sweb.eval.x86_64.astropy__astropy-12907:latest", ecr
        ) == f"{ecr}:astropy__astropy-12907"


class TestResolveImageSources:
    def test_dockerhub_1776_default(self):
        """Default param ``image_source = dockerhub_1776`` maps to Hub URLs."""
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["sweb.eval.x86_64.astropy__astropy-12907:latest"],
            {"image_source": "dockerhub_1776"},
        )
        assert mapping == {
            "sweb.eval.x86_64.astropy__astropy-12907:latest":
                "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        }

    def test_ecr_mirror_uses_ecr_repo_uri_param(self):
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["sweb.eval.x86_64.django__django-11133:latest"],
            {"image_source": "ecr_mirror", "ecr_repo_uri": ecr},
        )
        assert mapping == {
            "sweb.eval.x86_64.django__django-11133:latest":
                f"{ecr}:django__django-11133"
        }

    def test_ecr_mirror_falls_back_to_env_var(self, monkeypatch):
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        monkeypatch.setenv("BENCH_SWEBENCH_ECR_REPO", ecr)
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["sweb.eval.x86_64.django__django-11133:latest"],
            {"image_source": "ecr_mirror"},
        )
        assert mapping == {
            "sweb.eval.x86_64.django__django-11133:latest":
                f"{ecr}:django__django-11133"
        }

    def test_ecr_mirror_requires_repo_uri(self, monkeypatch):
        monkeypatch.delenv("BENCH_SWEBENCH_ECR_REPO", raising=False)
        with pytest.raises(ValueError, match="ecr_repo_uri"):
            SWEBenchAdapter().resolve_image_sources(
                ["sweb.eval.x86_64.django__django-11133:latest"],
                {"image_source": "ecr_mirror"},
            )

    def test_local_build_identity_map(self):
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["sweb.eval.x86_64.django__django-11133:latest"],
            {"image_source": "local_build"},
        )
        assert mapping == {
            "sweb.eval.x86_64.django__django-11133:latest":
                "sweb.eval.x86_64.django__django-11133:latest"
        }

    def test_unknown_image_source_raises(self):
        with pytest.raises(ValueError, match="Unknown image_source"):
            SWEBenchAdapter().resolve_image_sources(
                ["sweb.eval.x86_64.django__django-11133:latest"],
                {"image_source": "bogus"},
            )

    def test_env_var_overrides_when_param_missing(self, monkeypatch):
        """``BENCH_SWEBENCH_IMAGE_SOURCE`` env var picks the mode when param
        absent — lets the ECS task definition flip the AWS default to
        ``ecr_mirror`` without touching the benchmark TOML."""
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        monkeypatch.setenv("BENCH_SWEBENCH_IMAGE_SOURCE", "ecr_mirror")
        monkeypatch.setenv("BENCH_SWEBENCH_ECR_REPO", ecr)
        # ``image_source`` is NOT in params — env var should drive.
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["sweb.eval.x86_64.django__django-11133:latest"],
            {},
        )
        assert mapping == {
            "sweb.eval.x86_64.django__django-11133:latest":
                f"{ecr}:django__django-11133"
        }


# =============================================================================
# Pro variant — Scale AI's SWE-bench_Pro (deviations.md S21).
#
# Pro diverges from Verified in five places: dataset name, lowercase
# fail_to_pass/pass_to_pass, dockerhub_tag-based image naming, custom
# eval harness (not Princeton's swebench PyPI), and no published
# BM25/oracle augmented datasets. The tests below pin each branch.
# =============================================================================


def _fake_pro_rows() -> list[dict]:
    """One representative Pro row covering all required fields."""
    return [
        {
            "instance_id": "instance_flipt-io__flipt-deadbeef",
            "repo": "flipt-io/flipt",
            "repo_language": "go",
            "base_commit": "abc1234",
            "problem_statement": "Pro issue text",
            "patch": "diff --git a/x b/x\n@@\n+ok",
            "test_patch": "",
            "fail_to_pass": "['TestA', 'TestB']",
            "pass_to_pass": "['TestC']",
            "selected_test_files_to_run": "['./pkg/x_test.go']",
            "before_repo_set_cmd": "cd /app\nmake setup",
            "dockerhub_tag": "flipt-io.flipt-deadbeef",
            "requirements": "",
            "interface": "",
        }
    ]


class TestProVariantLoadTasks:
    """``variant=pro`` flips field casing + adds Pro-specific metadata."""

    def test_load_tasks_pro_propagates_lowercase_fields(self, monkeypatch):
        from cross_bench.benchmarks import swebench as swe

        rows = _fake_pro_rows()
        monkeypatch.setattr(
            swe, "load_dataset",
            lambda ds_name, *a, **k: _FakeHFDataset(rows),
        )
        tasks = list(SWEBenchAdapter().load_tasks({"variant": "pro"}))
        assert len(tasks) == 1
        md = tasks[0].metadata
        assert md["variant"] == "pro"
        assert md["fail_to_pass"] == "['TestA', 'TestB']"
        assert md["pass_to_pass"] == "['TestC']"
        # Uppercase aliases preserved for downstream code that read those.
        assert md["FAIL_TO_PASS"] == md["fail_to_pass"]
        assert md["PASS_TO_PASS"] == md["pass_to_pass"]
        assert md["selected_test_files_to_run"] == "['./pkg/x_test.go']"
        assert md["before_repo_set_cmd"] == "cd /app\nmake setup"
        assert md["dockerhub_tag"] == "flipt-io.flipt-deadbeef"
        assert md["repo_language"] == "go"
        assert md["requires_network"] is False

    def test_load_tasks_pro_propagates_requires_network_param(self, monkeypatch):
        from cross_bench.benchmarks import swebench as swe

        monkeypatch.setattr(
            swe, "load_dataset",
            lambda ds_name, *a, **k: _FakeHFDataset(_fake_pro_rows()),
        )
        tasks = list(SWEBenchAdapter().load_tasks(
            {"variant": "pro", "requires_network": True}
        ))
        assert tasks[0].metadata["requires_network"] is True

    def test_load_tasks_pro_rejects_bm25_retrieval(self, monkeypatch):
        """Pro has no published BM25/oracle datasets — only retrieval=none works."""
        from cross_bench.benchmarks import swebench as swe

        rows = _fake_pro_rows()
        monkeypatch.setattr(
            swe, "load_dataset",
            lambda ds_name, *a, **k: _FakeHFDataset(rows),
        )
        with pytest.raises(ValueError, match="retrieval.*pro"):
            list(SWEBenchAdapter().load_tasks(
                {"variant": "pro", "retrieval": "bm25"}
            ))

    def test_load_tasks_pro_rejects_oracle_retrieval(self, monkeypatch):
        from cross_bench.benchmarks import swebench as swe

        rows = _fake_pro_rows()
        monkeypatch.setattr(
            swe, "load_dataset",
            lambda ds_name, *a, **k: _FakeHFDataset(rows),
        )
        with pytest.raises(ValueError, match="retrieval.*pro"):
            list(SWEBenchAdapter().load_tasks(
                {"variant": "pro", "retrieval": "oracle"}
            ))

    def test_load_tasks_unknown_variant_raises(self, monkeypatch):
        from cross_bench.benchmarks import swebench as swe

        monkeypatch.setattr(swe, "load_dataset", lambda *a, **k: _FakeHFDataset([]))
        with pytest.raises(ValueError, match="Unknown SWE-bench variant"):
            list(SWEBenchAdapter().load_tasks({"variant": "bogus"}))


class TestProSandboxImage:
    """Pro images come from ``jefzda/sweap-images:<dockerhub_tag>``."""

    def test_pro_image_uses_dockerhub_tag(self):
        from cross_bench.models import Task, TaskGranularity
        task = Task(
            task_id="instance_flipt-io__flipt-deadbeef",
            benchmark="swebench",
            granularity=TaskGranularity.REPO,
            prompt="",
            metadata={
                "variant": "pro",
                "dockerhub_tag": "flipt-io.flipt-deadbeef",
                "repo": "flipt-io/flipt",
                "base_commit": "abc",
            },
            ground_truth="",
        )
        assert SWEBenchAdapter().get_sandbox_image(task) == (
            "jefzda/sweap-images:flipt-io.flipt-deadbeef"
        )

    def test_pro_image_missing_dockerhub_tag_raises(self):
        from cross_bench.models import Task, TaskGranularity
        task = Task(
            task_id="instance_x",
            benchmark="swebench",
            granularity=TaskGranularity.REPO,
            prompt="",
            metadata={"variant": "pro", "dockerhub_tag": ""},
            ground_truth="",
        )
        with pytest.raises(ValueError, match="dockerhub_tag"):
            SWEBenchAdapter().get_sandbox_image(task)


class TestProResolveImageSources:
    """Pro image URL routing across the three image_source modes.

    Pro images live under ``jefzda/sweap-images:<dockerhub_tag>`` on
    Docker Hub. They translate to ``<pro_ecr_repo_uri>:<dockerhub_tag>``
    when an ECR mirror is configured for Pro, else pass through.
    """

    def test_pro_image_passes_through_dockerhub_mode(self):
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["jefzda/sweap-images:flipt-io.flipt-deadbeef"],
            {"image_source": "dockerhub_1776"},
        )
        assert mapping == {
            "jefzda/sweap-images:flipt-io.flipt-deadbeef":
                "jefzda/sweap-images:flipt-io.flipt-deadbeef"
        }

    def test_pro_image_passes_through_ecr_mode_without_pro_ecr(self):
        """When pro_ecr_repo_uri is unset, Pro images fall back to Docker Hub.

        Noisy on AWS (NAT egress, anon-pull rate limits) but functional.
        Verified images in the same dispatch still use ECR.
        """
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["jefzda/sweap-images:flipt-io.flipt-deadbeef"],
            {"image_source": "ecr_mirror", "ecr_repo_uri": ecr},
        )
        assert mapping == {
            "jefzda/sweap-images:flipt-io.flipt-deadbeef":
                "jefzda/sweap-images:flipt-io.flipt-deadbeef"
        }

    def test_pro_image_translates_to_pro_ecr_when_set(self):
        """Configured pro_ecr_repo_uri → Pro images route to the Pro mirror."""
        pro_ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench-pro"
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["jefzda/sweap-images:flipt-io.flipt-deadbeef"],
            {
                "image_source": "ecr_mirror",
                "pro_ecr_repo_uri": pro_ecr,
                # ecr_repo_uri (Verified) not needed for Pro-only dispatch
                "ecr_repo_uri": "",
            },
        )
        assert mapping == {
            "jefzda/sweap-images:flipt-io.flipt-deadbeef":
                f"{pro_ecr}:flipt-io.flipt-deadbeef"
        }

    def test_pro_ecr_env_var_overrides_when_param_missing(self, monkeypatch):
        """``BENCH_SWEBENCH_PRO_ECR_REPO`` env var picks the Pro ECR.

        Lets the ECS task definition flip the AWS default without
        touching the benchmark TOML — same pattern as the Verified
        ``BENCH_SWEBENCH_ECR_REPO`` env var.
        """
        pro_ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench-pro"
        monkeypatch.setenv("BENCH_SWEBENCH_PRO_ECR_REPO", pro_ecr)
        mapping = SWEBenchAdapter().resolve_image_sources(
            ["jefzda/sweap-images:flipt-io.flipt-deadbeef"],
            {"image_source": "ecr_mirror"},  # no pro_ecr_repo_uri param
        )
        assert mapping == {
            "jefzda/sweap-images:flipt-io.flipt-deadbeef":
                f"{pro_ecr}:flipt-io.flipt-deadbeef"
        }

    def test_mixed_pro_and_verified_in_one_dispatch(self):
        """Both image kinds in one call route to their respective ECRs."""
        ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench"
        pro_ecr = "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench-pro"
        mapping = SWEBenchAdapter().resolve_image_sources(
            [
                "sweb.eval.x86_64.django__django-11133:latest",
                "jefzda/sweap-images:flipt-io.flipt-deadbeef",
            ],
            {
                "image_source": "ecr_mirror",
                "ecr_repo_uri": ecr,
                "pro_ecr_repo_uri": pro_ecr,
            },
        )
        assert mapping == {
            "sweb.eval.x86_64.django__django-11133:latest":
                f"{ecr}:django__django-11133",
            "jefzda/sweap-images:flipt-io.flipt-deadbeef":
                f"{pro_ecr}:flipt-io.flipt-deadbeef",
        }


class TestProEvalHelpers:
    """Pure-function helpers in swebench_pro_eval — no sandbox required."""

    def test_strip_binary_hunks_drops_binary_section(self):
        from cross_bench.benchmarks.swebench_pro_eval import _strip_binary_hunks
        patch = (
            "diff --git a/text.go b/text.go\n"
            "@@\n"
            "+ok\n"
            "diff --git a/img.png b/img.png\n"
            "Binary files a/img.png and b/img.png differ\n"
        )
        cleaned = _strip_binary_hunks(patch)
        assert "text.go" in cleaned
        assert "img.png" not in cleaned

    def test_strip_binary_hunks_drops_git_binary_patch(self):
        from cross_bench.benchmarks.swebench_pro_eval import _strip_binary_hunks
        patch = (
            "diff --git a/x b/x\n@@\n+ok\n"
            "diff --git a/blob b/blob\nGIT binary patch\n"
            "literal 100\ndeadbeef\n"
        )
        cleaned = _strip_binary_hunks(patch)
        assert "blob" not in cleaned
        assert "+ok" in cleaned

    def test_parse_selected_test_files_handles_stringified_list(self):
        from cross_bench.benchmarks.swebench_pro_eval import (
            _parse_selected_test_files,
        )
        assert _parse_selected_test_files("['a', 'b', 'c']") == ["a", "b", "c"]
        assert _parse_selected_test_files("[]") == []
        assert _parse_selected_test_files("") == []
        assert _parse_selected_test_files(None) == []
        # Real lists pass through.
        assert _parse_selected_test_files(["a", "b"]) == ["a", "b"]

    def test_parse_test_id_set_returns_set(self):
        from cross_bench.benchmarks.swebench_pro_eval import _parse_test_id_set
        assert _parse_test_id_set("['TestA', 'TestB']") == {"TestA", "TestB"}
        assert _parse_test_id_set("[]") == set()

    @pytest.mark.asyncio
    async def test_evaluate_pro_task_empty_required_tests_is_explicit_error(self):
        """Empty fail_to_pass + pass_to_pass → loud error, not silent False.

        Every published Pro row ships at least one fail_to_pass test, so
        an empty union means the row's metadata failed to parse. The
        guard must fire BEFORE any run-scripts clone or sandbox work —
        ``sandbox_mgr=None`` proves nothing downstream is touched. See
        deviations.md S25.
        """
        from cross_bench.benchmarks.swebench_pro_eval import evaluate_pro_task

        out = await evaluate_pro_task(
            instance_id="dummy__instance-1",
            metadata={"fail_to_pass": "[]", "pass_to_pass": "[]"},
            patch="diff --git a/x b/x\n+fix\n",
            sandbox_mgr=None,  # would raise AttributeError if reached
            sandbox=None,
        )
        assert out["error"] == "no_required_tests_parsed"
        assert out["pass"] is False
        assert out["resolved"] is False
        assert out["patch_successfully_applied"] is False

    @pytest.mark.asyncio
    async def test_evaluate_pro_task_missing_test_keys_is_explicit_error(self):
        """Metadata lacking the test-list keys entirely takes the same path."""
        from cross_bench.benchmarks.swebench_pro_eval import evaluate_pro_task

        out = await evaluate_pro_task(
            instance_id="dummy__instance-2",
            metadata={},
            patch="diff --git a/x b/x\n+fix\n",
            sandbox_mgr=None,
            sandbox=None,
        )
        assert out["error"] == "no_required_tests_parsed"
        assert out["resolved"] is False

    def test_create_entryscript_uses_last_line_of_before_cmd(self):
        """Upstream takes ``before_repo_set_cmd.strip().split('\\n')[-1]``."""
        from cross_bench.benchmarks.swebench_pro_eval import _create_entryscript
        es = _create_entryscript({
            "before_repo_set_cmd": "echo earlier_steps\necho cd /something\nmake setup",
            "selected_test_files_to_run": "['./x_test.go']",
            "base_commit": "abc123",
        })
        # Only the last line ("make setup") makes it in.
        assert "make setup" in es
        assert "echo earlier_steps" not in es

    def test_create_entryscript_executes_run_script_then_parser(self):
        from cross_bench.benchmarks.swebench_pro_eval import _create_entryscript
        es = _create_entryscript({
            "before_repo_set_cmd": "make setup",
            "selected_test_files_to_run": "['./a_test.go','./b_test.go']",
            "base_commit": "deadbeef",
        })
        # Reset → checkout → apply → before_cmd → run_script → parser.
        assert "git reset --hard deadbeef" in es
        assert "git checkout deadbeef" in es
        assert "git apply -v /workspace/patch.diff" in es
        assert "bash /workspace/run_script.sh ./a_test.go,./b_test.go" in es
        assert "python3 /workspace/parser.py" in es


class TestProRunScriptsResolution:
    """Verify the submodule-first → env override → runtime-clone fallback.

    The contract is documented in the module docstring of
    ``swebench_pro_eval``; this test class locks it. Drift here means
    a Pro run might silently use a different snapshot of upstream's
    run_scripts/ between local development (submodule) and AWS ECS
    (env override) — exactly the comparability footgun that
    deviations.md S21 / S22 exist to prevent.
    """

    def test_submodule_run_scripts_returns_none_outside_repo(
        self, tmp_path, monkeypatch
    ):
        """Without a .gitmodules marker anywhere up the tree, return None.

        This is the failure mode that triggers the runtime-clone
        fallback when CrossBench is installed via ``pip install
        cross-bench`` outside its source tree.
        """
        monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(tmp_path))
        from cross_bench.benchmarks.swebench_pro_eval import (
            _submodule_run_scripts,
        )
        # tmp_path has no .gitmodules and no third_party/swebench_pro.
        assert _submodule_run_scripts() is None

    def test_submodule_run_scripts_returns_dir_when_populated(
        self, tmp_path, monkeypatch
    ):
        """When the submodule has run_scripts/, return its absolute path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".gitmodules").write_text("[submodule]")
        run_scripts = repo / "third_party" / "swebench_pro" / "run_scripts"
        run_scripts.mkdir(parents=True)
        (run_scripts / "instance_x").mkdir()
        monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(repo))
        from cross_bench.benchmarks.swebench_pro_eval import (
            _submodule_run_scripts,
        )
        result = _submodule_run_scripts()
        assert result == run_scripts.resolve()

    def test_submodule_run_scripts_returns_none_for_empty_dir(
        self, tmp_path, monkeypatch
    ):
        """An empty run_scripts/ (interrupted submodule init) is not usable."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".gitmodules").write_text("[submodule]")
        (repo / "third_party" / "swebench_pro" / "run_scripts").mkdir(
            parents=True
        )
        monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(repo))
        from cross_bench.benchmarks.swebench_pro_eval import (
            _submodule_run_scripts,
        )
        assert _submodule_run_scripts() is None

    @pytest.mark.asyncio
    async def test_resolve_uses_submodule_when_present(
        self, tmp_path, monkeypatch
    ):
        """Submodule is the first-preference path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".gitmodules").write_text("[submodule]")
        run_scripts = repo / "third_party" / "swebench_pro" / "run_scripts"
        run_scripts.mkdir(parents=True)
        (run_scripts / "instance_foo").mkdir()
        monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(repo))
        # Even with an env override set, the submodule wins.
        monkeypatch.setenv("CROSS_BENCH_PRO_SCRIPTS_DIR", str(tmp_path / "ignored"))
        from cross_bench.benchmarks.swebench_pro_eval import (
            _resolve_run_scripts_root,
        )
        result = await _resolve_run_scripts_root()
        assert result == run_scripts.resolve()

    @pytest.mark.asyncio
    async def test_resolve_uses_env_override_when_submodule_missing(
        self, tmp_path, monkeypatch
    ):
        """Submodule absent → env override is the next-preference path."""
        # Force the repo-root walk to fail.
        repo = tmp_path / "norepo"
        repo.mkdir()
        monkeypatch.setenv("CROSS_BENCH_REPO_ROOT", str(repo))
        # Populate the env-override path with run_scripts/.
        env_root = tmp_path / "env_pro"
        env_run_scripts = env_root / "run_scripts"
        env_run_scripts.mkdir(parents=True)
        monkeypatch.setenv("CROSS_BENCH_PRO_SCRIPTS_DIR", str(env_root))
        from cross_bench.benchmarks.swebench_pro_eval import (
            _resolve_run_scripts_root,
        )
        result = await _resolve_run_scripts_root()
        assert result == env_run_scripts

