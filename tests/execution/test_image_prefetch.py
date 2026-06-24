"""Tests for the image pre-pull pre-flight (deviations.md S4).

Two layers:

1. ``SandboxManager.prefetch_images`` — unit-level coverage with the
   docker client monkeypatched. Verifies parallel pull dispatch,
   already-local short-circuit, error tolerance, and the return
   ``{image: status}`` contract.

2. Engine integration — a recording double for the sandbox manager
   asserts that ``_run_code_gen`` calls ``prefetch_images`` once with
   the deduplicated union of ``adapter.get_sandbox_image(task)`` across
   the task list, BEFORE the task loop runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.execution.engine import RunEngine
from cross_bench.execution.sandbox import SandboxManager
from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
)
from cross_bench.systems.base import LLMDriver, MemoryMiddleware, PromptStrategy


# ---------------------------------------------------------------------------
# 1. SandboxManager.prefetch_images — direct unit tests
# ---------------------------------------------------------------------------


class _FakeImageNotFound(Exception):
    """Stand-in for docker.errors.ImageNotFound — duck-typed by class name."""


class _FakeImagesAPI:
    """Records every get / pull call. Lets each test pre-program responses."""

    def __init__(
        self,
        present: set[str] | None = None,
        pull_raises: dict[str, Exception] | None = None,
        get_raises: dict[str, Exception] | None = None,
    ) -> None:
        self.present = set(present or [])
        self.pull_raises = pull_raises or {}
        self.get_raises = get_raises or {}
        self.pulled: list[str] = []
        self.got: list[str] = []

    def get(self, name: str):
        self.got.append(name)
        if name in self.get_raises:
            raise self.get_raises[name]
        if name in self.present:
            return object()
        raise _FakeImageNotFound(name)

    def pull(self, name: str):
        self.pulled.append(name)
        if name in self.pull_raises:
            raise self.pull_raises[name]
        return object()


class _FakeDockerClient:
    def __init__(self, images: _FakeImagesAPI) -> None:
        self.images = images
        self.containers = None  # never used by prefetch_images


@pytest.fixture
def _patch_docker(monkeypatch):
    """Install a fake ``docker.from_env`` + a fake ``docker.errors.ImageNotFound``
    so ``SandboxManager.__init__`` and ``prefetch_images`` don't need real
    docker. Returns the test's images fake so each test can pre-program it."""
    images = _FakeImagesAPI()

    fake_docker = type("DockerModule", (), {})()
    fake_docker.from_env = lambda: _FakeDockerClient(images)
    fake_docker.errors = type("Errors", (), {"ImageNotFound": _FakeImageNotFound})()

    import sys
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    return images


@pytest.mark.asyncio
async def test_prefetch_pulls_uncached_image(_patch_docker):
    """A never-seen image gets a ``pull`` call and status ``pulled``."""
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images(["sweb.eval.x86_64.django__django-11133:latest"])
    assert statuses == {
        "sweb.eval.x86_64.django__django-11133:latest": "pulled"
    }
    assert _patch_docker.pulled == [
        "sweb.eval.x86_64.django__django-11133:latest"
    ]


@pytest.mark.asyncio
async def test_prefetch_skips_already_local_image(_patch_docker):
    """``images.get`` succeeding means the image is local — no pull."""
    _patch_docker.present = {"sweb.eval.x86_64.django__django-11133:latest"}
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images([
        "sweb.eval.x86_64.django__django-11133:latest"
    ])
    assert statuses == {
        "sweb.eval.x86_64.django__django-11133:latest": "already_local"
    }
    assert _patch_docker.pulled == []  # no pull issued


@pytest.mark.asyncio
async def test_prefetch_dedupes_and_skips_empty(_patch_docker):
    """Duplicate images pulled once; empty string ignored."""
    mgr = SandboxManager()
    images = [
        "img-a:latest",
        "img-a:latest",  # duplicate
        "img-b:latest",
        "",  # empty — conversational adapter convention
    ]
    statuses = await mgr.prefetch_images(images)
    assert set(statuses.keys()) == {"img-a:latest", "img-b:latest"}
    # Exactly one pull per unique image.
    assert sorted(_patch_docker.pulled) == ["img-a:latest", "img-b:latest"]


@pytest.mark.asyncio
async def test_prefetch_failure_doesnt_block_other_pulls(_patch_docker):
    """One image's pull error must not abort the batch."""
    _patch_docker.pull_raises = {"broken:latest": RuntimeError("auth denied")}
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images(["broken:latest", "good:latest"])
    assert statuses["good:latest"] == "pulled"
    assert statuses["broken:latest"].startswith("error:RuntimeError:auth denied")
    # ``good:latest`` was still attempted.
    assert "good:latest" in _patch_docker.pulled


@pytest.mark.asyncio
async def test_prefetch_empty_input_returns_empty_dict(_patch_docker):
    mgr = SandboxManager()
    assert await mgr.prefetch_images([]) == {}
    assert await mgr.prefetch_images(["", "", ""]) == {}
    assert _patch_docker.pulled == []


# ---------------------------------------------------------------------------
# 2. Engine integration — _run_code_gen calls prefetch_images pre-loop
# ---------------------------------------------------------------------------


class _StubLLM(LLMDriver):
    async def complete(
        self, prompt: str, config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return "ok", StepRecord(
            step_index=0, action="completion", input_text=prompt, output_text="ok",
        )

    async def chat(
        self, messages: list[dict[str, str]], config: dict[str, Any]
    ) -> tuple[str, StepRecord]:
        return "ok", StepRecord(
            step_index=0, action="chat", input_text=str(messages), output_text="ok",
        )


class _PassthroughStrategy(PromptStrategy):
    def format(
        self,
        task: Task,
        benchmark_prompt: str,
        memory_context: str | None = None,
    ) -> str | list[dict[str, str]]:
        return benchmark_prompt


class _PerInstanceImageAdapter(BenchmarkAdapter):
    """Mimics SWE-bench: a different image per task. Engine should dedupe."""
    name = "test-per-instance"
    granularity = TaskGranularity.REPO

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        for tid in params["task_ids"]:
            yield Task(
                task_id=tid,
                benchmark=self.name,
                granularity=self.granularity,
                prompt="prompt",
                metadata={"image": f"img:{tid}"},
                ground_truth="",
            )

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        return {"pass": True}

    def get_sandbox_image(self, task: Task | None = None) -> str:
        if task is None:
            return "default:latest"
        return task.metadata["image"]


class _RecordingSandboxManager:
    """Stand-in that records the prefetch call and serves create/destroy."""

    def __init__(self) -> None:
        self.prefetch_calls: list[set[str]] = []

    async def prefetch_images(self, images, *, concurrency: int = 4):
        # Record AS A SET so test assertions don't depend on iter order.
        self.prefetch_calls.append(set(images))
        return {img: "pulled" for img in images}

    async def create(self, image: str, task: Task, **_: Any) -> Any:
        return None

    async def destroy(self, sandbox: Any) -> None:
        pass

    async def execute(self, sandbox: Any, cmd: str, timeout: int = 60) -> Any:
        raise AssertionError("No sandbox exec expected in these tests")


class _NoPrefetchSandboxManager:
    """Older-shape fake — no prefetch_images attribute. Engine must tolerate."""

    async def create(self, image: str, task: Task, **_: Any) -> Any:
        return None

    async def destroy(self, sandbox: Any) -> None:
        pass

    async def execute(self, sandbox: Any, cmd: str, timeout: int = 60) -> Any:
        raise AssertionError("No sandbox exec expected in these tests")


class _FakeArtifactStore:
    async def store_run_config(self, config: Any) -> None:
        pass

    async def store_task_result(self, run_id: str, result: Any) -> None:
        pass

    async def store_run_summary(self, result: Any) -> None:
        pass


class _FakeResultRepo:
    async def create_run(self, config: Any) -> None:
        pass

    async def save_task_result(self, result: Any) -> None:
        pass

    async def complete_run(self, run_id: str, metrics: dict, completed_at: Any) -> None:
        pass


def _make_config(task_ids: list[str]) -> RunConfig:
    return RunConfig(
        run_id="test-prefetch",
        benchmark="test-per-instance",
        benchmark_params={"task_ids": task_ids},
        execution_mode=ExecutionMode.SINGLE_SHOT,
        system=SystemConfig(
            llm=LLMConfig(provider="stub", model="stub"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
            memory=None,
        ),
        timeout_seconds=10,
    )


@pytest.mark.asyncio
async def test_engine_calls_prefetch_with_deduped_image_union(monkeypatch):
    """Engine must call prefetch_images once with the unique images across tasks."""
    adapter = _PerInstanceImageAdapter()
    mgr = _RecordingSandboxManager()

    # Stub out the registry lookups to return our test components.
    import cross_bench.execution.engine as engine_mod
    monkeypatch.setattr(engine_mod, "get_adapter", lambda _: adapter)
    monkeypatch.setattr(engine_mod, "get_llm_driver", lambda _: _StubLLM())
    monkeypatch.setattr(
        engine_mod, "get_prompt_strategy", lambda _: _PassthroughStrategy()
    )
    # Stub execute_single_shot so we don't actually try to run.
    from cross_bench.models import TaskResult, TaskStatus

    async def _stub_executor(*, task, **_kw):
        return TaskResult(
            task_id=task.task_id,
            run_id="test-prefetch",
            status=TaskStatus.PASS,
            steps=[],
            final_output="",
            evaluation={"pass": True},
        )

    monkeypatch.setattr(engine_mod, "execute_single_shot", _stub_executor)

    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),
        result_repo=_FakeResultRepo(),
        sandbox_mgr=mgr,  # type: ignore[arg-type]
    )

    await engine.execute_run(_make_config(task_ids=["t1", "t2", "t3", "t2"]))

    # Exactly one prefetch call, with the unique image set.
    assert len(mgr.prefetch_calls) == 1
    assert mgr.prefetch_calls[0] == {"img:t1", "img:t2", "img:t3"}


@pytest.mark.asyncio
async def test_engine_tolerates_sandbox_manager_without_prefetch(monkeypatch):
    """Backwards compat: a fake without prefetch_images must not crash the run."""
    adapter = _PerInstanceImageAdapter()
    mgr = _NoPrefetchSandboxManager()

    import cross_bench.execution.engine as engine_mod
    monkeypatch.setattr(engine_mod, "get_adapter", lambda _: adapter)
    monkeypatch.setattr(engine_mod, "get_llm_driver", lambda _: _StubLLM())
    monkeypatch.setattr(
        engine_mod, "get_prompt_strategy", lambda _: _PassthroughStrategy()
    )

    from cross_bench.models import TaskResult, TaskStatus

    async def _stub_executor(*, task, **_kw):
        return TaskResult(
            task_id=task.task_id,
            run_id="test-prefetch-bc",
            status=TaskStatus.PASS,
            steps=[],
            final_output="",
            evaluation={"pass": True},
        )

    monkeypatch.setattr(engine_mod, "execute_single_shot", _stub_executor)

    engine = RunEngine(
        artifact_store=_FakeArtifactStore(),
        result_repo=_FakeResultRepo(),
        sandbox_mgr=mgr,  # type: ignore[arg-type]
    )

    # No exception expected.
    await engine.execute_run(_make_config(task_ids=["t1"]))


# ---------------------------------------------------------------------------
# 3. Dict-mode prefetch_images (deviations.md S4): pull pullable URL,
#    retag locally as bare name so the sandbox launcher finds it.
# ---------------------------------------------------------------------------


class _TaggableFakeImage:
    """A fake docker Image with a recordable ``tag()``."""

    def __init__(self) -> None:
        self.tag_calls: list[str] = []

    def tag(self, name: str) -> None:
        self.tag_calls.append(name)


class _FakeImagesAPIWithTagging(_FakeImagesAPI):
    """Extends the legacy fake with retag support.

    ``get(pullable_url)`` after a successful pull returns a tag-able
    image stand-in. Records every tag() call so tests can assert the
    bare local name was applied.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._after_pull: dict[str, _TaggableFakeImage] = {}
        self.tag_calls: list[tuple[str, str]] = []  # (source_url, local_name)

    def pull(self, name: str, **_kw: Any):  # noqa: ANN001 — kwargs ignored for compat
        super().pull(name)
        img = _TaggableFakeImage()
        self.present.add(name)
        self._after_pull[name] = img
        return img

    def get(self, name: str):
        self.got.append(name)
        if name in self._after_pull:
            class _Recorder:
                def __init__(self, parent, source):
                    self._parent = parent
                    self._source = source
                def tag(self, local_name):
                    self._parent.tag_calls.append((self._source, local_name))
            return _Recorder(self, name)
        if name in self.get_raises:
            raise self.get_raises[name]
        if name in self.present:
            return object()
        raise _FakeImageNotFound(name)


@pytest.fixture
def _patch_docker_tagging(monkeypatch):
    """Like ``_patch_docker`` but with the tag-aware fake."""
    images = _FakeImagesAPIWithTagging()
    fake_docker = type("DockerModule", (), {})()
    fake_docker.from_env = lambda: _FakeDockerClient(images)
    fake_docker.errors = type("Errors", (), {"ImageNotFound": _FakeImageNotFound})()
    import sys
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    return images


@pytest.mark.asyncio
async def test_dict_mode_pulls_url_and_retags_to_local(_patch_docker_tagging):
    """When ``local_name != pullable_url`` the fake records:
    (1) a ``pull`` of the URL, (2) a ``tag`` of that URL to the bare name."""
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images({
        "sweb.eval.x86_64.astropy__astropy-12907:latest":
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
    })
    assert statuses == {
        "sweb.eval.x86_64.astropy__astropy-12907:latest": "pulled"
    }
    # Pull happened against the pullable URL.
    assert _patch_docker_tagging.pulled == [
        "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    ]
    # Retag bridged source -> local.
    assert _patch_docker_tagging.tag_calls == [(
        "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        "sweb.eval.x86_64.astropy__astropy-12907:latest",
    )]


@pytest.mark.asyncio
async def test_dict_mode_already_local_skips_pull_and_retag(_patch_docker_tagging):
    """When the bare local name is already in the daemon's cache, prefetch
    must skip the registry entirely — no pull, no retag."""
    _patch_docker_tagging.present = {
        "sweb.eval.x86_64.astropy__astropy-12907:latest"
    }
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images({
        "sweb.eval.x86_64.astropy__astropy-12907:latest":
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
    })
    assert statuses == {
        "sweb.eval.x86_64.astropy__astropy-12907:latest": "already_local"
    }
    assert _patch_docker_tagging.pulled == []
    assert _patch_docker_tagging.tag_calls == []


@pytest.mark.asyncio
async def test_dict_mode_identity_pair_no_retag(_patch_docker_tagging):
    """When ``local_name == pullable_url`` we pull but skip the no-op retag."""
    mgr = SandboxManager()
    statuses = await mgr.prefetch_images({"img-a:latest": "img-a:latest"})
    assert statuses == {"img-a:latest": "pulled"}
    assert _patch_docker_tagging.pulled == ["img-a:latest"]
    assert _patch_docker_tagging.tag_calls == []  # retag skipped


# ---------------------------------------------------------------------------
# 4. _ecr_auth_config_for — ECR URL detection + boto3 token decode
# ---------------------------------------------------------------------------


def test_ecr_auth_skips_non_ecr_urls():
    """Docker Hub / local URLs return None — docker SDK falls back to its
    own credential resolution path."""
    from cross_bench.execution.sandbox import _ecr_auth_config_for
    assert _ecr_auth_config_for("docker.io/swebench/sweb.eval...:latest") is None
    assert _ecr_auth_config_for("cross-bench-function:latest") is None
    assert _ecr_auth_config_for("ghcr.io/foo/bar:1.0") is None


def test_ecr_auth_decodes_token_for_ecr_url(monkeypatch):
    """ECR URL triggers a boto3 ``get_authorization_token`` call and the
    returned base64 ``user:pass`` is decoded into ``{username, password}``."""
    import base64, sys
    fake_token = base64.b64encode(b"AWS:secret-token-xyz").decode()

    class _FakeECR:
        def get_authorization_token(self):
            return {"authorizationData": [{"authorizationToken": fake_token}]}

    class _FakeBoto3:
        def client(self, name, region_name=None):
            assert name == "ecr"
            assert region_name == "us-east-1"
            return _FakeECR()

    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3())

    from cross_bench.execution.sandbox import _ecr_auth_config_for
    auth = _ecr_auth_config_for(
        "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench:astropy__astropy-12907"
    )
    assert auth == {"username": "AWS", "password": "secret-token-xyz"}


def test_ecr_auth_returns_none_on_boto3_failure(monkeypatch):
    """boto3 client failure (no AWS creds, throttled, etc.) falls back to
    None so the pull attempt still runs (docker SDK may have its own creds)."""
    import sys

    class _FakeBoto3:
        def client(self, name, region_name=None):
            raise RuntimeError("no AWS creds available")

    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3())

    from cross_bench.execution.sandbox import _ecr_auth_config_for
    assert _ecr_auth_config_for(
        "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench:x"
    ) is None
