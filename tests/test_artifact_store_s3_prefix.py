"""S3 key-prefix routing for ArtifactStore.

Lets dispatches land under a navigable layout like
``s3://<bucket>/runs/swebench_pro/<run_id>/...`` so operators can
``aws s3 ls`` by benchmark instead of needing the exact run UUID.

Tests cover three things:
1. Empty prefix preserves the pre-2026-05-25 flat layout byte-for-byte.
2. Non-empty prefix prepends to every S3 key (write + list).
3. Prefix normalisation: leading slash stripped, trailing slash added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cross_bench.artifact_store import ArtifactStore


class _RecordingS3:
    """Stand-in for boto3 S3 client that records uploads + list calls."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []  # (local, bucket, key)
        self.list_prefixes_seen: list[str] = []
        self.objects: dict[str, bytes] = {}  # key -> body

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        self.uploads.append((local, bucket, key))
        self.objects[key] = Path(local).read_bytes()

    class _Paginator:
        def __init__(self, parent: "_RecordingS3") -> None:
            self._parent = parent

        def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
            self._parent.list_prefixes_seen.append(Prefix)
            contents = [
                {"Key": k} for k in self._parent.objects
                if k.startswith(Prefix)
            ]
            yield {"Contents": contents}

    def get_paginator(self, name: str) -> "_RecordingS3._Paginator":
        assert name == "list_objects_v2"
        return self._Paginator(self)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"Body": _FakeBody(self.objects[Key])}


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _make_store(tmp_path: Path, *, prefix: str = "") -> tuple[ArtifactStore, _RecordingS3]:
    """Build a store wired to a recording S3 client.

    We can't easily patch ``boto3.client`` inside ``__init__`` without
    a monkeypatch, so we construct with backend='local' and inject the
    recorder by hand. The ``_write_json`` path only checks
    ``self._s3 is not None`` to decide whether to upload, so this is a
    faithful exercise of the S3 write branch.
    """
    store = ArtifactStore(
        backend="local",  # avoid boto3 client construction
        bucket="test-bucket",
        local_dir=str(tmp_path),
        s3_prefix=prefix,
    )
    recorder = _RecordingS3()
    store._s3 = recorder  # type: ignore[attr-defined]
    store._bucket = "test-bucket"
    return store, recorder


@pytest.mark.asyncio
async def test_empty_prefix_preserves_flat_layout(tmp_path):
    """Default behaviour: S3 keys are exactly ``<run_id>/...``."""
    store, recorder = _make_store(tmp_path)  # no prefix

    from cross_bench.models import (
        ExecutionMode,
        LLMConfig,
        PromptStrategyConfig,
        RunConfig,
        SystemConfig,
    )
    config = RunConfig(
        run_id="abc-123",
        benchmark="swebench",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.MINI_SWE_AGENT,
    )
    await store.store_run_config(config)

    assert len(recorder.uploads) == 1
    _local, bucket, key = recorder.uploads[0]
    assert bucket == "test-bucket"
    assert key == "abc-123/config.json"


@pytest.mark.asyncio
async def test_nonempty_prefix_prepends_to_s3_key(tmp_path):
    """``runs/swebench_pro/`` prefix → keys land under that root."""
    store, recorder = _make_store(tmp_path, prefix="runs/swebench_pro/")

    from cross_bench.models import (
        ExecutionMode,
        LLMConfig,
        PromptStrategyConfig,
        RunConfig,
        SystemConfig,
    )
    config = RunConfig(
        run_id="abc-123",
        benchmark="swebench",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
        ),
        execution_mode=ExecutionMode.MINI_SWE_AGENT,
    )
    await store.store_run_config(config)

    _local, _bucket, key = recorder.uploads[0]
    assert key == "runs/swebench_pro/abc-123/config.json"


def test_prefix_normalisation_strips_leading_slash(tmp_path):
    """``/runs/x/`` and ``runs/x/`` produce identical layout."""
    s1 = ArtifactStore(backend="local", local_dir=str(tmp_path), s3_prefix="/runs/x/")
    s2 = ArtifactStore(backend="local", local_dir=str(tmp_path), s3_prefix="runs/x/")
    assert s1._s3_prefix == "runs/x/"  # type: ignore[attr-defined]
    assert s2._s3_prefix == "runs/x/"  # type: ignore[attr-defined]


def test_prefix_normalisation_adds_trailing_slash(tmp_path):
    """Missing trailing slash is added so keys join cleanly."""
    store = ArtifactStore(backend="local", local_dir=str(tmp_path), s3_prefix="runs/x")
    assert store._s3_prefix == "runs/x/"  # type: ignore[attr-defined]


def test_prefix_empty_string_stays_empty(tmp_path):
    """Empty prefix is preserved (used for the flat back-compat layout)."""
    store = ArtifactStore(backend="local", local_dir=str(tmp_path), s3_prefix="")
    assert store._s3_prefix == ""  # type: ignore[attr-defined]


def test_local_dir_s3_url_is_rejected(tmp_path):
    """``--artifacts-dir s3://bucket`` must fail loud, not silently
    create a ``./s3:/bucket/...`` tree on the orchestrator's disk."""
    with pytest.raises(ValueError, match="local_dir must be a local filesystem path"):
        ArtifactStore(backend="local", local_dir="s3://cross-bench-dev-artifacts-3684f16")
    with pytest.raises(ValueError, match="local_dir must be a local filesystem path"):
        ArtifactStore(backend="s3", bucket="cross-bench-dev-artifacts-3684f16",
                      local_dir="s3://cross-bench-dev-artifacts-3684f16")


@pytest.mark.asyncio
async def test_list_task_results_honors_prefix(tmp_path):
    """Listing shard tasks looks under ``<prefix><run_id>/tasks/``."""
    store, recorder = _make_store(tmp_path, prefix="runs/swebench_pro/")
    # Seed a fake task object directly in the recorder's bucket.
    recorder.objects["runs/swebench_pro/run-1/tasks/t1.json"] = json.dumps(
        {"task_id": "t1", "pass": True}
    ).encode()

    results = await store.list_task_results("run-1")
    assert recorder.list_prefixes_seen == ["runs/swebench_pro/run-1/tasks/"]
    assert results == [{"task_id": "t1", "pass": True}]
