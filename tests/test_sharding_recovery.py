"""Tests for shard-manifest resilience: upfront write + --discover read.

Locks the recovery path used after the 2026-05-11 Mastra incident
(orchestrator SIGKILLed mid-run, no auto-merge, manual recovery
required hunting shard UUIDs in CloudWatch). With these helpers,
recovery is one CLI command:

    cross-bench merge-shards --parent <run_id> --discover --benchmark <name>

because the manifest is written BEFORE any ECS task is dispatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross_bench.artifact_store import ArtifactStore
from cross_bench.sharding import (
    _shard_run_id,
    read_shards_manifest,
    write_shards_manifest_upfront,
)


@pytest.fixture
def local_store(tmp_path: Path) -> ArtifactStore:
    """ArtifactStore writing to a temp local dir (no S3)."""
    return ArtifactStore(local_dir=str(tmp_path), bucket=None)


# ---------------------------------------------------------------------------
# write_shards_manifest_upfront: contract
# ---------------------------------------------------------------------------


def test_write_upfront_creates_shards_json(local_store: ArtifactStore, tmp_path: Path):
    """``shards.json`` lands in ``{parent_run_id}/shards.json``."""
    shard_ids = write_shards_manifest_upfront(
        parent_run_id="parent-A",
        conversation_ids=["conv-1", "conv-2", "conv-3"],
        store=local_store,
    )
    manifest_path = tmp_path / "parent-A" / "shards.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["parent_run_id"] == "parent-A"
    assert data["shard_run_ids"] == shard_ids
    assert data["conversation_ids"] == ["conv-1", "conv-2", "conv-3"]
    assert "dispatched_at" in data


def test_write_upfront_shard_ids_match_dispatch(local_store: ArtifactStore):
    """The shard IDs in the manifest must match what ``dispatch_shards``
    would compute, otherwise recovery merges the wrong shards."""
    shard_ids = write_shards_manifest_upfront(
        parent_run_id="parent-B",
        conversation_ids=["conv-26", "conv-30"],
        store=local_store,
    )
    expected = [
        _shard_run_id("parent-B", "conv-26"),
        _shard_run_id("parent-B", "conv-30"),
    ]
    assert shard_ids == expected


def test_write_upfront_optional_metadata(local_store: ArtifactStore, tmp_path: Path):
    """``benchmark`` and ``profiles`` land in the manifest when supplied."""
    write_shards_manifest_upfront(
        parent_run_id="parent-C",
        conversation_ids=["conv-1"],
        store=local_store,
        benchmark="locomo_original",
        profiles=["online", "mastra_embedded"],
    )
    data = json.loads((tmp_path / "parent-C" / "shards.json").read_text())
    assert data["benchmark"] == "locomo_original"
    assert data["profiles"] == ["online", "mastra_embedded"]


def test_write_upfront_omits_optional_when_absent(
    local_store: ArtifactStore, tmp_path: Path
):
    """No ``benchmark``/``profiles`` keys when caller didn't supply them.
    Keeps the manifest minimal for callers that don't need them."""
    write_shards_manifest_upfront(
        parent_run_id="parent-D",
        conversation_ids=["conv-1"],
        store=local_store,
    )
    data = json.loads((tmp_path / "parent-D" / "shards.json").read_text())
    assert "benchmark" not in data
    assert "profiles" not in data


# ---------------------------------------------------------------------------
# read_shards_manifest: round-trip + back-compat
# ---------------------------------------------------------------------------


async def test_read_roundtrip(local_store: ArtifactStore):
    """``read_shards_manifest`` returns what ``write_shards_manifest_upfront`` wrote."""
    write_shards_manifest_upfront(
        parent_run_id="parent-E",
        conversation_ids=["conv-1", "conv-2"],
        store=local_store,
        benchmark="locomo_original",
    )
    data = await read_shards_manifest("parent-E", local_store)
    assert data["parent_run_id"] == "parent-E"
    assert len(data["shard_run_ids"]) == 2
    assert data["benchmark"] == "locomo_original"


async def test_read_accepts_post_merge_shape(local_store: ArtifactStore, tmp_path: Path):
    """``merge_shards`` overwrites the upfront manifest with the richer
    post-merge shape (adds ``per_shard_task_counts`` / ``total_tasks``).
    ``read_shards_manifest`` must accept both."""
    # Simulate the post-merge shape written by ``merge_shards``.
    parent_dir = tmp_path / "parent-F"
    parent_dir.mkdir()
    (parent_dir / "shards.json").write_text(json.dumps({
        "parent_run_id": "parent-F",
        "shard_run_ids": ["s1", "s2"],
        "per_shard_task_counts": {"s1": 199, "s2": 105},
        "total_tasks": 304,
    }))
    data = await read_shards_manifest("parent-F", local_store)
    assert data["shard_run_ids"] == ["s1", "s2"]
    assert data["total_tasks"] == 304


async def test_read_missing_manifest_raises_filenotfound(local_store: ArtifactStore):
    """Missing manifest → ``FileNotFoundError`` so the CLI can surface
    a clear recovery message."""
    with pytest.raises(FileNotFoundError):
        await read_shards_manifest("nonexistent-parent", local_store)


async def test_read_malformed_manifest_raises_valueerror(
    local_store: ArtifactStore, tmp_path: Path
):
    """Manifest exists but lacks ``shard_run_ids`` → ValueError."""
    parent_dir = tmp_path / "parent-G"
    parent_dir.mkdir()
    (parent_dir / "shards.json").write_text(json.dumps({
        "parent_run_id": "parent-G",
        "some_unrelated_key": "garbage",
    }))
    with pytest.raises(ValueError, match="shard_run_ids"):
        await read_shards_manifest("parent-G", local_store)


# ---------------------------------------------------------------------------
# Determinism: upfront write + dispatch yield identical shard IDs
# ---------------------------------------------------------------------------


def test_shard_run_id_determinism():
    """``_shard_run_id`` MUST be deterministic — that's what makes the
    upfront manifest valid. Same (parent, conv) → same UUID5 forever."""
    a = _shard_run_id("locomo-mastra-20260511-093717", "conv-26")
    b = _shard_run_id("locomo-mastra-20260511-093717", "conv-26")
    assert a == b
    # Different parent → different UUID.
    c = _shard_run_id("different-parent", "conv-26")
    assert a != c
    # Different conv → different UUID.
    d = _shard_run_id("locomo-mastra-20260511-093717", "conv-30")
    assert a != d
