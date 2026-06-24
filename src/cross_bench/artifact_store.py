from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Not serializable: {type(obj)}")


_SECRET_KEY_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password")


def _safe_task_filename(task_id: str) -> str:
    """Flatten path-like task IDs to a single filename component.

    Preserves the original spelling everywhere except where the OS would
    misinterpret a separator. The full ``task_id`` still appears verbatim
    inside the JSON payload, so downstream consumers don't lose
    information.
    """
    return task_id.replace("/", "__").replace("\\", "__")


def _redact_secrets(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            kl = str(k).lower()
            if any(frag in kl for frag in _SECRET_KEY_FRAGMENTS) and isinstance(v, str) and v:
                out[k] = "<redacted>"
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(data, list):
        return [_redact_secrets(x) for x in data]
    return data


class ArtifactStore:
    """Writes run artifacts to local disk, optionally mirroring to S3.

    Layout under both backends is identical::

        {s3_prefix}{run_id}/config.json
        {s3_prefix}{run_id}/summary.json
        {s3_prefix}{run_id}/tasks/{task_id}.json

    ``s3_prefix`` is a key prefix applied only on the S3 side (local
    disk always uses the flat ``{run_id}/...`` layout — the local copy
    is a working cache). Empty prefix preserves the pre-2026-05-25 flat
    layout byte-for-byte for back-compat. Setting it to e.g.
    ``"runs/swebench_pro/"`` produces
    ``s3://{bucket}/runs/swebench_pro/{run_id}/...``, which makes
    ``aws s3 ls s3://{bucket}/runs/swebench_pro/`` list every Pro run
    without needing to know any UUID.

    When ``backend="s3"``, every write also uploads to
    ``s3://{bucket}/{s3_prefix}{key}``. The local copy stays at
    ``{local_dir}/{run_id}/`` and is the source of truth on disk.
    """

    def __init__(
        self,
        backend: str = "local",
        bucket: str = "",
        local_dir: str = "./artifacts",
        s3_prefix: str = "",
    ):
        # ``local_dir`` is a LOCAL filesystem path. An ``s3://...`` URL
        # collapses to ``s3:/...`` under ``pathlib.Path`` and then
        # ``mkdir(parents=True)`` happily creates a literal ``./s3:/...``
        # tree on disk — which fills the orchestrator's ephemeral disk
        # after a few task JSONs. Fail loud at construction instead.
        # S3 mirroring is configured via ``backend="s3"`` + ``bucket``,
        # not via the URL shape of ``local_dir``.
        if str(local_dir).startswith(("s3://", "s3:/")):
            raise ValueError(
                f"ArtifactStore(local_dir={local_dir!r}) — local_dir must be a "
                "local filesystem path. Pass backend='s3' + bucket=... to "
                "mirror to S3."
            )
        self._backend = backend
        self._bucket = bucket
        self._local_dir = Path(local_dir)
        # Normalise prefix so it never starts with '/' and always ends
        # with '/' (or is the empty string for back-compat).
        prefix = (s3_prefix or "").lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self._s3_prefix = prefix
        self._s3 = None
        if backend == "s3":
            if not bucket:
                raise ValueError(
                    "ArtifactStore(backend='s3', ...) requires a non-empty bucket"
                )
            import boto3

            self._s3 = boto3.client("s3")

    def _ensure_dir(self, *parts: str) -> Path:
        p = self._local_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _rel_key(self, path: Path) -> str:
        return path.relative_to(self._local_dir).as_posix()

    def _s3_key(self, path: Path) -> str:
        """Local path → full S3 key under the configured prefix.

        Prefix-empty case is byte-identical to ``_rel_key`` so callers
        without a prefix configured see the legacy flat layout.
        """
        return f"{self._s3_prefix}{self._rel_key(path)}"

    def _write_json(self, path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_serialize)
        logger.debug("artifact.saved path=%s", str(path))
        if self._s3 is not None:
            key = self._s3_key(path)
            self._s3.upload_file(str(path), self._bucket, key)
            logger.debug("artifact.s3_uploaded bucket=%s key=%s", self._bucket, key)

    async def store_run_config(self, config: Any) -> None:
        run_dir = self._ensure_dir(config.run_id)
        self._write_json(run_dir / "config.json", _redact_secrets(asdict(config)))

    async def store_run_summary(self, run_result: Any) -> None:
        run_dir = self._ensure_dir(run_result.run_id)
        # Include a redacted copy of the config so dashboards / compare
        # views can read model + memory provider without a second fetch.
        # ``config.json`` remains the source of truth.
        try:
            config_payload: Any = _redact_secrets(asdict(run_result.config))
        except Exception:
            config_payload = {}
        data = {
            "run_id": run_result.run_id,
            "benchmark": getattr(run_result.config, "benchmark", ""),
            "status": getattr(run_result.status, "value", str(run_result.status)),
            "aggregate_metrics": run_result.aggregate_metrics,
            "started_at": getattr(run_result, "started_at", None),
            "completed_at": getattr(run_result, "completed_at", None),
            "config": config_payload,
        }
        self._write_json(run_dir / "summary.json", data)

    async def store_task_result(self, run_id: str, result: Any) -> None:
        task_dir = self._ensure_dir(run_id, "tasks")
        # Upstream task IDs occasionally contain ``/`` (e.g. ``HumanEval/0``)
        # or ``\\``. Treating them as path separators would scatter task
        # JSONs into ad-hoc subdirs whose parents we never created — and
        # break ``list_task_results`` which globs only the top of
        # ``tasks/``. Sanitize to a single flat filename; the original
        # ``task_id`` is preserved inside the JSON itself.
        self._write_json(
            task_dir / f"{_safe_task_filename(result.task_id)}.json", asdict(result)
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    async def retrieve(self, key: str) -> str:
        """Return the JSON text for the given artifact ``key``.

        ``key`` is the path relative to the run root, e.g.
        ``"runs/abc/summary.json"`` or ``"abc/summary.json"``. For
        backwards compatibility both shapes are accepted — the
        ``"runs/"`` prefix is stripped when resolving local paths.
        """
        local_rel = key[len("runs/"):] if key.startswith("runs/") else key
        local_path = self._local_dir / local_rel
        if local_path.exists():
            return local_path.read_text()
        if self._s3 is not None:
            obj = self._s3.get_object(Bucket=self._bucket, Key=local_rel)
            return obj["Body"].read().decode()
        raise FileNotFoundError(local_path)

    async def list_task_results(self, run_id: str) -> list[dict[str, Any]]:
        """Return every ``tasks/*.json`` for ``run_id`` as parsed dicts.

        Prefers S3 when configured so a merger running outside the ECS
        container can pick up shard results that were never written to
        the merger's local disk. Honors ``s3_prefix`` so a merger
        configured with ``s3_prefix='runs/swebench_pro/'`` finds the
        right shards under the navigable layout.
        """
        results: list[dict[str, Any]] = []
        if self._s3 is not None:
            prefix = f"{self._s3_prefix}{run_id}/tasks/"
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    body = self._s3.get_object(Bucket=self._bucket, Key=obj["Key"])
                    results.append(json.loads(body["Body"].read()))
            return results

        tasks_dir = self._local_dir / run_id / "tasks"
        if not tasks_dir.exists():
            return results
        for p in sorted(tasks_dir.glob("*.json")):
            results.append(json.loads(p.read_text()))
        return results
