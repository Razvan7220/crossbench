"""Docker sandbox management for safe code execution."""

from __future__ import annotations

import asyncio
import base64
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


# AWS ECR image URL: <12-digit-account>.dkr.ecr.<region>.amazonaws.com/...
_ECR_URL_RE = re.compile(
    r"^(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
)


def _ecr_auth_config_for(pullable_url: str) -> dict[str, str] | None:
    """Return docker-SDK ``auth_config`` for an ECR URL, else ``None``.

    Non-ECR URLs (Docker Hub, local builds, etc.) return ``None`` and
    the docker SDK falls back to whatever credentials the daemon has
    configured (or no auth for anonymous pulls).

    ECR URLs trigger a fresh ``boto3 ecr.get_authorization_token()``
    call so each pull carries a current 12-h IAM session token. This
    sidesteps the long-running-host failure mode where the daemon's
    boot-time ``docker login`` token expires mid-run.
    """
    m = _ECR_URL_RE.match(pullable_url)
    if not m:
        return None
    try:
        import boto3
    except ImportError:
        logger.warning(
            "sandbox.ecr_auth_skipped",
            reason="boto3 not installed",
            url=pullable_url,
        )
        return None
    region = m.group("region")
    try:
        ecr = boto3.client("ecr", region_name=region)
        token_resp = ecr.get_authorization_token()
        auth_data = token_resp["authorizationData"][0]
        decoded = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
        username, _, password = decoded.partition(":")
        return {"username": username, "password": password}
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "sandbox.ecr_auth_failed",
            url=pullable_url,
            error=f"{type(e).__name__}:{e}",
        )
        return None


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False


@dataclass
class Sandbox:
    container_id: str
    image: str
    work_dir: str = "/workspace"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxManager:
    """Manages Docker containers for sandboxed benchmark evaluation.

    Security posture: containers run with ``--network=none`` (unless the
    benchmark sets ``requires_network``), an explicit memory cap, and a
    pinned CPU budget. The user inside the container is whatever the
    benchmark image declares — most CrossBench images are still root,
    so OS-level isolation comes from the container boundary, not from
    a non-root in-container user. If you add a non-root ``USER`` to a
    benchmark Dockerfile, also pass ``user=`` through ``kwargs`` to
    :meth:`create` so it actually takes effect.
    """

    def __init__(
        self,
        *,
        default_memory: str = "2g",
        default_cpus: float = 2.0,
        network_mode: str = "none",
    ):
        # Docker client is connected lazily on first actual sandbox use
        # (see the ``_docker`` property). Conversational benchmarks
        # (LoCoMo, ConvoMem, LongMemEval, …) never create a container, so
        # they must not require a running Docker daemon just to construct
        # the manager the engine always instantiates.
        self._docker_client: Any = None
        self._default_memory = default_memory
        self._default_cpus = default_cpus
        self._network_mode = network_mode
        self._active: dict[str, Sandbox] = {}

    @property
    def _docker(self) -> Any:
        """Lazily-constructed Docker client (``docker.from_env()``).

        Deferred so a no-sandbox run (conversational memory benchmarks)
        never touches Docker. Code-gen / SWE-bench paths hit this on the
        first ``create`` / ``prefetch_images`` call, where a daemon is
        genuinely required.
        """
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    async def create(self, image: str, task: Any, **kwargs: Any) -> Sandbox:
        """Create a new sandbox container."""
        network = kwargs.pop("network_mode", self._network_mode)
        # Terminal-Bench tasks may need network access
        if hasattr(task, "metadata") and task.metadata.get("requires_network"):
            network = "bridge"

        # Some images (notably Scale AI's SWE-bench Pro images at
        # ``jefzda/sweap-images:<tag>``) ship with a custom ENTRYPOINT
        # that runs and exits. Passing ``command="sleep infinity"`` to
        # such an image just gives the entrypoint extra args; the
        # container still exits → subsequent ``docker exec`` returns
        # ``409 container is not running``. Caller can opt into a bash
        # entrypoint override via ``entrypoint="/bin/bash"`` kwarg.
        # See deviations.md S21 for the Pro-specific need; Verified
        # path is byte-identical to the pre-2026-05-25 behaviour
        # because no entrypoint is set unless explicitly requested.
        entrypoint = kwargs.pop("entrypoint", None)
        if entrypoint == "/bin/bash":
            # ``sleep infinity`` runs as the arg to bash -c so the
            # container holds open regardless of what the image's
            # native ENTRYPOINT was.
            command: Any = ["-c", "sleep infinity"]
        else:
            command = "sleep infinity"
        loop = asyncio.get_event_loop()
        run_kwargs: dict[str, Any] = dict(
            image=image,
            command=command,
            detach=True,
            network_mode=network,
            mem_limit=kwargs.get("memory", self._default_memory),
            nano_cpus=int(kwargs.get("cpus", self._default_cpus) * 1e9),
            working_dir="/workspace",
            remove=False,
        )
        if entrypoint is not None:
            run_kwargs["entrypoint"] = entrypoint
        run_kwargs.update(kwargs)
        container = await loop.run_in_executor(
            None,
            lambda: self._docker.containers.run(**run_kwargs),
        )
        sandbox = Sandbox(
            container_id=container.id,
            image=image,
        )
        self._active[container.id] = sandbox
        logger.info("sandbox.created", container_id=container.short_id, image=image)
        return sandbox

    async def execute(
        self, sandbox: Sandbox, command: str, timeout: int = 60
    ) -> ExecutionResult:
        """Execute a command inside the sandbox."""
        import time

        loop = asyncio.get_event_loop()
        container = self._docker.containers.get(sandbox.container_id)

        t0 = time.perf_counter()
        try:
            exec_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: container.exec_run(
                        ["bash", "-c", command],
                        workdir=sandbox.work_dir,
                        demux=True,
                    ),
                ),
                timeout=timeout,
            )
            duration = (time.perf_counter() - t0) * 1000
            stdout_bytes, stderr_bytes = exec_result.output
            return ExecutionResult(
                exit_code=exec_result.exit_code,
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            duration = (time.perf_counter() - t0) * 1000
            logger.warning("sandbox.timeout", container_id=sandbox.container_id, command=command[:100])
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                duration_ms=duration,
                timed_out=True,
            )

    async def write_file(self, sandbox: Sandbox, path: str, content: str) -> None:
        """Write a file inside the sandbox container.

        ``put_archive`` requires the destination directory to already
        exist; with nested writes (e.g. ``/workspace/tests/sub/foo.py``)
        the parent dir is rarely there. We ``mkdir -p`` first so callers
        can write to arbitrary depths without staging intermediate
        directories themselves.
        """
        import io
        import shlex
        import tarfile

        loop = asyncio.get_event_loop()
        container = self._docker.containers.get(sandbox.container_id)

        # Create a tar archive with the file
        buf = io.BytesIO()
        data = content.encode("utf-8")
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.split("/")[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)

        dest_dir = "/".join(path.split("/")[:-1]) or "/"
        if dest_dir not in ("", "/"):
            # Best-effort mkdir; if the image lacks ``mkdir`` (rare), the
            # subsequent put_archive will surface the real error.
            await loop.run_in_executor(
                None,
                lambda: container.exec_run(
                    ["sh", "-c", f"mkdir -p {shlex.quote(dest_dir)}"],
                ),
            )
        await loop.run_in_executor(
            None, lambda: container.put_archive(dest_dir, buf)
        )

    async def apply_patch(self, sandbox: Sandbox, patch: str) -> bool:
        """Apply a git diff patch inside the sandbox."""
        await self.write_file(sandbox, "/workspace/solution.patch", patch)
        result = await self.execute(
            sandbox,
            "cd /workspace && git apply --allow-empty solution.patch",
            timeout=30,
        )
        return result.exit_code == 0

    async def destroy(self, sandbox: Sandbox) -> None:
        """Stop and remove the sandbox container."""
        loop = asyncio.get_event_loop()
        try:
            container = self._docker.containers.get(sandbox.container_id)
            await loop.run_in_executor(None, lambda: container.remove(force=True))
        except Exception:
            pass
        self._active.pop(sandbox.container_id, None)
        logger.info("sandbox.destroyed", container_id=sandbox.container_id[:12])

    async def destroy_all(self) -> None:
        """Clean up all active sandboxes."""
        for sandbox in list(self._active.values()):
            await self.destroy(sandbox)

    async def prefetch_images(
        self,
        images: "Iterable[str] | dict[str, str]",
        *,
        concurrency: int = 4,
    ) -> dict[str, str]:
        """Pull (in parallel) every image the run will need, before the task loop.

        SWE-bench Verified ships 500 per-instance Docker images and the per-task
        timeout (default 1800 s) would otherwise absorb a 5-15 min ``docker
        pull`` for each cold image. Pulling up-front separates infra cost from
        model performance: pull failures (auth, ECR offline, image missing)
        surface immediately with a clear status here, while the task loop sees
        only "image not available" rather than "model produced bad patch".

        Two input modes:

        - **Flat iterable** (legacy, conversational benchmarks): each item is
          both the pullable URL and the local-lookup name. Skips already-local;
          pulls remote; tags as itself (no-op).
        - **Mapping** (SWE-bench-style): ``{local_name: pullable_url}``.
          When ``local_name != pullable_url``, pulls the URL, then locally
          re-tags as ``local_name`` so the sandbox launcher's
          ``containers.run(image=local_name, ...)`` finds the image without
          needing the original registry URL. ECR pulls fetch fresh
          authentication via boto3 before the docker SDK call so the
          12-h IAM session-token expiry doesn't bite mid-run.

        Behaviour:
        - Already-local images (looked up by ``local_name``) report
          ``already_local`` and skip the registry entirely.
        - Successful pulls report ``pulled``.
        - Failures report ``error:<class>:<msg>`` and do **not** abort the
          batch — they may be transient, and the per-task ``create()`` will
          surface the real error if the image still isn't available when
          a task tries to use it.
        - ``concurrency`` caps simultaneous pulls. Default 4 is safe for
          docker-hub / ECR rate limits; bump for fast LANs.

        Args:
            images: Either an iterable of image tags (legacy mode) or a
                ``{local_name: pullable_url}`` mapping. Empty strings are
                tolerated (the empty string is skipped — conversational
                adapters return ``""`` from ``get_sandbox_image``).
            concurrency: Max parallel ``docker pull`` calls.

        Returns:
            ``{local_name: status}`` for every distinct non-empty entry.
            Useful for run-level logging and for tests that need to
            assert what was pulled.
        """
        # Normalize both input modes into a ``{local_name: pullable_url}`` map.
        if isinstance(images, dict):
            source_map: dict[str, str] = {k: v for k, v in images.items() if k}
        else:
            source_map = {img: img for img in images if img}
        if not source_map:
            return {}

        import docker as _docker

        semaphore = asyncio.Semaphore(max(1, concurrency))
        loop = asyncio.get_event_loop()

        async def _pull_one(local_name: str, pullable_url: str) -> tuple[str, str]:
            async with semaphore:
                # Already-local check first — avoids hitting the registry at
                # all for images previously pulled or pre-staged locally.
                try:
                    await loop.run_in_executor(
                        None, lambda: self._docker.images.get(local_name)
                    )
                    return local_name, "already_local"
                except _docker.errors.ImageNotFound:
                    pass
                except Exception as e:  # noqa: BLE001 — surface infra errors
                    return local_name, f"error:{type(e).__name__}:{e}"

                # Pull. ECR URLs get fresh auth_config so the IAM session-
                # token expiry doesn't bite mid-run on long-running ASG hosts.
                # Non-ECR URLs (Docker Hub, local) skip the kwarg so docker
                # SDK fakes that take only ``pull(name)`` stay compatible.
                auth_config = _ecr_auth_config_for(pullable_url)
                t0 = asyncio.get_event_loop().time()

                async def _do_pull(url: str, auth: dict[str, str] | None) -> None:
                    if auth is not None:
                        await loop.run_in_executor(
                            None,
                            lambda: self._docker.images.pull(url, auth_config=auth),
                        )
                    else:
                        await loop.run_in_executor(
                            None,
                            lambda: self._docker.images.pull(url),
                        )

                try:
                    await _do_pull(pullable_url, auth_config)
                except Exception as e:  # noqa: BLE001
                    # Fallback path: if the resolver rewrote the local image
                    # name to a different pullable URL (e.g. ECR mirror) and
                    # the pull failed (mirror not populated, ECR repo just
                    # provisioned, etc.), AND the original local name itself
                    # is a valid registry URL (like Pro's
                    # ``jefzda/sweap-images:<tag>``), retry once with the
                    # local name. Saves a noisy prefetch failure +
                    # subsequent lazy-pull-at-create-time on State 2
                    # deployments. See deviations.md S21.
                    if local_name != pullable_url and "/" in local_name:
                        fallback_url = local_name
                        try:
                            await _do_pull(fallback_url, None)
                            logger.info(
                                "sandbox.image_pulled_via_fallback",
                                local=local_name,
                                primary=pullable_url,
                                fallback=fallback_url,
                                primary_error=str(e)[:200],
                            )
                            elapsed = asyncio.get_event_loop().time() - t0
                            return local_name, "pulled"
                        except Exception as fallback_e:  # noqa: BLE001
                            return (
                                local_name,
                                f"error:{type(e).__name__}:{e} "
                                f"(fallback {type(fallback_e).__name__}:{fallback_e})",
                            )
                    return local_name, f"error:{type(e).__name__}:{e}"
                elapsed = asyncio.get_event_loop().time() - t0
                logger.info(
                    "sandbox.image_pulled",
                    local=local_name,
                    source=pullable_url,
                    elapsed_s=round(elapsed, 1),
                )

                # Retag if the local lookup name differs from the pullable URL.
                # No-op when they're identical (legacy iterable mode).
                if local_name != pullable_url:
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: self._docker.images.get(pullable_url).tag(local_name),
                        )
                    except Exception as e:  # noqa: BLE001
                        return local_name, f"error:retag:{type(e).__name__}:{e}"

                return local_name, "pulled"

        results = await asyncio.gather(
            *(_pull_one(local, url) for local, url in source_map.items())
        )
        status_map = dict(results)
        pulled = sum(1 for s in status_map.values() if s == "pulled")
        cached = sum(1 for s in status_map.values() if s == "already_local")
        errors = sum(1 for s in status_map.values() if s.startswith("error:"))
        logger.info(
            "sandbox.prefetch_summary",
            total=len(status_map),
            pulled=pulled,
            already_local=cached,
            errors=errors,
        )
        return status_map
