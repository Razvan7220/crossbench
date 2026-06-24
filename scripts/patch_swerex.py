"""Apply CrossBench's SWE-bench-Pro fixes to swerex.deployment.docker.

Runs at orchestrator image build time (see Dockerfile).  Edits
``swerex/deployment/docker.py`` in-place; idempotent (each substitution
short-circuits if the replacement is already present).

Why these patches exist
-----------------------
swerex 1.2.0's ``DockerDeployment`` builds a layered Python 3.11
standalone image on top of the Pro task container and spawns a swerex-
remote HTTP server inside it.  Out-of-the-box that flow breaks on five
unrelated quirks of the SWE-bench Pro task images, plus two
container-architecture mismatches that surface only when swerex's
client itself runs inside a container (which is the orchestrator's
ECS setup, but NOT the laptop/Docker-Desktop setup main's earlier
patches were validated on).  See ``deviations.md`` S23 for the full
story; the seven substitutions are summarised inline below.
"""
from __future__ import annotations

import pathlib
import sys


def _apply(src: str, label: str, old: str, new: str, *, sentinel: str) -> str:
    """Apply one substitution.  Idempotent via ``sentinel`` lookup."""
    if sentinel in src:
        print(f"  [{label}] already applied (sentinel present)", flush=True)
        return src
    if old not in src:
        raise SystemExit(f"  [{label}] OLD target not found in swerex docker.py")
    out = src.replace(old, new, 1)
    if out == src:
        raise SystemExit(f"  [{label}] replace returned identical string")
    print(f"  [{label}] applied", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    """Patch ``swerex/deployment/docker.py`` in place.

    When ``argv`` is empty (the production path: ``RUN python3
    /tmp/patch_swerex.py`` in the Dockerfile), the target file is
    looked up via ``import swerex.deployment.docker``.  For tests, pass
    an explicit path: ``python patch_swerex.py /tmp/fake/docker.py``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        path = pathlib.Path(argv[0])
    else:
        import swerex.deployment.docker as _m
        path = pathlib.Path(_m.__file__)
    text = path.read_text(encoding="utf-8")

    # --- Sub #1: bullseye builder pin ------------------------------------------
    # ``python:3.11-slim`` on Docker Hub now resolves to Debian Trixie (GLIBC
    # 2.38).  Pro task images are heterogeneous: openlibrary uses Bookworm
    # (GLIBC 2.36), ansible uses Ubuntu 20.04 (GLIBC 2.31).  Pin to bullseye
    # (Debian 11, GLIBC 2.31) — forward-compat covers every Pro image observed.
    text = _apply(
        text,
        "sub1: bullseye pin",
        "python:3.11-slim AS builder",
        "python:3.11-slim-bullseye AS builder",
        sentinel="python:3.11-slim-bullseye AS builder",
    )

    # --- Sub #2: truthy-check on python_standalone_dir -------------------------
    # Keep the escape hatch for callers that pass an empty string to disable the
    # standalone build.  SEAL does NOT pass empty (the standalone build is
    # required); this preserves upstream's API contract for other callers.
    text = _apply(
        text,
        "sub2: truthy check",
        "if self._config.python_standalone_dir is not None:",
        "if self._config.python_standalone_dir:  # empty string disables build",
        sentinel="# empty string disables build",
    )

    # --- Sub #3: PyPI index URL on swe-rex install -----------------------------
    # Pro images ship ``/root/.config/pip/pip.conf`` with
    # ``index-url = http://127.0.0.1:9876/`` (a devpi proxy that isn't running
    # during the swerex build).  Force public PyPI on the swe-rex install line.
    text = _apply(
        text,
        "sub3: pypi index",
        "pip3 install --no-cache-dir {PACKAGE_NAME}",
        "pip3 install --index-url https://pypi.org/simple/ --no-cache-dir {PACKAGE_NAME}",
        sentinel="--index-url https://pypi.org/simple/",
    )

    # --- Sub #4: --entrypoint= clear ------------------------------------------
    # Pro images set ``ENTRYPOINT=["/bin/bash"]``.  swerex's CMD is
    # ``["/bin/sh", "-c", "..."]``.  Without --entrypoint= clearing, docker
    # concatenates to ``/bin/bash /bin/sh -c "..."`` → bash reads ``/bin/sh``
    # as a script → ``cannot execute binary file``.  Use the equals form
    # (single-argv element) for unambiguous empty-value semantics.
    text = _apply(
        text,
        "sub4: --entrypoint= clear",
        '"docker",\n            "run",\n            "--rm",\n            "-p",',
        '"docker",\n            "run",\n            "--rm",\n            "--entrypoint=",\n            "-p",',
        sentinel='"--entrypoint=",',
    )

    # --- Sub #5: libssl.so.1.1 bundle from bullseye builder --------------------
    # swerex's stage-2 ``apt-get install -y libc6`` doesn't install openssl.
    # Bookworm runtime images ship libssl3 only (libssl.so.3); the standalone
    # Python was compiled against libssl1.1.  Copy the .so.1.1 libs from the
    # builder stage into the standalone Python's ``lib/``; LD_LIBRARY_PATH
    # already points there.
    text = _apply(
        text,
        "sub5: libssl bundle",
        'f"COPY --from=builder /root/python3.11 {self._config.python_standalone_dir}/python3.11\\n"',
        (
            'f"COPY --from=builder /root/python3.11 {self._config.python_standalone_dir}/python3.11\\n"\n'
            '            "COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so.1.1 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 /root/python3.11/lib/\\n"'
        ),
        sentinel="libssl.so.1.1 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1",
    )

    # --- Sub #6: share orchestrator's network namespace ------------------------
    # When the swerex client runs INSIDE a container that mounts
    # ``/var/run/docker.sock`` to spawn sibling containers (the ECS task
    # setup), the spawned container's ``-p 58335:8000`` port mapping lands
    # on the EC2 host's loopback — unreachable from the orchestrator's own
    # loopback.  swerex's client then probes ``http://127.0.0.1:58335`` and
    # times out forever.  The fix: when ``/.dockerenv`` is present, spawn
    # the sibling with ``--network=container:<orchestrator>`` so it shares
    # the orchestrator's network namespace.  Then ``127.0.0.1:8000`` (the
    # container-internal port) is directly reachable.
    #
    # ``--network=container`` is INCOMPATIBLE with ``-p`` (docker errors
    # ``conflicting options: port publishing and the container type network
    # mode``).  So when we activate network-share, we also drop the ``-p``
    # args from the docker run command — the find-free-port logic is then
    # vestigial.  We probe port 8000 directly via ``_probe_port`` below.
    text = _apply(
        text,
        "sub6: --network=container",
        (
            '        platform_arg = []\n'
            '        if self._config.platform is not None:\n'
            '            platform_arg = ["--platform", self._config.platform]\n'
            '        cmds = [\n'
            '            "docker",\n'
            '            "run",\n'
            '            "--rm",\n'
            '            "--entrypoint=",\n'
            '            "-p",\n'
            '            f"{self._config.port}:8000",\n'
        ),
        (
            '        platform_arg = []\n'
            '        if self._config.platform is not None:\n'
            '            platform_arg = ["--platform", self._config.platform]\n'
            '        # Sub #6 (CrossBench): share orchestrator net namespace when\n'
            '        # the client itself is containerized (docker-out-of-docker).\n'
            '        # Also DROP -p (incompatible with --network=container).\n'
            '        import os as _os, socket as _socket\n'
            '        _net_arg = []\n'
            '        _port_args = ["-p", f"{self._config.port}:8000"]\n'
            '        _probe_port = self._config.port\n'
            '        if _os.path.exists("/.dockerenv"):\n'
            '            try:\n'
            '                _net_arg = ["--network", f"container:{_socket.gethostname()}"]\n'
            '                _port_args = []\n'
            '                _probe_port = 8000\n'
            '            except Exception:\n'
            '                pass\n'
            '        cmds = [\n'
            '            "docker",\n'
            '            "run",\n'
            '            "--rm",\n'
            '            "--entrypoint=",\n'
            '            *_net_arg,\n'
            '            *_port_args,\n'
        ),
        sentinel="Sub #6 (CrossBench)",
    )

    # Companion to sub #6: route the probe to ``_probe_port`` instead of
    # ``self._config.port``.  When network=container is active, _probe_port
    # equals 8000 (container-internal); otherwise it's the find-free-port
    # picked above (laptop-Docker-Desktop path).
    text = _apply(
        text,
        "sub6b: probe port override",
        "RemoteRuntimeConfig(port=self._config.port, timeout=self._runtime_timeout, auth_token=token)",
        "RemoteRuntimeConfig(port=_probe_port, timeout=self._runtime_timeout, auth_token=token)",
        sentinel="port=_probe_port,",
    )

    # --- Sub #7: kill Popen target BEFORE reading stdout/stderr ----------------
    # swerex's _wait_until_alive does ``self._container_process.stdout.read()``
    # after a TimeoutError.  ``read()`` is BLOCKING until the pipe closes —
    # which only happens when the ``docker run`` Popen target exits.  If the
    # sibling container is alive but swerex-remote isn't responding (e.g. our
    # localhost-isolation bug above), .read() blocks until our outer
    # subprocess timeout (60 min) fires.  Empirical: a 180s startup timeout
    # ballooned into a 60-min hang in the seal scaffold.  Fix: kill the
    # Popen target FIRST so the pipe closes, THEN do the diagnostic .read().
    text = _apply(
        text,
        "sub7: kill before read",
        (
            "        except TimeoutError as e:\n"
            "            self.logger.error(\"Runtime did not start within timeout. Here's the output from the container process.\")\n"
            "            self.logger.error(self._container_process.stdout.read().decode())  # type: ignore\n"
            "            self.logger.error(self._container_process.stderr.read().decode())  # type: ignore\n"
            "            assert self._container_process is not None\n"
            "            await self.stop()\n"
            "            raise e\n"
        ),
        (
            "        except TimeoutError as e:\n"
            "            # Sub #7 (CrossBench): kill the docker-run Popen target BEFORE\n"
            "            # reading stdout/stderr.  Without this, .read() blocks on the\n"
            "            # still-alive `docker run` until our outer subprocess timeout\n"
            "            # (e.g. 60min) fires.\n"
            "            if self._container_process is not None:\n"
            "                try:\n"
            "                    self._container_process.kill()\n"
            "                    self._container_process.wait(timeout=5)\n"
            "                except Exception:\n"
            "                    pass\n"
            "            self.logger.error(\"Runtime did not start within timeout. Here's the output from the container process.\")\n"
            "            try:\n"
            "                self.logger.error(self._container_process.stdout.read().decode())  # type: ignore\n"
            "                self.logger.error(self._container_process.stderr.read().decode())  # type: ignore\n"
            "            except Exception:\n"
            "                pass\n"
            "            assert self._container_process is not None\n"
            "            await self.stop()\n"
            "            raise e\n"
        ),
        sentinel="Sub #7 (CrossBench)",
    )

    # --- Sub #8: Alpine target support ----------------------------------------
    # Empirical finding (2026-06-15): the ~14% Pro NoPrediction floor is NOT
    # protonmail-only. gravitational/teleport (~36 instances), future-architect/
    # vuls, flipt-io/flipt, and navidrome/navidrome ALL ship Alpine-based images
    # too. They were silently dying because swerex's stage-2 Dockerfile calls
    # ``apt-get install libc6`` which exits 127 on Alpine.
    #
    # Earlier gcompat approach was rejected because the glibc-compiled standalone
    # Python needs symbols musl doesn't ship even with gcompat. The cleaner fix:
    # SKIP the glibc standalone build for Alpine targets entirely. Alpine has
    # Python 3.11+ in its repos via ``apk add python3 py3-pip`` — and swe-rex is
    # pure Python with no C extensions, so it installs cleanly via pip on the
    # system Python.
    #
    # This sub injects an ``alpine_dockerfile`` property + a probe method, then
    # branches ``_build_image`` to use it when the target image is Alpine. The
    # detection runs ``docker run --rm --entrypoint= $BASE_IMAGE cat
    # /etc/os-release`` — ~1-2s overhead per task, ID=alpine substring check.
    text = _apply(
        text,
        "sub8: Alpine target support",
        old=(
            "    def _build_image(self) -> str:\n"
            "        \"\"\"Builds image, returns image ID.\"\"\"\n"
            "        dockerfile = self.glibc_dockerfile\n"
        ),
        new=(
            "    def _is_alpine_target(self) -> bool:\n"
            "        \"\"\"Sub #8 (CrossBench): probe BASE_IMAGE /etc/os-release for ID=alpine.\n"
            "\n"
            "        Returns False on any error (docker exec failure, image gone,\n"
            "        timeout) so the caller falls back to the glibc path safely.\n"
            "        \"\"\"\n"
            "        import subprocess as _sp\n"
            "        try:\n"
            "            out = _sp.check_output(\n"
            "                [\"docker\", \"run\", \"--rm\", \"--entrypoint=\",\n"
            "                 self._config.image, \"cat\", \"/etc/os-release\"],\n"
            "                stderr=_sp.DEVNULL, timeout=30,\n"
            "            ).decode(\"utf-8\", errors=\"ignore\")\n"
            "            return \"ID=alpine\" in out\n"
            "        except Exception:\n"
            "            return False\n"
            "\n"
            "    @property\n"
            "    def alpine_dockerfile(self) -> str:\n"
            "        \"\"\"Sub #8 (CrossBench): Alpine-native Dockerfile.\n"
            "\n"
            "        Skips the glibc Python standalone build. Uses Alpine's apk\n"
            "        python3 + py3-pip + pip install swe-rex. Net effect: same\n"
            "        runtime contract (swe-rex-remote on PATH, listening on :8000)\n"
            "        on musl libc instead of glibc.\n"
            "        \"\"\"\n"
            "        if self._config.platform:\n"
            "            platform_arg = f\"--platform={self._config.platform}\"\n"
            "        else:\n"
            "            platform_arg = \"\"\n"
            "        return (\n"
            "            \"ARG BASE_IMAGE\\n\\n\"\n"
            "            f\"FROM {platform_arg} $BASE_IMAGE\\n\"\n"
            "            # Alpine 3.17 ships Python 3.10, 3.18+ ships 3.11+; both work\n"
            "            # for swe-rex (pure Python, no version-specific C extensions).\n"
            "            \"RUN apk add --no-cache python3 py3-pip\\n\"\n"
            "            # --break-system-packages bypasses PEP 668 (Alpine 3.18+).\n"
            "            # Fallback chain handles Alpine 3.17 (no PEP 668) and the\n"
            "            # Pro images' broken /root/.config/pip/pip.conf via explicit\n"
            "            # --index-url override.\n"
            "            f\"RUN pip3 install --break-system-packages \"\n"
            "            f\"--index-url https://pypi.org/simple/ --no-cache-dir \"\n"
            "            f\"{PACKAGE_NAME} || \"\n"
            "            f\"pip3 install --index-url https://pypi.org/simple/ \"\n"
            "            f\"--no-cache-dir {PACKAGE_NAME}\\n\"\n"
            "            # apk's py3-pip puts scripts in /usr/bin/. swerex's start_cmd\n"
            "            # in docker.py:_get_swerex_start_cmd constructs the path as\n"
            "            # ``{python_standalone_dir}/python3.11/bin/{REMOTE_EXECUTABLE_NAME}``\n"
            "            # (default python_standalone_dir='/root'). Make that path resolvable\n"
            "            # by symlinking from /usr/bin so the SAME start_cmd works on Alpine\n"
            "            # without further patching of _get_swerex_start_cmd.\n"
            "            \"RUN mkdir -p /root/python3.11/bin\\n\"\n"
            "            f\"RUN ln -sf $(which {REMOTE_EXECUTABLE_NAME}) \"\n"
            "            f\"/root/python3.11/bin/{REMOTE_EXECUTABLE_NAME}\\n\"\n"
            "            \"RUN ln -sf $(which python3) /root/python3.11/bin/python3\\n\"\n"
            "            f\"RUN /root/python3.11/bin/{REMOTE_EXECUTABLE_NAME} --version\\n\"\n"
            "        )\n"
            "\n"
            "    def _build_image(self) -> str:\n"
            "        \"\"\"Builds image, returns image ID.\"\"\"\n"
            "        # Sub #8 (CrossBench): branch on target distro. Alpine images\n"
            "        # can't run the glibc standalone Python, so we apt-get-skip\n"
            "        # entirely and use Alpine's system Python via apk.\n"
            "        if self._is_alpine_target():\n"
            "            dockerfile = self.alpine_dockerfile\n"
            "        else:\n"
            "            dockerfile = self.glibc_dockerfile\n"
        ),
        sentinel="Sub #8 (CrossBench)",
    )

    path.write_text(text, encoding="utf-8")
    print(
        "swerex docker.py patched "
        "(bullseye + empty-disable + pypi-index + entrypoint-clear + "
        "libssl-copy + net-share + read-after-kill + alpine-support)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
