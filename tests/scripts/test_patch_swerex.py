"""Tests for ``scripts/patch_swerex.py``.

The patch script applies CrossBench's 7-substitution fix to a copy of
``swerex/deployment/docker.py``.  These tests use a hand-built fixture
that includes every target string from upstream swerex 1.2.0, so they
exercise the same substitution logic the orchestrator's image-build
step runs without requiring swerex installed in the test environment.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH_SCRIPT = REPO_ROOT / "scripts" / "patch_swerex.py"


# A minimal fixture that mirrors the load-bearing strings from
# swerex 1.2.0's docker.py.  Indent matches upstream so multi-line
# search strings (subs 4, 6, 7) land cleanly.
_FIXTURE_DOCKER_PY = textwrap.dedent('''
    """Vendored mini-fixture mirroring swerex 1.2.0 docker.py."""

    import subprocess

    PACKAGE_NAME = "swe-rex"
    REMOTE_EXECUTABLE_NAME = "swerex-remote"


    class DockerDeployment:
        def __init__(self, config):
            self._config = config

        @property
        def glibc_dockerfile(self) -> str:
            platform_arg = ""
            return (
                "ARG BASE_IMAGE\\n\\n"
                f"FROM {platform_arg} python:3.11-slim AS builder\\n"
                f"COPY --from=builder /root/python3.11 {self._config.python_standalone_dir}/python3.11\\n"
                f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {PACKAGE_NAME}\\n\\n"
            )

        async def start(self):
            self._pull_image()
            if self._config.python_standalone_dir is not None:
                image_id = self._build_image()
            else:
                image_id = self._config.image
            assert self._container_name is None
            self._container_name = "x"
            token = "t"
            platform_arg = []
            if self._config.platform is not None:
                platform_arg = ["--platform", self._config.platform]
            cmds = [
                "docker",
                "run",
                "--rm",
                "-p",
                f"{self._config.port}:8000",
                *platform_arg,
                *self._config.docker_args,
                "--name",
                self._container_name,
                image_id,
            ]
            self._container_process = subprocess.Popen(cmds)
            self._runtime = RemoteRuntime.from_config(
                RemoteRuntimeConfig(port=self._config.port, timeout=self._runtime_timeout, auth_token=token)
            )

        async def _wait_until_alive(self, timeout: float = 10.0):
            try:
                return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime_timeout)
            except TimeoutError as e:
                self.logger.error("Runtime did not start within timeout. Here's the output from the container process.")
                self.logger.error(self._container_process.stdout.read().decode())  # type: ignore
                self.logger.error(self._container_process.stderr.read().decode())  # type: ignore
                assert self._container_process is not None
                await self.stop()
                raise e
''')


def _run_patch_script(target: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run patch_swerex.py with the fixture as target."""
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def fresh_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write the unmodified fixture into a tmp file."""
    p = tmp_path / "docker.py"
    p.write_text(_FIXTURE_DOCKER_PY, encoding="utf-8")
    return p


def test_patch_script_runs_clean_on_fresh_fixture(fresh_fixture: pathlib.Path) -> None:
    """All 7 (well, 8 including 6b) substitutions apply on first run."""
    result = _run_patch_script(fresh_fixture)
    assert result.returncode == 0, (
        f"patch script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Every numbered substitution should report 'applied' (not 'already applied').
    for label in (
        "sub1: bullseye pin",
        "sub2: truthy check",
        "sub3: pypi index",
        "sub4: --entrypoint= clear",
        "sub5: libssl bundle",
        "sub6: --network=container",
        "sub6b: probe port override",
        "sub7: kill before read",
    ):
        assert f"[{label}] applied" in result.stdout, (
            f"expected '{label}' to apply (output: {result.stdout!r})"
        )


def test_patch_is_idempotent(fresh_fixture: pathlib.Path) -> None:
    """Re-running the script after a successful patch is a no-op."""
    first = _run_patch_script(fresh_fixture)
    assert first.returncode == 0
    second = _run_patch_script(fresh_fixture)
    assert second.returncode == 0
    # Every line should now report 'already applied'.
    for label in (
        "sub1: bullseye pin",
        "sub2: truthy check",
        "sub3: pypi index",
        "sub4: --entrypoint= clear",
        "sub5: libssl bundle",
        "sub6: --network=container",
        "sub6b: probe port override",
        "sub7: kill before read",
    ):
        assert f"[{label}] already applied" in second.stdout, (
            f"expected '{label}' to be already-applied on 2nd run; got: {second.stdout!r}"
        )


def test_patched_fixture_is_valid_python(fresh_fixture: pathlib.Path) -> None:
    """The patched file must still parse cleanly as Python."""
    result = _run_patch_script(fresh_fixture)
    assert result.returncode == 0
    src = fresh_fixture.read_text(encoding="utf-8")
    ast.parse(src)  # raises SyntaxError on failure


def test_patched_fixture_contains_all_expected_markers(fresh_fixture: pathlib.Path) -> None:
    """Each substitution leaves a unique post-patch marker in the file."""
    result = _run_patch_script(fresh_fixture)
    assert result.returncode == 0
    src = fresh_fixture.read_text(encoding="utf-8")
    expected_markers = [
        "python:3.11-slim-bullseye AS builder",                       # sub1
        "# empty string disables build",                              # sub2
        "--index-url https://pypi.org/simple/",                       # sub3
        '"--entrypoint=",',                                           # sub4
        "libssl.so.1.1 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1",   # sub5
        "Sub #6 (CrossBench)",                                        # sub6
        'container:{_socket.gethostname()}',                          # sub6
        "_port_args = []",                                            # sub6 (drop -p when net=container)
        "*_port_args,",                                               # sub6 (port args spread in cmds list)
        "port=_probe_port,",                                          # sub6b
        "Sub #7 (CrossBench)",                                        # sub7
        "self._container_process.kill()",                             # sub7
    ]
    for marker in expected_markers:
        assert marker in src, f"missing post-patch marker: {marker!r}"


def test_sub6_drops_port_publishing_when_network_container_active(fresh_fixture: pathlib.Path) -> None:
    """Regression lock for the 2026-05-27 net-fix run failure mode.

    Docker errors out on ``conflicting options: port publishing and the
    container type network mode`` when ``--network=container:...`` is
    combined with ``-p host:8000``.  The patched start() must:
    - keep ``-p`` arg in cmds when ``/.dockerenv`` is absent (laptop)
    - drop ``-p`` arg when ``/.dockerenv`` is present (containerized
      orchestrator → shared net namespace via ``--network=container``)
    """
    result = _run_patch_script(fresh_fixture)
    assert result.returncode == 0
    src = fresh_fixture.read_text(encoding="utf-8")
    # Both branches must be present in the patched source — fixture has both
    # an unpatched control path (laptop) and the new in-container branch.
    assert '_port_args = ["-p", f"{self._config.port}:8000"]' in src, (
        "laptop path must still set port args"
    )
    assert "_port_args = []" in src, "containerized path must drop port args"
    # And the raw `-p` literal in the cmds list itself should be gone —
    # replaced by the spread.
    assert '"-p",\n            f"{self._config.port}:8000",' not in src, (
        "raw -p literal should be replaced by *_port_args spread"
    )
    assert "*_port_args," in src, "cmds list must spread port args dynamically"


def test_patch_script_fails_loudly_when_target_string_missing(tmp_path: pathlib.Path) -> None:
    """If swerex changes upstream and a target string disappears, the script
    must error out (not silently no-op)."""
    truncated = tmp_path / "docker.py"
    truncated.write_text(
        "def nothing_to_see_here(): pass\n", encoding="utf-8"
    )
    result = _run_patch_script(truncated)
    assert result.returncode != 0, (
        "patch script should fail when no targets match"
    )
    # The script raises SystemExit with a clear "OLD target not found" message.
    assert "OLD target not found" in (result.stderr + result.stdout)


def test_patch_script_has_no_external_imports_beyond_swerex() -> None:
    """The script should import only ``pathlib``, ``sys``, and ``swerex``.

    Lockdown so a future contributor doesn't add an unguarded import
    that breaks the build (the script runs at orchestrator image-build
    time, before the full env is available)."""
    src = PATCH_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level_imports: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_imports.add(node.module.split(".")[0])
    # Top-level imports only.  In-function imports (swerex.deployment.docker)
    # are deliberately deferred and don't show up here.
    assert top_level_imports.issubset({"__future__", "pathlib", "sys"}), (
        f"unexpected top-level imports in patch_swerex.py: {top_level_imports}"
    )
