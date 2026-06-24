from pathlib import Path
import logging
import os
import shutil
import stat
import subprocess
from typing import Any
from cross_bench.models import Task

logger = logging.getLogger(__name__)


def _crossbench_repo_root() -> Path:
    """``src/cross_bench/benchmarks/repobench.py`` -> CrossBench checkout root."""
    return Path(__file__).resolve().parents[3]


def _resolve_synthetic_workspace_dir(raw: str) -> Path:
    """Resolve ``synthetic_workspace_dir`` relative to the CrossBench repo root."""
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = _crossbench_repo_root() / p
    return p.resolve()


def _sanitize_repo_dir_name(repo_name: str) -> str:
    """Single path segment under ``synthetic_workspace_dir`` — no traversal."""
    s = (repo_name or "").strip() or "unknown"
    for ch in ("/", "\\", os.sep):
        s = s.replace(ch, "_")
    s = s.replace("\x00", "")
    if s in (".", ".."):
        return f"repo_{s}"
    return s[:240]


def _safe_repo_relative_path(rel: str) -> Path | None:
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or rel.startswith("/"):
        return None
    p = Path(rel)
    if p.is_absolute():
        return None
    if ".." in p.parts:
        return None
    return p


def _force_rmtree(path: Path) -> None:
    """Remove a tree even when some entries are not owned by the current user."""

    def _onerror(func: Any, p: str, _exc_info: Any) -> None:
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            func(p)
        else:
            raise

    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def _ensure_writable_workspace_base(base: Path) -> None:
    """Create ``base`` and verify the current user can write there."""
    base.mkdir(parents=True, exist_ok=True)
    probe = base / ".crossbench_write_probe"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        try:
            subprocess.run(["chown", "-R", f"{os.getuid()}:{os.getgid()}", base], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.exception("materialize_synthetic_workspace chown failed path=%s", base)
            raise PermissionError(
                f"Cannot write to synthetic workspace {base!s} ({exc}). "
                "If this directory is bind-mounted into a Docker container, fix ownership "
                f"on the host, e.g. chown -R \"$USER:$USER\" {base!s}"
            ) from exc


def _materialize_synthetic_workspace(task: Task) -> None:
    """Write snippet + target files under ``synthetic_workspace_dir`` / repo.

    Sets ``task.metadata["synthetic_workspace_tear_path"]`` to the repo
    root so :meth:`RepoBenchAdapter.teardown_task` can ``rmtree`` it.
    """
    meta = task.metadata or {}
    raw_base = meta.get("synthetic_workspace_dir")
    if raw_base is None:
        return
    base_str = str(raw_base).strip()
    if not base_str:
        return

    base = _resolve_synthetic_workspace_dir(base_str)
    repo_key = _sanitize_repo_dir_name(str(meta.get("repo_name", "unknown")))
    repo_root = (base / repo_key).resolve()
    # Ensure repo_root stays under base (after resolve, symlinks could escape).
    try:
        repo_root.relative_to(base)
    except ValueError:
        return

    _ensure_writable_workspace_base(base)
    meta.pop("synthetic_workspace_tear_path", None)
    # A prior task (or a container with this dir bind-mounted) may have left a
    # root-owned tree on the bind mount.
    _force_rmtree(repo_root)
    repo_root.mkdir(parents=True, exist_ok=True)
    for snippet in meta.get("memory_snippets") or []:
        rel = _safe_repo_relative_path(str(snippet.get("path", "")))
        if rel is None:
            continue
        dest = repo_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(str(snippet.get("snippet") or ""), encoding="utf-8")

    file_rel = _safe_repo_relative_path(str(meta.get("file_path", "")))
    if file_rel is not None:
        main_dest = repo_root / file_rel
        main_dest.parent.mkdir(parents=True, exist_ok=True)
        imp = str(meta.get("import_statement") or "").rstrip()
        body_raw = meta.get("all_code")
        if isinstance(body_raw, str) and body_raw.strip():
            body = body_raw
        else:
            body = str(meta.get("cropped_code") or "")
        if imp and body.strip():
            content = f"{imp}\n\n{body}"
        elif imp:
            content = imp
        else:
            content = body
        main_dest.write_text(content, encoding="utf-8")

    meta["synthetic_workspace_tear_path"] = str(repo_root)
    # Scope string for filesystem-backed memory: the resolved workspace root a
    # middleware should use (files live under ``{base}/{repo_key}/...``).
    meta["code_workspace"] = str(base.resolve())


def _clone_task_repo(
    repo: str,
    commit: str,
    dest: Path,
    timeout_s: float = 180,
) -> None:
    """Shallow-clone ``repo`` at ``commit`` into ``dest`` (GitHub HTTPS).

    Uses the fetch-single-commit trick so only the objects reachable from
    ``commit`` are transferred — typically 10–100 MB instead of the full
    history.  ``dest`` must not exist or be empty.

    Raises ``subprocess.CalledProcessError`` on clone/checkout failure.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "init", str(dest)],
        check=True, capture_output=True, timeout=timeout_s,
    )
    # Remove any stale "origin" remote left by a previous partial run before adding
    # a fresh one — git remote add fails with exit 128 if origin already exists.
    subprocess.run(
        ["git", "-C", str(dest), "remote", "remove", "origin"],
        capture_output=True,  # ignore failure — remote may not exist
    )
    subprocess.run(
        ["git", "-C", str(dest), "remote", "add", "origin", url],
        check=True, capture_output=True, timeout=timeout_s,
    )
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit],
        check=True, capture_output=True, timeout=timeout_s,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "FETCH_HEAD"],
        check=True, capture_output=True, timeout=timeout_s,
    )


def _teardown_synthetic_workspace(task: Task) -> None:
    meta = task.metadata or {}
    meta.pop("code_workspace", None)
    tear = meta.pop("synthetic_workspace_tear_path", None)
    if not tear:
        return
    root = Path(str(tear))
    if root.exists():
        _force_rmtree(root)