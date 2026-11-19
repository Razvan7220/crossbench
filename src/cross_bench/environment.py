"""Best-effort capture of the software/hardware environment of a run.

The artifact store writes this next to ``config.json`` and ``summary.json``
as ``environment.json``.  ``config.json`` records what the run *asked for*
(model names, profiles, memory params); this module records what the run
actually executed on:

* the exact command line and working directory;
* Python, OS/kernel, package versions, git revision and lockfile hashes;
* CPU model, logical/physical cores, affinity, frequency, governor and
  cgroup CPU/memory limits;
* GPU inventory (NVIDIA / ROCm / PCI display controllers);
* relevant runtime environment variables (secrets redacted);
* Ollama server version, installed model digests and the referenced models'
  Modelfiles/parameters (including ``num_thread``, which is baked into the
  derived local models).

Everything is best-effort: missing tools or files yield partial data rather
than failing the benchmark run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1

_SECRET_FRAGMENTS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "database_url",
    "db_url",
)

_SECRET_FLAGS = {
    "--api-key",
    "--apikey",
    "--token",
    "--password",
    "--secret",
    "--db-url",
    "--database-url",
}

_ENV_EXACT = {
    "TZ",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "PYTHONHASHSEED",
    "KUBERNETES_SERVICE_HOST",
}

_ENV_PREFIXES = (
    "BENCH_",
    "OLLAMA_",
    "CUDA_",
    "NVIDIA_",
    "ROCR_",
    "HIP_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "HF_",
    "TRANSFORMERS_",
    "TOKENIZERS_",
    "MEM0_",
    "PYTHON",
)

_MODEL_INFO_SUFFIXES = (
    ".embedding_length",
    ".context_length",
    ".block_count",
    ".attention.head_count",
    ".pooling_type",
    ".rope.freq_base",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except (OSError, ValueError):
        return None


def _run(
    cmd: list[str],
    *,
    timeout: float = 2.0,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run a short diagnostic command, returning raw stdout/stderr."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError:
        return {"cmd": cmd, "ok": False, "error": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "ok": False, "error": f"timeout after {timeout}s"}
    except OSError as exc:
        return {"cmd": cmd, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "cmd": cmd,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _expand_cpu_list(spec: str | None) -> list[int]:
    if not spec:
        return []
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = part.split("-", 1)
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        except ValueError:
            continue
    return sorted(set(cpus))


def _format_cpu_ranges(cpus: Iterable[int]) -> str | None:
    ordered = sorted(set(cpus))
    if not ordered:
        return None
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _cpu_list_payload(spec: str | None) -> dict[str, Any] | None:
    if not spec:
        return None
    return {"raw": spec.strip(), "cpus": _expand_cpu_list(spec)}


def _redact_url_credentials(value: str) -> str:
    if "://" not in value:
        return value
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return value
    if parts.username is None and parts.password is None and not parts.query:
        return value

    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parts.username is not None or parts.password is not None:
        netloc = f"<redacted>@{host}"
    if port is not None:
        netloc = f"{netloc}:{port}"

    query = parts.query
    if query:
        query_pairs: list[str] = []
        for item in query.split("&"):
            key, sep, val = item.partition("=")
            if any(frag in key.lower() for frag in _SECRET_FRAGMENTS):
                val = "<redacted>"
            query_pairs.append(f"{key}{sep}{val}")
        query = "&".join(query_pairs)

    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _redact_env_value(key: str, value: str) -> str:
    lowered = key.lower()
    if any(frag in lowered for frag in _SECRET_FRAGMENTS):
        return "<redacted>"
    return _redact_url_credentials(value)


def _redact_command(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if arg in _SECRET_FLAGS:
            redacted.append(arg)
            redact_next = True
            continue
        if "=" in arg:
            key, sep, _value = arg.partition("=")
            if key in _SECRET_FLAGS or any(frag in key.lower() for frag in _SECRET_FRAGMENTS):
                redacted.append(f"{key}{sep}<redacted>")
                continue
        redacted.append(_redact_url_credentials(arg))
    return redacted


def _collect_command(argv: list[str] | None, cwd: Path) -> dict[str, Any]:
    command = _redact_command(list(argv or sys.argv))
    return {
        "argv": command,
        "shell": shlex.join(command),
        "executable": sys.executable,
        "cwd": str(cwd),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }


# ---------------------------------------------------------------------------
# OS / Python / packages
# ---------------------------------------------------------------------------


def _parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    text = _read_text(path)
    if not text:
        return {}
    data: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        data[key.strip()] = value
    return data


def _collect_container() -> dict[str, Any]:
    hints: list[str] = []
    if Path("/.dockerenv").exists():
        hints.append("docker")
    if Path("/run/.containerenv").exists():
        hints.append("podman")
    cgroup = _read_text(Path("/proc/1/cgroup")) or ""
    for marker in ("docker", "kubepods", "containerd", "podman", "lxc"):
        if marker in cgroup:
            hints.append(marker)
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        hints.append("kubernetes")
    return {
        "containerized": bool(hints),
        "hints": sorted(set(hints)),
    }


def _collect_platform() -> dict[str, Any]:
    uname = platform.uname()
    libc = platform.libc_ver()
    return {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "platform": platform.platform(),
        "libc": {"lib": libc[0], "version": libc[1]} if libc[0] else None,
        "hostname": socket.gethostname(),
        "os_release": _parse_os_release(),
        "container": _collect_container(),
    }


def _collect_python() -> dict[str, Any]:
    return {
        "version": sys.version,
        "version_info": list(sys.version_info),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "virtualenv": sys.prefix != sys.base_prefix,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def _collect_packages() -> dict[str, str]:
    from importlib import metadata

    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata.get("Name")
            version = dist.version
        except Exception:
            continue
        if name and version:
            packages[name] = version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _collect_git(cwd: Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []
    env_root = os.environ.get("CROSS_BENCH_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    if cwd is not None:
        candidates.append(cwd)
    candidates.append(Path(__file__).resolve().parents[2])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        probe = _run(["git", "rev-parse", "--show-toplevel"], cwd=candidate)
        if not probe.get("ok"):
            continue
        root = probe["stdout"].strip()
        if not root or root in seen:
            continue
        seen.add(root)

        info: dict[str, Any] = {"available": True, "root": root}
        for key, args in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("branch", ["git", "branch", "--show-current"]),
            ("describe", ["git", "describe", "--always", "--dirty", "--tags"]),
            ("remote", ["git", "remote", "get-url", "origin"]),
        ):
            result = _run(args, cwd=root)
            value = result["stdout"].strip() if result.get("ok") else None
            if key == "remote" and value:
                # A remote can embed a PAT (``https://ghp_...@github.com``).
                value = _redact_url_credentials(value)
            info[key] = value

        status = _run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root,
        )
        lines = [line for line in status["stdout"].splitlines() if line] if status.get("ok") else []
        info["dirty"] = bool(lines)
        info["status"] = lines[:200]
        return info
    return {"available": False}


def _collect_project_files(
    cwd: Path | None = None,
    git_root: str | None = None,
) -> dict[str, Any]:
    roots: list[Path] = []
    if git_root:
        roots.append(Path(git_root))
    if cwd is not None:
        roots.append(cwd)
    roots.append(Path(__file__).resolve().parents[2])

    files: dict[str, Any] = {}
    for root in roots:
        for name in (
            "pyproject.toml",
            "constraints.txt",
            "requirements.lock",
            "uv.lock",
            "Dockerfile",
        ):
            if name in files:
                continue
            path = root / name
            if not path.is_file():
                continue
            files[name] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        if files:
            break
    return files


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def _parse_cpuinfo(path: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {}
    info: dict[str, Any] = {}
    physical_cores: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "model name" and "model" not in info:
            info["model"] = value
        elif key == "vendor_id" and "vendor" not in info:
            info["vendor"] = value
        elif key == "cpu cores" and "cpu_cores" not in info:
            info["cpu_cores"] = _to_int(value)
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
            if physical_id is not None and core_id is not None:
                physical_cores.add((physical_id, core_id))
    if physical_cores:
        info["physical_cores"] = len(physical_cores)
    return info


def _parse_cgroup_cpu_max(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    parts = text.split()
    if len(parts) != 2:
        return None
    quota_s, period_s = parts
    result: dict[str, Any] = {
        "quota_us": None if quota_s == "max" else _to_int(quota_s),
        "period_us": _to_int(period_s),
    }
    if quota_s != "max" and result["quota_us"] and result["period_us"]:
        result["limit_cores"] = result["quota_us"] / result["period_us"]
    else:
        result["limit_cores"] = None
    return result


def _parse_cgroup_memory_max(text: str | None) -> int | None:
    if text is None:
        return None
    value = text.strip()
    if value == "max":
        return None
    parsed = _to_int(value)
    if parsed is None or parsed >= 1 << 60:
        return None
    return parsed


def _collect_cgroup() -> dict[str, Any]:
    root = Path("/sys/fs/cgroup")
    data: dict[str, Any] = {"root": str(root)}

    cpu_max = _read_text(root / "cpu.max")
    is_v2 = bool(cpu_max) or (root / "cgroup.controllers").exists()
    if is_v2:
        data["version"] = "v2"
        if cpu_max:
            data["cpu_max"] = _parse_cgroup_cpu_max(cpu_max)
            data["cpu_max_raw"] = cpu_max.strip()
        else:
            data["cpu_max"] = None
        data["cpuset"] = _cpu_list_payload(_read_text(root / "cpuset.cpus.effective"))
        data["memory_max"] = _parse_cgroup_memory_max(_read_text(root / "memory.max"))
        data["memory_current"] = _to_int(_read_text(root / "memory.current"))
        cpu_weight = _read_text(root / "cpu.weight")
        if cpu_weight:
            data["cpu_weight"] = _to_int(cpu_weight)
        return data

    cpu_dir = root / "cpu"
    quota = _read_text(cpu_dir / "cpu.cfs_quota_us")
    period = _read_text(cpu_dir / "cpu.cfs_period_us")
    if quota is not None and period is not None:
        data["version"] = "v1"
        quota_i = _to_int(quota)
        period_i = _to_int(period)
        data["cpu_cfs_quota_us"] = quota_i
        data["cpu_cfs_period_us"] = period_i
        data["cpu_max"] = {
            "quota_us": None if quota_i is not None and quota_i < 0 else quota_i,
            "period_us": period_i,
            "limit_cores": (
                quota_i / period_i if quota_i is not None and quota_i > 0 and period_i else None
            ),
        }
        data["cpuset"] = _cpu_list_payload(_read_text(root / "cpuset" / "cpuset.cpus"))
        data["memory_max"] = _parse_cgroup_memory_max(
            _read_text(root / "memory" / "memory.limit_in_bytes")
        )
        data["memory_current"] = _to_int(_read_text(root / "memory" / "memory.usage_in_bytes"))
        return data

    data["version"] = None
    return data


def _collect_cpu() -> dict[str, Any]:
    info: dict[str, Any] = _parse_cpuinfo()
    info.setdefault("model", platform.processor() or None)
    info["logical_cores"] = os.cpu_count()

    affinity: list[int] | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        info["physical_cores"] = psutil.cpu_count(logical=False)
        try:
            affinity = sorted(psutil.Process().cpu_affinity())
        except Exception:
            affinity = None
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                info["frequency_mhz"] = {
                    "current": freq.current,
                    "min": freq.min,
                    "max": freq.max,
                }
        except Exception:
            pass
    except Exception:
        try:
            info["physical_cores"] = _parse_cpuinfo().get("physical_cores")
        except Exception:
            pass

    if affinity is None:
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
    if affinity is not None:
        info["affinity"] = affinity
        info["affinity_count"] = len(affinity)
        info["affinity_ranges"] = _format_cpu_ranges(affinity)

    governor = _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    if governor:
        info["governor"] = governor.strip()
    numa = _read_text(Path("/sys/devices/system/node/online"))
    if numa:
        info["numa_nodes"] = numa.strip()
    return info


def _collect_memory() -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        data.update(
            total_bytes=vm.total,
            available_bytes=vm.available,
            used_bytes=vm.used,
            percent=vm.percent,
        )
        swap = psutil.swap_memory()
        data["swap"] = {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "free_bytes": swap.free,
            "percent": swap.percent,
        }
        return data
    except Exception:
        pass

    text = _read_text(Path("/proc/meminfo"))
    if not text:
        return data
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if not parts:
            continue
        parsed = _to_int(parts[0])
        if parsed is None:
            continue
        values[key.strip()] = parsed * 1024
    if "MemTotal" in values:
        data["total_bytes"] = values["MemTotal"]
    if "MemAvailable" in values:
        data["available_bytes"] = values["MemAvailable"]
    return data


def _collect_gpus() -> dict[str, Any]:
    data: dict[str, Any] = {
        "nvidia": [],
        "rocm": None,
        "pci_display_controllers": [],
    }

    if shutil.which("nvidia-smi"):
        query = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            timeout=3.0,
        )
        if query.get("ok"):
            fields = [
                "index",
                "name",
                "uuid",
                "memory_total_mib",
                "driver_version",
                "compute_cap",
            ]
            data["nvidia"] = [
                dict(zip(fields, row)) for row in csv.reader(io.StringIO(query["stdout"])) if row
            ]
        else:
            data["nvidia_error"] = query.get("error") or query.get("stderr")

        full = _run(["nvidia-smi"], timeout=3.0)
        if full.get("ok"):
            match = re.search(r"CUDA Version:\s*([0-9.]+)", full["stdout"])
            if match:
                data["cuda_version"] = match.group(1)

    if shutil.which("rocm-smi"):
        rocm = _run(
            [
                "rocm-smi",
                "--showproductname",
                "--showmeminfo",
                "vram",
                "--showdriverversion",
            ],
            timeout=3.0,
        )
        data["rocm"] = (rocm.get("stdout") or rocm.get("stderr") or "").strip() or None

    if shutil.which("lspci"):
        pci = _run(["lspci", "-nn"], timeout=3.0)
        if pci.get("ok"):
            data["pci_display_controllers"] = [
                line
                for line in pci["stdout"].splitlines()
                if re.search(r"VGA compatible|3D controller|Display controller", line, re.I)
            ]
    return data


def _collect_processes(names: tuple[str, ...] = ("ollama",)) -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    processes: list[dict[str, Any]] = []
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        comm = _read_text(pid_dir / "comm")
        if not comm or not any(name in comm.lower() for name in names):
            continue
        status = _read_text(pid_dir / "status") or ""
        fields: dict[str, str] = {}
        for line in status.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
        cmdline_raw = _read_text(pid_dir / "cmdline") or ""
        cmdline = [part for part in cmdline_raw.split("\0") if part]
        processes.append(
            {
                "pid": int(pid_dir.name),
                "comm": comm.strip(),
                "cmdline": cmdline,
                "cpus_allowed_list": fields.get("Cpus_allowed_list"),
                "mems_allowed_list": fields.get("Mems_allowed_list"),
            }
        )
    return sorted(processes, key=lambda item: item["pid"])


def _load_average() -> list[float] | None:
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Runtime environment variables
# ---------------------------------------------------------------------------


def _collect_environment_variables(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    selected: dict[str, str] = {}
    for key, value in source.items():
        if key in _ENV_EXACT or key.startswith(_ENV_PREFIXES):
            selected[key] = _redact_env_value(key, value)
    return dict(sorted(selected.items()))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _normalise_ollama_base_url(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip()
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, path, "", "")).rstrip("/")


def _looks_like_ollama(url: str | None) -> bool:
    if not url:
        return False
    try:
        parts = urlsplit(url if "://" in url else f"http://{url}")
        return parts.port == 11434
    except ValueError:
        return False


def _ollama_targets(config: Any) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}

    def add(url: str | None, model: str | None, provider: str | None) -> None:
        if provider != "ollama" and not _looks_like_ollama(url):
            return
        base = _normalise_ollama_base_url(url or "http://localhost:11434")
        if not base:
            return
        targets.setdefault(base, set())
        if model:
            targets[base].add(str(model))

    system = getattr(config, "system", None)
    llm = getattr(system, "llm", None)
    if llm is not None:
        extra = getattr(llm, "extra", None) or {}
        add(
            extra.get("base_url") or extra.get("ollama_base_url") or extra.get("api_base"),
            getattr(llm, "model", None),
            getattr(llm, "provider", None),
        )

    embedding = getattr(system, "embedding", None)
    if embedding is not None:
        params = getattr(embedding, "params", None) or {}
        add(
            params.get("base_url") or params.get("ollama_base_url") or params.get("api_base"),
            getattr(embedding, "model", None),
            getattr(embedding, "provider", None),
        )

    memory = getattr(system, "memory", None)
    if memory is not None and getattr(memory, "enabled", True):
        params = getattr(memory, "params", None) or {}
        for provider_key, url_key, model_key in (
            ("llm_provider", "llm_base_url", "llm_model"),
            ("embedder_provider", "embedder_base_url", "embedder_model"),
            ("embedding_provider", "embedding_base_url", "embedding_model"),
        ):
            add(params.get(url_key), params.get(model_key), params.get(provider_key))

    return targets


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 2.0,
) -> tuple[Any | None, str | None]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _filter_ollama_show(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    model_info = data.get("model_info") or {}
    filtered_info: dict[str, Any] = {}
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if key.startswith("general.") or any(
                key.endswith(suffix) for suffix in _MODEL_INFO_SUFFIXES
            ):
                filtered_info[key] = value
    return {
        "details": data.get("details"),
        "parameters": data.get("parameters"),
        "modelfile": data.get("modelfile"),
        "capabilities": data.get("capabilities"),
        "model_info": filtered_info,
    }


def collect_ollama_info(config: Any, *, timeout: float = 2.0) -> dict[str, Any]:
    result: dict[str, Any] = {"probed": False, "targets": []}
    cli = shutil.which("ollama")
    if cli:
        version = _run(["ollama", "--version"], timeout=2.0)
        result["cli"] = {
            "path": cli,
            "version": ((version.get("stdout") or version.get("stderr") or "").strip() or None),
        }

    targets = _ollama_targets(config)
    if not targets:
        return result

    result["probed"] = True
    result["processes"] = _collect_processes(("ollama",))
    for base_url, models in list(targets.items())[:3]:
        target: dict[str, Any] = {
            "base_url": base_url,
            "referenced_models": {},
        }

        version, error = _http_json(f"{base_url}/api/version", timeout=timeout)
        if error:
            target["version_error"] = error
        else:
            target["version"] = version

        tags, error = _http_json(f"{base_url}/api/tags", timeout=timeout)
        if error:
            target["tags_error"] = error
        elif isinstance(tags, dict):
            target["installed_models"] = [
                {
                    key: model.get(key)
                    for key in (
                        "name",
                        "model",
                        "digest",
                        "size",
                        "modified_at",
                        "details",
                    )
                }
                for model in tags.get("models", [])
                if isinstance(model, dict)
            ]

        loaded, error = _http_json(f"{base_url}/api/ps", timeout=timeout)
        if error:
            target["ps_error"] = error
        elif isinstance(loaded, dict):
            target["loaded_models"] = [
                {
                    key: model.get(key)
                    for key in (
                        "name",
                        "model",
                        "digest",
                        "size",
                        "size_vram",
                        "expires_at",
                        "details",
                    )
                }
                for model in loaded.get("models", [])
                if isinstance(model, dict)
            ]

        for model in sorted(models):
            show, error = _http_json(
                f"{base_url}/api/show",
                method="POST",
                payload={"model": model},
                timeout=timeout,
            )
            if error:
                target["referenced_models"][model] = {"error": error}
            else:
                target["referenced_models"][model] = _filter_ollama_show(show)

        result["targets"].append(target)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_environment(
    config: Any,
    *,
    command: list[str] | None = None,
    cwd: str | Path | None = None,
    probe_ollama: bool = True,
) -> dict[str, Any]:
    """Collect a reproducible snapshot of the run environment.

    Args:
        config: The :class:`~cross_bench.models.RunConfig` being executed.
        command: Command line to record. Defaults to :data:`sys.argv`.
        cwd: Working directory to record and use for git discovery.
        probe_ollama: Probe configured Ollama endpoints over HTTP. Tests can
            disable this to stay offline; production runs leave it on.
    """
    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    git = _collect_git(cwd_path)
    git_root = git.get("root") if git.get("available") else None

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": getattr(config, "run_id", None),
        "benchmark": getattr(config, "benchmark", None),
        "collected_at": datetime.now(UTC).isoformat(),
        "command": _collect_command(command, cwd_path),
        "software": {
            "python": _collect_python(),
            "platform": _collect_platform(),
            "packages": _collect_packages(),
            "git": git,
            "project_files": _collect_project_files(
                cwd_path,
                git_root=git_root,
            ),
        },
        "hardware": {
            "cpu": _collect_cpu(),
            "memory": _collect_memory(),
            "gpus": _collect_gpus(),
            "cgroup": _collect_cgroup(),
            "load_average": _load_average(),
            "processes": _collect_processes(("ollama",)),
        },
        "environment": _collect_environment_variables(),
        "ollama": collect_ollama_info(config) if probe_ollama else {"probed": False},
    }
