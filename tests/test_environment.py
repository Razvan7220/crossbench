"""Tests for the per-run environment snapshot.

``environment.json`` sits next to ``config.json`` / ``summary.json`` and
answers the reproducibility questions ``config.json`` cannot: which exact
command ran, which package versions / git revision were used, how many
cores the process could actually use, and which Ollama model digests +
``num_thread`` parameters were loaded.

Everything here is offline: the Ollama HTTP layer is monkeypatched and
``probe_ollama=False`` short-circuits live probing in the integration
test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cross_bench import environment as env_mod
from cross_bench.environment import (
    _normalise_ollama_base_url,
    collect_environment,
    collect_ollama_info,
)
from cross_bench.models import (
    LLMConfig,
    MemoryConfig,
    PromptStrategyConfig,
    RunConfig,
    SystemConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(*, memory: MemoryConfig | None = None) -> RunConfig:
    return RunConfig(
        run_id="env-test-run",
        benchmark="humaneval",
        system=SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
            memory=memory,
        ),
    )


# ---------------------------------------------------------------------------
# collect_environment — shape, redaction, hardware
# ---------------------------------------------------------------------------


def test_collect_environment_records_reproducibility_fields(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "super-secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("BENCH_ARTIFACTS_BUCKET", "bench-artifacts")
    monkeypatch.setenv("UNRELATED_VAR", "not-captured")

    snapshot = collect_environment(
        _config(),
        command=[
            "cross-bench",
            "run",
            "--api-key",
            "sk-secret",
            "--db-url",
            "postgresql://user:pass@db.example:5432/bench",
            "-p",
            "online",
        ],
        probe_ollama=False,
    )

    assert snapshot["schema_version"] == env_mod.SCHEMA_VERSION
    assert snapshot["run_id"] == "env-test-run"
    assert snapshot["benchmark"] == "humaneval"

    # The command line is what makes a run reproducible beyond config.json.
    assert snapshot["command"]["argv"][:3] == ["cross-bench", "run", "--api-key"]
    assert snapshot["command"]["argv"][3] == "<redacted>"
    assert snapshot["command"]["argv"][4] == "--db-url"
    assert snapshot["command"]["argv"][5] == "<redacted>"
    assert snapshot["command"]["shell"].startswith("cross-bench run")

    # Software: Python / packages / git / lockfile hashes.
    software = snapshot["software"]
    assert software["python"]["version_info"][0] == 3
    assert "cross-bench" in software["packages"]
    assert "available" in software["git"]
    assert software["project_files"]["pyproject.toml"]["sha256"]

    # Hardware: enough detail to explain latency differences between hosts.
    hardware = snapshot["hardware"]
    assert hardware["cpu"]["logical_cores"] >= 1
    assert hardware["cpu"]["affinity_count"] >= 1
    assert hardware["cpu"]["affinity_ranges"]
    assert hardware["memory"]["total_bytes"] > 0
    assert set(hardware["gpus"]) >= {"nvidia", "rocm", "pci_display_controllers"}
    assert "cgroup" in hardware

    # Selected environment variables are captured, secrets are redacted,
    # and unrelated variables stay out of the artifact.
    environment = snapshot["environment"]
    assert environment["OLLAMA_API_KEY"] == "<redacted>"
    assert environment["HF_TOKEN"] == "<redacted>"
    assert environment["BENCH_ARTIFACTS_BUCKET"] == "bench-artifacts"
    assert "UNRELATED_VAR" not in environment

    assert snapshot["ollama"] == {"probed": False}


def test_git_remote_credentials_are_redacted(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return {"cmd": cmd, "ok": True, "stdout": "/repo\n", "stderr": ""}
        if cmd[:4] == ["git", "remote", "get-url", "origin"]:
            return {
                "cmd": cmd,
                "ok": True,
                "stdout": "https://ghp_secret@github.com/org/repo.git\n",
                "stderr": "",
            }
        return {"cmd": cmd, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(env_mod, "_run", fake_run)
    snapshot = collect_environment(_config(), cwd=Path("/repo"), probe_ollama=False)

    assert snapshot["software"]["git"]["remote"] == "https://<redacted>@github.com/org/repo.git"


def test_cpu_helpers_parse_ranges_and_cgroup_limits():
    assert env_mod._expand_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert env_mod._format_cpu_ranges([0, 1, 2, 3, 8, 10, 11]) == "0-3,8,10-11"
    assert env_mod._parse_cgroup_cpu_max("400000 100000") == {
        "quota_us": 400000,
        "period_us": 100000,
        "limit_cores": 4.0,
    }
    assert env_mod._parse_cgroup_cpu_max("max 100000") == {
        "quota_us": None,
        "period_us": 100000,
        "limit_cores": None,
    }


# ---------------------------------------------------------------------------
# Ollama inventory
# ---------------------------------------------------------------------------


def test_normalise_ollama_url_strips_v1_and_credentials():
    assert _normalise_ollama_base_url("http://localhost:11434/v1/") == ("http://localhost:11434")
    assert _normalise_ollama_base_url("localhost:11434") == "http://localhost:11434"
    assert _normalise_ollama_base_url("http://user:pass@localhost:11434/v1") == (
        "http://localhost:11434"
    )
    assert _normalise_ollama_base_url("https://ollama.example.com/api") == (
        "https://ollama.example.com/api"
    )


def test_collect_ollama_info_captures_digests_and_model_parameters(monkeypatch):
    monkeypatch.setattr(env_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(env_mod, "_collect_processes", lambda _names=(): [])
    calls: list[tuple[str, str]] = []

    def fake_http(
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float = 2.0,
    ):
        calls.append((method, url))
        if url.endswith("/api/version"):
            return {"version": "0.40.16"}, None
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "gte-qwen2:8t",
                        "model": "gte-qwen2:8t",
                        "digest": "sha256:abc123",
                        "size": 3558625664,
                        "modified_at": "2026-11-17T13:56:00Z",
                        "details": {
                            "format": "gguf",
                            "embedding_length": 1536,
                            "quantization_level": "F16",
                        },
                    },
                    {
                        "name": "qwen2.5:7b-instruct-8t",
                        "digest": "sha256:def456",
                        "details": {"quantization_level": "Q4_K_M"},
                    },
                ]
            }, None
        if url.endswith("/api/ps"):
            return {"models": []}, None
        if url.endswith("/api/show"):
            assert payload is not None
            return {
                "details": {"embedding_length": 1536},
                "parameters": "num_thread 8\nnum_ctx 8192\n",
                "modelfile": "FROM /models/gte.gguf\n",
                "capabilities": ["embedding"],
                "model_info": {
                    "qwen2.embedding_length": 1536,
                    "qwen2.pooling_type": 1,
                    "general.architecture": "qwen2",
                    "tokenizer.ggml.tokens": ["a", "b"],  # noisy, filtered out
                },
            }, None
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(env_mod, "_http_json", fake_http)

    memory = MemoryConfig(
        provider="mem0v2_embedded",
        params={
            "llm_provider": "ollama",
            "llm_model": "qwen2.5:7b-instruct-8t",
            "llm_base_url": "http://localhost:11434/v1",
            "embedder_provider": "ollama",
            "embedder_model": "gte-qwen2:8t",
            "embedder_base_url": "http://localhost:11434/v1",
        },
    )

    info = collect_ollama_info(_config(memory=memory))

    assert info["probed"] is True
    assert len(info["targets"]) == 1
    target = info["targets"][0]
    # The native Ollama API is probed without the OpenAI-compatible /v1 suffix.
    assert target["base_url"] == "http://localhost:11434"
    assert target["version"] == {"version": "0.40.16"}
    assert [m["name"] for m in target["installed_models"]] == [
        "gte-qwen2:8t",
        "qwen2.5:7b-instruct-8t",
    ]
    assert target["installed_models"][0]["digest"] == "sha256:abc123"
    assert set(target["referenced_models"]) == {
        "gte-qwen2:8t",
        "qwen2.5:7b-instruct-8t",
    }

    show = target["referenced_models"]["gte-qwen2:8t"]
    assert "num_thread 8" in show["parameters"]
    assert show["model_info"]["qwen2.embedding_length"] == 1536
    assert show["model_info"]["qwen2.pooling_type"] == 1
    assert show["model_info"]["general.architecture"] == "qwen2"
    # No giant tokenizer arrays or other noise in the artifact.
    assert "tokenizer.ggml.tokens" not in show["model_info"]

    # Every /api/show call goes to the native API, never to /v1.
    show_calls = [url for method, url in calls if url.endswith("/api/show")]
    assert show_calls and all("/v1" not in url for url in show_calls)


def test_collect_ollama_info_does_not_probe_non_ollama_configs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no HTTP probe expected for non-Ollama configs")

    monkeypatch.setattr(env_mod, "_http_json", forbidden)
    monkeypatch.setattr(env_mod.shutil, "which", lambda _name: None)

    info = collect_ollama_info(_config())
    assert info == {"probed": False, "targets": []}


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434/v1", True),
        ("https://api.openai.com/v1", False),
        (None, False),
    ],
)
def test_looks_like_ollama(url, expected):
    assert env_mod._looks_like_ollama(url) is expected
