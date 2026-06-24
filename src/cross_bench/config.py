"""Platform configuration loading from TOML files and environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import tomli

from cross_bench.models import (
    EmbeddingConfig,
    ExecutionMode,
    LLMConfig,
    MemoryConfig,
    PromptStrategyConfig,
    RunConfig,
    SystemConfig,
)

_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

# Support BENCH_CONFIGS_DIR env var (used in Docker where the package is
# installed into site-packages but configs/ lives at /app/configs).
# Falls back to the path relative to source tree for local dev.
CONFIGS_DIR = Path(
    os.environ.get("BENCH_CONFIGS_DIR", "")
    or Path(__file__).resolve().parent.parent.parent / "configs"
)


# Vars already warned about this process — one warning per variable, not
# one per ``${VAR}`` occurrence (a single missing OPENAI_API_KEY can be
# referenced by dozens of profile TOMLs).
_warned_missing_env_vars: set[str] = set()

logger = structlog.get_logger()


def _interpolate_env(value: Any) -> Any:
    """Replace ${VAR} placeholders with environment variable values.

    A ``${VAR}`` that is unset in the environment and has no
    ``:default`` interpolates to ``""`` — config loading must not
    hard-fail just because a secret isn't present (unit tests and
    profile-listing tooling load every TOML without secrets). But a
    silently-empty ``api_key`` surfaces later as a cryptic provider
    401 / daemon-init failure far from the cause, so the miss is
    logged loudly, once per variable per process.
    """
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2)  # None if no :default
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            if var_name not in _warned_missing_env_vars:
                _warned_missing_env_vars.add(var_name)
                logger.warning(
                    "config.env_var_missing",
                    var=var_name,
                    hint=(
                        "interpolated to empty string — set it in the "
                        "environment or .env if this run needs it"
                    ),
                )
            return ""
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and interpolate environment variables."""
    with open(path, "rb") as f:
        data = tomli.load(f)
    return _interpolate_env(data)


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge(merged[k], v)
        else:
            merged[k] = v
    return merged


PROFILES_DIR = CONFIGS_DIR / "profiles"

# Names of built-in profiles (resolved from configs/profiles/<name>.toml)
BUILTIN_PROFILES = ("local", "online", "claude", "o3_mini", "gpt4o_mini", "no_memory",
                    "mem0v1_api", "mem0v1_embedded",
                    "mem0v2_api", "mem0v2_embedded",
                    "evermemos_api", "evermemos_embedded",
                    "graphiti", "hindsight_embedded", "mastra_embedded",
                    "memos_embedded",
                    "gpt41mini", "swebench_agent",
                    "swebench_gold", "swebench_pro_gold",
                    "gemini_3_flash", "qwen3_coder_480b",
                    "seal_haiku45", "seal_prompt_caching", "seal_full_trace",
                    "seal_str_replace_nudge")


def list_profiles() -> list[str]:
    """Return available profile names."""
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.toml"))


def load_run_config(
    config_path: Path | None = None,
    *,
    benchmark: str | None = None,
    profiles: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Build a RunConfig from layered TOML files + CLI overrides.

    Precedence (highest wins):
        CLI overrides > run config file > profiles > benchmark defaults > platform defaults

    Args:
        config_path: Path to a run-specific TOML file.
        benchmark:   Benchmark name (loads ``configs/benchmarks/<name>.toml``).
        profiles:    Profile names to stack — each is merged in order
                     (e.g. ``["online", "claude", "evermemos_api"]``).
        overrides:   Arbitrary dict merged last.
    """
    # Layer 1: platform defaults
    default_path = CONFIGS_DIR / "default.toml"
    merged: dict[str, Any] = {}
    if default_path.exists():
        merged = load_toml(default_path)

    # Layer 2: benchmark defaults
    # Resolved here (before profiles) because the docstring contract is
    # "profiles override benchmark defaults" — so the benchmark TOML must
    # land BEFORE the profile loop. (Pre-2026-05-21 the code had layers 2
    # and 3 swapped, silently letting the benchmark TOML overwrite any
    # ``[benchmark] key = value`` set by a profile mixin. That made
    # ``-p swebench_retrieval_bm25`` a no-op against ``swebench.toml``'s
    # ``retrieval = "none"`` default, which we discovered while
    # debugging the BM25 n=50 dispatch — see the per-task ``config.json``
    # showing ``benchmark_params.retrieval = 'none'``.)
    bench_name = benchmark
    if config_path:
        run_data = load_toml(config_path)
        bench_name = bench_name or run_data.get("run", {}).get("benchmark")

    if bench_name:
        bench_cfg_path = CONFIGS_DIR / "benchmarks" / f"{bench_name}.toml"
        if bench_cfg_path.exists():
            merged = _merge(merged, load_toml(bench_cfg_path))

    # Layer 3: profiles — merged in order (later profiles override earlier ones)
    for profile in profiles or []:
        if Path(profile).is_absolute():
            # Allow absolute-path profiles (e.g. tempfile written by parallel_runner)
            profile_path = Path(profile)
        else:
            profile_path = PROFILES_DIR / f"{profile}.toml"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile '{profile}' not found. Available: {list_profiles()}. Path: {profile_path}"
            )
        merged = _merge(merged, load_toml(profile_path))

    # Layer 4: run config file
    if config_path:
        merged = _merge(merged, load_toml(config_path))

    # Layer 5: CLI overrides
    if overrides:
        merged = _merge(merged, overrides)

    # Layer 6: bench-LLM → memory inheritance.
    # Embedded memory systems run their own internal LLM/embedder for fact
    # extraction, embeddings, agentic retrieval, etc. By default we want
    # those internal calls to use the same model the user already chose for
    # the bench (i.e. the [system.llm] / [system.embedding] block).
    # Otherwise stacking ``-p local -p mem0v2_embedded`` would silently use
    # OpenAI internally even though the bench answer model is Ollama —
    # exactly the kind of axis confusion the leaderboard mustn't have.
    #
    # The user can still diverge by stacking an explicit override mixin
    # like ``-p mem_internal_gpt4o_mini`` AFTER the memory profile; the
    # deep-merge already preserves whichever keys the override sets.
    merged = _inherit_bench_llm_into_memory(merged)

    return _build_run_config(merged)


# ---------------------------------------------------------------------------
# Bench-LLM → embedded-memory inheritance
# ---------------------------------------------------------------------------

# Provider-name suffix that means "memory algorithm runs in-process".
# These are the providers whose internal LLM/embedder choice is ours to
# decide; API memory providers run server-side and their internal LLM is
# vendor-managed.
_EMBEDDED_PROVIDER_SUFFIXES: tuple[str, ...] = ("_embedded",)

# Per-middleware translation between the canonical bench LLM keys
# (``provider``/``model``/``base_url`` under ``[system.llm]``) and the
# memory params each middleware actually reads. Adding a new middleware
# to the inheritance contract is a one-line entry here.
#
# Format: ``provider_name -> {bench_key: memory_param_key, ...}``.
# Keys not listed are simply not inherited; the middleware keeps its own
# defaults for them.
_MEMORY_LLM_KEY_MAP: dict[str, dict[str, str]] = {
    "mem0v1_embedded": {
        "llm_provider": "llm_provider",
        "llm_model": "llm_model",
        "llm_base_url": "llm_base_url",
        "emb_provider": "embedder_provider",
        "emb_model": "embedder_model",
    },
    "mem0v2_embedded": {
        "llm_provider": "llm_provider",
        "llm_model": "llm_model",
        "llm_base_url": "llm_base_url",
        "emb_provider": "embedder_provider",
        "emb_model": "embedder_model",
        # AWS Bedrock fields. Inherited when the bench LLM is
        # ``anthropic_bedrock`` (normalized to ``aws_bedrock`` for mem0).
        # mem0's ``AWSBedrockConfig`` reads ``aws_region`` + ``aws_profile``
        # off the LLM config; without these the bedrock client can't be
        # constructed and fact extraction silently empties the store.
        "aws_region": "aws_region",
        "aws_profile": "aws_profile",
    },
    # evermemos uses ``llm_model`` + ``llm_base_url`` + ``embedding_model``
    # (no separate provider key — the model+base_url combo is OpenAI-
    # compatible). We don't infer ``llm_api_key`` here; the profile
    # carries it via ``${OPENAI_API_KEY}`` interpolation.
    "evermemos_embedded": {
        "llm_model": "llm_model",
        "llm_base_url": "llm_base_url",
        "emb_model": "embedding_model",
    },
    # MemOS uses the same shape as mem0: provider + model + base_url for
    # the extractor LLM, plus embedder for the ``general_text`` backend.
    # Naive backend doesn't use the embedder at retrieval time but the
    # LLM is still required at config-validation time.
    "memos_embedded": {
        "llm_provider": "llm_provider",
        "llm_model": "llm_model",
        "llm_base_url": "llm_base_url",
        "emb_provider": "embedder_provider",
        "emb_model": "embedder_model",
    },
}


def _is_embedded_memory(provider: str) -> bool:
    return any(provider.endswith(s) for s in _EMBEDDED_PROVIDER_SUFFIXES)


# Bench-layer provider names that map to a standard memory-side
# provider name. Empty values left untouched.
_BENCH_TO_MEMORY_PROVIDER: dict[str, str] = {
    # CrossBench's ``generic_openai`` is "OpenAI SDK with a custom
    # base_url" — used by ``local.toml`` to talk to Ollama's
    # OpenAI-compat endpoint. The memory side just needs ``openai`` +
    # the base_url override to do the same.
    "generic_openai": "openai",
    # ``anthropic_bedrock`` is the CrossBench LLM driver that routes
    # Anthropic Messages calls through AWS Bedrock (IAM auth, no static
    # API key). Mem0 / MemOS / etc. spell the same provider as
    # ``aws_bedrock`` — translate so the internal-LLM config inherits
    # cleanly. Without this, mem0 raises ``Unsupported LLM provider:
    # anthropic_bedrock`` at ``Memory.from_config()`` time and every
    # task ingests an empty store.
    "anthropic_bedrock": "aws_bedrock",
}


def _normalize_provider_for_memory(name: Any) -> Any:
    if not isinstance(name, str) or not name:
        return name
    return _BENCH_TO_MEMORY_PROVIDER.get(name, name)


def _inherit_bench_llm_into_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Copy bench LLM/embedder choice into memory params if absent.

    Applies to embedded-memory providers. Operates on the merged config dict
    (post profile-stack).

    Only fills keys the memory params have NOT already set — explicit user choice
    always wins.
    """
    sys = data.get("system") or {}
    mem = sys.get("memory")
    if not isinstance(mem, dict):
        return data

    provider = mem.get("provider", "")
    if not provider or not _is_embedded_memory(provider):
        return data

    key_map = _MEMORY_LLM_KEY_MAP.get(provider)
    if key_map is None:
        # Provider opted into inheritance but has no translation table yet.
        return data

    bench_llm = sys.get("llm") or {}
    bench_emb = sys.get("embedding") or {}

    # Memory params live two ways in TOML — top-level under
    # ``[system.memory]`` or nested under ``[system.memory.params]``.
    # Look in both before deciding whether to inherit.
    nested_params = mem.get("params") if isinstance(mem.get("params"), dict) else None
    has_key = lambda k: k in mem or (nested_params is not None and k in nested_params)

    # Source values from the bench-side blocks. ``base_url`` lives in the
    # ``LLMConfig.extra`` bag, which in TOML is a flat ``[system.llm]``
    # key — so we read it directly off ``sys["llm"]``.
    #
    # Normalize the provider name: CrossBench's ``generic_openai``
    # routes OpenAI-SDK calls through a custom ``base_url`` (used by the
    # ``local`` profile for Ollama's OpenAI-compat endpoint). Mem0 / etc.
    # don't know that name — but they do support ``openai`` + a
    # ``*_base_url`` override, which is the same trick. Translate so the
    # internal-LLM config inherits cleanly.
    bench_llm_provider = _normalize_provider_for_memory(bench_llm.get("provider"))

    sources = {
        "llm_provider": bench_llm_provider,
        "llm_model": bench_llm.get("model"),
        "llm_base_url": bench_llm.get("base_url"),
        "emb_provider": bench_emb.get("provider"),
        "emb_model": bench_emb.get("model"),
        # Bedrock-specific fields live as flat keys on ``[system.llm]``
        # in TOML (they end up in ``LLMConfig.extra`` post-build, but
        # this inheritance step runs on the raw merged dict). Only the
        # mem0 entry in ``_MEMORY_LLM_KEY_MAP`` requests them today —
        # other providers ignore them via key_map filtering.
        "aws_region": bench_llm.get("aws_region"),
        "aws_profile": bench_llm.get("aws_profile"),
    }

    inherited: dict[str, Any] = {}
    for bench_key, mem_key in key_map.items():
        if has_key(mem_key):
            continue  # explicit override — leave alone
        val = sources.get(bench_key)
        if val is None or val == "":
            continue
        inherited[mem_key] = val

    if not inherited:
        return data

    # Write inherited values back. We always write at the top level of
    # ``[system.memory]`` (not into ``params``) — the loader flattens
    # both shapes into ``MemoryConfig.params`` so middleware reads work
    # the same either way. Top-level keeps the inheritance visible in
    # debug dumps.
    new_data = {**data, "system": {**sys, "memory": {**mem, **inherited}}}
    return new_data


def _build_run_config(data: dict[str, Any]) -> RunConfig:
    run = data.get("run", {})
    sys = data.get("system", {})
    bench_params = data.get("benchmark", {})

    llm_data = sys.get("llm", {})
    llm = LLMConfig(
        provider=llm_data.get("provider", "openai"),
        model=llm_data.get("model", "gpt-4o"),
        temperature=llm_data.get("temperature", 0.0),
        max_tokens=llm_data.get("max_tokens", 4096),
        context_window=llm_data.get("context_window", 128_000),
        top_p=llm_data.get("top_p", 1.0),
        seed=llm_data.get("seed"),
        extra={k: v for k, v in llm_data.items()
               if k not in ("provider", "model", "temperature", "max_tokens", "context_window", "top_p", "seed")},
    )

    mem_data = sys.get("memory")
    memory = None
    if mem_data and mem_data.get("provider"):
        # Flatten: if TOML uses [system.memory.params] section, the parsed dict
        # has {"provider", "enabled", "params": {...}}.  Extract and merge.
        raw_params = {k: v for k, v in mem_data.items() if k not in ("provider", "enabled")}
        if "params" in raw_params and isinstance(raw_params["params"], dict):
            # Merge the nested params dict into the top level
            nested = raw_params.pop("params")
            raw_params.update(nested)
        memory = MemoryConfig(
            provider=mem_data["provider"],
            enabled=mem_data.get("enabled", True),
            params=raw_params,
        )

    emb_data = sys.get("embedding")
    embedding = None
    if emb_data and emb_data.get("provider"):
        embedding = EmbeddingConfig(
            provider=emb_data["provider"],
            model=emb_data.get("model", ""),
            params={k: v for k, v in emb_data.items() if k not in ("provider", "model")},
        )

    ps_data = sys.get("prompt_strategy", {})
    prompt_strategy = PromptStrategyConfig(
        type=ps_data.get("type", "zero_shot"),
        params={k: v for k, v in ps_data.items() if k != "type"},
    )

    system = SystemConfig(
        llm=llm,
        memory=memory,
        embedding=embedding,
        prompt_strategy=prompt_strategy,
    )

    return RunConfig(
        benchmark=run.get("benchmark", "humaneval"),
        system=system,
        execution_mode=ExecutionMode(run.get("execution_mode", "single_shot")),
        max_agent_steps=run.get("max_agent_steps", 20),
        timeout_seconds=run.get("timeout_seconds", 300),
        sandbox_image=run.get("sandbox_image", "") or os.environ.get("BENCH_SANDBOX_IMAGE", ""),
        benchmark_params=bench_params,
        tags=run.get("tags", []),
    )
