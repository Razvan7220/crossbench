"""Test configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest

from cross_bench.config import (
    BUILTIN_PROFILES,
    _interpolate_env,
    load_run_config,
    load_toml,
)


def test_env_interpolation():
    os.environ["TEST_KEY_XYZ"] = "secret_value"
    result = _interpolate_env({"key": "${TEST_KEY_XYZ}", "nested": {"k": "${TEST_KEY_XYZ}"}})
    assert result["key"] == "secret_value"
    assert result["nested"]["k"] == "secret_value"
    del os.environ["TEST_KEY_XYZ"]


def test_env_interpolation_missing_var_warns_once_and_returns_empty(monkeypatch):
    """Unset ``${VAR}`` with no default → ``""`` + a one-time warning.

    The empty-string fallback is load-bearing (config loading must not
    hard-fail without secrets — this very test suite loads every
    profile TOML without them), but the miss must be loud: a silently
    empty ``api_key`` otherwise surfaces as a cryptic provider 401 far
    from the cause.
    """
    import cross_bench.config as config_mod

    monkeypatch.delenv("CB_TEST_MISSING_VAR", raising=False)
    config_mod._warned_missing_env_vars.discard("CB_TEST_MISSING_VAR")

    result = _interpolate_env("api_key=${CB_TEST_MISSING_VAR}")
    assert result == "api_key="
    # Recorded for warn-once dedup — the observable side effect of the
    # warning path.
    assert "CB_TEST_MISSING_VAR" in config_mod._warned_missing_env_vars

    # Second occurrence: same value, dedup set unchanged.
    assert _interpolate_env("${CB_TEST_MISSING_VAR}") == ""
    assert "CB_TEST_MISSING_VAR" in config_mod._warned_missing_env_vars


def test_env_interpolation_default_suppresses_missing_warning(monkeypatch):
    """``${VAR:default}`` is an explicit opt-in to absence — no warning."""
    import cross_bench.config as config_mod

    monkeypatch.delenv("CB_TEST_DEFAULTED_VAR", raising=False)
    config_mod._warned_missing_env_vars.discard("CB_TEST_DEFAULTED_VAR")

    assert _interpolate_env("${CB_TEST_DEFAULTED_VAR:fallback}") == "fallback"
    assert "CB_TEST_DEFAULTED_VAR" not in config_mod._warned_missing_env_vars


def test_env_interpolation_set_var_does_not_warn(monkeypatch):
    import cross_bench.config as config_mod

    monkeypatch.setenv("CB_TEST_SET_VAR", "v")
    config_mod._warned_missing_env_vars.discard("CB_TEST_SET_VAR")

    assert _interpolate_env("${CB_TEST_SET_VAR}") == "v"
    assert "CB_TEST_SET_VAR" not in config_mod._warned_missing_env_vars


def test_load_toml(tmp_path):
    toml_file = tmp_path / "test.toml"
    toml_file.write_text("""
[run]
benchmark = "humaneval"
execution_mode = "single_shot"

[system.llm]
provider = "openai"
model = "gpt-4o"
temperature = 0.0
""")
    data = load_toml(toml_file)
    assert data["run"]["benchmark"] == "humaneval"
    assert data["system"]["llm"]["model"] == "gpt-4o"


def test_load_run_config_from_file(tmp_path):
    toml_file = tmp_path / "run.toml"
    toml_file.write_text("""
[run]
benchmark = "mbpp"
execution_mode = "single_shot"

[system.llm]
provider = "anthropic"
model = "claude-sonnet-4-6"

[system.prompt_strategy]
type = "chain_of_thought"
""")
    config = load_run_config(toml_file)
    assert config.benchmark == "mbpp"
    assert config.system.llm.provider == "anthropic"
    assert config.system.prompt_strategy.type == "chain_of_thought"


def test_load_run_config_with_overrides(tmp_path):
    toml_file = tmp_path / "run.toml"
    toml_file.write_text("""
[run]
benchmark = "humaneval"

[system.llm]
provider = "openai"
model = "gpt-4o"

[system.prompt_strategy]
type = "zero_shot"
""")
    config = load_run_config(
        toml_file,
        overrides={"run": {"execution_mode": "agentic"}, "benchmark": {"limit": 5}},
    )
    assert config.execution_mode.value == "agentic"
    assert config.benchmark_params.get("limit") == 5


# ---------------------------------------------------------------------------
# Precedence: profile mixin must override the benchmark TOML's [benchmark]
# defaults. Locks the 2026-05-21 fix where the layer order was swapped and
# the benchmark TOML was silently overwriting any [benchmark] key set by a
# profile (e.g. -p swebench_retrieval_bm25 setting retrieval=bm25 was being
# clobbered by configs/benchmarks/swebench.toml's retrieval=none default).
# ---------------------------------------------------------------------------


def test_profile_overrides_benchmark_default_for_benchmark_keys(tmp_path, monkeypatch):
    """A profile mixin's ``[benchmark]`` keys must beat the benchmark TOML's
    ``[benchmark]`` defaults.

    Without this guarantee, ``-p swebench_retrieval_bm25`` would be a no-op
    against ``configs/benchmarks/swebench.toml``'s ``retrieval = "none"`` —
    exactly the bug that wasted a full n=50 SWE-bench dispatch on
    2026-05-21 before being diagnosed by reading the per-task config.json.
    """
    import cross_bench.config as config_mod

    fake_root = tmp_path / "configs"
    (fake_root / "benchmarks").mkdir(parents=True)
    (fake_root / "profiles").mkdir()

    # Benchmark default sets retrieval=none — mirrors swebench.toml.
    (fake_root / "benchmarks" / "fakebench.toml").write_text(
        '[run]\nbenchmark = "fakebench"\n\n[benchmark]\nretrieval = "none"\n'
    )
    # Profile mixin tries to override it to bm25.
    (fake_root / "profiles" / "fake_retrieval_bm25.toml").write_text(
        '[benchmark]\nretrieval = "bm25"\n'
    )

    monkeypatch.setattr(config_mod, "CONFIGS_DIR", fake_root)
    monkeypatch.setattr(config_mod, "PROFILES_DIR", fake_root / "profiles")

    cfg = config_mod.load_run_config(
        benchmark="fakebench",
        profiles=["fake_retrieval_bm25"],
    )
    assert cfg.benchmark_params.get("retrieval") == "bm25", (
        "Profile mixin's [benchmark] retrieval should override the "
        "benchmark TOML's [benchmark] retrieval, but got "
        f"{cfg.benchmark_params.get('retrieval')!r}"
    )


def test_builtin_profiles_must_not_set_scaffold_run_keys():
    """Built-in profile TOMLs MUST NOT set ``[run] execution_mode`` /
    ``max_agent_steps`` / ``timeout_seconds``.

    Profiles declare WHO the run uses (which LLM / which memory).
    Scaffold (``execution_mode`` etc.) is a BENCHMARK concern — each
    ``configs/benchmarks/<name>.toml`` declares its own. With profile >
    benchmark precedence (locked by the test above), any profile that
    sets a scaffold key silently clobbers the benchmark TOML's choice.

    Locks the 2026-05-21 fallout: ``mem0v2_embedded.toml`` had
    ``execution_mode = "conversational"`` for LoCoMo compatibility,
    which silently broke SWE-bench × mem0v2 by routing through
    one-shot patch generation instead of the mini-SWE-agent loop
    (250-step agentic, with Option-E episodic memory). The
    50-task n=50 BM25 retest produced 4/50 resolved (8%) under the
    wrong scaffold — worse than the broken-BM25 control's 5/50 (10%)
    on the proper scaffold.

    If you genuinely need a scaffold override for a non-default
    benchmark stack, encode it in a benchmark-specific profile mixin
    (e.g. ``swebench_step_50.toml``) and stack it explicitly.
    """
    import cross_bench.config as config_mod

    forbidden_keys = ("execution_mode", "max_agent_steps", "timeout_seconds")
    # Carve-out: profiles prefixed ``swebench_steps_`` are explicitly
    # named step-cap mixins (e.g. ``swebench_steps_100.toml``). The
    # ``max_agent_steps`` key is allowed on those; everything else
    # (``execution_mode``, ``timeout_seconds``) stays locked because
    # those genuinely change scaffold semantics and were the actual
    # source of the 2026-05-21 regression. The naming prefix makes
    # the intent obvious to anyone stacking ``-p`` flags — they're
    # opting into a benchmark-affiliated tunable, not a memory mixin
    # silently flipping scaffold.
    def _is_allowed(profile_name: str, key: str) -> bool:
        if key == "max_agent_steps" and profile_name.startswith("swebench_steps_"):
            return True
        return False

    offenders: list[str] = []

    for profile_path in sorted(config_mod.PROFILES_DIR.glob("*.toml")):
        data = config_mod.load_toml(profile_path)
        run_block = data.get("run", {})
        if not isinstance(run_block, dict):
            continue
        for k in forbidden_keys:
            if k in run_block and not _is_allowed(profile_path.stem, k):
                offenders.append(f"{profile_path.name}:[run].{k}={run_block[k]!r}")

    assert not offenders, (
        "Profile TOMLs set scaffold-domain [run] keys (these belong in "
        "configs/benchmarks/<name>.toml, not in profiles — see the bug "
        "trail at deviations.md): " + "; ".join(offenders)
    )


def test_swebench_steps_profiles_set_max_agent_steps():
    """``swebench_steps_<N>.toml`` profiles cap the agent at N steps."""
    import cross_bench.config as config_mod

    expected = {
        "swebench_steps_100": 100,
        "swebench_steps_150": 150,
        "swebench_steps_200": 200,
        "swebench_steps_250": 250,
    }
    for name, want in expected.items():
        path = config_mod.PROFILES_DIR / f"{name}.toml"
        assert path.exists(), f"missing profile {path}"
        data = config_mod.load_toml(path)
        assert data["run"]["max_agent_steps"] == want


def test_swebench_steps_100_overrides_benchmark_default(monkeypatch):
    """Stacking ``-p swebench_steps_100`` on a SWE-bench run produces
    ``max_agent_steps = 100`` even though ``swebench.toml`` sets 250.

    Layer precedence: profile > benchmark > default. The carve-out in
    ``test_builtin_profiles_must_not_set_scaffold_run_keys`` allows
    these specific profiles to do this.
    """
    import cross_bench.config as config_mod

    cfg = config_mod.load_run_config(
        benchmark="swebench",
        profiles=["swebench_steps_100"],
    )
    assert cfg.max_agent_steps == 100


def test_swebench_scaffold_survives_memory_mixin_stack():
    """End-to-end: ``-p online -p mem0v2_embedded -b swebench`` must
    materialize ``execution_mode = "mini_swe_agent"`` — the scaffold
    declared by ``configs/benchmarks/swebench.toml``.

    Concrete regression: under the wrong scaffold the 50-task BM25
    retest at run ``swebench-mem0v2-bm25-n50-real-20260521-141741``
    produced ``execution_mode: "conversational"`` and ran SWE-bench
    as one-shot patch generation (no agent loop, no Option-E
    episodic memory). This test fails fast if any future profile
    re-introduces a scaffold override.
    """
    import cross_bench.config as config_mod

    cfg = config_mod.load_run_config(
        benchmark="swebench",
        profiles=["online", "mem0v2_embedded"],
    )
    assert cfg.execution_mode.value == "mini_swe_agent", (
        f"Expected SWE-bench to materialize execution_mode=mini_swe_agent, "
        f"got {cfg.execution_mode.value!r}. A stacked profile is likely "
        f"clobbering the benchmark TOML's [run] scaffold choice."
    )
    assert cfg.max_agent_steps == 250, (
        f"Expected max_agent_steps=250 from swebench.toml, "
        f"got {cfg.max_agent_steps}."
    )
    assert cfg.timeout_seconds == 3600, (
        f"Expected timeout_seconds=3600 from swebench.toml, "
        f"got {cfg.timeout_seconds}."
    )


def test_locomo_scaffold_survives_memory_mixin_stack():
    """Sibling test to the SWE-bench one: ``-p online -p mem0v2_embedded
    -b locomo_original`` must still materialize
    ``execution_mode = "conversational"``. The profile scrub
    didn't break the historical conversational-memory path —
    that scaffold comes from ``configs/benchmarks/locomo_original.toml``
    after the scrub, not from the profile."""
    import cross_bench.config as config_mod

    cfg = config_mod.load_run_config(
        benchmark="locomo_original",
        profiles=["online", "mem0v2_embedded"],
    )
    assert cfg.execution_mode.value == "conversational", (
        f"Expected LoCoMo to materialize execution_mode=conversational, "
        f"got {cfg.execution_mode.value!r}."
    )


def test_cli_override_beats_profile(tmp_path, monkeypatch):
    """CLI override layer must win over a profile that already overrode the
    benchmark default. Locks the full precedence chain:
    CLI > run-config > profile > benchmark > platform."""
    import cross_bench.config as config_mod

    fake_root = tmp_path / "configs"
    (fake_root / "benchmarks").mkdir(parents=True)
    (fake_root / "profiles").mkdir()

    (fake_root / "benchmarks" / "fakebench.toml").write_text(
        '[run]\nbenchmark = "fakebench"\n\n[benchmark]\nretrieval = "none"\n'
    )
    (fake_root / "profiles" / "fake_retrieval_bm25.toml").write_text(
        '[benchmark]\nretrieval = "bm25"\n'
    )

    monkeypatch.setattr(config_mod, "CONFIGS_DIR", fake_root)
    monkeypatch.setattr(config_mod, "PROFILES_DIR", fake_root / "profiles")

    cfg = config_mod.load_run_config(
        benchmark="fakebench",
        profiles=["fake_retrieval_bm25"],
        overrides={"benchmark": {"retrieval": "oracle"}},
    )
    # CLI override 'oracle' beats both profile 'bm25' and benchmark 'none'.
    assert cfg.benchmark_params.get("retrieval") == "oracle"


# ---------------------------------------------------------------------------
# Bench-LLM → memory inheritance (embedded)
# ---------------------------------------------------------------------------

from cross_bench.config import _inherit_bench_llm_into_memory


def test_inherit_fills_mem0v2_from_bench_openai():
    """Stack ``-p online -p mem0v2_embedded`` → mem0 internal = gpt-4o."""
    data = {
        "system": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "memory": {
                "provider": "mem0v2_embedded",
                "user_id": "bench_mem0v2",
                # NO llm_*/embedder_* keys — should inherit
            },
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    assert mem["llm_provider"] == "openai"
    assert mem["llm_model"] == "gpt-4o"
    assert mem["embedder_provider"] == "openai"
    assert mem["embedder_model"] == "text-embedding-3-small"
    # untouched keys still there
    assert mem["user_id"] == "bench_mem0v2"


def test_inherit_translates_generic_openai_to_openai():
    """``-p local`` (generic_openai → ollama-via-OpenAI-compat) inherits as ``openai``."""
    data = {
        "system": {
            "llm": {
                "provider": "generic_openai",
                "model": "qwen3:7b",
                "base_url": "http://localhost:11434/v1",
            },
            "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
            "memory": {"provider": "mem0v2_embedded"},
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    assert mem["llm_provider"] == "openai"  # normalized
    assert mem["llm_model"] == "qwen3:7b"
    assert mem["llm_base_url"] == "http://localhost:11434/v1"
    assert mem["embedder_provider"] == "ollama"
    assert mem["embedder_model"] == "nomic-embed-text"


def test_inherit_translates_anthropic_bedrock_to_aws_bedrock():
    """``haiku45_bedrock`` (anthropic_bedrock + aws_region) inherits as ``aws_bedrock``.

    Mem0 v2.0.0's LLM registry only knows ``aws_bedrock``; sending
    ``anthropic_bedrock`` raises ``Unsupported LLM provider`` at
    ``Memory.from_config()`` time. This test locks the translation and
    the aws_region/aws_profile threading. Without it, every ECS shard
    of a mem0v2 × Bedrock run crashes at engine init with no summary
    written (observed empirically on
    ``pro-smoke-mem0v2-s100-r1-20260526-022235``).
    """
    data = {
        "system": {
            "llm": {
                "provider": "anthropic_bedrock",
                "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "aws_region": "us-east-1",
                "aws_profile": "research-dev",
            },
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "memory": {"provider": "mem0v2_embedded"},
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    assert mem["llm_provider"] == "aws_bedrock"  # normalized for mem0
    assert mem["llm_model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert mem["aws_region"] == "us-east-1"
    assert mem["aws_profile"] == "research-dev"


def test_explicit_memory_keys_win_over_inheritance():
    """Override mixin (e.g. mem_internal_gpt4o_mini) must beat inheritance."""
    data = {
        "system": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "memory": {
                "provider": "mem0v2_embedded",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",  # explicit divergence
            },
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    assert mem["llm_model"] == "gpt-4o-mini"  # NOT overwritten to gpt-4o
    # embedder still inherited (no explicit value)
    assert mem["embedder_provider"] == "openai"


def test_no_inherit_for_api_memory():
    """``mem0v2_api`` runs server-side; we don't inject any LLM config."""
    data = {
        "system": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "memory": {"provider": "mem0v2_api", "api_key": "x"},
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    assert "llm_provider" not in mem
    assert "llm_model" not in mem


def test_no_inherit_for_unmapped_embedded_provider():
    """Unknown embedded provider → no inheritance (don't guess key names)."""
    data = {
        "system": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "memory": {"provider": "novel_embedded"},
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    assert "llm_provider" not in out["system"]["memory"]


def test_inherit_respects_nested_params_shape():
    """Some profiles use ``[system.memory.params]``; inheritance must read both."""
    data = {
        "system": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "memory": {
                "provider": "mem0v2_embedded",
                "params": {"llm_model": "gpt-4o-mini"},  # nested explicit override
            },
        }
    }
    out = _inherit_bench_llm_into_memory(data)
    mem = out["system"]["memory"]
    # Explicit nested key blocks inheritance for that key
    assert "llm_model" not in mem  # didn't write top-level since nested has it
    assert mem["params"]["llm_model"] == "gpt-4o-mini"
    # Other keys with no nested explicit value still inherit at top level
    assert mem.get("llm_provider") == "openai"


# ---------------------------------------------------------------------------
# embed_1536 profile — one 1536-dim embedder for every memory architecture
# ---------------------------------------------------------------------------

_EMBEDDED_MEMORY_PROFILES = [
    "mem0v1_embedded",
    "mem0v2_embedded",
    "evermemos_embedded",
    "graphiti",
    "hindsight_embedded",
    "mastra_embedded",
    "memos_embedded",
    "rag",
]


def test_embed_1536_listed_in_builtin_profiles():
    assert "embed_1536" in BUILTIN_PROFILES


@pytest.mark.parametrize("memory_profile", _EMBEDDED_MEMORY_PROFILES)
def test_embed_1536_forces_gte_embedding_for_every_memory_mixin(memory_profile):
    """Stacking ``embed_1536`` last must override every memory mixin."""
    cfg = load_run_config(
        benchmark="locomo_original",
        profiles=["local", memory_profile, "embed_1536"],
    )

    assert cfg.system.embedding.model == "gte-qwen2:8t"
    params = cfg.system.memory.params
    assert params["embedding_model"] == "gte-qwen2:8t"
    assert params["embedder_model"] == "gte-qwen2:8t"
    assert params["embedder_provider"] == "ollama"
    assert params["embedder_base_url"] == "http://localhost:11434"
    assert params["embedder_dims"] == 1536
    assert params["embedding_base_url"] == "http://localhost:11434/v1"


def test_embed_1536_keeps_8thread_llm_when_stacked_last():
    """The embedding profile must not clobber the 8-thread chat model."""
    cfg = load_run_config(
        benchmark="locomo_original",
        profiles=[
            "local",
            "mem0v2_embedded",
            "ollama_memory",
            "ollama_8threads",
            "embed_1536",
        ],
    )

    assert cfg.system.llm.model == "qwen2.5:7b-instruct-8t"
    assert cfg.system.embedding.model == "gte-qwen2:8t"
    params = cfg.system.memory.params
    assert params["llm_model"] == "qwen2.5:7b-instruct-8t"
    assert params["embedder_model"] == "gte-qwen2:8t"
    assert params["embedder_dims"] == 1536
