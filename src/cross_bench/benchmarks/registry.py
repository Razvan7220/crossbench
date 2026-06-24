"""Benchmark adapter discovery and registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cross_bench.benchmarks.base import BenchmarkAdapter

_REGISTRY: dict[str, type[BenchmarkAdapter]] = {}

# Alias map: ``cli_name -> canonical_adapter_name``. Lets a benchmark TOML
# present a friendly variant name (``swebench_pro``) that resolves to the
# adapter the variant lives inside (``swebench``). Keeps the CLI surface
# discoverable — ``cross-bench list-benchmarks`` shows both names — while
# the adapter implementation stays one class.
#
# Adding a new alias is one line below. The benchmark TOML at
# ``configs/benchmarks/<cli_name>.toml`` then sets ``[run] benchmark =
# "<canonical>"`` plus whatever variant params (``[benchmark] variant =
# "pro"``) the adapter dispatches on.
_ALIASES: dict[str, str] = {
    "swebench_pro": "swebench",
    "swebench_pro_gold": "swebench",
    "swebench_pro_seal": "swebench",
    "swebench_gold": "swebench",
}


def register_adapter(cls: type[BenchmarkAdapter]) -> type[BenchmarkAdapter]:
    """Class decorator to register a benchmark adapter."""
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> BenchmarkAdapter:
    """Instantiate a benchmark adapter by name (or alias)."""
    # Resolve alias first so ``swebench_pro`` -> ``swebench``.
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        _ensure_builtins_loaded()
    if canonical not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {sorted(_REGISTRY.keys())} "
            f"(aliases: {sorted(_ALIASES.keys())})"
        )
    return _REGISTRY[canonical]()


def list_benchmarks() -> list[str]:
    """Return sorted list of registered benchmark names (including aliases)."""
    _ensure_builtins_loaded()
    return sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))


_BUILTINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    """Import all built-in adapter modules to trigger registration."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Each module's top-level @register_adapter call populates _REGISTRY
    import cross_bench.benchmarks.bigcodebench  # noqa: F401
    import cross_bench.benchmarks.convomem  # noqa: F401
    import cross_bench.benchmarks.crosscodeeval  # noqa: F401
    import cross_bench.benchmarks.humaneval  # noqa: F401
    import cross_bench.benchmarks.humaneval_plus  # noqa: F401
    import cross_bench.benchmarks.livecodebench  # noqa: F401
    import cross_bench.benchmarks.locomo_everos  # noqa: F401
    import cross_bench.benchmarks.locomo_original  # noqa: F401
    import cross_bench.benchmarks.longmemeval  # noqa: F401
    import cross_bench.benchmarks.mbpp  # noqa: F401
    import cross_bench.benchmarks.mbpp_plus  # noqa: F401
    import cross_bench.benchmarks.repobench  # noqa: F401
    import cross_bench.benchmarks.swebench  # noqa: F401
    import cross_bench.benchmarks.terminalbench  # noqa: F401
    _BUILTINS_LOADED = True
