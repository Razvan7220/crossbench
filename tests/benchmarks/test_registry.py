"""Test benchmark adapter registry."""

from cross_bench.benchmarks import get_adapter, list_benchmarks
from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import _ALIASES
from cross_bench.models import TaskGranularity


def test_list_benchmarks():
    benchmarks = list_benchmarks()
    assert "humaneval" in benchmarks
    assert "humaneval_plus" in benchmarks
    assert "mbpp" in benchmarks
    assert "mbpp_plus" in benchmarks
    assert "swebench" in benchmarks
    assert "swebench_pro" in benchmarks  # alias surfaced in the list
    assert "swebench_pro_gold" in benchmarks
    assert "swebench_pro_seal" in benchmarks
    assert "swebench_gold" in benchmarks
    assert "bigcodebench" in benchmarks
    assert "livecodebench" in benchmarks
    assert "crosscodeeval" in benchmarks
    assert "repobench" in benchmarks
    assert "terminalbench" in benchmarks
    assert "convomem" in benchmarks
    assert "locomo_original" in benchmarks
    assert "locomo_everos" in benchmarks
    assert "longmemeval" in benchmarks
    # 14 real adapters + 4 aliases (swebench_pro, swebench_pro_gold,
    # swebench_pro_seal, swebench_gold)
    assert len(benchmarks) == 18


def test_get_adapter():
    adapter = get_adapter("humaneval")
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "humaneval"
    assert adapter.granularity == TaskGranularity.FUNCTION


def test_get_unknown_raises():
    import pytest
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_adapter("nonexistent_benchmark")


def test_swebench_pro_alias_resolves_to_swebench_adapter():
    """``-b swebench_pro`` flows to the same SWE-bench adapter as ``-b swebench``.

    The benchmark TOML ``configs/benchmarks/swebench_pro.toml`` sets
    ``[benchmark] variant = "pro"`` so the adapter dispatches on Pro
    behaviour, but the implementation class is one. See deviations.md S21.
    """
    pro = get_adapter("swebench_pro")
    plain = get_adapter("swebench")
    assert type(pro) is type(plain)
    # Both report the canonical name (the alias is a routing concern,
    # not an identity concern).
    assert pro.name == "swebench"


def test_alias_listed_in_unknown_error_message():
    """Unknown benchmarks should hint at available aliases for discovery."""
    import pytest
    with pytest.raises(KeyError, match="aliases"):
        get_adapter("definitely_not_a_benchmark")


def test_all_adapters_instantiate():
    # Names that are aliases resolve to the canonical adapter, whose
    # ``adapter.name`` is the canonical name — not the alias. So we
    # check identity-via-resolution rather than ``adapter.name == name``.
    for name in list_benchmarks():
        adapter = get_adapter(name)
        canonical = _ALIASES.get(name, name)
        assert adapter.name == canonical
        assert isinstance(adapter.granularity, TaskGranularity)
        assert isinstance(adapter.get_sandbox_image(), str)
        assert isinstance(adapter.get_default_params(), dict)
