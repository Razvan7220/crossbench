"""Benchmark adapters for coding evaluation."""

from cross_bench.benchmarks.base import BenchmarkAdapter
from cross_bench.benchmarks.registry import get_adapter, list_benchmarks, register_adapter

__all__ = ["BenchmarkAdapter", "get_adapter", "list_benchmarks", "register_adapter"]
