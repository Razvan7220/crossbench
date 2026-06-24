"""Tests for memory-system internal-token rollup in ``RunEngine._aggregate``.

Per-task counters are stamped onto each ``TaskResult.evaluation`` as
``memory_tokens_in / _out / _llm_calls / _io_tokens_in / _io_tokens_out``
by the per-task wiring in ``_run_conversational`` / ``_run_code_gen``.
This test exercises the aggregator's side of the contract: sums to
``total_memory_*`` keys plus ``avg_memory_tokens_per_task``.

See deviations.md "Platform / cross-cutting" entry on internal-token
accounting.
"""

from __future__ import annotations

from cross_bench.execution.engine import (
    RunEngine,
    _snapshot_memory_stats,
    _stamp_memory_delta,
)
from cross_bench.models import StepRecord, TaskResult, TaskStatus


def _result(eval_dict):
    return TaskResult(
        task_id="t",
        run_id="r",
        status=TaskStatus.PASS,
        steps=[
            StepRecord(
                step_index=0, action="chat", input_text="", output_text="",
                tokens_in=0, tokens_out=0, latency_ms=0.0,
            )
        ],
        final_output="",
        evaluation=eval_dict,
    )


# ---------------------------------------------------------------------------
# Aggregator: totals + avg_per_task
# ---------------------------------------------------------------------------


def test_aggregate_sums_memory_token_columns_across_tasks():
    results = [
        _result({
            "memory_tokens_in": 100, "memory_tokens_out": 50,
            "memory_llm_calls": 2,
            "memory_io_tokens_in": 200, "memory_io_tokens_out": 80,
        }),
        _result({
            "memory_tokens_in": 30, "memory_tokens_out": 20,
            "memory_llm_calls": 1,
            "memory_io_tokens_in": 40, "memory_io_tokens_out": 25,
        }),
    ]
    metrics = RunEngine._aggregate(results)
    assert metrics["total_memory_tokens_in"] == 130.0
    assert metrics["total_memory_tokens_out"] == 70.0
    assert metrics["total_memory_llm_calls"] == 3.0
    assert metrics["total_memory_io_tokens_in"] == 240.0
    assert metrics["total_memory_io_tokens_out"] == 105.0


def test_aggregate_avg_memory_tokens_per_task():
    """avg = (in + out) / total_tasks, mirroring the bench-LLM
    ``avg_tokens_per_task``."""
    results = [
        _result({"memory_tokens_in": 100, "memory_tokens_out": 50}),
        _result({"memory_tokens_in": 30, "memory_tokens_out": 20}),
    ]
    metrics = RunEngine._aggregate(results)
    # (100+30) + (50+20) = 200; 200 / 2 = 100
    assert metrics["avg_memory_tokens_per_task"] == 100.0


def test_aggregate_avg_memory_io_tokens_per_task():
    results = [
        _result({"memory_io_tokens_in": 200, "memory_io_tokens_out": 80}),
        _result({"memory_io_tokens_in": 40, "memory_io_tokens_out": 25}),
    ]
    metrics = RunEngine._aggregate(results)
    # (200+40) + (80+25) = 345; 345 / 2 = 172.5
    assert metrics["avg_memory_io_tokens_per_task"] == 172.5


def test_aggregate_zero_when_no_memory_keys_stamped():
    """A no-memory run still gets the ``total_memory_*`` columns
    populated as 0.0 (not None) so the leaderboard column shape is
    stable regardless of which memory system was active."""
    results = [_result({}), _result({})]
    metrics = RunEngine._aggregate(results)
    assert metrics["total_memory_tokens_in"] == 0.0
    assert metrics["total_memory_tokens_out"] == 0.0
    assert metrics["total_memory_llm_calls"] == 0.0
    assert metrics["total_memory_io_tokens_in"] == 0.0
    assert metrics["total_memory_io_tokens_out"] == 0.0


def test_aggregate_generic_avg_loop_produces_avg_memory_keys():
    """The engine's generic numeric averager additionally produces
    ``avg_memory_tokens_in`` / ``_out`` / ``_llm_calls`` / IO keys
    via the ``for key in all_eval_keys`` loop."""
    results = [
        _result({"memory_tokens_in": 100, "memory_tokens_out": 50}),
        _result({"memory_tokens_in": 30, "memory_tokens_out": 20}),
    ]
    metrics = RunEngine._aggregate(results)
    assert metrics["avg_memory_tokens_in"] == 65.0  # (100+30)/2
    assert metrics["avg_memory_tokens_out"] == 35.0  # (50+20)/2


# ---------------------------------------------------------------------------
# _snapshot_memory_stats + _stamp_memory_delta helpers
# ---------------------------------------------------------------------------


class _FakeMemory:
    def __init__(self, stats: dict[str, int]) -> None:
        self._stats = stats

    def get_internal_token_stats(self) -> dict[str, int]:
        return dict(self._stats)


def test_snapshot_memory_stats_none_returns_zeros():
    s = _snapshot_memory_stats(None)
    assert s == {
        "tokens_in": 0, "tokens_out": 0, "llm_calls": 0,
        "io_tokens_in": 0, "io_tokens_out": 0,
    }


def test_snapshot_memory_stats_reads_get_internal_token_stats():
    fake = _FakeMemory({
        "tokens_in": 10, "tokens_out": 5, "llm_calls": 1,
        "io_tokens_in": 20, "io_tokens_out": 8,
    })
    s = _snapshot_memory_stats(fake)
    assert s["tokens_in"] == 10
    assert s["tokens_out"] == 5
    assert s["llm_calls"] == 1


def test_snapshot_memory_stats_swallows_exceptions():
    class _Broken:
        def get_internal_token_stats(self):
            raise RuntimeError("boom")

    s = _snapshot_memory_stats(_Broken())
    assert s == {
        "tokens_in": 0, "tokens_out": 0, "llm_calls": 0,
        "io_tokens_in": 0, "io_tokens_out": 0,
    }


def test_snapshot_memory_stats_missing_method_returns_zeros():
    """A middleware that predates the contract returns zeros."""
    s = _snapshot_memory_stats(object())
    assert s["tokens_in"] == 0


def test_stamp_memory_delta_computes_after_minus_before():
    evaluation: dict = {}
    before = {
        "tokens_in": 5, "tokens_out": 2, "llm_calls": 1,
        "io_tokens_in": 10, "io_tokens_out": 3,
    }
    after = {
        "tokens_in": 12, "tokens_out": 7, "llm_calls": 3,
        "io_tokens_in": 25, "io_tokens_out": 9,
    }
    _stamp_memory_delta(evaluation, before, after)
    assert evaluation["memory_tokens_in"] == 7
    assert evaluation["memory_tokens_out"] == 5
    assert evaluation["memory_llm_calls"] == 2
    assert evaluation["memory_io_tokens_in"] == 15
    assert evaluation["memory_io_tokens_out"] == 6


def test_stamp_memory_delta_clamps_negative_at_zero():
    """A counter that decreased mid-task (middleware bug) shouldn't
    emit a negative — clamp at zero, surface as if no work happened."""
    evaluation: dict = {}
    before = {"tokens_in": 10, "tokens_out": 5, "llm_calls": 2,
              "io_tokens_in": 0, "io_tokens_out": 0}
    after = {"tokens_in": 5, "tokens_out": 2, "llm_calls": 1,
             "io_tokens_in": 0, "io_tokens_out": 0}
    _stamp_memory_delta(evaluation, before, after)
    assert evaluation["memory_tokens_in"] == 0
    assert evaluation["memory_tokens_out"] == 0
    assert evaluation["memory_llm_calls"] == 0


def test_stamp_memory_delta_additive_mode_adds_to_existing():
    """``additive=True`` rolls a second delta onto the first — used
    by the conversational flow to attach the conversation's
    ingestion delta to the first task's evaluation alongside its
    own per-task delta."""
    evaluation = {
        "memory_tokens_in": 5, "memory_tokens_out": 3,
        "memory_llm_calls": 1,
        "memory_io_tokens_in": 0, "memory_io_tokens_out": 0,
    }
    before = {k: 0 for k in (
        "tokens_in", "tokens_out", "llm_calls",
        "io_tokens_in", "io_tokens_out",
    )}
    after = {
        "tokens_in": 100, "tokens_out": 50, "llm_calls": 4,
        "io_tokens_in": 200, "io_tokens_out": 80,
    }
    _stamp_memory_delta(evaluation, before, after, additive=True)
    assert evaluation["memory_tokens_in"] == 105
    assert evaluation["memory_tokens_out"] == 53
    assert evaluation["memory_llm_calls"] == 5
    assert evaluation["memory_io_tokens_in"] == 200
    assert evaluation["memory_io_tokens_out"] == 80
