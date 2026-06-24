"""Tests for ``RunEngine._aggregate`` summary keys.

Specifically locks the LLM-judge + latency surfacing the user asked for:

* ``pass_rate_f1`` and ``pass_rate_llm_judge`` are top-level summary keys
  whenever the underlying per-task evaluation has ``f1_pass`` /
  ``llm_judge_correct`` (so a glance at ``summary.json`` answers
  "what fraction did each grader say was correct?").
* ``avg_search_latency_ms``, ``avg_inference_latency_ms``,
  ``avg_llm_judge_latency_ms``, and ``avg_total_latency_ms`` come out
  of the auto-averaging path automatically.
"""

from __future__ import annotations

from cross_bench.execution.engine import RunEngine
from cross_bench.models import StepRecord, TaskResult, TaskStatus


def _result(*, eval_dict, status=TaskStatus.PASS):
    return TaskResult(
        task_id="t",
        run_id="r",
        status=status,
        steps=[StepRecord(step_index=0, action="chat", input_text="", output_text="",
                          tokens_in=0, tokens_out=0, latency_ms=0.0)],
        final_output="",
        evaluation=eval_dict,
    )


def test_aggregate_surfaces_pass_rate_f1_and_llm_judge():
    results = [
        _result(eval_dict={"f1_pass": 1, "llm_judge_correct": 1}),
        _result(eval_dict={"f1_pass": 1, "llm_judge_correct": 0}),
        _result(eval_dict={"f1_pass": 0, "llm_judge_correct": 1}),
        _result(eval_dict={"f1_pass": 0, "llm_judge_correct": 0}),
    ]
    metrics = RunEngine._aggregate(results)
    assert metrics["pass_rate_f1"] == 0.5  # 2 / 4
    assert metrics["pass_rate_llm_judge"] == 0.5  # 2 / 4


def test_aggregate_omits_pass_rate_llm_judge_when_judge_not_run():
    """If LLM-judge is disabled, ``pass_rate_llm_judge`` must NOT appear —
    a missing key is honest about not being measured."""
    results = [
        _result(eval_dict={"f1_pass": 1}),
        _result(eval_dict={"f1_pass": 0}),
    ]
    metrics = RunEngine._aggregate(results)
    assert "pass_rate_f1" in metrics
    assert "pass_rate_llm_judge" not in metrics


def test_aggregate_omits_pass_rate_f1_when_no_f1_pass_keys():
    """Symmetric: a primary='llm_judge' run that doesn't store f1_pass
    leaves the F1 row out."""
    results = [
        _result(eval_dict={"llm_judge_correct": 1}),
    ]
    metrics = RunEngine._aggregate(results)
    assert "pass_rate_f1" not in metrics
    assert metrics["pass_rate_llm_judge"] == 1.0


def test_aggregate_averages_latency_metrics():
    """search / inference / judge / total latencies all auto-average via
    the ``avg_<key>`` path. We don't add them by name; we trust the
    generic loop to pick them up so adding a new latency key is free."""
    results = [
        _result(eval_dict={
            "search_latency_ms": 100.0,
            "inference_latency_ms": 500.0,
            "llm_judge_latency_ms": 200.0,
            "total_latency_ms": 800.0,
        }),
        _result(eval_dict={
            "search_latency_ms": 200.0,
            "inference_latency_ms": 600.0,
            "llm_judge_latency_ms": 300.0,
            "total_latency_ms": 1100.0,
        }),
    ]
    metrics = RunEngine._aggregate(results)
    assert metrics["avg_search_latency_ms"] == 150.0
    assert metrics["avg_inference_latency_ms"] == 550.0
    assert metrics["avg_llm_judge_latency_ms"] == 250.0
    assert metrics["avg_total_latency_ms"] == 950.0


def test_aggregate_handles_partial_judge_coverage():
    """If only some tasks ran the judge (e.g. a partial outage), the
    aggregate divides by tasks that DID run it, not by total. Explicit
    is better than mean-of-zeros."""
    results = [
        _result(eval_dict={"f1_pass": 1, "llm_judge_correct": 1}),
        _result(eval_dict={"f1_pass": 0}),  # judge didn't run on this one
        _result(eval_dict={"f1_pass": 1, "llm_judge_correct": 0}),
    ]
    metrics = RunEngine._aggregate(results)
    assert metrics["pass_rate_f1"] == 2 / 3
    assert metrics["pass_rate_llm_judge"] == 0.5  # 1 / 2 measured


# ---------------------------------------------------------------------------
# Schema stability — declared metrics are always present (None when
# unmeasured) so summary.json columns are stable across memory systems.
# ---------------------------------------------------------------------------


class _SchemaAdapter:
    """Tiny stand-in adapter that declares a fixed metric schema."""

    def __init__(self, keys: list[str]):
        self._keys = keys

    def metrics_schema(self) -> list[str]:
        return list(self._keys)


def test_aggregate_pre_seeds_declared_schema_with_none():
    """Adapter-declared keys absent from any task's evaluation surface
    as ``None`` (JSON null) — honest 'not measured' rather than the
    silent omission of a column."""
    adapter = _SchemaAdapter([
        "pass_rate_llm_judge",  # nobody ran it this time
        "avg_total_latency_ms",  # nobody set a timing
        "pass_rate_f1",          # this one IS measured below
    ])
    results = [_result(eval_dict={"f1_pass": 1})]
    metrics = RunEngine._aggregate(results, adapter=adapter)

    # Measured key gets its real value
    assert metrics["pass_rate_f1"] == 1.0
    # Unmeasured-but-declared keys are explicitly None
    assert metrics["pass_rate_llm_judge"] is None
    assert metrics["avg_total_latency_ms"] is None


def test_aggregate_schema_stable_across_memory_systems():
    """Switching memory systems must NOT change the column set —
    same adapter schema, same keys in summary.json."""
    adapter = _SchemaAdapter([
        "pass_rate_f1", "pass_rate_llm_judge",
        "avg_search_latency_ms", "avg_total_latency_ms",
    ])
    # Memory system A — produces F1 + search latency
    results_A = [_result(eval_dict={
        "f1_pass": 1,
        "search_latency_ms": 100.0,
        "total_latency_ms": 500.0,
    })]
    # Memory system B — same adapter, but no F1 measurement (e.g. an
    # eval crashed) and no judge column
    results_B = [_result(eval_dict={
        "search_latency_ms": 200.0,
        "total_latency_ms": 700.0,
    })]
    metrics_A = RunEngine._aggregate(results_A, adapter=adapter)
    metrics_B = RunEngine._aggregate(results_B, adapter=adapter)

    # Column sets are identical even though measured values differ
    declared = {"pass_rate_f1", "pass_rate_llm_judge",
                "avg_search_latency_ms", "avg_total_latency_ms"}
    assert declared <= set(metrics_A.keys())
    assert declared <= set(metrics_B.keys())
    # B has unmeasured F1 → None; A has it measured
    assert metrics_A["pass_rate_f1"] == 1.0
    assert metrics_B["pass_rate_f1"] is None
