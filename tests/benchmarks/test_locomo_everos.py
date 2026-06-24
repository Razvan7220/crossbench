# ruff: noqa: E501
"""LoCoMo-EverOS adapter tests.

Cover:

* Registration / defaults (filter_categories = [5], system = "mem0",
  num_runs = 3, 30-s interval; default baseline = ``"auto"`` which
  resolves per-sample from the conversation's ``session_1_date_time``).
* :func:`interpolate_timestamps` — 30-s cadence within a session,
  +1 day between sessions, anchored at the resolved baseline.
* :func:`_resolve_baseline` — ``"auto"`` reads
  ``session_1_date_time``; explicit ISO overrides; unparseable falls
  back to ``datetime(2024, 1, 1)``.
* Per-system answer prompt selection (mem0 / memos / cot / zep) and
  the anti-abstention rule.
* Cat-2 date hint applied; cat-5 excluded by default; cat-5 pass-through
  (no MCQ) when explicitly re-enabled.
* ``load_conversations`` emits synthetic ISO-8601 per-turn timestamps and
  preserves image fields + dia_id.
* ``judge_answer`` — LLM-judge is mocked, then 3-run aggregation
  (mean$\\pm$std) is verified across 3-CORRECT, 2-of-3, 1-of-3, 0-of-3
  scenarios.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks import locomo_everos as everos_mod
from cross_bench.benchmarks.locomo_everos import (
    ANSWER_PROMPTS,
    SYSTEMS,
    LoCoMoEverosAdapter,
    _parse_baseline,
    _parse_locomo_session_date_time,
    _resolve_baseline,
    interpolate_timestamps,
)
from cross_bench.models import Task, TaskGranularity

# ---------------------------------------------------------------------------
# Sample factory — shape mirrors data/locomo10.json; re-declared here so
# tests are independent of test_locomo_original.py.
# ---------------------------------------------------------------------------


def _sample(
    sample_id: str = "sample_0",
    speaker_a: str = "Caroline",
    speaker_b: str = "Melanie",
    qa_items: list[dict[str, Any]] | None = None,
    conversation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LoCoMo-shaped sample with 2 sessions and 5 QAs (one per category)."""
    if conversation is None:
        conversation = {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "session_1_date_time": "1:00 pm on 1 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": speaker_a, "text": "I went hiking."},
                {"dia_id": "D1:2", "speaker": speaker_b, "text": "Where?"},
                {
                    "dia_id": "D1:3",
                    "speaker": speaker_a,
                    "text": "The Appalachian Trail.",
                    "img_url": "http://example/hike.jpg",
                    "blip_caption": "a photo of a forest trail",
                    "query": "appalachian trail",
                },
            ],
            "session_2_date_time": "3:00 pm on 8 May 2023",
            "session_2": [
                {"dia_id": "D2:1", "speaker": speaker_b, "text": "How was the weekend?"},
                {"dia_id": "D2:2", "speaker": speaker_a, "text": "Finished a book on gardening."},
            ],
        }
    if qa_items is None:
        qa_items = [
            {"question": "What outdoor activities?", "answer": "hiking, gardening", "category": 1, "evidence": ["D1:1", "D2:2"]},
            {"question": "When did Caroline hike?", "answer": "May 2023", "category": 2, "evidence": ["D1:1"]},
            {"question": "Is gardening outdoor?", "answer": "Likely yes; usually outdoors", "category": 3, "evidence": ["D2:2"]},
            {"question": "Where did Caroline hike?", "answer": "Appalachian Trail", "category": 4, "evidence": ["D1:3"]},
            {"question": "Which mountain?", "answer": "Mount Everest", "category": 5, "evidence": []},
        ]
    return {"sample_id": sample_id, "conversation": conversation, "qa": qa_items}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "locomo"
    d.mkdir()
    (d / "locomo10.json").write_text(
        json.dumps([_sample("sample_0"), _sample("sample_1")]),
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Registration & defaults
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("locomo_everos")
    assert isinstance(a, LoCoMoEverosAdapter)
    assert a.name == "locomo_everos"
    assert a.granularity == TaskGranularity.CONVERSATIONAL
    assert a.requires_sandbox is False


def test_default_params_match_everos_conventions():
    p = LoCoMoEverosAdapter().get_default_params()
    # EverOS + mem0ai both drop cat 5.
    assert p["filter_categories"] == [5]
    # Default per-system prompt.
    assert p["system"] == "mem0"
    # 3-run mean±std is the EverOS signature.
    assert p["num_runs"] == 3
    # Synthetic timeline is anchored per-sample (``"auto"`` resolves to
    # the conversation's session_1_date_time); 30s cadence.
    assert p["timestamp"]["baseline"] == "auto"
    assert p["timestamp"]["interval_seconds"] == 30
    # LLM-judge config shape.
    assert p["judge"]["provider"] == "openai"
    assert p["judge"]["model"] == "gpt-4o-mini"


def test_systems_are_canonical_four():
    # mem0, memos, cot, zep — forked from mem0 by EverOS.
    assert set(SYSTEMS) == {"mem0", "memos", "cot", "zep"}


# ---------------------------------------------------------------------------
# Timestamp interpolation — the smoking-gun EverOS-only feature.
# ---------------------------------------------------------------------------


def test_parse_baseline_accepts_iso_8601():
    assert _parse_baseline("2024-01-01T00:00:00") == datetime(2024, 1, 1)
    assert _parse_baseline("2024-06-15T09:30:00") == datetime(2024, 6, 15, 9, 30, 0)


def test_parse_baseline_rejects_garbage():
    with pytest.raises(ValueError, match="Invalid timestamp baseline"):
        _parse_baseline("not-a-date")


def test_interpolate_timestamps_30s_within_session():
    conv = _sample()["conversation"]
    baseline = datetime(2024, 1, 1)
    out = interpolate_timestamps(conv, baseline, interval_seconds=30)

    # 3 turns in session_1 + 2 in session_2 = 5 tuples.
    assert len(out) == 5

    # Session 1: starts at 2024-01-01 (offset 0 days), +30s per turn.
    assert out[0] == (1, 0, datetime(2024, 1, 1, 0, 0, 0))
    assert out[1] == (1, 1, datetime(2024, 1, 1, 0, 0, 30))
    assert out[2] == (1, 2, datetime(2024, 1, 1, 0, 1, 0))


def test_interpolate_timestamps_plus_1_day_between_sessions():
    conv = _sample()["conversation"]
    baseline = datetime(2024, 1, 1)
    out = interpolate_timestamps(conv, baseline, interval_seconds=30)
    # Session 2: starts one day later, +30s per turn.
    assert out[3] == (2, 0, datetime(2024, 1, 2, 0, 0, 0))
    assert out[4] == (2, 1, datetime(2024, 1, 2, 0, 0, 30))


def test_interpolate_timestamps_honors_custom_baseline_and_interval():
    conv = _sample()["conversation"]
    baseline = datetime(2023, 6, 15, 12, 0, 0)
    out = interpolate_timestamps(conv, baseline, interval_seconds=60)
    assert out[0][2] == datetime(2023, 6, 15, 12, 0, 0)
    assert out[1][2] == datetime(2023, 6, 15, 12, 1, 0)
    # Session 2 still increments by 1 day.
    assert out[3][2] == datetime(2023, 6, 16, 12, 0, 0)


def test_interpolate_timestamps_empty_conversation():
    assert interpolate_timestamps({}, datetime(2024, 1, 1), 30) == []


# ---------------------------------------------------------------------------
# Baseline resolution — _resolve_baseline + _parse_locomo_session_date_time
# (Anchors synthetic timestamps to the source conversation clock, mirroring
# the trap-fix pattern Mastra M38 / Hindsight H46 / MemOS O24 use.)
# ---------------------------------------------------------------------------


def test_parse_locomo_session_date_time_with_comma():
    assert _parse_locomo_session_date_time("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)


def test_parse_locomo_session_date_time_without_comma():
    # snap-research's mirror ships the no-comma variant on some samples;
    # observed in the test fixture and in real data. Both must parse.
    assert _parse_locomo_session_date_time("1:00 pm on 1 May 2023") == datetime(2023, 5, 1, 13, 0)


def test_parse_locomo_session_date_time_uppercase_am_pm():
    assert _parse_locomo_session_date_time("11:30 AM on 4 July 2024") == datetime(2024, 7, 4, 11, 30)


def test_parse_locomo_session_date_time_returns_none_on_empty():
    assert _parse_locomo_session_date_time("") is None
    assert _parse_locomo_session_date_time("   ") is None


def test_parse_locomo_session_date_time_returns_none_on_unparseable():
    assert _parse_locomo_session_date_time("not-a-date") is None


def test_resolve_baseline_auto_reads_session_1_date_time():
    conv = _sample()["conversation"]
    # Fixture: session_1_date_time = "1:00 pm on 1 May 2023".
    assert _resolve_baseline(conv, "auto") == datetime(2023, 5, 1, 13, 0)


def test_resolve_baseline_none_treated_as_auto():
    conv = _sample()["conversation"]
    assert _resolve_baseline(conv, None) == datetime(2023, 5, 1, 13, 0)


def test_resolve_baseline_explicit_iso_overrides_auto():
    conv = _sample()["conversation"]
    assert _resolve_baseline(conv, "2024-01-01T00:00:00") == datetime(2024, 1, 1)


def test_resolve_baseline_auto_unparseable_falls_back_to_2024_01_01(caplog):
    conv = {"session_1_date_time": "garbage"}
    with caplog.at_level("WARNING"):
        baseline = _resolve_baseline(conv, "auto")
    assert baseline == datetime(2024, 1, 1)
    assert "unparseable" in caplog.text.lower()


def test_resolve_baseline_auto_missing_session_date_falls_back():
    # No session_1_date_time at all — same fall-back path.
    assert _resolve_baseline({}, "auto") == datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Per-system answer prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_answer_prompts_include_anti_abstention_rule(system):
    prompt = ANSWER_PROMPTS[system]
    # Anti-abstention — biases the answerer AWAY from category-5-correct behavior.
    assert "NEVER say" in prompt
    assert "not mentioned" in prompt
    assert "not specified" in prompt


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_answer_prompts_include_temporal_slop_rules(system):
    prompt = ANSWER_PROMPTS[system]
    # 14-day / 50% tolerance — Mem0-lineage signature.
    assert "14 days" in prompt
    assert "50%" in prompt


def test_mem0_prompt_exposes_single_memories_slot():
    p = ANSWER_PROMPTS["mem0"]
    assert "{memories}" in p
    assert "{memories_working}" not in p  # that's memos-specific


def test_memos_prompt_exposes_two_tier_memory_slots():
    p = ANSWER_PROMPTS["memos"]
    assert "{memories_working}" in p
    assert "{memories_longterm}" in p


def test_zep_prompt_exposes_facts_and_episodes_slots():
    p = ANSWER_PROMPTS["zep"]
    assert "{facts}" in p
    assert "{episodes}" in p


def test_cot_prompt_is_a_baseline_with_full_history():
    p = ANSWER_PROMPTS["cot"]
    # CoT: single {memories} slot, labelled as conversation history.
    assert "{memories}" in p
    assert "Conversation history" in p


# ---------------------------------------------------------------------------
# format_prompt — per-system selection, cat-2 hint, cat-5 pass-through
# ---------------------------------------------------------------------------


def _task(
    category: int,
    question: str = "Q?",
    answer: str = "A",
    system: str = "mem0",
    sample_id: str = "sample_0",
    qa_index: int = 0,
) -> Task:
    return Task(
        task_id=f"{sample_id}_q{qa_index}",
        benchmark="locomo_everos",
        granularity=TaskGranularity.CONVERSATIONAL,
        prompt=question,
        metadata={
            "conversation_id": sample_id,
            "sample_id": sample_id,
            "qa_index": qa_index,
            "category": category,
            "category_name": {1: "multi_hop", 2: "temporal", 3: "commonsense", 4: "single_hop", 5: "adversarial"}.get(category, ""),
            "evidence": [],
            "system": system,
            "reference_timestamp": "2024-01-02T00:00:30",
            "conversation": {},
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
        },
        ground_truth=answer,
    )


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_format_prompt_returns_per_system_template(system):
    t = _task(category=4, system=system, question="Where did Caroline hike?")
    out = LoCoMoEverosAdapter().format_prompt(t)
    # Question is substituted.
    assert "Where did Caroline hike?" in out
    # Memory slots are preserved as literal Python format strings for the
    # downstream backend to fill.
    if system == "mem0":
        assert "{memories}" in out
    elif system == "memos":
        assert "{memories_working}" in out
        assert "{memories_longterm}" in out
    elif system == "zep":
        assert "{facts}" in out
        assert "{episodes}" in out
    elif system == "cot":
        assert "{memories}" in out


def test_format_prompt_cat2_appends_date_hint_before_substitution():
    t = _task(category=2, question="When did Caroline hike?")
    out = LoCoMoEverosAdapter().format_prompt(t)
    assert "Use DATE of CONVERSATION" in out


def test_format_prompt_cat5_is_pass_through_no_mcq():
    # EverOS doesn't exercise cat 5 (filter_categories=[5] default); if
    # a caller re-enables it, we pass the question verbatim — deliberately
    # no MCQ conversion (that would be the paper-faithful path).
    t = _task(category=5, question="Which mountain?")
    out = LoCoMoEverosAdapter().format_prompt(t)
    assert "Which mountain?" in out
    assert "(a)" not in out  # no MCQ decoration
    assert "Not mentioned in the conversation" not in out  # no abstention option


def test_format_prompt_unknown_system_raises():
    t = _task(category=1, system="unknown")
    with pytest.raises(ValueError, match="Unknown system"):
        LoCoMoEverosAdapter().format_prompt(t)


# ---------------------------------------------------------------------------
# load_tasks — EverOS defaults, metadata shape, cat-5 exclusion
# ---------------------------------------------------------------------------


def test_load_tasks_excludes_cat5_by_default(data_dir: Path):
    tasks = list(LoCoMoEverosAdapter().load_tasks({"data_path": str(data_dir)}))
    # Sample has 5 QAs; cat 5 excluded by default => 4 per sample x 2 samples = 8.
    assert len(tasks) == 8
    assert all(t.metadata["category"] != 5 for t in tasks)


def test_load_tasks_explicit_empty_filter_categories_includes_cat5(data_dir: Path):
    tasks = list(LoCoMoEverosAdapter().load_tasks({
        "data_path": str(data_dir),
        "filter_categories": [],  # explicit: keep everything, including cat 5
    }))
    assert len(tasks) == 10
    assert any(t.metadata["category"] == 5 for t in tasks)


def test_load_tasks_metadata_includes_system_and_reference_timestamp(data_dir: Path):
    tasks = list(LoCoMoEverosAdapter().load_tasks({
        "data_path": str(data_dir),
        "system": "zep",
    }))
    first = tasks[0]
    assert first.benchmark == "locomo_everos"
    assert first.metadata["system"] == "zep"
    # Reference timestamp is the last-session synthetic timestamp.
    # Default ``baseline = "auto"`` resolves to session_1_date_time =
    # "1:00 pm on 1 May 2023" -> 2023-05-01T13:00:00. Sample has 2
    # sessions of 3+2 turns; session_2's last turn is at baseline +
    # 1 day + 30s = 2023-05-02T13:00:30.
    assert first.metadata["reference_timestamp"] == "2023-05-02T13:00:30"


def test_load_tasks_unknown_system_raises(data_dir: Path):
    with pytest.raises(ValueError, match="Unknown system"):
        list(LoCoMoEverosAdapter().load_tasks({
            "data_path": str(data_dir),
            "system": "not-a-real-system",
        }))


def test_load_tasks_respects_limit(data_dir: Path):
    tasks = list(LoCoMoEverosAdapter().load_tasks({
        "data_path": str(data_dir),
        "limit": 3,
    }))
    assert len(tasks) == 3


def test_load_tasks_honors_custom_baseline(data_dir: Path):
    tasks = list(LoCoMoEverosAdapter().load_tasks({
        "data_path": str(data_dir),
        "timestamp": {"baseline": "2023-06-15T12:00:00", "interval_seconds": 60},
    }))
    # Session 2 last turn: baseline + 1 day + 1 * 60s = 2023-06-16T12:01:00.
    assert tasks[0].metadata["reference_timestamp"] == "2023-06-16T12:01:00"


# ---------------------------------------------------------------------------
# load_conversations — synthetic ISO-8601 per-turn timestamps
# ---------------------------------------------------------------------------


def test_load_conversations_emits_iso_timestamps_30s_apart(data_dir: Path):
    convs = LoCoMoEverosAdapter().load_conversations({"data_path": str(data_dir)})
    msgs = convs[0]["messages"]
    # Default ``baseline = "auto"`` resolves to session_1_date_time =
    # "1:00 pm on 1 May 2023" -> 2023-05-01T13:00:00.
    assert msgs[0]["timestamp"] == "2023-05-01T13:00:00"
    assert msgs[1]["timestamp"] == "2023-05-01T13:00:30"
    assert msgs[2]["timestamp"] == "2023-05-01T13:01:00"
    # Session 2 starts +1 day (EverOS's per-session offset = sess_idx
    # days, NOT the actual session_2_date_time).
    assert msgs[3]["timestamp"] == "2023-05-02T13:00:00"
    assert msgs[4]["timestamp"] == "2023-05-02T13:00:30"


def test_load_conversations_preserves_original_session_date(data_dir: Path):
    convs = LoCoMoEverosAdapter().load_conversations({"data_path": str(data_dir)})
    msg = convs[0]["messages"][0]
    # Backends that want the human-readable session date can find it here.
    assert msg["session_date_time"] == "1:00 pm on 1 May 2023"


def test_load_conversations_preserves_dia_id_and_image_fields(data_dir: Path):
    convs = LoCoMoEverosAdapter().load_conversations({"data_path": str(data_dir)})
    msg = convs[0]["messages"][2]  # D1:3 — the image turn
    assert msg["dia_id"] == "D1:3"
    assert msg["img_url"] == "http://example/hike.jpg"
    assert msg["blip_caption"] == "a photo of a forest trail"
    assert msg["img_query"] == "appalachian trail"


def test_load_conversations_metadata_surfaces_timeline_config(data_dir: Path):
    convs = LoCoMoEverosAdapter().load_conversations({"data_path": str(data_dir)})
    meta = convs[0]["metadata"]
    # Resolved baseline is session_1_date_time from the fixture.
    assert meta["timestamp_baseline"] == "2023-05-01T13:00:00"
    assert meta["turn_interval_seconds"] == 30
    assert meta["num_sessions"] == 2


# ---------------------------------------------------------------------------
# judge_answer — 3-run mean±std aggregation (LLM judge mocked)
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Mock for ``llm_judge`` returning canned labels in order."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
        llm_config: dict[str, Any] | None = None,
        num_runs: int = 3,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "llm_config": llm_config,
            "num_runs": num_runs,
            "system_prompt": system_prompt,
        })
        # Take the first ``num_runs`` labels from the canned list so tests
        # can override ``num_runs`` without redoing the fixture.
        labels = self.labels[:num_runs]
        num_correct = sum(1 for lab in labels if lab == "CORRECT")
        passed = num_correct > num_runs / 2
        return {
            "pass": passed,
            "correct": int(passed),
            "num_correct": num_correct,
            "num_runs": num_runs,
            "raw_judgments": labels,
        }


@pytest.mark.asyncio
async def test_judge_all_correct_mean_1_std_0(monkeypatch):
    fake = _FakeJudge(["CORRECT", "CORRECT", "CORRECT"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._judge_config = {"provider": "openai", "model": "gpt-4o-mini"}
    a._num_runs = 3

    t = _task(category=1, question="Q", answer="A")
    out = await a.judge_answer(t, "A paraphrase")

    assert out["pass"] is True
    assert out["correct"] == 1
    assert out["accuracy_mean"] == 1.0
    assert out["accuracy_std"] == 0.0
    assert out["num_correct"] == 3
    assert out["num_runs"] == 3
    assert out["raw_judgments"] == ["CORRECT", "CORRECT", "CORRECT"]


@pytest.mark.asyncio
async def test_judge_all_wrong_mean_0_std_0(monkeypatch):
    fake = _FakeJudge(["WRONG", "WRONG", "WRONG"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 3
    t = _task(category=1)
    out = await a.judge_answer(t, "nope")

    assert out["pass"] is False
    assert out["correct"] == 0
    assert out["accuracy_mean"] == 0.0
    assert out["accuracy_std"] == 0.0


@pytest.mark.asyncio
async def test_judge_2_of_3_computes_bessel_std(monkeypatch):
    fake = _FakeJudge(["CORRECT", "CORRECT", "WRONG"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 3
    t = _task(category=1)
    out = await a.judge_answer(t, "partial")

    # mean = 2/3 ≈ 0.667
    assert out["accuracy_mean"] == pytest.approx(0.667, abs=1e-3)
    # Bessel-corrected sample variance for [1, 1, 0]:
    #   mean = 0.6667, deviations squared = (0.333)^2, (0.333)^2, (0.667)^2
    #   sum = 0.1111 + 0.1111 + 0.4444 = 0.6667
    #   sample variance = 0.6667 / 2 = 0.3333
    #   std = sqrt(0.3333) ≈ 0.577
    assert out["accuracy_std"] == pytest.approx(0.577, abs=1e-3)
    # Majority vote: 2 of 3 correct => pass.
    assert out["pass"] is True


@pytest.mark.asyncio
async def test_judge_1_of_3_fails_majority(monkeypatch):
    fake = _FakeJudge(["CORRECT", "WRONG", "WRONG"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 3
    t = _task(category=2)
    out = await a.judge_answer(t, "close-ish")

    # 1 of 3 -> majority is WRONG, pass=False.
    assert out["pass"] is False
    assert out["accuracy_mean"] == pytest.approx(0.333, abs=1e-3)


@pytest.mark.asyncio
async def test_judge_respects_num_runs_override(monkeypatch):
    # 5 labels available; adapter configured for 5 runs.
    fake = _FakeJudge(["CORRECT"] * 5)
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 5
    t = _task(category=1)
    out = await a.judge_answer(t, "X")

    assert out["num_runs"] == 5
    assert out["num_correct"] == 5
    assert fake.calls[0]["num_runs"] == 5


@pytest.mark.asyncio
async def test_judge_uses_everos_judge_system_prompt(monkeypatch):
    fake = _FakeJudge(["CORRECT", "CORRECT", "CORRECT"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 3
    t = _task(category=1)
    _ = await a.judge_answer(t, "X")

    # Judge system prompt should be the EverOS / mem0-lineage one —
    # partial-credit + 14-day / 50% slop.
    sys_prompt = fake.calls[0]["system_prompt"] or ""
    assert "14 days" in sys_prompt
    assert "50%" in sys_prompt


@pytest.mark.asyncio
async def test_judge_surfaces_category_and_system(monkeypatch):
    fake = _FakeJudge(["CORRECT", "CORRECT", "CORRECT"])
    monkeypatch.setattr(everos_mod, "llm_judge", fake)

    a = LoCoMoEverosAdapter()
    a._num_runs = 3
    t = _task(category=2, system="zep")
    out = await a.judge_answer(t, "X")

    assert out["category"] == 2
    assert out["category_name"] == "temporal"
    assert out["system"] == "zep"


# ---------------------------------------------------------------------------
# Dataset resolution — re-uses locomo_original's helpers
# ---------------------------------------------------------------------------


def test_load_dataset_explicit_file_path(tmp_path: Path):
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps([_sample("sample_x")]), encoding="utf-8")
    tasks = list(LoCoMoEverosAdapter().load_tasks({
        "file_path": str(custom),
        "data_path": str(tmp_path / "nope"),
    }))
    # 1 sample * 4 non-cat-5 QAs = 4.
    assert len(tasks) == 4
    assert all(t.metadata["sample_id"] == "sample_x" for t in tasks)


def test_load_dataset_missing_file_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import urllib.request

    def _boom(*_args, **_kwargs):
        raise OSError("simulated offline")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)

    with pytest.raises(FileNotFoundError, match="LoCoMo dataset not found"):
        list(LoCoMoEverosAdapter().load_tasks({
            "data_path": str(tmp_path / "nope"),
        }))
