"""Tests for Hindsight Embedded middleware tz-awareness invariant.

Locks the regression that took down every LoCoMo conv-* shard on the
2026-05-10 dev-AWS run:

    Traceback (most recent call last):
      File "hindsight_embedded.py", line 467, in _build_links
        hours_diff = abs((unit.occurred_at - other.occurred_at).total_seconds()) / 3600.0
    TypeError: can't subtract offset-naive and offset-aware datetimes

Root cause: ``_parse_date`` returned naive ``datetime`` on the
``fromisoformat`` branch (and passthrough for any caller-supplied naive
``datetime``), while ``MemoryUnit`` defaults / ``datetime.now(timezone.utc)``
elsewhere are aware. Any cross-subtraction crashed.

Fix: ``_parse_date`` now always returns aware UTC when non-None.

Also locks the 2026-05-12 follow-up fixes (H46-H50) for the post-fix
3% pass-rate observed on the 2026-05-11 dev-AWS run:

- H46: ``_parse_date`` handles LoCoMo's natural-language format
  "1:56 pm on 8 May, 2023" so ``post_response`` no longer falls back
  to ``datetime.now()`` for every ingestion.
- H47: ``HINDSIGHT_ANSWER_PROMPT`` demands ``FINAL ANSWER:`` so
  ``_extract_final_answer`` can strip CoT before F1 grading.
- H48: ``_EXTRACT_FACTS_USER`` prompt expanded with examples,
  coreference resolution, USER ASSERTIONS rule, and temporal
  anchoring guidance.
- H49: ``pre_prompt`` caps recall output at ``recall_max_tokens``
  (default 3000) — mirrors upstream's token-budget recall.
- H50: ``_recall`` applies recency boost in combined scoring —
  mirrors upstream's ``apply_combined_scoring`` shape.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cross_bench.evaluation.provider_prompts import HINDSIGHT_ANSWER_PROMPT
from cross_bench.systems.memory.hindsight_embedded import _parse_date


# ---------------------------------------------------------------------------
# Core invariant: non-None result is always tz-aware UTC
# ---------------------------------------------------------------------------


def test_parse_date_iso_string_no_tz_returns_aware_utc():
    """``fromisoformat("2023-05-08")`` is naive — fix stamps UTC."""
    result = _parse_date("2023-05-08")
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 8


def test_parse_date_iso_string_with_z_returns_aware_utc():
    """``Z`` suffix → ``fromisoformat`` already produces aware."""
    result = _parse_date("2023-05-08T16:30:00Z")
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0


def test_parse_date_iso_string_with_explicit_offset_preserved():
    """Explicit offset is honoured (not overwritten to UTC)."""
    result = _parse_date("2023-05-08T16:30:00+05:30")
    assert result is not None
    assert result.tzinfo is not None
    # Offset is preserved by ``fromisoformat`` — the fix only stamps
    # UTC when no tz info is present.
    assert result.utcoffset().total_seconds() == 5.5 * 3600


def test_parse_date_naive_datetime_input_stamped_utc():
    """Caller-supplied naive ``datetime`` gets a UTC stamp."""
    result = _parse_date(datetime(2023, 5, 8, 16, 30))
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0


def test_parse_date_aware_datetime_input_passthrough():
    """Caller-supplied aware ``datetime`` is preserved verbatim."""
    aware = datetime(2023, 5, 8, 16, 30, tzinfo=timezone.utc)
    result = _parse_date(aware)
    assert result is not None
    assert result.tzinfo is timezone.utc


def test_parse_date_yyyymmdd_branch_returns_aware_utc():
    """Pre-fix this branch was already aware; lock the contract."""
    # Force ``fromisoformat`` to fail by giving an oddly-formatted
    # date that the regex still matches. ``fromisoformat`` rejects
    # single-digit month/day; regex accepts both.
    result = _parse_date("2023-5-8 some trailing garbage")
    assert result is not None
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 8


def test_parse_date_returns_none_for_empty():
    assert _parse_date(None) is None
    assert _parse_date("") is None
    assert _parse_date("null") is None


def test_parse_date_returns_none_for_garbage():
    assert _parse_date("not a date") is None


# ---------------------------------------------------------------------------
# End-to-end: ``_parse_date`` results never crash datetime subtraction
# ---------------------------------------------------------------------------


def test_parse_date_subtraction_against_aware_now_does_not_crash():
    """``_apply_temporal_search`` / recency boost do
    ``datetime.now(timezone.utc) - u.occurred_at``. Pre-fix this
    crashed for any iso-string-without-tz input."""
    now_aware = datetime.now(timezone.utc)
    parsed = _parse_date("2023-05-08")
    assert parsed is not None
    # Must not raise.
    delta = now_aware - parsed
    assert delta.total_seconds() > 0


def test_parse_date_cross_subtraction_does_not_crash():
    """``_build_links`` does ``unit.occurred_at - other.occurred_at``
    across pairs of units. With the fix, any two ``_parse_date``
    results can be subtracted regardless of which branch produced
    each one."""
    a = _parse_date("2023-05-08")  # fromisoformat branch
    b = _parse_date(datetime(2023, 5, 9, 12, 0))  # naive-datetime input branch
    c = _parse_date("2023-05-10T08:00:00Z")  # aware fromisoformat branch
    d = _parse_date("2023-05-11")  # YYYY-MM-DD regex branch (via single-digit fail-through)
    assert a is not None and b is not None and c is not None and d is not None
    # All pairwise subtractions must succeed.
    for lhs in (a, b, c, d):
        for rhs in (a, b, c, d):
            delta = lhs - rhs
            assert isinstance(delta.total_seconds(), float)


# ---------------------------------------------------------------------------
# H46: LoCoMo natural-language timestamp parsing
# ---------------------------------------------------------------------------


def test_parse_date_handles_locomo_pm_format():
    """LoCoMo timestamp shape: ``"1:56 pm on 8 May, 2023"``."""
    result = _parse_date("1:56 pm on 8 May, 2023")
    assert result is not None
    assert result.tzinfo is not None
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 8
    assert result.hour == 13  # 1 pm = 13:00 (24h)
    assert result.minute == 56


def test_parse_date_handles_locomo_am_format():
    """AM hour stays under 12."""
    result = _parse_date("9:30 am on 12 July, 2023")
    assert result is not None
    assert result.year == 2023
    assert result.month == 7
    assert result.day == 12
    assert result.hour == 9
    assert result.minute == 30


def test_parse_date_handles_locomo_noon_edge_case():
    """12 pm = noon (12:00 in 24h), 12 am = midnight (00:00)."""
    noon = _parse_date("12:00 pm on 1 January, 2023")
    midnight = _parse_date("12:00 am on 1 January, 2023")
    assert noon is not None and noon.hour == 12
    assert midnight is not None and midnight.hour == 0


def test_parse_date_handles_locomo_no_comma():
    """Comma between day-month and year is optional in upstream data."""
    result = _parse_date("3:15 pm on 4 December 2024")
    assert result is not None
    assert result.year == 2024
    assert result.month == 12
    assert result.day == 4
    assert result.hour == 15


def test_parse_date_handles_locomo_uppercase_ampm():
    result = _parse_date("11:45 AM on 30 June, 2023")
    assert result is not None
    assert result.hour == 11
    assert result.minute == 45


def test_parse_date_rejects_locomo_lookalikes():
    """Garbage that contains 'pm on' words but isn't a date should
    not silently parse."""
    assert _parse_date("the meeting is set for tomorrow pm on the agenda") is None
    assert _parse_date("pm on May") is None


# ---------------------------------------------------------------------------
# H47: HINDSIGHT_ANSWER_PROMPT demands FINAL ANSWER format
# ---------------------------------------------------------------------------


def test_hindsight_answer_prompt_demands_final_answer_marker():
    assert "FINAL ANSWER:" in HINDSIGHT_ANSWER_PROMPT


def test_hindsight_answer_prompt_demands_cat5_abstention_substring():
    """Cat-5 (adversarial) requires one of LoCoMo's two abstention
    substrings."""
    assert "no information available" in HINDSIGHT_ANSWER_PROMPT


def test_hindsight_answer_prompt_warns_against_today_substitution():
    """Without this guidance, the model substitutes today's date for
    LoCoMo events from 2023 — the documented failure mode."""
    assert "NOT today" in HINDSIGHT_ANSWER_PROMPT or "not today" in HINDSIGHT_ANSWER_PROMPT.lower()


def test_hindsight_answer_prompt_renders_with_placeholders():
    """``format`` substitutes ``{context}`` and ``{question}`` cleanly."""
    rendered = HINDSIGHT_ANSWER_PROMPT.format(
        context="[experience] User attended LGBTQ support group (score: 0.5, date=2023-05-08)",
        question="When did Caroline attend the LGBTQ support group?",
    )
    assert "When did Caroline" in rendered
    assert "date=2023-05-08" in rendered
    assert "FINAL ANSWER:" in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered


# ---------------------------------------------------------------------------
# H48: _EXTRACT_FACTS_USER prompt has worked examples + temporal anchoring
# ---------------------------------------------------------------------------


def test_extract_facts_prompt_has_temporal_anchoring_guidance():
    """The prompt must tell the LLM to resolve relative time
    expressions against ``reference_time``, NOT against today."""
    from cross_bench.systems.memory.hindsight_embedded import _EXTRACT_FACTS_USER
    p = _EXTRACT_FACTS_USER
    # Check rule and examples are present.
    assert "REFERENCE TIME" in p
    assert "{reference_time}" in p
    # The "not today" anti-pattern must be stated.
    assert "NOT against today" in p
    # Worked examples.
    assert "2023-05-06" in p or "2023-05-07" in p
    assert "WORKED EXAMPLES" in p


def test_extract_facts_prompt_has_coreference_rule():
    from cross_bench.systems.memory.hindsight_embedded import _EXTRACT_FACTS_USER
    assert "Coreference" in _EXTRACT_FACTS_USER or "coreference" in _EXTRACT_FACTS_USER.lower()


def test_extract_facts_prompt_has_user_assertions_rule():
    from cross_bench.systems.memory.hindsight_embedded import _EXTRACT_FACTS_USER
    assert "USER ASSERTIONS" in _EXTRACT_FACTS_USER


# ---------------------------------------------------------------------------
# H49: pre_prompt applies token budget (recall_max_tokens)
# ---------------------------------------------------------------------------


def test_recall_max_tokens_default_is_3000():
    """Default budget mirrors upstream's ``max_tokens=4096`` ballpark
    with some headroom for the answer prompt scaffolding."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory.hindsight_embedded import HindsightEmbeddedMiddleware
    cfg = MemoryConfig(provider="hindsight_embedded", params={})
    mw = HindsightEmbeddedMiddleware(cfg)
    assert mw._recall_max_tokens == 3000


def test_retrieval_limit_default_bumped_to_50():
    """Old default of 10 produced ~280 tokens of context — model
    starved. New default 50 lifts the count cap; the token-budget
    cap is the real stopper."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory.hindsight_embedded import HindsightEmbeddedMiddleware
    cfg = MemoryConfig(provider="hindsight_embedded", params={})
    mw = HindsightEmbeddedMiddleware(cfg)
    assert mw._retrieval_limit == 50


def test_recall_max_tokens_param_override():
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory.hindsight_embedded import HindsightEmbeddedMiddleware
    cfg = MemoryConfig(provider="hindsight_embedded", params={"recall_max_tokens": 1500})
    mw = HindsightEmbeddedMiddleware(cfg)
    assert mw._recall_max_tokens == 1500


# ---------------------------------------------------------------------------
# H50: recency boost in combined scoring
# ---------------------------------------------------------------------------


def test_recency_boost_default_weight_is_02():
    """Mirrors upstream ``apply_combined_scoring(ce * recency_boost(
    0.2) * temporal_boost(0.2))`` — see deviations.md H19."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory.hindsight_embedded import HindsightEmbeddedMiddleware
    cfg = MemoryConfig(provider="hindsight_embedded", params={})
    mw = HindsightEmbeddedMiddleware(cfg)
    assert mw._recency_boost_weight == pytest.approx(0.2)
    assert mw._recency_half_life_days == pytest.approx(30)


def test_recency_boost_disabled_when_weight_zero():
    """``recency_boost_weight=0`` opts out of the boost (e.g. for
    apples-to-apples comparison with the pre-fix run)."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory.hindsight_embedded import HindsightEmbeddedMiddleware
    cfg = MemoryConfig(provider="hindsight_embedded", params={"recency_boost_weight": 0})
    mw = HindsightEmbeddedMiddleware(cfg)
    assert mw._recency_boost_weight == 0


# ---------------------------------------------------------------------------
# Hot-fix: post_response tolerates LLM-emitted nulls for 'who' / 'fact_type'
# ---------------------------------------------------------------------------
#
# The 2026-05-12 Hindsight run failed at minute 3 with
#     TypeError: 'NoneType' object is not iterable
# at hindsight_embedded.py::post_response when iterating ``entity_names``.
# Root cause: the expanded H48 extraction prompt encourages the LLM to
# emit ``"who": null`` for unattributed facts, but
# ``raw.get("who", [])`` only returns the default when the key is
# MISSING — an explicit null bypasses the default. Same trap for
# ``fact_type``. All 10 shards exited 1; the run produced 0 summaries.


async def _run_post_response_with_extracted(monkeypatch, extracted_facts: list[dict]):
    """Helper: run ``post_response`` with a stubbed extractor that
    returns the supplied raw facts. Returns the middleware so the
    caller can inspect ``_units``."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory import hindsight_embedded as hm

    cfg = MemoryConfig(provider="hindsight_embedded", params={
        "llm_api_key": "stub",
        "enable_consolidation": "false",
    })
    mw = hm.HindsightEmbeddedMiddleware(cfg)

    async def _stub_extract(self, text, ref_time):
        return list(extracted_facts)

    async def _stub_embed_batch(self, texts):
        return [[0.0] for _ in texts]

    async def _stub_resolve_entity(self, name, ref_time):
        return f"ent:{name}"

    monkeypatch.setattr(hm.HindsightEmbeddedMiddleware, "_extract_facts", _stub_extract)
    monkeypatch.setattr(hm.HindsightEmbeddedMiddleware, "_embed_batch", _stub_embed_batch)
    monkeypatch.setattr(hm.HindsightEmbeddedMiddleware, "_resolve_entity", _stub_resolve_entity)

    await mw.post_response(
        prompt="",
        response="any text",
        context={"timestamp": "1:56 pm on 8 May, 2023"},
    )
    return mw


async def test_post_response_tolerates_who_null(monkeypatch):
    """``"who": null`` (explicit JSON null) used to crash with
    TypeError; now it should be normalized to no entities."""
    mw = await _run_post_response_with_extracted(monkeypatch, [
        {"fact_text": "User attended a support group.",
         "fact_type": "experience",
         "who": None,
         "when": "2023-05-07",
         "cause_of": None},
    ])
    units = list(mw._units.values())
    assert len(units) == 1
    assert units[0].entity_ids == []


async def test_post_response_tolerates_who_missing(monkeypatch):
    """``"who"`` key missing entirely — same defensive treatment."""
    mw = await _run_post_response_with_extracted(monkeypatch, [
        {"fact_text": "User attended a support group.",
         "fact_type": "experience",
         "when": "2023-05-07",
         "cause_of": None},
    ])
    units = list(mw._units.values())
    assert len(units) == 1
    assert units[0].entity_ids == []


async def test_post_response_tolerates_who_bare_string(monkeypatch):
    """Some LLMs emit ``"who": "Caroline"`` instead of
    ``"who": ["Caroline"]``. Normalize to singleton list."""
    mw = await _run_post_response_with_extracted(monkeypatch, [
        {"fact_text": "Caroline went to the museum.",
         "fact_type": "experience",
         "who": "Caroline",
         "when": "2023-05-07",
         "cause_of": None},
    ])
    units = list(mw._units.values())
    assert len(units) == 1
    assert units[0].entity_ids == ["ent:Caroline"]


async def test_post_response_handles_who_list_normally(monkeypatch):
    """Sanity: normal list shape still works."""
    mw = await _run_post_response_with_extracted(monkeypatch, [
        {"fact_text": "Caroline met Bob.",
         "fact_type": "experience",
         "who": ["Caroline", "Bob"],
         "when": "2023-05-07",
         "cause_of": None},
    ])
    units = list(mw._units.values())
    assert len(units) == 1
    assert set(units[0].entity_ids) == {"ent:Caroline", "ent:Bob"}


async def test_post_response_tolerates_fact_type_null(monkeypatch):
    """``"fact_type": null`` used to silently produce a ``MemoryUnit``
    with ``fact_type=None``, which breaks the answer-prompt rendering
    (``[None]`` in the bracketed fact-type prefix). Normalize to the
    default ``"world"``."""
    mw = await _run_post_response_with_extracted(monkeypatch, [
        {"fact_text": "Some fact.",
         "fact_type": None,
         "who": [],
         "when": None,
         "cause_of": None},
    ])
    units = list(mw._units.values())
    assert len(units) == 1
    assert units[0].fact_type == "world"
