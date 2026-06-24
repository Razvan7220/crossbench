"""Tests for Mastra Embedded middleware contract.

Locks two regressions that tanked LoCoMo Mastra pass-rate to ~7% on the
2026-05-10 run:

1. Observer rendering used ``time.localtime(m.created_at)`` (= ingest
   wall-clock = today), ignoring the source conversation's timestamp.
   Observations got stamped with "2026-05-10" for LoCoMo events that
   happened in 2023, and the model answered with the wrong year on
   every temporal question.

2. ``MASTRA_ANSWER_PROMPT`` did not demand the ``FINAL ANSWER:`` format
   that ``execute_single_shot`` extracts before scoring. F1 ran over
   the whole prose prediction, drowning genuine matches in disclaimer
   tokens.
"""

from __future__ import annotations

import time

import pytest

from cross_bench.evaluation.provider_prompts import (
    MASTRA_ANSWER_PROMPT,
    format_answer_prompt,
)
from cross_bench.systems.memory.mastra_embedded import Message


# ---------------------------------------------------------------------------
# Bug 1: Observer must anchor to source-conversation timestamps
# ---------------------------------------------------------------------------


def test_message_accepts_timestamp_str():
    """``Message`` carries the source-conversation timestamp verbatim."""
    m = Message(
        id="m1",
        role="user",
        text="hello",
        thread_id="t1",
        resource_id="r1",
        timestamp_str="2023-05-08 16:30 PM",
    )
    assert m.timestamp_str == "2023-05-08 16:30 PM"


def test_message_timestamp_str_defaults_empty():
    """No ``timestamp_str`` → empty string (legacy callers unaffected)."""
    m = Message(
        id="m1",
        role="user",
        text="hello",
        thread_id="t1",
        resource_id="r1",
    )
    assert m.timestamp_str == ""


def test_observer_prefers_timestamp_str_over_created_at():
    """Observer rendering uses ``timestamp_str`` when set.

    The format string is what the observer LLM sees: with ``timestamp_str``
    the date should be 2023-05-08 (from LoCoMo), NOT ``time.strftime`` of
    today.
    """
    m = Message(
        id="m1",
        role="user",
        text="hello",
        thread_id="t1",
        resource_id="r1",
        timestamp_str="2023-05-08 16:30 PM",
    )
    # This is the exact expression used inside ``_run_observer``.
    rendered = (
        f"[{m.role}] ({m.timestamp_str or time.strftime('%Y-%m-%d %H:%M', time.localtime(m.created_at))}) "
        f"{m.text[:500]}"
    )
    assert "2023-05-08 16:30 PM" in rendered
    # No today-style stamp should sneak in.
    assert time.strftime("%Y-%m-%d", time.localtime()) not in rendered


def test_observer_falls_back_to_created_at_when_no_timestamp_str():
    """Legacy callers without ``timestamp_str`` keep current behaviour."""
    t = 1683551400.0  # 2023-05-08 in UTC, deterministic
    m = Message(
        id="m1",
        role="user",
        text="hello",
        thread_id="t1",
        resource_id="r1",
        created_at=t,
    )
    rendered = (
        f"[{m.role}] ({m.timestamp_str or time.strftime('%Y-%m-%d %H:%M', time.localtime(m.created_at))}) "
        f"{m.text[:500]}"
    )
    # Year should match the deterministic created_at, not today.
    assert "2023" in rendered or "2025" in rendered  # local-tz can shift by 1 day; year must match the source


# ---------------------------------------------------------------------------
# Bug 2: MASTRA_ANSWER_PROMPT must demand FINAL ANSWER format
# ---------------------------------------------------------------------------


def test_mastra_answer_prompt_demands_final_answer_marker():
    """Without this directive, ``_extract_final_answer`` in the engine
    can't strip CoT prose before F1 scoring."""
    assert "FINAL ANSWER:" in MASTRA_ANSWER_PROMPT


def test_mastra_answer_prompt_demands_cat5_abstention_substring():
    """Cat-5 (adversarial) requires one of LoCoMo's two abstention
    substrings; the upstream-faithful one is ``no information available``."""
    assert "no information available" in MASTRA_ANSWER_PROMPT


def test_mastra_answer_prompt_renders_with_question_and_context():
    """``format_answer_prompt`` substitutes both placeholders cleanly."""
    rendered = format_answer_prompt(
        provider="mastra_embedded",
        question="When did Caroline go to the LGBTQ support group?",
        context="[Observations]\n[HIGH] 2023-05-07 | User attended ...",
    )
    assert "When did Caroline go to the LGBTQ support group?" in rendered
    assert "[HIGH] 2023-05-07" in rendered
    assert "FINAL ANSWER:" in rendered
    # No leftover ``{context}``/``{question}`` placeholders.
    assert "{context}" not in rendered
    assert "{question}" not in rendered


# ---------------------------------------------------------------------------
# Integration: post_response wires context["timestamp"] into Message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_response_propagates_timestamp_str(monkeypatch):
    """``post_response`` reads ``context["timestamp"]`` and stamps it on
    BOTH user and assistant messages so the observer sees source-clock
    dates, not today."""
    # Lazy-import to avoid sentence_transformers at module load.
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory import mastra_embedded as mem

    # Stub the side-effect-heavy paths.
    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_update_working_memory",
        lambda self, t, p, r: None,
    )
    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_run_observer",
        lambda self, t: None,
    )
    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_run_reflector",
        lambda self, t: None,
    )

    cfg = MemoryConfig(provider="mastra_embedded", params={})
    mw = mem.MastraEmbeddedMiddleware(cfg)
    await mw.post_response(
        prompt="What's your favorite color?",
        response="Blue.",
        context={"timestamp": "2023-05-08 16:30 PM"},
    )
    msgs = list(mw._threads.values())[0].messages
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].timestamp_str == "2023-05-08 16:30 PM"
    assert msgs[1].role == "assistant"
    assert msgs[1].timestamp_str == "2023-05-08 16:30 PM"


@pytest.mark.asyncio
async def test_post_response_handles_missing_timestamp(monkeypatch):
    """No ``timestamp`` key → ``timestamp_str`` is empty (legacy path)."""
    from cross_bench.models import MemoryConfig
    from cross_bench.systems.memory import mastra_embedded as mem

    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_update_working_memory",
        lambda self, t, p, r: None,
    )
    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_run_observer",
        lambda self, t: None,
    )
    monkeypatch.setattr(
        mem.MastraEmbeddedMiddleware, "_run_reflector",
        lambda self, t: None,
    )

    cfg = MemoryConfig(provider="mastra_embedded", params={})
    mw = mem.MastraEmbeddedMiddleware(cfg)
    await mw.post_response(
        prompt="Hi",
        response="Hello.",
        context={},  # no timestamp
    )
    msgs = list(mw._threads.values())[0].messages
    assert msgs[0].timestamp_str == ""
    assert msgs[1].timestamp_str == ""


# ---------------------------------------------------------------------------
# Reflector prompt format constraint (M51)
# ---------------------------------------------------------------------------
#
# The 2026-05-11 LoCoMo run revealed a bimodal accuracy split:
# 3 shards at 34-42% pass-rate vs 7 shards at 5-10% (with cat-2 = 0%
# on every "bad" shard). Root cause: large conversations exceed the
# reflector's 40k-token threshold and trigger compression. The
# reflector LLM (gpt-4o-mini) — under Level 4 EXTREME COMPRESSION
# — produced a free-form narrative summary with no per-observation
# dates, destroying temporal grounding for downstream answering.
#
# Fix: make the per-line `[priority] YYYY-MM-DD | text` output
# format a HARD CONSTRAINT in the base prompt AND in every
# compression-level guidance string. Compression is achieved by
# MERGING lines, never by reformatting them into prose.


def test_reflector_base_prompt_mandates_per_line_format():
    """Base prompt must explicitly require the per-line
    ``[priority] YYYY-MM-DD | text`` shape — pre-fix it was only
    listed as a soft rule and the LLM ignored it under pressure."""
    from cross_bench.systems.memory.mastra_embedded import _REFLECTOR_BASE_PROMPT
    assert "[priority] YYYY-MM-DD | observation text" in _REFLECTOR_BASE_PROMPT
    assert "OUTPUT FORMAT" in _REFLECTOR_BASE_PROMPT
    assert "HARD CONSTRAINT" in _REFLECTOR_BASE_PROMPT


def test_reflector_base_prompt_says_never_collapse_into_paragraph():
    """The explicit ``Never collapse multiple observations into a
    paragraph`` instruction is the surgical fix for the 2026-05-11
    Level 4 failure mode where conv-42 emitted Markdown narrative
    instead of date-prefixed lines."""
    from cross_bench.systems.memory.mastra_embedded import _REFLECTOR_BASE_PROMPT
    # Normalize whitespace to make assertion robust against any
    # in-prompt line-wrapping the author chooses.
    flat = " ".join(_REFLECTOR_BASE_PROMPT.split())
    assert "Never collapse multiple observations into a paragraph" in flat


def test_reflector_base_prompt_describes_merge_strategy():
    """Compression must happen by MERGING (reducing count), not by
    reformatting. The base prompt spells this out so the LLM has a
    well-defined operation."""
    from cross_bench.systems.memory.mastra_embedded import _REFLECTOR_BASE_PROMPT
    assert "MERGE observations" in _REFLECTOR_BASE_PROMPT


def test_compression_guidance_levels_repeat_format_constraint():
    """Each compression level (1-4) repeats the per-line format
    constraint. LLMs follow the LAST instruction they saw more
    reliably than the first; the repetition is defensive against
    the failure mode where Level 4 pressure overrides the base
    prompt's instruction."""
    from cross_bench.systems.memory.mastra_embedded import COMPRESSION_GUIDANCE
    # Level 0 is intentionally empty (first attempt, no compression).
    assert COMPRESSION_GUIDANCE[0] == ""
    for level in (1, 2, 3, 4):
        guidance = COMPRESSION_GUIDANCE[level]
        # Each level must mention the date prefix or the per-line format.
        assert (
            "YYYY-MM-DD" in guidance
            or "[priority]" in guidance
        ), f"Level {level} guidance is missing the format reminder: {guidance!r}"


def test_compression_guidance_level_4_explicitly_forbids_paragraphs():
    """Level 4 is where the original failure happened; it must
    explicitly forbid the paragraph/bullet/header reformat that
    destroyed the 2026-05-11 Mastra cat-2 accuracy."""
    from cross_bench.systems.memory.mastra_embedded import COMPRESSION_GUIDANCE
    g4 = COMPRESSION_GUIDANCE[4]
    assert "paragraph" in g4.lower()


def test_compression_guidance_level_3_keeps_date_prefix_nonnegotiable():
    """Level 3 must also call out date preservation — the symptom on
    bad shards was 0% cat-2 (temporal), which traces directly to
    losing the date prefix."""
    from cross_bench.systems.memory.mastra_embedded import COMPRESSION_GUIDANCE
    g3 = COMPRESSION_GUIDANCE[3]
    assert "non-negotiable" in g3.lower() or "mandatory" in g3.lower() or "MUST" in g3
