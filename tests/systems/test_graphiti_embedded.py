"""Regression tests for the Graphiti/Zep answer-prompt fixes.

The 2026-05-21 ``locomo-graphiti-20260521-202532`` run produced
pass_rate=4.8% (cat-5=0.4%, cat-2=0%) despite working retrieval
(509ms search, 318 retrieval tokens per task). Diagnosis: the model
emitted free-form prose like ``"The retrieved memories do not specify a
date for when Caroline attended the LGBTQ support group."`` because
``GRAPHITI_ANSWER_PROMPT`` ended with the soft directive ``"Answer
concisely and directly based on the memories above."`` — no
``FINAL ANSWER:`` marker for ``_extract_final_answer`` to strip
CoT preamble, no cat-5 abstention substring instruction.

Same trap as:
- L1 — every memory-system prompt needs ``FINAL ANSWER:`` extraction
- M39 — Mastra abstention substring + FINAL ANSWER format
- H47 — Hindsight abstention substring + FINAL ANSWER format

These tests lock the graphiti/zep prompts so the regression can't
quietly come back.
"""

from __future__ import annotations

from cross_bench.evaluation.provider_prompts import (
    GRAPHITI_ANSWER_PROMPT,
    ZEP_ANSWER_PROMPT,
)


# ---------------------------------------------------------------------------
# GRAPHITI_ANSWER_PROMPT
# ---------------------------------------------------------------------------


def test_graphiti_answer_prompt_demands_final_answer_marker():
    """``_extract_final_answer`` (locomo_original.py) strips CoT
    preamble before F1 scoring, but only when the model emits a
    ``FINAL ANSWER:`` line. Without it the entire prose is graded.
    """
    assert "FINAL ANSWER:" in GRAPHITI_ANSWER_PROMPT


def test_graphiti_answer_prompt_demands_cat5_abstention_substring():
    """Cat-5 (adversarial) F1 is the substring check for one of
    LoCoMo's two upstream-accepted abstention phrases. The model must
    be told to emit one of them literally on missing answers.
    """
    assert "no information available" in GRAPHITI_ANSWER_PROMPT


def test_graphiti_answer_prompt_renders_with_placeholders():
    """``format`` substitutes ``{context}`` and ``{question}``
    cleanly; the FINAL ANSWER directive survives substitution."""
    rendered = GRAPHITI_ANSWER_PROMPT.format(
        context="[fact] Caroline attended LGBTQ support group (score: 0.5)",
        question="When did Caroline attend the LGBTQ support group?",
    )
    assert "When did Caroline" in rendered
    assert "Caroline attended LGBTQ support group" in rendered
    assert "FINAL ANSWER:" in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered


def test_graphiti_answer_prompt_preserves_bi_temporal_framing():
    """The graphiti-specific framing — bi-temporal knowledge graph,
    facts vs entity summaries — has to survive the FINAL ANSWER
    addition. It's the load-bearing graphiti-vs-others distinction
    in the answer prompt."""
    assert "temporally-aware knowledge graph" in GRAPHITI_ANSWER_PROMPT
    assert "bi-temporal" in GRAPHITI_ANSWER_PROMPT
    assert "Facts:" in GRAPHITI_ANSWER_PROMPT
    assert "Entity summaries:" in GRAPHITI_ANSWER_PROMPT


# ---------------------------------------------------------------------------
# ZEP_ANSWER_PROMPT — shared bug, shared fix
# ---------------------------------------------------------------------------


def test_zep_answer_prompt_demands_final_answer_marker():
    assert "FINAL ANSWER:" in ZEP_ANSWER_PROMPT


def test_zep_answer_prompt_demands_cat5_abstention_substring():
    assert "no information available" in ZEP_ANSWER_PROMPT


def test_zep_answer_prompt_renders_with_placeholders():
    rendered = ZEP_ANSWER_PROMPT.format(
        facts="event_time=2023-05-08 — Caroline attended group",
        entities="Caroline — LGBTQ ally",
        question="When did Caroline attend the LGBTQ support group?",
    )
    assert "FINAL ANSWER:" in rendered
    assert "When did Caroline" in rendered
    assert "{facts}" not in rendered
    assert "{entities}" not in rendered
    assert "{question}" not in rendered


def test_zep_answer_prompt_preserves_facts_entities_framing():
    """The zep-specific FACTS / ENTITIES split (vs graphiti's
    inline format) has to survive the FINAL ANSWER addition."""
    assert "FACTS:" in ZEP_ANSWER_PROMPT
    assert "ENTITIES:" in ZEP_ANSWER_PROMPT
    assert "event_time" in ZEP_ANSWER_PROMPT
