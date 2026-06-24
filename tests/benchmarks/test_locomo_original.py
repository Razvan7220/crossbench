# ruff: noqa: E501 - strings below are byte-exact copies of upstream prompts
# from ``snap-research/locomo/task_eval/gpt_utils.py`` and ``evaluation.py``;
# splitting them would defeat the regression check's purpose.
"""LoCoMo adapter tests — lock byte-exact fidelity to upstream.

The prompt strings below are copied verbatim from
``snap-research/locomo/task_eval/gpt_utils.py`` and the F1 / normalization
from ``task_eval/evaluation.py`` so the regression assertions fail loudly if
we ever drift.

We stage the locomo10.json fixture under ``tmp_path`` and walk the full
dataflow: JSON -> load_tasks -> format_prompt -> judge_answer, asserting on
the exact input, transformations and outputs at each stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks.locomo_original import (
    _CAT2_DATE_HINT,
    _CAT5_ABSTENTION_OPTION,
    _CAT5_MCQ_TEMPLATE,
    _CONV_START_PROMPT,
    _QA_PROMPT,
    _QA_PROMPT_CAT_5,
    CATEGORY_NAMES,
    LoCoMoOriginalAdapter,
    _adversarial_correct,
    _build_conversation_context,
    _build_qa_prompt,
    _cat5_answer_key,
    _cat5_mcq,
    _evidence_recall,
    _f1_multi,
    _f1_score,
    _format_turn,
    _get_cat_5_answer,
    _normalize_answer,
    _rewrite_question,
    _score_qa,
    _session_keys_sorted,
    _speaker_pair,
)
from cross_bench.models import Task, TaskGranularity

# ---------------------------------------------------------------------------
# Byte-exact upstream strings. Copied verbatim from
# snap-research/locomo/task_eval/gpt_utils.py (HEAD of main as of commit
# 91c1069, 2024-08-13) and task_eval/evaluation.py. Any drift must fail.
# ---------------------------------------------------------------------------

# CONV_START_PROMPT — gpt_utils.py line 51. Preserves the upstream typo
# "wriiten" (sic) on purpose.
_UPSTREAM_CONV_START = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

# QA_PROMPT — gpt_utils.py lines 25-29. Used for categories 1-4.
_UPSTREAM_QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

# QA_PROMPT_CAT_5 — gpt_utils.py lines 31-35. Used for category 5.
_UPSTREAM_QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# Cat-2 date hint — gpt_utils.py line 244.
_UPSTREAM_CAT2_HINT = " Use DATE of CONVERSATION to answer with an approximate date."

# Cat-5 MCQ template — gpt_utils.py line 246.
_UPSTREAM_CAT5_TEMPLATE = " Select the correct answer: (a) {} (b) {}. "

# Cat-5 abstention option — gpt_utils.py lines 248, 251.
_UPSTREAM_ABSTENTION_OPTION = "Not mentioned in the conversation"


# ---------------------------------------------------------------------------
# Sample factory — shape mirrors data/locomo10.json
# ---------------------------------------------------------------------------


def _sample(
    sample_id: str = "sample_0",
    speaker_a: str = "Caroline",
    speaker_b: str = "Melanie",
    qa_items: list[dict[str, Any]] | None = None,
    conversation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One upstream-shaped LoCoMo sample with 2 sessions and 5 QA pairs.

    Covers one question per reasoning category so the dataflow tests can
    exercise every branch of _score_qa / _rewrite_question.
    """
    if conversation is None:
        conversation = {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "session_1_date_time": "1:00 pm on 1 May 2023",
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": speaker_a,
                    "text": "I went hiking on Saturday.",
                },
                {
                    "dia_id": "D1:2",
                    "speaker": speaker_b,
                    "text": "Where did you go?",
                },
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
                {
                    "dia_id": "D2:1",
                    "speaker": speaker_b,
                    "text": "How was your weekend?",
                },
                {
                    "dia_id": "D2:2",
                    "speaker": speaker_a,
                    "text": "I finished a book on gardening.",
                },
            ],
        }

    if qa_items is None:
        qa_items = [
            {
                "question": "What outdoor activities does Caroline enjoy?",
                "answer": "hiking, gardening",
                "category": 1,
                "evidence": ["D1:1", "D2:2"],
            },
            {
                "question": "When did Caroline go hiking?",
                "answer": "May 2023",
                "category": 2,
                "evidence": ["D1:1"],
            },
            {
                "question": "Is gardening typically an outdoor activity?",
                "answer": "Likely yes; gardening is usually done outdoors",
                "category": 3,
                "evidence": ["D2:2"],
            },
            {
                "question": "Where did Caroline hike?",
                "answer": "Appalachian Trail",
                "category": 4,
                "evidence": ["D1:3"],
            },
            {
                "question": "What mountain did Melanie climb?",
                "answer": "Mount Everest",
                "category": 5,
                "evidence": [],
            },
        ]

    return {
        "sample_id": sample_id,
        "conversation": conversation,
        "qa": qa_items,
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write a minimal ``locomo10.json`` fixture under ``tmp_path``."""
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

    a = get_adapter("locomo_original")
    assert isinstance(a, LoCoMoOriginalAdapter)
    assert a.name == "locomo_original"
    assert a.granularity == TaskGranularity.CONVERSATIONAL
    assert a.requires_sandbox is False


def test_default_params_match_upstream_defaults():
    p = LoCoMoOriginalAdapter().get_default_params()
    assert p["data_path"] == "data/locomo"
    assert p["prompt_style"] == "question"
    assert p["filter_categories"] == []
    # Upstream does not use LLM-as-judge; it thresholds F1 at 0.5 (implicit
    # "non-trivial overlap" expectation).
    assert p["judge"]["f1_threshold"] == 0.5


def test_category_names_match_paper():
    # Locked mapping — integer codes are a contract with
    # snap-research/locomo's qa[i]["category"]. Do not renumber.
    assert CATEGORY_NAMES == {
        1: "multi_hop",
        2: "temporal",
        3: "commonsense",
        4: "single_hop",
        5: "adversarial",
    }


# ---------------------------------------------------------------------------
# Byte-exact prompt constants
# ---------------------------------------------------------------------------


def test_conv_start_prompt_preserves_upstream_typo():
    # "wriiten" is a typo in upstream gpt_utils.py line 51 — we preserve it
    # intentionally so prompts are byte-identical.
    assert "wriiten" in _CONV_START_PROMPT
    assert "written" not in _CONV_START_PROMPT  # sanity: no accidental fix


def test_conv_start_prompt_matches_upstream():
    assert _CONV_START_PROMPT == _UPSTREAM_CONV_START


def test_qa_prompt_matches_upstream():
    assert _QA_PROMPT == _UPSTREAM_QA_PROMPT


def test_qa_prompt_cat5_matches_upstream():
    assert _QA_PROMPT_CAT_5 == _UPSTREAM_QA_PROMPT_CAT_5


def test_cat2_date_hint_matches_upstream():
    assert _CAT2_DATE_HINT == _UPSTREAM_CAT2_HINT


def test_cat5_mcq_template_matches_upstream():
    assert _CAT5_MCQ_TEMPLATE == _UPSTREAM_CAT5_TEMPLATE


def test_cat5_abstention_option_matches_upstream():
    assert _CAT5_ABSTENTION_OPTION == _UPSTREAM_ABSTENTION_OPTION


def test_build_qa_prompt_selects_cat5_template():
    assert _build_qa_prompt(5, "Q?") == _UPSTREAM_QA_PROMPT_CAT_5.format("Q?")


@pytest.mark.parametrize("category", [1, 2, 3, 4])
def test_build_qa_prompt_uses_generic_template_for_non_cat5(category):
    assert _build_qa_prompt(category, "Q?") == _UPSTREAM_QA_PROMPT.format("Q?")


# ---------------------------------------------------------------------------
# Normalization — evaluation.py::normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_strips_commas():
    assert _normalize_answer("cat, dog") == "cat dog"


def test_normalize_lowercases():
    assert _normalize_answer("HELLO World") == "hello world"


def test_normalize_strips_punctuation():
    assert _normalize_answer("hello, world!") == "hello world"


def test_normalize_strips_articles_a_an_the():
    # Upstream strips a|an|the — matches SQuAD.
    assert _normalize_answer("a cat and the dog") == "cat dog"


def test_normalize_strips_and_as_article():
    # LoCoMo upstream *also* treats 'and' as an article — non-standard
    # vs SQuAD, but we preserve the behavior.
    assert _normalize_answer("salt and pepper") == "salt pepper"


def test_normalize_preserves_word_boundaries_for_short_articles():
    # "a" inside "apple" must NOT be stripped — word boundary required.
    assert _normalize_answer("an apple") == "apple"
    assert _normalize_answer("the android") == "android"
    assert _normalize_answer("band") == "band"


def test_normalize_collapses_whitespace():
    assert _normalize_answer("  hello   world  ") == "hello world"


# ---------------------------------------------------------------------------
# F1 — evaluation.py::f1_score
# ---------------------------------------------------------------------------


def test_f1_exact_match_is_1():
    assert _f1_score("hello world", "hello world") == pytest.approx(1.0)


def test_f1_no_overlap_is_0():
    assert _f1_score("apple", "banana") == 0.0


def test_f1_empty_prediction_is_0():
    assert _f1_score("", "anything") == 0.0


def test_f1_empty_ground_truth_is_0():
    assert _f1_score("anything", "") == 0.0


def test_f1_ignores_commas_in_singletoken_match():
    # Commas stripped during normalization -> exact match.
    assert _f1_score("cat, dog", "cat dog") == pytest.approx(1.0)


def test_f1_applies_porter_stemmer():
    # "running" and "run" share the same Porter stem.
    assert _f1_score("running", "run") == pytest.approx(1.0)
    assert _f1_score("gardening", "gardens") == pytest.approx(1.0)


def test_f1_partial_overlap_matches_hand_calculation():
    # pred tokens (after normalize+stem): ["cat"]     (articles stripped)
    # gt tokens  (after normalize+stem): ["cat","dog"]
    # num_same=1, precision=1/1=1.0, recall=1/2=0.5
    # F1 = 2*1.0*0.5/(1.0+0.5) = 0.6666...
    assert _f1_score("the cat", "cat and dog") == pytest.approx(2 / 3)


def test_f1_returns_float_not_int():
    # evaluation.py::f1_score returns int ``0`` on no-overlap; our wrapper
    # returns ``0.0`` as a float everywhere for consistent serialization.
    out = _f1_score("apple", "banana")
    assert isinstance(out, float)


# ---------------------------------------------------------------------------
# Multi-answer F1 — evaluation.py::f1 (for category 1)
# ---------------------------------------------------------------------------


def test_f1_multi_exact_match_is_1():
    assert _f1_multi("cat, dog", "cat, dog") == pytest.approx(1.0)


def test_f1_multi_partial_answer_halved():
    # pred=["cat"], gt=["cat","dog"]:
    # per_gt = [max(F1("cat","cat"))=1.0, max(F1("cat","dog"))=0.0]
    # mean = 0.5
    assert _f1_multi("cat", "cat, dog") == pytest.approx(0.5)


def test_f1_multi_reordered_predictions_still_match():
    # Upstream takes the max over pred fragments for each gt fragment -
    # so order doesn't matter.
    assert _f1_multi("dog, cat", "cat, dog") == pytest.approx(1.0)


def test_f1_multi_empty_prediction_is_0():
    # split("") -> [""], and F1("", ...) = 0.
    assert _f1_multi("", "cat, dog") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Adversarial (cat 5) — evaluation.py::eval_question_answering l.217-221
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "no information available",
    "No Information Available",
    "There is no information available in the conversation.",
    "not mentioned",
    "The information is not mentioned",
    "NOT MENTIONED in conversation",
])
def test_adversarial_correct_detects_abstention_phrases(text):
    assert _adversarial_correct(text) is True


@pytest.mark.parametrize("text", [
    "Mount Everest",
    "I don't know",  # upstream does NOT accept "I don't know"
    "unknown",
    "",
    "some random response",
])
def test_adversarial_rejects_non_abstention(text):
    assert _adversarial_correct(text) is False


# ---------------------------------------------------------------------------
# _score_qa dispatch — each category goes to the right metric
# ---------------------------------------------------------------------------


def test_score_qa_cat1_uses_multi_answer_f1():
    # pred matches one of two gts -> 0.5 under multi-answer F1
    assert _score_qa(1, "cat", "cat, dog") == pytest.approx(0.5)


def test_score_qa_cat2_uses_single_f1():
    # Single-phrase F1 — 100% overlap after normalization & stemming.
    assert _score_qa(2, "May 2023", "may 2023") == pytest.approx(1.0)


def test_score_qa_cat3_splits_on_semicolon_before_scoring():
    # Upstream: ``answer.split(';')[0].strip()`` before grading.
    # pred "likely no" should FULLY match the pre-split ground truth.
    golden = "Likely no; though she likes reading"
    assert _score_qa(3, "likely no", golden) == pytest.approx(1.0)


def test_score_qa_cat3_only_first_semicolon_fragment_scored():
    # Prediction matching ONLY the after-semicolon part must NOT get credit
    # because upstream discards it. We pick a prediction whose Porter stems
    # are disjoint from "likely no" -> {"like", "no"}.
    golden = "likely no; though she enjoys reading books"
    assert _score_qa(3, "she enjoys reading books", golden) == pytest.approx(0.0)


def test_score_qa_cat4_uses_single_f1():
    assert _score_qa(4, "Appalachian Trail", "Appalachian Trail") == pytest.approx(1.0)


def test_score_qa_cat5_abstention_scores_1():
    assert _score_qa(5, "no information available", "Mount Everest") == 1.0


def test_score_qa_cat5_non_abstention_scores_0():
    assert _score_qa(5, "Mount Everest", "Mount Everest") == 0.0


def test_score_qa_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown LoCoMo category"):
        _score_qa(99, "x", "y")


# ---------------------------------------------------------------------------
# _get_cat_5_answer — upstream gpt_utils.py::get_cat_5_answer
# ---------------------------------------------------------------------------


@pytest.fixture
def cat5_key() -> dict[str, str]:
    """A sample cat-5 answer_key; shape matches what ``_cat5_mcq`` produces."""
    return {"a": _CAT5_ABSTENTION_OPTION, "b": "Mount Everest"}


@pytest.mark.parametrize("raw, expected_key", [
    ("a", "a"),
    ("A", "a"),   # case-insensitive — upstream ``strip().lower()``
    (" a ", "a"),  # whitespace-stripped
    ("b", "b"),
    ("B", "b"),
    # Anything length 1 that isn't 'a' maps to 'b' — upstream does ``'a' in
    # model_prediction`` which is False for any non-'a' single char.
    ("x", "b"),
])
def test_get_cat_5_answer_length_1_resolves_to_option(raw, expected_key, cat5_key):
    assert _get_cat_5_answer(raw, cat5_key) == cat5_key[expected_key]


@pytest.mark.parametrize("raw, expected_key", [
    ("(a)", "a"),
    ("(A)", "a"),   # case-insensitive
    (" (a) ", "a"),  # whitespace-stripped to length 3
    ("(b)", "b"),
    ("(B)", "b"),
    # Any other 3-char string maps to 'b' — upstream does ``'(a)' in ...``
    # which only matches the literal ``(a)`` substring.
    ("abc", "b"),
])
def test_get_cat_5_answer_length_3_resolves_to_option(raw, expected_key, cat5_key):
    assert _get_cat_5_answer(raw, cat5_key) == cat5_key[expected_key]


@pytest.mark.parametrize("raw", [
    "not mentioned in the conversation",
    "Mount Everest",
    "(a) Not mentioned",  # length > 3 -> passthrough
    "a)",  # length 2 -> passthrough (neither branch fires)
    "ab",  # length 2 -> passthrough
    "",   # length 0 -> passthrough
])
def test_get_cat_5_answer_otherwise_passes_through(raw, cat5_key):
    # For any input NOT of length 1 or 3, upstream returns the raw (but
    # lowercased + stripped) prediction unchanged.
    assert _get_cat_5_answer(raw, cat5_key) == raw.strip().lower()


def test_cat5_answer_key_matches_cat5_mcq_for_same_seed():
    # _cat5_answer_key must produce the SAME mapping that _cat5_mcq embedded
    # at prompt time — the judge relies on this to recover answer_key without
    # re-running prompt assembly.
    _, mcq_key = _cat5_mcq("Q?", "ANSWER", seed="stable-seed")
    key = _cat5_answer_key("ANSWER", seed="stable-seed")
    assert key == mcq_key


def test_cat5_answer_key_covers_both_options_across_seeds():
    # Across enough seeds, both {a: abstention, b: answer} and
    # {a: answer, b: abstention} orderings must appear — proves the helper
    # is actually seeded (not a fixed mapping).
    orderings: set[tuple[str, str]] = set()
    for i in range(30):
        key = _cat5_answer_key("ANSWER", seed=f"seed-{i}")
        orderings.add((key["a"], key["b"]))
    assert (_CAT5_ABSTENTION_OPTION, "ANSWER") in orderings
    assert ("ANSWER", _CAT5_ABSTENTION_OPTION) in orderings


# ---------------------------------------------------------------------------
# _evidence_recall — upstream evaluation.py::eval_question_answering l.228-237
# ---------------------------------------------------------------------------


def test_evidence_recall_no_retrieved_ids_returns_one():
    # Matches upstream ``all_recall.append(1)`` when no context key is set.
    assert _evidence_recall(None, ["D1:3"]) == 1.0


def test_evidence_recall_empty_retrieved_ids_returns_one():
    assert _evidence_recall([], ["D1:3"]) == 1.0


def test_evidence_recall_empty_evidence_returns_one():
    # Upstream: ``len(line['evidence']) > 0`` gates the computation.
    assert _evidence_recall(["D1:3"], []) == 1.0


def test_evidence_recall_dialog_mode_full_hit():
    # Retrieved IDs don't start with 'S' -> dialog-literal match mode.
    recall = _evidence_recall(["D1:1", "D1:3", "D2:2"], ["D1:1", "D2:2"])
    assert recall == pytest.approx(1.0)


def test_evidence_recall_dialog_mode_partial_hit():
    # 1 of 2 evidence items covered -> 0.5.
    recall = _evidence_recall(["D1:1", "D1:3"], ["D1:1", "D2:2"])
    assert recall == pytest.approx(0.5)


def test_evidence_recall_dialog_mode_no_hit():
    recall = _evidence_recall(["D9:9"], ["D1:1", "D2:2"])
    assert recall == pytest.approx(0.0)


def test_evidence_recall_session_mode_full_hit():
    # Retrieved IDs start with 'S' -> session match mode.
    # Upstream: strip leading 'S' from retrieved, strip leading 'D' from each
    # evidence's session prefix.
    # evidence D1:3 -> session "1"; D2:2 -> session "2". Both covered.
    recall = _evidence_recall(["S1", "S2", "S3"], ["D1:3", "D2:2"])
    assert recall == pytest.approx(1.0)


def test_evidence_recall_session_mode_partial_hit():
    # Only session 1 retrieved; evidence from sessions 1 and 2.
    recall = _evidence_recall(["S1"], ["D1:3", "D2:2"])
    assert recall == pytest.approx(0.5)


def test_evidence_recall_session_mode_no_hit():
    recall = _evidence_recall(["S9"], ["D1:3", "D2:2"])
    assert recall == pytest.approx(0.0)


def test_evidence_recall_mode_detection_uses_first_id():
    # Upstream: ``line[eval_key + '_context'][0].startswith('S')``
    # The FIRST id decides the mode for the whole list.
    recall = _evidence_recall(["S1", "D9:9"], ["D1:3"])
    # Session mode -> "1" in ["1", "9:9"[1:]="9:9"]? Not quite — the
    # ``e[1:]`` strips only the leading 'S' or 'D'. "D9:9"[1:] = "9:9".
    # evidence "D1:3".split(':')[0][1:] = "1". "1" not in ["1", "9:9"] ...
    # wait, "1" IS in that list. So recall = 1.0.
    assert recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _rewrite_question — per-category prompt munging
# ---------------------------------------------------------------------------


def test_rewrite_cat1_is_passthrough():
    q, key = _rewrite_question(1, "Who is Alice?", "Alice", task_id="t")
    assert q == "Who is Alice?"
    assert key is None


def test_rewrite_cat2_appends_date_hint():
    q, key = _rewrite_question(2, "When did Alice visit?", "May 2023", task_id="t")
    assert q == "When did Alice visit?" + _UPSTREAM_CAT2_HINT
    assert key is None


def test_rewrite_cat3_is_passthrough():
    q, key = _rewrite_question(3, "Is gardening outdoor?", "Yes", task_id="t")
    assert q == "Is gardening outdoor?"
    assert key is None


def test_rewrite_cat4_is_passthrough():
    q, key = _rewrite_question(4, "Where did Caroline go?", "Trail", task_id="t")
    assert q == "Where did Caroline go?"
    assert key is None


def test_rewrite_cat5_is_mcq_with_both_options():
    q, key = _rewrite_question(5, "What mountain?", "Everest", task_id="sample_0_q4")
    assert q.startswith("What mountain?")
    assert "(a)" in q and "(b)" in q
    assert _UPSTREAM_ABSTENTION_OPTION in q
    assert "Everest" in q
    # Key maps {'a':..., 'b':...} and always covers both options.
    assert key is not None
    assert set(key.keys()) == {"a", "b"}
    assert _UPSTREAM_ABSTENTION_OPTION in key.values()
    assert "Everest" in key.values()


def test_rewrite_cat5_is_deterministic_for_same_seed():
    q1, k1 = _rewrite_question(5, "Q?", "A", task_id="fixed-seed")
    q2, k2 = _rewrite_question(5, "Q?", "A", task_id="fixed-seed")
    assert q1 == q2
    assert k1 == k2


def test_rewrite_cat5_varies_across_seeds():
    # Enough seeds that BOTH orderings are sampled — prove determinism is
    # a function of the seed, not a fixed choice.
    orderings: set[tuple[str, str]] = set()
    for i in range(30):
        _, key = _rewrite_question(5, "Q?", "A", task_id=f"seed-{i}")
        assert key is not None
        orderings.add((key["a"], key["b"]))
    # Both orderings (A-first vs abstention-first) must appear.
    assert (_UPSTREAM_ABSTENTION_OPTION, "A") in orderings
    assert ("A", _UPSTREAM_ABSTENTION_OPTION) in orderings


def test_cat5_mcq_returns_byte_exact_upstream_template():
    # The question suffix must match upstream's literal format string.
    q, key = _cat5_mcq("Q?", "ANSWER", seed="stable")
    a_opt = key["a"]
    b_opt = key["b"]
    expected_suffix = _UPSTREAM_CAT5_TEMPLATE.format(a_opt, b_opt)
    assert q == "Q?" + expected_suffix


# ---------------------------------------------------------------------------
# _format_turn — gpt_utils.py::get_input_context turn rendering
# ---------------------------------------------------------------------------


def test_format_turn_basic():
    out = _format_turn({"speaker": "Alice", "text": "hello"})
    # Upstream: ``speaker + ' said, "' + text + '"\n' + '\n'`` when no caption.
    assert out == 'Alice said, "hello"\n\n'


def test_format_turn_with_blip_caption():
    out = _format_turn({
        "speaker": "Alice",
        "text": "look",
        "blip_caption": "a photo of a cat",
    })
    # Upstream appends ``' and shared %s.' % blip_caption`` before the final \n.
    assert out == 'Alice said, "look"\n and shared a photo of a cat.\n'


def test_format_turn_handles_missing_speaker_text():
    # Defensive: upstream would KeyError, we default to "".
    out = _format_turn({})
    assert out == ' said, ""\n\n'


# ---------------------------------------------------------------------------
# _session_keys_sorted / _speaker_pair
# ---------------------------------------------------------------------------


def test_session_keys_sorted_ignores_date_time_keys():
    conv = {
        "session_1": [],
        "session_1_date_time": "...",
        "session_10": [],
        "session_2": [],
        "speaker_a": "X",
    }
    assert _session_keys_sorted(conv) == [1, 2, 10]


def test_speaker_pair_prefers_explicit_keys():
    conv = {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1": [{"speaker": "Bob", "text": "!"}],
    }
    assert _speaker_pair(conv) == ("Caroline", "Melanie")


def test_speaker_pair_falls_back_to_session_1():
    conv = {
        "session_1": [
            {"speaker": "Caroline", "text": "hi"},
            {"speaker": "Melanie", "text": "hi"},
            {"speaker": "Caroline", "text": "bye"},
        ],
    }
    # First two distinct speakers, ordered by first appearance — stable
    # (upstream uses set iteration which is nondeterministic).
    assert _speaker_pair(conv) == ("Caroline", "Melanie")


def test_speaker_pair_with_no_sessions():
    assert _speaker_pair({}) == ("", "")


# ---------------------------------------------------------------------------
# _build_conversation_context — prepend + reverse-chronological layout
# ---------------------------------------------------------------------------


def test_build_context_empty_conversation():
    assert _build_conversation_context({}) == ""


def test_build_context_renders_sessions_in_reverse_chronological_order():
    # Session 2 is chronologically LATER, but upstream's prepend strategy
    # causes it to appear FIRST in the byte output.
    conv = _sample()["conversation"]
    out = _build_conversation_context(conv)

    s1_date = "DATE: 1:00 pm on 1 May 2023"
    s2_date = "DATE: 3:00 pm on 8 May 2023"
    assert s1_date in out
    assert s2_date in out
    # Session 2's block comes FIRST — reverse chronological.
    assert out.index(s2_date) < out.index(s1_date)


def test_build_context_preserves_turn_order_within_session():
    conv = _sample()["conversation"]
    out = _build_conversation_context(conv)
    # Within session_1, the speaker order in the rendered byte layout
    # follows the ORIGINAL (ascending) turn order:
    # D1:1 (Caroline) -> D1:2 (Melanie) -> D1:3 (Caroline).
    t1 = out.index('Caroline said, "I went hiking on Saturday."')
    t2 = out.index('Melanie said, "Where did you go?"')
    t3 = out.index('Caroline said, "The Appalachian Trail."')
    assert t1 < t2 < t3


def test_build_context_includes_blip_caption_rider():
    conv = _sample()["conversation"]
    out = _build_conversation_context(conv)
    assert "and shared a photo of a forest trail." in out


# ---------------------------------------------------------------------------
# load_tasks — yields one Task per QA with conversation_id == sample_id
# ---------------------------------------------------------------------------


def test_load_tasks_yields_one_task_per_qa(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({"data_path": str(data_dir)}))
    # 2 samples * 5 QA = 10 tasks.
    assert len(tasks) == 10


def test_load_tasks_preserves_upstream_fields(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({"data_path": str(data_dir)}))
    first = tasks[0]

    assert first.task_id == "sample_0_q0"
    assert first.benchmark == "locomo_original"
    assert first.granularity == TaskGranularity.CONVERSATIONAL
    assert first.prompt == "What outdoor activities does Caroline enjoy?"
    assert first.ground_truth == "hiking, gardening"
    # conversation_id == sample_id so memory is shared across a sample's QAs.
    assert first.metadata["conversation_id"] == "sample_0"
    assert first.metadata["sample_id"] == "sample_0"
    assert first.metadata["qa_index"] == 0
    assert first.metadata["category"] == 1
    assert first.metadata["category_name"] == "multi_hop"
    assert first.metadata["evidence"] == ["D1:1", "D2:2"]
    assert first.metadata["speaker_a"] == "Caroline"
    assert first.metadata["speaker_b"] == "Melanie"
    # Full conversation rides on each task so ``format_prompt`` can
    # reconstruct the full-history prompt.
    assert "session_1" in first.metadata["conversation"]


def test_load_tasks_category_names_set_for_each_task(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({"data_path": str(data_dir)}))
    names = [t.metadata["category_name"] for t in tasks[:5]]
    # Fixture has one question per category in order 1..5.
    assert names == ["multi_hop", "temporal", "commonsense", "single_hop", "adversarial"]


def test_load_tasks_filter_categories_excludes_matching(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({
        "data_path": str(data_dir),
        "filter_categories": [5],  # drop adversarial
    }))
    assert all(t.metadata["category"] != 5 for t in tasks)
    # Original was 10; fixture has 2 adversarial (one per sample).
    assert len(tasks) == 8


def test_load_tasks_respects_limit(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({
        "data_path": str(data_dir),
        "limit": 3,
    }))
    assert len(tasks) == 3


def test_load_tasks_prompt_style_flows_into_metadata(data_dir: Path):
    tasks = list(LoCoMoOriginalAdapter().load_tasks({
        "data_path": str(data_dir),
        "prompt_style": "full_history",
    }))
    assert all(t.metadata["prompt_style"] == "full_history" for t in tasks)


def test_load_tasks_coerces_int_answer_to_string(tmp_path: Path):
    # Upstream sometimes stores ``answer`` as int (e.g. ``2022``) — we must
    # stringify it so downstream F1 can tokenize.
    sample = _sample(qa_items=[{
        "question": "What year?",
        "answer": 2022,
        "category": 2,
        "evidence": ["D1:1"],
    }])
    d = tmp_path / "locomo"
    d.mkdir()
    (d / "locomo10.json").write_text(json.dumps([sample]), encoding="utf-8")

    tasks = list(LoCoMoOriginalAdapter().load_tasks({"data_path": str(d)}))
    assert tasks[0].ground_truth == "2022"


# ---------------------------------------------------------------------------
# load_conversations — flatten sessions in ascending chronological order
# ---------------------------------------------------------------------------


def test_load_conversations_one_per_sample(data_dir: Path):
    convs = LoCoMoOriginalAdapter().load_conversations({"data_path": str(data_dir)})
    assert len(convs) == 2
    assert {c["conversation_id"] for c in convs} == {"sample_0", "sample_1"}


def test_load_conversations_sessions_ascending_turns_in_order(data_dir: Path):
    convs = LoCoMoOriginalAdapter().load_conversations({"data_path": str(data_dir)})
    conv = convs[0]
    # 3 turns in session_1 + 2 in session_2 = 5 messages.
    assert len(conv["messages"]) == 5
    # Ascending session order — session_1 first, session_2 after.
    assert [m["session_index"] for m in conv["messages"]] == [1, 1, 1, 2, 2]
    # Turn index is zero-based within each session.
    assert [m["turn_index"] for m in conv["messages"]] == [0, 1, 2, 0, 1]


def test_load_conversations_preserves_dia_id_and_image_fields(data_dir: Path):
    convs = LoCoMoOriginalAdapter().load_conversations({"data_path": str(data_dir)})
    msg = convs[0]["messages"][2]  # the image turn (D1:3)
    assert msg["dia_id"] == "D1:3"
    assert msg["img_url"] == "http://example/hike.jpg"
    assert msg["blip_caption"] == "a photo of a forest trail"
    assert msg["img_query"] == "appalachian trail"


def test_load_conversations_speaker_flows_to_both_id_and_name(data_dir: Path):
    convs = LoCoMoOriginalAdapter().load_conversations({"data_path": str(data_dir)})
    msg = convs[0]["messages"][0]
    # Upstream uses the literal speaker name as identity — we mirror that.
    assert msg["speaker_id"] == "Caroline"
    assert msg["speaker_name"] == "Caroline"
    assert msg["timestamp"] == "1:00 pm on 1 May 2023"


def test_load_conversations_metadata_carries_sample_shape(data_dir: Path):
    convs = LoCoMoOriginalAdapter().load_conversations({"data_path": str(data_dir)})
    meta = convs[0]["metadata"]
    assert meta["sample_id"] == "sample_0"
    assert meta["speaker_a"] == "Caroline"
    assert meta["speaker_b"] == "Melanie"
    assert meta["num_sessions"] == 2


# ---------------------------------------------------------------------------
# format_prompt — question vs full_history modes
# ---------------------------------------------------------------------------


def _task_from_qa(
    qa: dict[str, Any],
    sample_id: str = "sample_0",
    qa_index: int = 0,
    prompt_style: str = "question",
    conversation: dict[str, Any] | None = None,
) -> Task:
    cat = int(qa["category"])
    return Task(
        task_id=f"{sample_id}_q{qa_index}",
        benchmark="locomo_original",
        granularity=TaskGranularity.CONVERSATIONAL,
        prompt=qa["question"],
        metadata={
            "conversation_id": sample_id,
            "sample_id": sample_id,
            "qa_index": qa_index,
            "category": cat,
            "category_name": CATEGORY_NAMES.get(cat, "unknown"),
            "evidence": qa.get("evidence", []),
            "prompt_style": prompt_style,
            "conversation": conversation or _sample()["conversation"],
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
        },
        ground_truth=str(qa["answer"]),
    )


def test_format_prompt_question_mode_cat1_is_verbatim():
    qa = _sample()["qa"][0]
    t = _task_from_qa(qa, qa_index=0)
    assert LoCoMoOriginalAdapter().format_prompt(t) == qa["question"]


def test_format_prompt_question_mode_cat2_appends_date_hint():
    qa = _sample()["qa"][1]
    t = _task_from_qa(qa, qa_index=1)
    out = LoCoMoOriginalAdapter().format_prompt(t)
    assert out == qa["question"] + _UPSTREAM_CAT2_HINT


def test_format_prompt_question_mode_cat5_is_mcq():
    qa = _sample()["qa"][4]
    t = _task_from_qa(qa, qa_index=4)
    out = LoCoMoOriginalAdapter().format_prompt(t)
    assert out.startswith(qa["question"])
    assert "(a)" in out and "(b)" in out
    assert _UPSTREAM_ABSTENTION_OPTION in out
    assert "Mount Everest" in out


def test_format_prompt_full_history_byte_exact_assembly():
    qa = _sample()["qa"][3]  # cat 4, no question rewrites
    t = _task_from_qa(qa, qa_index=3, prompt_style="full_history")
    out = LoCoMoOriginalAdapter().format_prompt(t)

    # Upstream: start_prompt + query_conv + '\n\n' + QA_PROMPT.format(question)
    start = _UPSTREAM_CONV_START.format("Caroline", "Melanie")
    conv_ctx = _build_conversation_context(_sample()["conversation"])
    qa_block = _UPSTREAM_QA_PROMPT.format(qa["question"])
    assert out == start + conv_ctx + "\n\n" + qa_block


def test_format_prompt_full_history_cat2_injects_date_hint_into_qa_block():
    qa = _sample()["qa"][1]  # cat 2
    t = _task_from_qa(qa, qa_index=1, prompt_style="full_history")
    out = LoCoMoOriginalAdapter().format_prompt(t)
    # The cat-2 hint rides INSIDE the QA_PROMPT slot (same as upstream).
    assert _UPSTREAM_CAT2_HINT in out
    # Must still use the generic QA_PROMPT (not cat-5 template).
    assert "Answer with exact words from the context" in out


def test_format_prompt_full_history_cat5_uses_cat5_template():
    qa = _sample()["qa"][4]  # cat 5
    t = _task_from_qa(qa, qa_index=4, prompt_style="full_history")
    out = LoCoMoOriginalAdapter().format_prompt(t)
    # Cat-5 uses QA_PROMPT_CAT_5 (no "exact words from the context" clause).
    assert "Answer with exact words from the context" not in out
    assert "Based on the above context, answer the following question." in out


def test_format_prompt_unknown_style_raises():
    qa = _sample()["qa"][0]
    t = _task_from_qa(qa, prompt_style="not-a-style")
    with pytest.raises(NotImplementedError):
        LoCoMoOriginalAdapter().format_prompt(t)


# ---------------------------------------------------------------------------
# judge_answer — per-category F1 / substring check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_cat1_multi_answer_f1_perfect_passes():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # cat 1, "hiking, gardening"
    out = await a.judge_answer(t, "hiking, gardening")
    assert out["pass"] is True
    assert out["correct"] == 1
    assert out["f1"] == pytest.approx(1.0)
    assert out["category"] == 1
    assert out["category_name"] == "multi_hop"
    assert out["is_adversarial"] is False


@pytest.mark.asyncio
async def test_judge_cat1_single_answer_in_multi_gt_half_score():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    out = await a.judge_answer(t, "hiking")
    # F1_multi: one of two ground truths covered -> 0.5 exactly.
    assert out["f1"] == pytest.approx(0.5)
    # Default threshold is 0.5 -> >= 0.5 -> pass=True.
    assert out["pass"] is True


@pytest.mark.asyncio
async def test_judge_cat2_with_date_hint_prompt_still_scored_by_f1():
    # The date hint is a prompt-time transformation; judging only looks at
    # the answer vs ground truth with single-phrase F1.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][1], qa_index=1)  # cat 2, "May 2023"
    out = await a.judge_answer(t, "May 2023")
    assert out["f1"] == pytest.approx(1.0)
    assert out["pass"] is True


@pytest.mark.asyncio
async def test_judge_cat3_splits_on_semicolon_before_scoring():
    # Ground truth has a semicolon — only first fragment is graded.
    # Fixture qa[2].answer = "Likely yes; gardening is usually done outdoors";
    # first fragment stems to ["like", "ye"].
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][2], qa_index=2)  # cat 3
    # Prediction has zero stem overlap with "Likely yes" -> F1 = 0.
    out = await a.judge_answer(t, "gardening is outdoors")
    assert out["f1"] == pytest.approx(0.0)
    assert out["pass"] is False


@pytest.mark.asyncio
async def test_judge_cat4_single_phrase_f1():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][3], qa_index=3)  # cat 4, "Appalachian Trail"
    out = await a.judge_answer(t, "Appalachian Trail")
    assert out["f1"] == pytest.approx(1.0)
    assert out["pass"] is True


@pytest.mark.asyncio
async def test_judge_cat5_abstention_scores_perfect_pass():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5
    out = await a.judge_answer(t, "no information available in conversation")
    # Cat-5 uses substring check -> score is exactly 1.0.
    assert out["f1"] == pytest.approx(1.0)
    assert out["pass"] is True
    assert out["is_adversarial"] is True


@pytest.mark.asyncio
async def test_judge_cat5_false_positive_scores_zero():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)
    out = await a.judge_answer(t, "Mount Everest")
    # Cat-5: even matching the (bogus) gold answer gets 0 — upstream requires
    # abstention, not guessing.
    assert out["f1"] == pytest.approx(0.0)
    assert out["pass"] is False


@pytest.mark.asyncio
async def test_judge_cat5_pass_requires_score_equal_to_one():
    # Cat-5 branch ignores the f1_threshold — must be 1.0 exactly (the
    # substring check is binary, so any 0 < score < 1 is impossible, but the
    # threshold override path is worth locking).
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.1}  # low threshold should not help
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)
    out = await a.judge_answer(t, "I don't know")  # not an abstention phrase
    assert out["pass"] is False


@pytest.mark.asyncio
async def test_judge_surfaces_f1_threshold_from_config():
    a = LoCoMoOriginalAdapter()
    # With a higher threshold, 0.5 F1 should NOT pass.
    a._judge_config = {"f1_threshold": 0.75}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # cat 1
    out = await a.judge_answer(t, "hiking")  # F1_multi = 0.5
    assert out["f1"] == pytest.approx(0.5)
    assert out["pass"] is False
    assert out["f1_threshold"] == 0.75


# ---------------------------------------------------------------------------
# LLM-as-judge metric — opt-in via ``judge.llm_judge.enabled``
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_skips_llm_judge_when_disabled():
    """Default LoCoMo run shouldn't pay for judge calls."""
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}  # no llm_judge block
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    out = await a.judge_answer(t, "hiking")
    assert "llm_judge_correct" not in out
    assert "llm_judge_latency_ms" not in out


@pytest.mark.asyncio
async def test_judge_runs_llm_judge_and_surfaces_metrics(monkeypatch):
    """When ``llm_judge.enabled = true``, judge_answer adds the J score
    AND a per-call latency, without changing the F1 contract."""
    captured: dict[str, object] = {}

    async def fake_judge(*, question, golden_answer, generated_answer,
                         llm_config, num_runs):
        captured.update(
            question=question,
            golden=golden_answer,
            generated=generated_answer,
            num_runs=num_runs,
        )
        return {
            "pass": True,
            "correct": 1,
            "num_correct": 3,
            "num_runs": 3,
            "raw_judgments": ["CORRECT", "CORRECT", "CORRECT"],
        }

    monkeypatch.setattr(
        "cross_bench.benchmarks.llm_judge.llm_judge", fake_judge
    )

    a = LoCoMoOriginalAdapter()
    a._judge_config = {
        "f1_threshold": 0.5,
        "llm_judge": {"enabled": True, "num_runs": 3, "model": "gpt-4o-mini"},
    }
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # cat 1
    out = await a.judge_answer(t, "hiking")  # partial F1, NOT cleared

    # J score appended
    assert out["llm_judge_correct"] == 1
    assert out["llm_judge_num_correct"] == 3
    assert out["llm_judge_num_runs"] == 3
    assert isinstance(out["llm_judge_latency_ms"], (int, float))
    assert out["llm_judge_latency_ms"] >= 0.0

    # F1 still primary by default — pass should track the F1 verdict
    # (F1=0.5 below the 0.5 threshold gate uses ``>=``? Let's verify
    # the actual contract: pass is True iff f1 >= threshold). Either
    # way, ``f1_pass`` reports the F1 verdict regardless of judge.
    assert "f1_pass" in out
    # The mock was called with what the adapter saw.
    assert captured["num_runs"] == 3


@pytest.mark.asyncio
async def test_judge_primary_metric_llm_judge_overrides_pass(monkeypatch):
    """``primary_metric = 'llm_judge'`` flips ``pass`` to track the J
    verdict, while F1 numbers remain in the result for analysis."""
    async def fake_judge(**kwargs):
        return {
            "pass": False,
            "correct": 0,
            "num_correct": 0,
            "num_runs": 1,
            "raw_judgments": ["WRONG"],
        }

    monkeypatch.setattr(
        "cross_bench.benchmarks.llm_judge.llm_judge", fake_judge
    )

    a = LoCoMoOriginalAdapter()
    a._judge_config = {
        "f1_threshold": 0.5,
        "primary_metric": "llm_judge",
        "llm_judge": {"enabled": True, "num_runs": 1},
    }
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    # Echo the gold so F1 = 1.0 — but the (mocked) judge says WRONG.
    gold = str(t.ground_truth)
    out = await a.judge_answer(t, gold)
    assert out["f1_pass"] == 1            # F1 still passed
    assert out["llm_judge_correct"] == 0  # judge said wrong
    assert out["pass"] is False           # primary = llm_judge wins
    assert out["correct"] == 0


@pytest.mark.asyncio
async def test_judge_skips_llm_judge_for_cat5(monkeypatch):
    """Cat-5 (adversarial) is excluded from the LLM judge per upstream
    Mem0 v2's ``evaluate_llm_judge`` (``if int(category) == 5: continue``).
    LoCoMo cat-5 gold is ``answer=None`` → empty string, so the judge
    would compare "" against "no information available" and label WRONG
    on every correctly-abstaining task — carrying no honest signal.
    """
    calls: list[dict[str, object]] = []

    async def fake_judge(**kwargs):
        calls.append(dict(kwargs))
        return {
            "pass": False,
            "correct": 0,
            "num_correct": 0,
            "num_runs": 1,
            "raw_judgments": ["WRONG"],
        }

    monkeypatch.setattr(
        "cross_bench.benchmarks.llm_judge.llm_judge", fake_judge
    )

    a = LoCoMoOriginalAdapter()
    a._judge_config = {
        "f1_threshold": 0.5,
        "llm_judge": {"enabled": True, "num_runs": 1},
    }
    # qa[4] is cat-5 in the test fixture
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)
    out = await a.judge_answer(t, "no information available")

    # Judge backend must NOT have been called for cat-5
    assert calls == [], f"LLM judge unexpectedly called for cat-5: {calls!r}"

    # No llm_judge_* keys leak into the result — aggregator excludes
    # them from the headline ``pass_rate_llm_judge`` average via the
    # ``isinstance(..., (int, float))`` filter on missing keys.
    assert "llm_judge_correct" not in out
    assert "llm_judge_num_correct" not in out
    assert "llm_judge_num_runs" not in out
    assert "llm_judge_latency_ms" not in out
    assert "llm_judge_error" not in out

    # F1 path is untouched — abstention substring still drives cat-5.
    assert out["category"] == 5
    assert out["f1"] == pytest.approx(1.0)
    assert out["f1_pass"] == 1


@pytest.mark.asyncio
async def test_judge_runs_llm_judge_for_non_cat5_categories(monkeypatch):
    """Sanity check the other side of the cat-5 skip — cats 1-4 still
    invoke the judge so the skip doesn't accidentally disable J score
    for the categories where it carries signal."""
    call_count = {"n": 0}

    async def fake_judge(**kwargs):
        call_count["n"] += 1
        return {
            "pass": True,
            "correct": 1,
            "num_correct": 1,
            "num_runs": 1,
            "raw_judgments": ["CORRECT"],
        }

    monkeypatch.setattr(
        "cross_bench.benchmarks.llm_judge.llm_judge", fake_judge
    )

    a = LoCoMoOriginalAdapter()
    a._judge_config = {
        "f1_threshold": 0.5,
        "llm_judge": {"enabled": True, "num_runs": 1},
    }
    for qa_idx in range(4):  # cats 1..4
        t = _task_from_qa(_sample()["qa"][qa_idx], qa_index=qa_idx)
        out = await a.judge_answer(t, str(t.ground_truth))
        assert "llm_judge_correct" in out, f"cat={out['category']} missing key"

    assert call_count["n"] == 4


@pytest.mark.asyncio
async def test_judge_records_latency_even_on_judge_error(monkeypatch):
    """If the judge backend blows up, we still record how long the call
    took before the error so total_latency_ms is honest."""
    async def boom(**kwargs):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(
        "cross_bench.benchmarks.llm_judge.llm_judge", boom
    )

    a = LoCoMoOriginalAdapter()
    a._judge_config = {
        "f1_threshold": 0.5,
        "llm_judge": {"enabled": True, "num_runs": 1},
    }
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    out = await a.judge_answer(t, "hiking")
    assert "llm_judge_error" in out
    assert "llm_judge_latency_ms" in out
    assert out["llm_judge_latency_ms"] >= 0.0
    # Primary metric (F1) still drives ``pass``; judge error doesn't
    # crash the per-task evaluation.
    assert "pass" in out


# ---------------------------------------------------------------------------
# judge_answer — cat-5 terse-reply resolution (`get_cat_5_answer` port)
# ---------------------------------------------------------------------------


def _abstention_letter(golden: str, task_id: str) -> str:
    """Return which letter ('a' or 'b') maps to the abstention option for
    the given seed. Used by the terse-reply judge tests so we can feed the
    correct letter regardless of the MCQ's randomized ordering."""
    key = _cat5_answer_key(golden, seed=task_id)
    return "a" if key["a"] == _CAT5_ABSTENTION_OPTION else "b"


@pytest.mark.asyncio
async def test_judge_cat5_terse_letter_answer_resolves_to_abstention_and_passes():
    # Model replies with just the letter ("a" or "b") corresponding to the
    # abstention option. Upstream's get_cat_5_answer resolves this to the
    # full option text before the substring check -> pass. Without the
    # resolution fix we'd fail (score 0) for this terse reply.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5
    abstention = _abstention_letter(str(t.ground_truth), t.task_id)

    out = await a.judge_answer(t, abstention)  # e.g. "a" or "b"
    assert out["pass"] is True
    assert out["f1"] == pytest.approx(1.0)
    # resolved_prediction is surfaced when the raw prediction differs.
    assert out["resolved_prediction"] == _CAT5_ABSTENTION_OPTION


@pytest.mark.asyncio
async def test_judge_cat5_terse_letter_pointing_to_wrong_option_fails():
    # Opposite of the previous test: reply with the letter that maps to the
    # real answer (not the abstention). Should resolve to the answer text,
    # which doesn't contain "not mentioned" / "no information available",
    # so scores 0.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5
    abstention = _abstention_letter(str(t.ground_truth), t.task_id)
    wrong = "b" if abstention == "a" else "a"

    out = await a.judge_answer(t, wrong)
    assert out["pass"] is False
    assert out["f1"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_judge_cat5_parenthesized_letter_resolves_same_as_letter():
    # "(a)" / "(b)" (length 3) resolves via the same mechanism.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5
    abstention = _abstention_letter(str(t.ground_truth), t.task_id)

    out = await a.judge_answer(t, f"({abstention})")
    assert out["pass"] is True
    assert out["resolved_prediction"] == _CAT5_ABSTENTION_OPTION


@pytest.mark.asyncio
async def test_judge_cat5_full_text_reply_bypasses_resolution():
    # Full-text replies (length != 1 and != 3) pass through unchanged.
    # The raw text is substring-checked directly.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)
    out = await a.judge_answer(t, "No information available in this chat.")
    assert out["pass"] is True
    # resolved_prediction key is ONLY surfaced when resolution changed the
    # text — a full-text reply shouldn't trigger it.
    assert "resolved_prediction" not in out


@pytest.mark.asyncio
async def test_judge_cat5_resolution_is_deterministic_across_calls():
    # Calling the judge twice with the same task + same terse reply must
    # produce the same resolved_prediction (the answer_key is deterministic
    # from task_id).
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)
    abstention = _abstention_letter(str(t.ground_truth), t.task_id)

    out1 = await a.judge_answer(t, abstention)
    out2 = await a.judge_answer(t, abstention)
    assert out1["resolved_prediction"] == out2["resolved_prediction"]
    assert out1["pass"] == out2["pass"]


# ---------------------------------------------------------------------------
# judge_answer — evidence recall (upstream evaluation.py::eval_question_answering
# recall branch at lines 228-237)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_evidence_recall_defaults_to_one_without_retrieved_ids():
    # When the engine hasn't populated retrieved_context_ids, upstream's
    # ``all_recall.append(1)`` fallback applies -> recall = 1.0.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][3], qa_index=3)  # cat 4, evidence=[D1:3]
    out = await a.judge_answer(t, "Appalachian Trail")
    assert out["evidence_recall"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_judge_evidence_recall_dialog_mode_partial_hit():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    qa = _sample()["qa"][0]  # cat 1, evidence=[D1:1, D2:2]
    t = _task_from_qa(qa, qa_index=0)
    # Engine surfaces the retrieved dialog labels; only 1 of 2 evidence
    # items covered -> recall = 0.5.
    t.metadata["retrieved_context_ids"] = ["D1:1", "D1:3"]

    out = await a.judge_answer(t, "hiking, gardening")
    assert out["evidence_recall"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_judge_evidence_recall_session_mode():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    qa = _sample()["qa"][0]  # cat 1, evidence=[D1:1, D2:2]
    t = _task_from_qa(qa, qa_index=0)
    # Session-mode retrieval: IDs start with 'S'. Evidence spans sessions
    # 1 and 2; only session 1 retrieved -> recall = 0.5.
    t.metadata["retrieved_context_ids"] = ["S1"]

    out = await a.judge_answer(t, "hiking, gardening")
    assert out["evidence_recall"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_judge_evidence_recall_empty_evidence_is_one():
    # Cat 5 QAs have empty evidence -> recall defaults to 1.0 even if
    # retrieved_context_ids is populated (upstream gates on
    # ``len(line['evidence']) > 0``).
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5, evidence=[]
    t.metadata["retrieved_context_ids"] = ["D9:9"]

    out = await a.judge_answer(t, "no information available")
    assert out["evidence_recall"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_judge_evidence_recall_rounded_to_three_places():
    # Display formatting — upstream computes as float; we round to 3dp for
    # JSON output stability.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    qa = {
        "question": "Q?",
        "answer": "A",
        "category": 1,
        "evidence": ["D1:1", "D1:2", "D2:1"],  # 3 evidence items
    }
    t = _task_from_qa(qa, qa_index=0)
    t.metadata["retrieved_context_ids"] = ["D1:1"]  # 1 of 3 covered -> 1/3

    out = await a.judge_answer(t, "A")
    # 1/3 rounded to 3 decimals.
    assert out["evidence_recall"] == pytest.approx(round(1 / 3, 3))


# ---------------------------------------------------------------------------
# evidence_recall_measured flag — distinguishes genuine measurement (engine
# plumbed IDs, evidence non-empty) from the upstream-faithful 1.0 default.
# Downstream aggregators should filter on this flag to avoid mixing
# measured recall with unmeasured perfect scores.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_evidence_recall_measured_is_false_without_plumbed_ids():
    # Engine never populated retrieved_context_ids -> not measured.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # cat 1, non-empty evidence

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["evidence_recall_measured"] is False
    # Recall itself still reports the upstream-faithful 1.0 default.
    assert out["evidence_recall"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_judge_evidence_recall_measured_is_true_when_ids_and_evidence_present():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # evidence=[D1:1, D2:2]
    t.metadata["retrieved_context_ids"] = ["D1:1"]

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["evidence_recall_measured"] is True


@pytest.mark.asyncio
async def test_judge_evidence_recall_measured_is_false_with_empty_evidence():
    # Cat 5 has empty evidence -> recall is the default 1.0, but it is
    # NOT a measurement: reporting it as "measured" would inflate
    # cat-5-heavy aggregates. Flag reflects this.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][4], qa_index=4)  # cat 5, evidence=[]
    t.metadata["retrieved_context_ids"] = ["D9:9"]  # IDs present, but irrelevant

    out = await a.judge_answer(t, "no information available")

    assert out["evidence_recall_measured"] is False
    assert out["evidence_recall"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_judge_evidence_recall_measured_true_even_with_empty_retrieval():
    # Retriever ran and returned nothing -> measured=True, recall=0.0
    # (nothing retrieved ≠ nothing attempted). This is the crucial
    # distinction from the default 1.0 fallback.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)  # evidence=[D1:1, D2:2]
    t.metadata["retrieved_context_ids"] = []

    out = await a.judge_answer(t, "hiking, gardening")

    # Current _evidence_recall treats empty list as the 1.0 fallback for
    # upstream fidelity (upstream would IndexError on []), but the
    # ``measured`` flag still distinguishes this from the no-plumbing case.
    # NOTE: this is the "measured=True, recall=1.0 default" edge case —
    # aggregators looking only at ``evidence_recall_measured`` still know
    # the retriever was invoked.
    assert out["evidence_recall_measured"] is True


# ---------------------------------------------------------------------------
# retrieved_context_ids echo — surfaced in judge output for debugging and
# downstream aggregation; copy of task.metadata, not a pointer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_echoes_retrieved_context_ids():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    t.metadata["retrieved_context_ids"] = ["D1:1", "D2:2"]

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["retrieved_context_ids"] == ["D1:1", "D2:2"]


@pytest.mark.asyncio
async def test_judge_echoes_empty_list_when_not_plumbed():
    # Consistent with the ``measured`` flag: when no IDs are plumbed,
    # the echo is ``[]``, not a missing key. Makes the output schema
    # stable for downstream consumers.
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["retrieved_context_ids"] == []


@pytest.mark.asyncio
async def test_judge_echoes_empty_list_when_retriever_returned_nothing():
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    t.metadata["retrieved_context_ids"] = []

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["retrieved_context_ids"] == []


@pytest.mark.asyncio
async def test_judge_echoes_ids_in_input_order():
    # Contract: the middleware is responsible for ordering (rank order,
    # deduplicated). The adapter just surfaces them unchanged so
    # downstream tools can do rank-aware analysis (MRR, Recall@K).
    a = LoCoMoOriginalAdapter()
    a._judge_config = {"f1_threshold": 0.5}
    t = _task_from_qa(_sample()["qa"][0], qa_index=0)
    t.metadata["retrieved_context_ids"] = ["D5:5", "D1:1", "D3:3"]

    out = await a.judge_answer(t, "hiking, gardening")

    assert out["retrieved_context_ids"] == ["D5:5", "D1:1", "D3:3"]


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def test_load_dataset_explicit_file_path_wins_over_data_path(tmp_path: Path):
    # A custom JSON path trumps data_path lookup.
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps([_sample("sample_x")]), encoding="utf-8")
    tasks = list(LoCoMoOriginalAdapter().load_tasks({
        "file_path": str(custom),
        "data_path": str(tmp_path / "nope"),
    }))
    assert len(tasks) == 5  # 1 sample * 5 QAs
    assert all(t.metadata["sample_id"] == "sample_x" for t in tasks)


def test_load_dataset_missing_file_raises(tmp_path: Path, monkeypatch):
    # Force the download path to fail so we get a clean FileNotFoundError.
    # chdir so the well-known relative fallback paths (``data/locomo/``,
    # ``locomo/``, ``./locomo10.json``) don't accidentally hit a cached copy
    # from a previous run in the real project root.
    monkeypatch.chdir(tmp_path)

    import urllib.request

    def _boom(*_args, **_kwargs):
        raise OSError("simulated offline")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)

    with pytest.raises(FileNotFoundError, match="LoCoMo dataset not found"):
        list(LoCoMoOriginalAdapter().load_tasks({
            "data_path": str(tmp_path / "nope"),
        }))


def test_load_dataset_rejects_non_array_json(tmp_path: Path):
    bad = tmp_path / "locomo10.json"
    bad.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        list(LoCoMoOriginalAdapter().load_tasks({"file_path": str(bad)}))


# ---------------------------------------------------------------------------
# End-to-end dataflow — JSON -> load_tasks -> format_prompt -> judge_answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_self_consistency_per_category(data_dir: Path):
    """Feeding the ground truth back in as the answer should pass for
    non-adversarial categories; for cat 5 it must FAIL because the ground
    truth is the distractor, not the abstention phrase."""
    a = LoCoMoOriginalAdapter()
    tasks = list(a.load_tasks({"data_path": str(data_dir)}))

    # Map (sample_id, qa_index) -> task for deterministic probing.
    by_q = {(t.metadata["sample_id"], t.metadata["qa_index"]): t for t in tasks}

    # cat 1-4 self-consistency: exact ground truth -> F1 = 1.0 -> pass.
    for qa_idx in (0, 1, 2, 3):
        t = by_q[("sample_0", qa_idx)]
        # For cat 3 we must echo only the first semicolon fragment — that's
        # what upstream grades against.
        golden = str(t.ground_truth)
        if t.metadata["category"] == 3:
            golden = golden.split(";")[0].strip()
        result = await a.judge_answer(t, golden)
        assert result["pass"] is True, (qa_idx, result)
        assert math.isclose(result["f1"], 1.0, abs_tol=1e-6)

    # cat 5: echoing ground truth fails (must say "no information available"
    # or "not mentioned").
    t5 = by_q[("sample_0", 4)]
    wrong = await a.judge_answer(t5, str(t5.ground_truth))
    assert wrong["pass"] is False
    correct = await a.judge_answer(t5, "There is no information available")
    assert correct["pass"] is True


def test_end_to_end_prompt_contains_rewritten_question_and_context(data_dir: Path):
    a = LoCoMoOriginalAdapter()
    tasks = list(a.load_tasks({
        "data_path": str(data_dir),
        "prompt_style": "full_history",
    }))
    cat2_task = next(t for t in tasks if t.metadata["category"] == 2)
    prompt = a.format_prompt(cat2_task)

    # The prompt bundles all three stages of the dataflow:
    #   1. CONV_START_PROMPT with speaker names filled in.
    assert "Caroline and Melanie" in prompt
    #   2. Reverse-chronological conversation context.
    assert "DATE: 1:00 pm on 1 May 2023" in prompt
    assert "DATE: 3:00 pm on 8 May 2023" in prompt
    assert prompt.index("8 May 2023") < prompt.index("1 May 2023")
    #   3. QA prompt with the cat-2 date hint appended to the question.
    assert _UPSTREAM_CAT2_HINT in prompt
