# ruff: noqa: E501 - the strings below are byte-exact copies of upstream
# AnsweringEvaluation.scala templates; splitting them would defeat the
# regression check's purpose.
"""ConvoMem adapter tests — lock byte-exact fidelity to upstream.

The prompt strings below are copied verbatim from
``SalesforceAIResearch/ConvoMem/src/main/scala/com/salesforce/crmmembench/questions/evidence/generation/AnsweringEvaluation.scala``
so the regression assertions fail loudly if we ever drift.

Tests mock Google / OpenAI / Anthropic SDKs so the judge path exercises our
prompt construction and routing without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks.convomem import (
    CATEGORIES,
    DEFAULT_JUDGE_MAX_RETRIES,
    DEFAULT_JUDGE_MODEL,
    EVIDENCE_COUNT_RANGES,
    ConvoMemAdapter,
    _abstention_prompt,
    _build_judge_prompt,
    _default_prompt,
    _parse_judge_label,
    _rubric_prompt,
    _temporal_prompt,
    _user_facts_multi_prompt,
    _user_facts_single_prompt,
)
from cross_bench.models import Task, TaskGranularity

# ---------------------------------------------------------------------------
# Byte-exact upstream strings. Copied from AnsweringEvaluation.scala on main.
# ---------------------------------------------------------------------------

_UPSTREAM_DEFAULT_GUIDELINES = """**Crucial Guidelines for your judgment:**

1.  **Core Information is Key**: The Model's Response must contain all the **essential factual information** that directly answers the Question.
2.  **Equivalence Counts**: Phrasing doesn't need to be identical. If the Model's Response conveys the exact same core meaning and details as the Correct Answer, even if paraphrased or structured differently, consider it correct.
3.  **Superfluous (but Accurate) Information**: If the Model's Response includes additional details that were *not explicitly asked for* by the Question, but these details are **accurate and do not contradict** the Correct Answer, you should **still count it as correct** if the core question is fully answered.
4.  **Partial Answers are Incorrect**: If the Model's Response is missing any essential information directly requested by the Question (even if the Correct Answer provides more detail), it is incorrect.
5.  **Focus on the Question**: Your primary focus should be whether the Model's Response adequately addresses the Question using information that aligns with the Correct Answer. Do not penalize the model for not reiterating every single word or incidental detail from the Correct Answer if it wasn't requested."""


def _upstream_default(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        f"""I will provide you with a **Question**, a **Correct Answer**, and a **Model's Response**. Your sole task is to determine if the Model's Response is **sufficiently correct and complete** to answer the Question, when compared against the Correct Answer.

{_UPSTREAM_DEFAULT_GUIDELINES}

**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**

Question: {question}
Correct Answer: {correct_answer}
Model Response: {model_answer}

Answer (RIGHT/WRONG):"""
    )


_UPSTREAM_RUBRIC_GUIDELINES = """Guidelines:
- The model's response must fully satisfy ALL criteria in the rubric
- Partial fulfillment of criteria should result in INCORRECT
- The response must demonstrate clear understanding of the context
- Missing any important aspect mentioned in the rubric means INCORRECT"""


def _upstream_rubric(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        f"""You are a judge evaluating if a model's response meets the criteria in a rubric.

Question asked: {question}

Evaluation rubric: {correct_answer}

Model's response: {model_answer}

{_UPSTREAM_RUBRIC_GUIDELINES}

Evaluate whether the model's response satisfies the rubric criteria.

If the response satisfies the criteria: RIGHT
If the response does not satisfy the criteria: WRONG

Respond with only RIGHT or WRONG."""
    )


_UPSTREAM_TEMPORAL_GUIDELINES = """Guidelines:
- Time calculations must be accurate within reasonable bounds
- Accept different but equivalent time expressions (3 months = 90 days = 12 weeks)
- The model must show correct temporal reasoning
- Off-by-one errors in day counts are NOT acceptable"""


def _upstream_temporal(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        f"""You are a judge determining if a model's temporal answer is correct.

Question: {question}

Correct answer: {correct_answer}

Model's answer: {model_answer}

{_UPSTREAM_TEMPORAL_GUIDELINES}

Temporal answers often have legitimate variations. Consider if the model's answer represents the same time period or demonstrates correct temporal reasoning.

If the temporal answer is correct: RIGHT
If the temporal answer is incorrect: WRONG

Respond with only RIGHT or WRONG."""
    )


_UPSTREAM_ABSTENTION_GUIDELINES = """**Critical Guidelines for Abstention Evaluation:**

1. **Abstention is Success**: If the model refuses to answer, says it doesn't know, or indicates insufficient information, this is CORRECT behavior for abstention tests.

2. **Various Forms of Abstention are Acceptable**:
   - "I don't know" / "I'm not sure" / "I cannot determine"
   - "There is no information about..." / "This wasn't mentioned..."
   - "I don't have enough information to answer..."
   - "Based on the conversation, I cannot tell..."
   - "This information wasn't provided..."
   - Asking for clarification or more information

3. **Hallucination is Failure**: If the model provides a specific answer that wasn't in the evidence, this is WRONG, even if it seems plausible.

4. **Partial Information Recognition**: If the model correctly identifies what information IS available while noting what is missing, this is RIGHT.

5. **Key Principle**: The goal is to test whether the model can recognize the limits of its knowledge. Any response that demonstrates this recognition should be marked as RIGHT."""


def _upstream_abstention(question: str, model_answer: str) -> str:
    return (
        f"""You are evaluating an ABSTENTION test where the model should recognize it lacks sufficient information to answer the question.

**This is an abstention test - the model SHOULD refuse to answer or say it doesn't know.**

{_UPSTREAM_ABSTENTION_GUIDELINES}

**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**

Question: {question}
Expected behavior: The model should indicate it cannot answer due to insufficient information
Model Response: {model_answer}

Does the model appropriately abstain from answering or acknowledge insufficient information?

Answer (RIGHT/WRONG):"""
    )


def _upstream_userfacts_single(
    question: str, evidence_text: str, correct_answer: str, model_answer: str,
) -> str:
    # Note: upstream's Scala source has a trailing space after
    # "**Question Asked:**" (line ends with ``** ``). We preserve it byte-
    # exactly via the concatenated literal below.
    return (
        "You are evaluating whether a model's response correctly answers a question using information from an evidence message.\n"
        "\n"
        "**Your Task:**\n"
        "Determine if the Model's Response is RIGHT or WRONG based on the following criteria.\n"
        "\n"
        "**Question Asked:** \n"
        f"{question}\n"
        "\n"
        "**Evidence Message Available to the Model:**\n"
        f"{evidence_text}\n"
        "\n"
        "**Correct Answer (what the response should convey):**\n"
        f"{correct_answer}\n"
        "\n"
        "**Model's Response to Evaluate:**\n"
        f"{model_answer}\n"
        "\n"
        "**Evaluation Criteria:**\n"
        "\n"
        "When the response is RIGHT:\n"
        "- The core answer matches the Correct Answer (exact phrasing not required)\n"
        "- The response demonstrates understanding of the information from the evidence message\n"
        "- The response correctly addresses the question asked\n"
        "- If the model includes accurate additional details not in the Correct Answer, that's still RIGHT\n"
        "\n"
        "When the response is WRONG:\n"
        "- The core answer contradicts or significantly differs from the Correct Answer\n"
        "- Critical information from the evidence message is missing or misunderstood\n"
        "- The response shows confusion about the facts\n"
        "- The response fails to answer the question asked\n"
        "\n"
        "**Important Notes:**\n"
        "- The model doesn't need to quote the evidence message verbatim\n"
        "- Paraphrasing or summarization is expected and acceptable\n"
        "- Focus on whether the essential information has been correctly conveyed\n"
        "\n"
        "**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n"
        "\n"
        "Answer (RIGHT/WRONG):"
    )


def _upstream_userfacts_multi(
    question: str, evidence_block: str, correct_answer: str, model_answer: str,
) -> str:
    return (
        "You are evaluating whether a model's response correctly answers a question using information from multiple evidence messages.\n"
        "\n"
        "**Your Task:**\n"
        "Determine if the Model's Response is RIGHT or WRONG based on the following criteria.\n"
        "\n"
        "**Question Asked:** \n"
        f"{question}\n"
        "\n"
        "**Evidence Messages Available to the Model:**\n"
        f"{evidence_block}\n"
        "\n"
        "**Correct Answer (what the response should convey):**\n"
        f"{correct_answer}\n"
        "\n"
        "**Model's Response to Evaluate:**\n"
        f"{model_answer}\n"
        "\n"
        "**Evaluation Criteria:**\n"
        "\n"
        "When the response is RIGHT:\n"
        "- The core answer matches the Correct Answer (exact phrasing not required)\n"
        "- Information from most evidence messages is reasonably reflected in the response\n"
        "- The response shows understanding and synthesis of the multiple pieces of information\n"
        "- Minor omissions are acceptable if the main answer is correct\n"
        "\n"
        "When the response is WRONG:\n"
        "- The core answer contradicts or significantly differs from the Correct Answer\n"
        "- Critical information from multiple evidence messages is completely missing\n"
        "- The response shows misunderstanding or confusion about the facts\n"
        "- The response fails to answer the question asked\n"
        "\n"
        "**Important Notes:**\n"
        "- The model doesn't need to explicitly cite each evidence message\n"
        "- Focus on whether the essential information has been incorporated, not perfect completeness\n"
        "- Some paraphrasing or summarization is expected and acceptable\n"
        "- If the model includes accurate additional details not in the Correct Answer, that's still RIGHT\n"
        "\n"
        "**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n"
        "\n"
        "Answer (RIGHT/WRONG):"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _evidence_item(
    question: str = "What color do I use for hot leads?",
    answer: str = "Green",
    message_evidences: list[dict[str, str]] | None = None,
    conversations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "message_evidences": message_evidences if message_evidences is not None else [
            {"speaker": "User", "text": "I use green for hot leads."},
        ],
        "conversations": conversations if conversations is not None else [
            {
                "messages": [
                    {"speaker": "User", "text": "Hey!"},
                    {"speaker": "Assistant", "text": "Hi there!"},
                    {"speaker": "User", "text": "I use green for hot leads."},
                ],
            },
        ],
    }


def _write_category(
    root: Path,
    category: str,
    k: int,
    items: list[dict[str, Any]],
) -> None:
    """Write one batched_*.json under
    ``root/core_benchmark/evidence_questions/<category>/<k>_evidence``.
    """
    d = root / "core_benchmark" / "evidence_questions" / category / f"{k}_evidence"
    d.mkdir(parents=True)
    (d / "batched_0.json").write_text(
        json.dumps({"evidence_items": items}), encoding="utf-8",
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Minimal ConvoMem layout covering 3 categories + one abstention."""
    _write_category(tmp_path, "user_evidence", 1, [_evidence_item()])
    _write_category(tmp_path, "user_evidence", 2, [
        _evidence_item(
            question="What two colors do I use?",
            answer="Green and red",
            message_evidences=[
                {"speaker": "User", "text": "Green for hot."},
                {"speaker": "User", "text": "Red for cold."},
            ],
        ),
    ])
    _write_category(tmp_path, "preference_evidence", 1, [
        _evidence_item(
            question="What tone should I use?",
            answer="Formal and concise, no emojis.",  # rubric
        ),
    ])
    _write_category(tmp_path, "abstention_evidence", 1, [
        _evidence_item(
            question="Where did I leave my car keys?",
            answer="UNANSWERABLE",
        ),
    ])
    return tmp_path


def _task_from_item(
    category: str,
    evidence_count: int,
    item: dict[str, Any],
    idx: int = 0,
) -> Task:
    qid = f"{category}/{evidence_count}_evidence/{idx}"
    return Task(
        task_id=qid,
        benchmark="convomem",
        granularity=TaskGranularity.CONVERSATIONAL,
        prompt=str(item.get("question", "")),
        metadata={
            "conversation_id": qid,
            "category": category,
            "evidence_count": evidence_count,
            "message_evidences": list(item.get("message_evidences", []) or []),
            "conversations": list(item.get("conversations", []) or []),
            "is_abstention": category == "abstention_evidence",
        },
        ground_truth=str(item.get("answer", "")),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("convomem")
    assert isinstance(a, ConvoMemAdapter)
    assert a.name == "convomem"
    assert a.granularity == TaskGranularity.CONVERSATIONAL
    assert a.requires_sandbox is False
    assert a.get_sandbox_image() == ""


def test_categories_match_upstream():
    # These are the exact directory names on HF Salesforce/ConvoMem under
    # core_benchmark/evidence_questions/.
    assert CATEGORIES == (
        "user_evidence",
        "assistant_facts_evidence",
        "changing_evidence",
        "abstention_evidence",
        "preference_evidence",
        "implicit_connection_evidence",
    )


def test_evidence_count_ranges_match_upstream():
    # Taken directly from the HF dataset README.
    assert EVIDENCE_COUNT_RANGES == {
        "user_evidence": (1, 2, 3, 4, 5, 6),
        "assistant_facts_evidence": (1, 2, 3, 4, 5, 6),
        "changing_evidence": (2, 3, 4, 5, 6),
        "abstention_evidence": (1, 2, 3),
        "preference_evidence": (1, 2),
        "implicit_connection_evidence": (1, 2, 3),
    }


def test_default_params_defaults_to_upstream_gemini_flash():
    p = ConvoMemAdapter().get_default_params()
    # Upstream default is Gemini.flash which is pinned to gemini-2.5-flash.
    assert p["judge"]["provider"] == "google"
    assert p["judge"]["model"] == "gemini-2.5-flash"
    # Upstream's Config.Evaluation.ANSWER_VERIFICATION_MAX_RETRIES = 20.
    assert p["judge"]["max_retries"] == 20
    assert DEFAULT_JUDGE_MODEL == "gemini-2.5-flash"
    assert DEFAULT_JUDGE_MAX_RETRIES == 20


# ---------------------------------------------------------------------------
# Byte-exact upstream prompt templates
# ---------------------------------------------------------------------------


def test_default_prompt_matches_upstream_byte_exact():
    out = _default_prompt("Q", "A", "R")
    assert out == _upstream_default("Q", "A", "R")


def test_rubric_prompt_matches_upstream_byte_exact():
    out = _rubric_prompt("Q", "Rubric text", "R")
    assert out == _upstream_rubric("Q", "Rubric text", "R")


def test_temporal_prompt_matches_upstream_byte_exact():
    out = _temporal_prompt("Q", "A", "R")
    assert out == _upstream_temporal("Q", "A", "R")


def test_abstention_prompt_matches_upstream_byte_exact():
    # correct_answer is deliberately omitted from the template body.
    out = _abstention_prompt("Q", "A (ignored)", "R")
    assert out == _upstream_abstention("Q", "R")
    assert "A (ignored)" not in out
    # The abstention prompt does not expose the correct answer to the judge.
    assert "Correct Answer:" not in out


def test_user_facts_single_prompt_matches_upstream_byte_exact():
    evidences = [{"speaker": "User", "text": "I use green for hot leads."}]
    out = _user_facts_single_prompt("Q", "A", "R", evidences)
    assert out == _upstream_userfacts_single(
        "Q", "I use green for hot leads.", "A", "R",
    )


def test_user_facts_single_prompt_handles_empty_evidence():
    # Upstream ``.headOption.map(_.text).getOrElse("")`` — default to empty.
    out = _user_facts_single_prompt("Q", "A", "R", [])
    assert out == _upstream_userfacts_single("Q", "", "A", "R")


def test_user_facts_multi_prompt_matches_upstream_byte_exact():
    evidences = [
        {"speaker": "User", "text": "Green for hot."},
        {"speaker": "User", "text": "Red for cold."},
    ]
    expected_block = "Evidence Message 1: Green for hot.\nEvidence Message 2: Red for cold."
    out = _user_facts_multi_prompt("Q", "A", "R", evidences)
    assert out == _upstream_userfacts_multi("Q", expected_block, "A", "R")
    assert "Evidence Message 1:" in out and "Evidence Message 2:" in out


# ---------------------------------------------------------------------------
# Category → evaluator routing
# ---------------------------------------------------------------------------


def test_routing_user_evidence_single_uses_user_facts_single():
    ev = [{"speaker": "User", "text": "hello"}]
    out = _build_judge_prompt(
        "user_evidence", 1, "Q", "A", "R", ev,
    )
    assert out == _user_facts_single_prompt("Q", "A", "R", ev)


def test_routing_user_evidence_multi_uses_user_facts_multi():
    ev = [{"speaker": "User", "text": "a"}, {"speaker": "User", "text": "b"}]
    out = _build_judge_prompt(
        "user_evidence", 2, "Q", "A", "R", ev,
    )
    assert out == _user_facts_multi_prompt("Q", "A", "R", ev)


def test_routing_abstention_evidence_uses_abstention_prompt():
    out = _build_judge_prompt(
        "abstention_evidence", 1, "Q", "A", "R", [],
    )
    assert out == _abstention_prompt("Q", "A", "R")


@pytest.mark.parametrize("category", ["preference_evidence", "implicit_connection_evidence"])
def test_routing_rubric_categories_use_rubric_prompt(category):
    out = _build_judge_prompt(category, 1, "Q", "Rubric", "R", [])
    assert out == _rubric_prompt("Q", "Rubric", "R")


@pytest.mark.parametrize("category", ["assistant_facts_evidence", "changing_evidence"])
def test_routing_default_categories_use_default_prompt(category):
    out = _build_judge_prompt(category, 3, "Q", "A", "R", [])
    assert out == _default_prompt("Q", "A", "R")


def test_routing_unknown_category_falls_through_to_default():
    # Upstream's ``getAnsweringEvaluation`` default is DefaultAnsweringEvaluation.
    out = _build_judge_prompt("brand_new_category", 1, "Q", "A", "R", [])
    assert out == _default_prompt("Q", "A", "R")


# ---------------------------------------------------------------------------
# _parse_judge_label — mirrors EvaluationUtils.verifyAnswerCorrectnessWithPrompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("RIGHT", True),
    ("right", True),
    ("Right.", True),
    ("The answer is right.", True),
    ("  RIGHT  ", True),
    ("WRONG", False),
    ("wrong", False),
    ("wrong!", False),
    # Both tokens: upstream logs a warning and returns False.
    ("RIGHT and WRONG", False),
    ("wrong, not right", False),
    ("maybe right but maybe wrong", False),
])
def test_parse_judge_label_contains_right_or_wrong(text, expected):
    assert _parse_judge_label(text) is expected


@pytest.mark.parametrize("text", ["maybe", "", "   ", "unclear", "yes"])
def test_parse_judge_label_none_when_neither_present(text):
    # Upstream: neither 'right' nor 'wrong' → retry. We signal with None.
    assert _parse_judge_label(text) is None


# ---------------------------------------------------------------------------
# load_tasks / load_conversations
# ---------------------------------------------------------------------------


def test_load_tasks_preserves_upstream_fields(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({"data_path": str(data_dir)}))
    # Fixture has 4 items total (1 per dir).
    assert len(tasks) == 4

    # Categories surfaced verbatim.
    cats = {t.metadata["category"] for t in tasks}
    assert cats == {"user_evidence", "preference_evidence", "abstention_evidence"}

    # user_evidence k=1 task
    ue1 = next(t for t in tasks if t.metadata["category"] == "user_evidence"
               and t.metadata["evidence_count"] == 1)
    assert ue1.prompt == "What color do I use for hot leads?"
    assert ue1.ground_truth == "Green"
    # conversation_id == task_id so the engine isolates each item's memory store.
    assert ue1.metadata["conversation_id"] == ue1.task_id
    assert ue1.metadata["is_abstention"] is False
    # Evidence + conversations ride along as-is.
    assert ue1.metadata["message_evidences"] == [
        {"speaker": "User", "text": "I use green for hot leads."},
    ]


def test_load_tasks_marks_abstention(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({"data_path": str(data_dir)}))
    abs_task = next(t for t in tasks if t.metadata["category"] == "abstention_evidence")
    assert abs_task.metadata["is_abstention"] is True


def test_load_tasks_filter_categories(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(data_dir),
        "filter_categories": ["abstention_evidence", "preference_evidence"],
    }))
    assert {t.metadata["category"] for t in tasks} == {"user_evidence"}


def test_load_tasks_filter_evidence_counts(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(data_dir),
        "filter_evidence_counts": [1],
    }))
    assert all(t.metadata["evidence_count"] == 1 for t in tasks)
    assert len(tasks) == 3  # k=2 user_evidence filtered out


def test_load_tasks_restrict_categories(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(data_dir),
        "categories": ["user_evidence"],
    }))
    assert {t.metadata["category"] for t in tasks} == {"user_evidence"}


def test_load_tasks_restrict_evidence_counts_silently_skips_invalid(data_dir: Path):
    # preference_evidence only ships k=1,2 — asking for k=3 shouldn't explode.
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(data_dir),
        "categories": ["preference_evidence"],
        "evidence_counts": [1, 3, 99],
    }))
    assert len(tasks) == 1
    assert tasks[0].metadata["evidence_count"] == 1


def test_load_tasks_respects_limit(data_dir: Path):
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(data_dir), "limit": 2,
    }))
    assert len(tasks) == 2


def test_load_tasks_rejects_unknown_category(data_dir: Path):
    with pytest.raises(ValueError, match="Unknown category"):
        list(ConvoMemAdapter().load_tasks({
            "data_path": str(data_dir),
            "categories": ["made_up"],
        }))


def test_load_conversations_one_per_item(data_dir: Path):
    convs = ConvoMemAdapter().load_conversations({"data_path": str(data_dir)})
    assert len(convs) == 4
    # conversation_id matches task_id so engine runs are isolated per item.
    assert all(c["conversation_id"].count("/") == 2 for c in convs)


def test_load_conversations_speaker_mapping(data_dir: Path):
    convs = ConvoMemAdapter().load_conversations({"data_path": str(data_dir)})
    # Find the user_evidence k=1 conversation and check its first turn.
    user_conv = next(
        c for c in convs
        if c["metadata"]["category"] == "user_evidence"
        and c["metadata"]["evidence_count"] == 1
    )
    first = user_conv["messages"][0]
    # Upstream uses "User" / "Assistant" casing — preserved as-is.
    assert first["speaker_id"] == "User"
    assert first["speaker_name"] == "User"
    assert first["content"] == "Hey!"


def test_load_conversations_flattens_all_conversations(tmp_path: Path):
    item = _evidence_item(conversations=[
        {"messages": [{"speaker": "User", "text": "a1"}]},
        {"messages": [
            {"speaker": "User", "text": "b1"},
            {"speaker": "Assistant", "text": "b2"},
        ]},
    ])
    _write_category(tmp_path, "user_evidence", 1, [item])
    convs = ConvoMemAdapter().load_conversations({"data_path": str(tmp_path)})
    assert [m["content"] for m in convs[0]["messages"]] == ["a1", "b1", "b2"]


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def test_evidence_file_accepts_bare_array(tmp_path: Path):
    # Support both {"evidence_items": [...]} and a raw array at the top level.
    # (The upstream format is a dict, but some downstream exports drop the key.)
    d = tmp_path / "core_benchmark" / "evidence_questions" / "user_evidence" / "1_evidence"
    d.mkdir(parents=True)
    (d / "bare.json").write_text(
        json.dumps([_evidence_item()]), encoding="utf-8",
    )
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(tmp_path),
        "categories": ["user_evidence"],
        "evidence_counts": [1],
    }))
    assert len(tasks) == 1


def test_evidence_file_rejects_wrong_toplevel_shape(tmp_path: Path):
    d = tmp_path / "core_benchmark" / "evidence_questions" / "user_evidence" / "1_evidence"
    d.mkdir(parents=True)
    (d / "bad.json").write_text('{"items": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_items"):
        list(ConvoMemAdapter().load_tasks({
            "data_path": str(tmp_path),
            "categories": ["user_evidence"],
            "evidence_counts": [1],
        }))


def test_load_dataset_alternate_evidence_questions_layout(tmp_path: Path):
    # Allow users to mirror only ``evidence_questions/…`` (no ``core_benchmark/``).
    d = tmp_path / "evidence_questions" / "user_evidence" / "1_evidence"
    d.mkdir(parents=True)
    (d / "batched_0.json").write_text(
        json.dumps({"evidence_items": [_evidence_item()]}), encoding="utf-8",
    )
    tasks = list(ConvoMemAdapter().load_tasks({
        "data_path": str(tmp_path),
        "categories": ["user_evidence"],
        "evidence_counts": [1],
    }))
    assert len(tasks) == 1


def test_load_dataset_missing_without_hf_raises(tmp_path: Path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated: no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FileNotFoundError, match="huggingface_hub"):
        list(ConvoMemAdapter().load_tasks({
            "data_path": str(tmp_path / "nope"),
            "categories": ["user_evidence"],
            "evidence_counts": [1],
        }))


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


def test_format_prompt_returns_question_verbatim():
    a = ConvoMemAdapter()
    t = _task_from_item("user_evidence", 1, _evidence_item())
    assert a.format_prompt(t) == t.prompt


# ---------------------------------------------------------------------------
# judge_answer — stubs each provider SDK to avoid network
# ---------------------------------------------------------------------------


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeGeminiModels:
    async def generate_content(self, **kwargs):
        _FakeGeminiClient.last_call.update(kwargs)
        # Counter lives on the class so it persists across Client() instances;
        # upstream retries reinstantiate the client on each call, but the
        # response stream should advance monotonically for test simplicity.
        idx = _FakeGeminiClient.call_count
        _FakeGeminiClient.call_count += 1
        text = _FakeGeminiClient.responses[
            min(idx, len(_FakeGeminiClient.responses) - 1)
        ]
        return _FakeGeminiResponse(text)


class _FakeGeminiAio:
    def __init__(self):
        self.models = _FakeGeminiModels()


class _FakeGeminiClient:
    last_kwargs: dict[str, Any] = {}
    last_call: dict[str, Any] = {}
    responses: list[str] = ["RIGHT"]
    call_count: int = 0

    def __init__(self, **kwargs):
        _FakeGeminiClient.last_kwargs = kwargs
        _FakeGeminiClient.last_call = {}
        self.aio = _FakeGeminiAio()


@pytest.fixture
def patched_gemini(monkeypatch):
    """Install a fake ``google.genai`` module with a ``Client`` stub."""
    import sys
    import types

    _FakeGeminiClient.responses = ["RIGHT"]
    _FakeGeminiClient.call_count = 0
    google_mod = sys.modules.get("google") or types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _FakeGeminiClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    setattr(google_mod, "genai", genai_mod)
    return _FakeGeminiClient


@pytest.mark.asyncio
async def test_judge_uses_gemini_by_default(patched_gemini):
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "gk-test"}
    t = _task_from_item("user_evidence", 1, _evidence_item())
    out = await a.judge_answer(t, "I use green for hot leads.")

    # Provider auto-detected from model prefix.
    assert patched_gemini.last_kwargs == {"api_key": "gk-test"}
    assert patched_gemini.last_call["model"] == "gemini-2.5-flash"

    # Prompt is the upstream user_facts single template.
    expected_prompt = _upstream_userfacts_single(
        t.prompt,
        t.metadata["message_evidences"][0]["text"],
        t.ground_truth,
        "I use green for hot leads.",
    )
    assert patched_gemini.last_call["contents"] == expected_prompt

    # Upstream semantics: judge returned "RIGHT" → pass.
    assert out["pass"] is True
    assert out["label"] is True
    assert out["correct"] == 1
    assert out["judge_label"] is True
    assert out["judge_response"] == "RIGHT"
    assert out["judge_model"] == "gemini-2.5-flash"
    assert out["judge_attempts"] == 1
    assert out["category"] == "user_evidence"
    assert out["evidence_count"] == 1


@pytest.mark.asyncio
async def test_judge_wrong_response_fails(patched_gemini):
    patched_gemini.responses = ["WRONG"]
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k"}
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    out = await a.judge_answer(t, "blue")
    assert out["pass"] is False
    assert out["label"] is False
    assert out["judge_label"] is False


@pytest.mark.asyncio
async def test_judge_ambiguous_response_counts_as_wrong(patched_gemini):
    patched_gemini.responses = ["RIGHT then WRONG"]
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k"}
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    out = await a.judge_answer(t, "x")
    # Upstream: both tokens present → False (with warning).
    assert out["pass"] is False
    assert out["judge_label"] is False
    assert out["judge_attempts"] == 1


@pytest.mark.asyncio
async def test_judge_retries_on_invalid_response(patched_gemini):
    # First call: no right/wrong → retry. Second call: RIGHT → done.
    patched_gemini.responses = ["hmmm", "RIGHT"]
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k", "max_retries": 5}
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    out = await a.judge_answer(t, "x")
    assert out["pass"] is True
    assert out["judge_attempts"] == 2


@pytest.mark.asyncio
async def test_judge_exhausts_retries_returns_none_label(patched_gemini):
    patched_gemini.responses = ["hmm", "what", "huh"]
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k", "max_retries": 3}
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    out = await a.judge_answer(t, "x")
    # Upstream returns None after exhaustion; we surface judge_label=None but
    # treat the test as failed for metrics.
    assert out["judge_label"] is None
    assert out["pass"] is False
    assert out["correct"] == 0
    assert out["judge_attempts"] == 3


@pytest.mark.asyncio
async def test_judge_routes_abstention_to_abstention_prompt(patched_gemini):
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k"}
    item = _evidence_item(
        question="Where did I hide my keys?",
        answer="UNANSWERABLE",
        message_evidences=[],
    )
    t = _task_from_item("abstention_evidence", 1, item)
    await a.judge_answer(t, "I don't have that information.")

    prompt = patched_gemini.last_call["contents"]
    assert prompt == _upstream_abstention(
        t.prompt, "I don't have that information.",
    )
    # The ground-truth is not exposed to the abstention judge.
    assert t.ground_truth not in prompt


@pytest.mark.asyncio
async def test_judge_routes_preference_to_rubric_prompt(patched_gemini):
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k"}
    t = _task_from_item("preference_evidence", 1, _evidence_item(
        question="What tone should I use?",
        answer="Formal, concise.",
    ))
    await a.judge_answer(t, "I'll keep it formal.")

    assert patched_gemini.last_call["contents"] == _upstream_rubric(
        t.prompt, t.ground_truth, "I'll keep it formal.",
    )


@pytest.mark.asyncio
async def test_judge_routes_user_evidence_k2_to_multi_prompt(patched_gemini):
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gemini-2.5-flash", "api_key": "k"}
    item = _evidence_item(
        question="What two colors?",
        answer="Green and red.",
        message_evidences=[
            {"speaker": "User", "text": "Green for hot."},
            {"speaker": "User", "text": "Red for cold."},
        ],
    )
    t = _task_from_item("user_evidence", 2, item)
    await a.judge_answer(t, "Green and red.")

    expected_block = "Evidence Message 1: Green for hot.\nEvidence Message 2: Red for cold."
    assert patched_gemini.last_call["contents"] == _upstream_userfacts_multi(
        t.prompt, expected_block, t.ground_truth, "Green and red.",
    )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


class _FakeOpenAIMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content: str):
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAICompletion:
    def __init__(self, content: str):
        self.choices = [_FakeOpenAIChoice(content)]


class _FakeOpenAIChatCompletions:
    def __init__(self, capture: dict[str, Any], response: str):
        self._capture = capture
        self._response = response

    async def create(self, **kwargs):
        self._capture.update(kwargs)
        return _FakeOpenAICompletion(self._response)


class _FakeOpenAIChat:
    def __init__(self, capture: dict[str, Any], response: str):
        self.completions = _FakeOpenAIChatCompletions(capture, response)


class _FakeAsyncOpenAI:
    last_kwargs: dict[str, Any] = {}
    last_call: dict[str, Any] = {}
    response: str = "RIGHT"

    def __init__(self, **kwargs):
        _FakeAsyncOpenAI.last_kwargs = kwargs
        _FakeAsyncOpenAI.last_call = {}
        self.chat = _FakeOpenAIChat(
            _FakeAsyncOpenAI.last_call, _FakeAsyncOpenAI.response,
        )


@pytest.mark.asyncio
async def test_judge_explicit_openai_provider_uses_openai(monkeypatch):
    import openai
    _FakeAsyncOpenAI.response = "RIGHT"
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    a = ConvoMemAdapter()
    a._judge_config = {
        "provider": "openai", "model": "gpt-4o-2024-08-06",
        "api_key": "sk-test", "base_url": "https://api.openai.com/v1",
    }
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    out = await a.judge_answer(t, "answer")

    assert _FakeAsyncOpenAI.last_call["model"] == "gpt-4o-2024-08-06"
    assert _FakeAsyncOpenAI.last_call["temperature"] == 0.0
    assert out["pass"] is True
    assert out["judge_model"] == "gpt-4o-2024-08-06"


@pytest.mark.asyncio
async def test_judge_model_prefix_infers_provider(monkeypatch):
    import openai
    _FakeAsyncOpenAI.response = "RIGHT"
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    # Non-gemini, non-claude model → OpenAI provider.
    a = ConvoMemAdapter()
    a._judge_config = {"model": "gpt-4o", "api_key": "sk"}
    t = _task_from_item("assistant_facts_evidence", 3, _evidence_item())
    await a.judge_answer(t, "x")
    assert _FakeAsyncOpenAI.last_call["model"] == "gpt-4o"
