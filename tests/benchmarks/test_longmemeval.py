"""LongMemEval adapter tests — lock byte-exact fidelity to upstream.

The prompt strings below are copied verbatim from
``xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py::get_anscheck_prompt``
so the regression assertions fail loudly if we ever drift.

Tests mock ``openai.AsyncOpenAI`` so the judge path exercises our prompt
construction and parameter choices (temperature=0, max_tokens=10) without
touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks.longmemeval import (
    JUDGE_MODEL_ZOO,
    QUESTION_TYPES,
    LongMemEvalAdapter,
    _extract_label,
    _get_anscheck_prompt,
    _is_abstention,
    _render_history,
)
from cross_bench.models import Task, TaskGranularity

# ---------------------------------------------------------------------------
# Byte-exact upstream prompt strings — do not edit. Copied from
# xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py (commit: main).
# ---------------------------------------------------------------------------

# ruff: noqa: E501 — strings below are byte-exact copies of upstream prompts;
# splitting them would defeat the regression check's purpose.
_UPSTREAM_GENERIC = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_UPSTREAM_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_UPSTREAM_KU = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_UPSTREAM_PREF = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_UPSTREAM_ABST = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _instance(
    question_id: str = "gpt4_abc123",
    question_type: str = "single-session-user",
    question: str = "What did I order?",
    answer: str = "A shell necklace.",
    question_date: str = "2023/04/10 (Mon) 23:07",
) -> dict[str, Any]:
    """One upstream-shaped instance with a 2-session haystack."""
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "question_date": question_date,
        "haystack_session_ids": ["sess_1", "sess_2"],
        "haystack_dates": ["2023/03/15 (Wed) 14:30", "2023/04/01 (Sat) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Hi assistant!"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "I ordered a shell necklace.",
                 "has_answer": True},
            ],
            [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I can't check weather directly."},
            ],
        ],
        "answer_session_ids": ["sess_1"],
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """On-disk ``data/longmemeval/longmemeval_oracle.json`` fixture."""
    d = tmp_path / "longmemeval"
    d.mkdir()
    (d / "longmemeval_oracle.json").write_text(
        json.dumps([_instance(), _instance(question_id="gpt4_def456_abs",
                                           question="Where did I hide my keys?",
                                           answer="The question is unanswerable.")]),
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered():
    from cross_bench.benchmarks import get_adapter

    a = get_adapter("longmemeval")
    assert isinstance(a, LongMemEvalAdapter)
    assert a.name == "longmemeval"
    assert a.granularity == TaskGranularity.CONVERSATIONAL
    assert a.requires_sandbox is False
    assert a.get_sandbox_image() == ""


def test_default_params_match_upstream_defaults():
    p = LongMemEvalAdapter().get_default_params()
    assert p["variant"] == "oracle"
    assert p["filter_question_types"] == []
    assert p["only_abstention"] is False
    # Default judge alias is gpt-4o — upstream's preferred evaluator per model_zoo.
    assert p["judge"]["model"] == "gpt-4o"


def test_judge_model_zoo_matches_upstream_pinned_ids():
    # These are the exact version-pinned ids the upstream model_zoo resolves.
    assert JUDGE_MODEL_ZOO["gpt-4o"] == "gpt-4o-2024-08-06"
    assert JUDGE_MODEL_ZOO["gpt-4o-mini"] == "gpt-4o-mini-2024-07-18"
    assert JUDGE_MODEL_ZOO["llama-3.1-70b-instruct"] == \
        "meta-llama/Meta-Llama-3.1-70B-Instruct"


def test_question_types_are_upstream_text_labels():
    # No numeric enum: upstream uses these six text labels.
    assert QUESTION_TYPES == (
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    )


# ---------------------------------------------------------------------------
# _get_anscheck_prompt — byte-exact match against upstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qtype", [
    "single-session-user", "single-session-assistant", "multi-session",
])
def test_generic_template_matches_upstream(qtype):
    out = _get_anscheck_prompt(qtype, "Q", "A", "R", abstention=False)
    assert out == _UPSTREAM_GENERIC.format("Q", "A", "R")


def test_temporal_template_matches_upstream():
    out = _get_anscheck_prompt("temporal-reasoning", "Q", "A", "R", abstention=False)
    assert out == _UPSTREAM_TEMPORAL.format("Q", "A", "R")


def test_knowledge_update_template_matches_upstream():
    out = _get_anscheck_prompt("knowledge-update", "Q", "A", "R", abstention=False)
    assert out == _UPSTREAM_KU.format("Q", "A", "R")


def test_preference_template_matches_upstream():
    # Note: uses ``Rubric:`` not ``Correct Answer:`` — upstream distinction.
    out = _get_anscheck_prompt("single-session-preference", "Q", "A", "R", abstention=False)
    assert out == _UPSTREAM_PREF.format("Q", "A", "R")
    assert "Rubric:" in out and "Correct Answer:" not in out


def test_abstention_template_overrides_question_type():
    # Abstention prompt is used regardless of question_type.
    for qtype in QUESTION_TYPES:
        out = _get_anscheck_prompt(qtype, "Q", "Explain", "R", abstention=True)
        assert out == _UPSTREAM_ABST.format("Q", "Explain", "R")


def test_get_anscheck_prompt_unknown_type_raises():
    with pytest.raises(NotImplementedError):
        _get_anscheck_prompt("made-up-type", "Q", "A", "R", abstention=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_abstention_on_abs_suffix():
    assert _is_abstention("gpt4_abc_abs")
    assert _is_abstention("whatever_abs_trailing")  # upstream uses 'in', not endswith
    assert not _is_abstention("gpt4_abc")


@pytest.mark.parametrize("text,label", [
    ("yes", True),
    ("Yes", True),
    ("YES", True),
    ("yes.", True),
    ("The answer is yes", True),
    ("no", False),
    ("No.", False),
    ("Maybe", False),
    ("", False),
])
def test_extract_label_matches_upstream(text, label):
    # Upstream: ``'yes' in eval_response.lower()``. Exactly this check.
    assert _extract_label(text) is label


# ---------------------------------------------------------------------------
# _render_history — upstream long-context serialization
# ---------------------------------------------------------------------------


def test_render_history_json_strips_has_answer_and_sorts_by_date():
    sessions = [
        [{"role": "user", "content": "hi", "has_answer": True}],
        [{"role": "assistant", "content": "hello"}],
    ]
    # Dates out of order on input; upstream sorts ascending.
    dates = ["2024-05-02", "2024-05-01"]
    out = _render_history(sessions, dates, history_format="json", useronly=False)

    assert "has_answer" not in out  # stripped before serialization
    # Session 1 is the earlier date (2024-05-01).
    assert "### Session 1:" in out
    assert "Session Date: 2024-05-01" in out
    # Order of session blocks follows ascending date.
    assert out.index("2024-05-01") < out.index("2024-05-02")


def test_render_history_nl_format_per_turn():
    sessions = [[
        {"role": "user", "content": "  hi  "},
        {"role": "assistant", "content": "hello"},
    ]]
    out = _render_history(sessions, ["2024-01-01"], history_format="nl")
    # Turns rendered as ``role: content`` with surrounding blank lines,
    # content stripped — matching upstream formatting.
    assert "user: hi" in out
    assert "assistant: hello" in out


def test_render_history_useronly_drops_assistant_turns():
    sessions = [[
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]]
    out = _render_history(sessions, ["2024-01-01"], history_format="nl", useronly=True)
    assert "user: q1" in out
    assert "a1" not in out


def test_render_history_rejects_unknown_format():
    with pytest.raises(NotImplementedError):
        _render_history([[{"role": "user", "content": "x"}]], ["d"], history_format="xml")


# ---------------------------------------------------------------------------
# load_tasks / load_conversations
# ---------------------------------------------------------------------------


def test_load_tasks_preserves_upstream_fields(data_dir: Path):
    a = LongMemEvalAdapter()
    tasks = list(a.load_tasks({"data_path": str(data_dir), "variant": "oracle"}))
    # data_dir fixture writes 2 instances: one regular + one abstention.
    assert len(tasks) == 2

    t = tasks[0]
    assert t.task_id == "gpt4_abc123"
    assert t.benchmark == "longmemeval"
    assert t.granularity == TaskGranularity.CONVERSATIONAL
    assert t.prompt == "What did I order?"
    assert t.ground_truth == "A shell necklace."
    # conversation_id == question_id so the engine ingests each haystack into
    # its own memory store (matches upstream per-question evaluation).
    assert t.metadata["conversation_id"] == "gpt4_abc123"
    assert t.metadata["question_type"] == "single-session-user"
    assert t.metadata["is_abstention"] is False
    assert t.metadata["question_date"] == "2023/04/10 (Mon) 23:07"
    assert t.metadata["haystack_session_ids"] == ["sess_1", "sess_2"]
    assert t.metadata["answer_session_ids"] == ["sess_1"]
    # Haystack rides along unchanged — no reshaping.
    assert t.metadata["haystack_sessions"][0][0] == {
        "role": "user", "content": "Hi assistant!"}


def test_load_tasks_marks_abstention(data_dir: Path):
    tasks = list(LongMemEvalAdapter().load_tasks({"data_path": str(data_dir)}))
    abs_task = [t for t in tasks if t.task_id.endswith("_abs")][0]
    assert abs_task.metadata["is_abstention"] is True


def test_load_tasks_filter_question_types(data_dir: Path):
    # Both fixture instances are single-session-user → filtering it returns nothing.
    tasks = list(LongMemEvalAdapter().load_tasks({
        "data_path": str(data_dir),
        "filter_question_types": ["single-session-user"],
    }))
    assert tasks == []


def test_load_tasks_only_abstention_filter(data_dir: Path):
    tasks = list(LongMemEvalAdapter().load_tasks({
        "data_path": str(data_dir),
        "only_abstention": True,
    }))
    assert len(tasks) == 1
    assert tasks[0].task_id.endswith("_abs")


def test_load_tasks_respects_limit(data_dir: Path):
    tasks = list(LongMemEvalAdapter().load_tasks({
        "data_path": str(data_dir),
        "limit": 1,
    }))
    assert len(tasks) == 1


def test_load_conversations_one_per_question(data_dir: Path):
    convs = LongMemEvalAdapter().load_conversations({"data_path": str(data_dir)})
    assert len(convs) == 2
    assert {c["conversation_id"] for c in convs} == {"gpt4_abc123", "gpt4_def456_abs"}


def test_load_conversations_preserves_has_answer_flag(data_dir: Path):
    convs = LongMemEvalAdapter().load_conversations({"data_path": str(data_dir)})
    conv = convs[0]
    # The third turn of session 1 is the evidence turn.
    evidence_turns = [m for m in conv["messages"] if m.get("has_answer")]
    assert len(evidence_turns) == 1
    assert evidence_turns[0]["content"] == "I ordered a shell necklace."


def test_load_conversations_roles_mapped_to_speaker_fields(data_dir: Path):
    convs = LongMemEvalAdapter().load_conversations({"data_path": str(data_dir)})
    msg = convs[0]["messages"][0]
    # Upstream role flows into both speaker_id and speaker_name.
    assert msg["speaker_id"] == "user"
    assert msg["speaker_name"] == "user"
    assert msg["timestamp"] == "2023/03/15 (Wed) 14:30"
    assert msg["session_id"] == "sess_1"


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def test_load_dataset_unknown_variant_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="variant must be one of"):
        list(LongMemEvalAdapter().load_tasks({
            "data_path": str(data_dir),
            "variant": "xl",  # not a real variant
        }))


def test_load_dataset_explicit_file_path(tmp_path: Path):
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps([_instance()]), encoding="utf-8")
    tasks = list(LongMemEvalAdapter().load_tasks({"file_path": str(custom)}))
    assert len(tasks) == 1


def test_load_dataset_missing_without_hf_raises(tmp_path: Path, monkeypatch):
    # Simulate missing huggingface_hub so the local-missing branch falls through.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated: no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FileNotFoundError, match="huggingface_hub"):
        list(LongMemEvalAdapter().load_tasks({
            "data_path": str(tmp_path / "nope"),
            "variant": "oracle",
        }))


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


def _task_from_instance(inst: dict[str, Any]) -> Task:
    return Task(
        task_id=inst["question_id"],
        benchmark="longmemeval",
        granularity=TaskGranularity.CONVERSATIONAL,
        prompt=inst["question"],
        metadata={
            "conversation_id": inst["question_id"],
            "question_type": inst["question_type"],
            "is_abstention": _is_abstention(inst["question_id"]),
            "question_date": inst["question_date"],
            "haystack_sessions": inst["haystack_sessions"],
            "haystack_dates": inst["haystack_dates"],
            "haystack_session_ids": inst["haystack_session_ids"],
            "answer_session_ids": inst["answer_session_ids"],
        },
        ground_truth=inst["answer"],
    )


def test_format_prompt_question_mode_is_verbatim():
    a = LongMemEvalAdapter()
    t = _task_from_instance(_instance())
    assert a.format_prompt(t) == t.prompt


def test_format_prompt_question_cot_suffix():
    a = LongMemEvalAdapter()
    t = _task_from_instance(_instance())
    t.metadata["prompt_style"] = "question_cot"
    # Matches the upstream no-retrieval CoT variant: "{question}Answer step by step."
    assert a.format_prompt(t) == f"{t.prompt}Answer step by step."


def test_format_prompt_full_history_matches_upstream_template():
    a = LongMemEvalAdapter()
    t = _task_from_instance(_instance())
    t.metadata["prompt_style"] = "full_history"
    out = a.format_prompt(t)
    # Contains upstream scaffolding strings verbatim.
    assert "I will give you several history chats between you and a user." in out
    assert "History Chats:" in out
    assert "Current Date: 2023/04/10 (Mon) 23:07" in out
    assert f"Question: {t.prompt}" in out
    assert out.endswith("Answer:")


def test_format_prompt_full_history_cot_uses_step_template():
    a = LongMemEvalAdapter()
    t = _task_from_instance(_instance())
    t.metadata["prompt_style"] = "full_history_cot"
    out = a.format_prompt(t)
    assert "Answer the question step by step" in out
    assert out.endswith("Answer (step by step):")


# ---------------------------------------------------------------------------
# judge_answer — the crown-jewel upstream-fidelity test
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, captured: dict[str, Any], response: str = "yes"):
        self._captured = captured
        self._response = response

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeCompletion(self._response)


class _FakeChat:
    def __init__(self, captured: dict[str, Any], response: str = "yes"):
        self.completions = _FakeChatCompletions(captured, response)


class _FakeAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI``. Captures constructor + call kwargs."""

    last_init: dict[str, Any] = {}
    last_call: dict[str, Any] = {}
    response_text: str = "yes"

    def __init__(self, **kwargs):
        _FakeAsyncOpenAI.last_init = kwargs
        _FakeAsyncOpenAI.last_call = {}
        self.chat = _FakeChat(_FakeAsyncOpenAI.last_call, _FakeAsyncOpenAI.response_text)


@pytest.fixture
def patched_openai(monkeypatch):
    import openai
    _FakeAsyncOpenAI.response_text = "yes"
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    return _FakeAsyncOpenAI


@pytest.mark.asyncio
async def test_judge_builds_prompt_via_get_anscheck_prompt(patched_openai):
    a = LongMemEvalAdapter()
    # Mirror what load_tasks stashes.
    a._judge_config = {"model": "gpt-4o", "api_key": "sk-test"}
    task = _task_from_instance(_instance(question_type="temporal-reasoning"))

    out = await a.judge_answer(task, "The answer is 18 days.")

    # The judge prompt fed to OpenAI is exactly upstream's temporal template.
    call = patched_openai.last_call
    msgs = call["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    expected = _UPSTREAM_TEMPORAL.format(task.prompt, task.ground_truth,
                                          "The answer is 18 days.")
    assert msgs[0]["content"] == expected

    # Upstream invocation params — temperature=0, max_tokens=10, n=1.
    assert call["temperature"] == 0
    assert call["max_tokens"] == 10
    assert call["n"] == 1
    # Model alias resolved to the pinned upstream id.
    assert call["model"] == "gpt-4o-2024-08-06"

    # Result shape — mirrors upstream's autoeval_label plus CrossBench adds.
    assert out["pass"] is True
    assert out["correct"] == 1
    assert out["label"] is True
    assert out["autoeval_label"] == {"model": "gpt-4o-2024-08-06", "label": True}
    assert out["judge_model"] == "gpt-4o-2024-08-06"
    assert out["question_type"] == "temporal-reasoning"
    assert out["is_abstention"] is False


@pytest.mark.asyncio
async def test_judge_uses_abstention_template_on_abs_question(patched_openai):
    a = LongMemEvalAdapter()
    a._judge_config = {"model": "gpt-4o", "api_key": "sk-test"}
    task = _task_from_instance(_instance(
        question_id="gpt4_xyz_abs",
        question_type="multi-session",  # ignored when abstention=True
        question="Where did I hide my car keys?",
        answer="The question is unanswerable.",
    ))

    await a.judge_answer(task, "I don't have that information.")

    prompt = patched_openai.last_call["messages"][0]["content"]
    expected = _UPSTREAM_ABST.format(task.prompt, task.ground_truth,
                                      "I don't have that information.")
    assert prompt == expected


@pytest.mark.asyncio
async def test_judge_label_negative_for_no_response(monkeypatch, patched_openai):
    _FakeAsyncOpenAI.response_text = "no, the response is incorrect"
    a = LongMemEvalAdapter()
    a._judge_config = {"model": "gpt-4o", "api_key": "sk-test"}
    task = _task_from_instance(_instance())

    out = await a.judge_answer(task, "wrong answer")
    assert out["pass"] is False
    assert out["label"] is False
    assert out["correct"] == 0


@pytest.mark.asyncio
async def test_judge_accepts_literal_model_ids_unchanged(patched_openai):
    # A model name not in JUDGE_MODEL_ZOO passes through verbatim — this is the
    # escape hatch for custom vLLM-served judges (upstream supports it via
    # the local API path).
    a = LongMemEvalAdapter()
    a._judge_config = {"model": "llama-3.1-70b-instruct", "api_key": "EMPTY"}
    await a.judge_answer(_task_from_instance(_instance()), "answer")
    # Alias in zoo → resolved to HuggingFace id.
    assert patched_openai.last_call["model"] == "meta-llama/Meta-Llama-3.1-70B-Instruct"


@pytest.mark.asyncio
async def test_judge_surfaces_sdk_failure(monkeypatch):
    """When the OpenAI call raises, judge returns a failure dict with shape."""
    import openai

    class _ExplodingOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": self})()

        async def create(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(openai, "AsyncOpenAI", _ExplodingOpenAI)

    a = LongMemEvalAdapter()
    a._judge_config = {"model": "gpt-4o", "api_key": "sk"}
    out = await a.judge_answer(_task_from_instance(_instance()), "x")
    assert out["pass"] is False
    assert out["correct"] == 0
    assert "judge_error" in out
    assert out["judge_model"] == "gpt-4o-2024-08-06"
