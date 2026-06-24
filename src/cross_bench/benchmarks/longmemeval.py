"""LongMemEval adapter — faithful port of xiaowu0162/LongMemEval (ICLR 2025).

Evaluates long-term conversational memory across six question types, plus
abstention. The six types come straight from the upstream benchmark:

* ``single-session-user``
* ``single-session-assistant``
* ``single-session-preference``
* ``multi-session``
* ``temporal-reasoning``
* ``knowledge-update``

Questions whose ``question_id`` ends in ``_abs`` are **abstention** cases —
the model is supposed to recognise the question as unanswerable.

Dataset — three canonical JSON files hosted at
``xiaowu0162/longmemeval-cleaned`` on HuggingFace:

* ``longmemeval_s_cleaned.json``  — LongMemEval_S (~115k tokens / ~40 sessions)
* ``longmemeval_m_cleaned.json``  — LongMemEval_M (~500 sessions)
* ``longmemeval_oracle.json``     — only the evidence sessions per question

Each instance is self-contained and ships its own haystack (sessions, dates,
session ids, evidence session ids). We surface every question as a
:class:`Task` whose ``conversation_id`` equals the ``question_id`` so the
conversational engine ingests each haystack into a fresh memory store,
matching the upstream protocol of independent per-question evaluation.

Judging reproduces ``src/evaluation/evaluate_qa.py`` verbatim:

* Five ``question_type``-conditioned prompt templates + an abstention template
* Single call at ``temperature=0`` / ``max_tokens=10``
* Label = ``"yes" in response.lower()``
* Default judge model: ``gpt-4o-2024-08-06`` (same alias resolution upstream
  uses in its ``model_zoo``).

References:
    https://github.com/xiaowu0162/LongMemEval
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
    https://arxiv.org/abs/2410.10813
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cross_bench.benchmarks.base import ConversationalBenchmarkAdapter
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream constants — DO NOT rename or renumber. These are contracts with
# xiaowu0162/LongMemEval's evaluate_qa.py / print_qa_metrics.py.
# ---------------------------------------------------------------------------

QUESTION_TYPES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

# Upstream ``model_zoo`` in src/evaluation/evaluate_qa.py. Only these aliases
# are accepted as the ``judge_model``; the values are the concrete, version-
# pinned model ids required for reproducibility.
JUDGE_MODEL_ZOO: dict[str, str] = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

# The canonical {variant: filename} mapping on HF xiaowu0162/longmemeval-cleaned.
_VARIANT_FILES: dict[str, str] = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

_HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"


# ---------------------------------------------------------------------------
# Prompt templates — reproduced BYTE-EXACT from upstream
# src/evaluation/evaluate_qa.py::get_anscheck_prompt.  Any whitespace or
# wording change here breaks faithfulness, so the strings are covered by
# regression tests.
# ---------------------------------------------------------------------------

_TEMPLATE_GENERIC = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)

_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes "
    "the model correctly identify the question as unanswerable? Answer yes or "
    "no only."
)

# Long-context reading prompt — upstream src/generation/run_generation.py
# ``prepare_prompt``, ``retriever_type="no-retrieval"`` branch omitted because
# that path is just ``{question}`` with an optional CoT suffix.
_ANSWER_TEMPLATE_DIRECT = (
    "I will give you several history chats between you and a user. Please "
    "answer the question based on the relevant chat history.\n\n\nHistory "
    "Chats:\n\n{history}\n\nCurrent Date: {question_date}\nQuestion: "
    "{question}\nAnswer:"
)

_ANSWER_TEMPLATE_COT = (
    "I will give you several history chats between you and a user. Please "
    "answer the question based on the relevant chat history. Answer the "
    "question step by step: first extract all the relevant information, and "
    "then reason over the information to get the answer.\n\n\nHistory "
    "Chats:\n\n{history}\n\nCurrent Date: {question_date}\nQuestion: "
    "{question}\nAnswer (step by step):"
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no I/O, easy to unit-test)
# ---------------------------------------------------------------------------

def _get_anscheck_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
) -> str:
    """Build the judge prompt.

    Mirrors ``src/evaluation/evaluate_qa.py::get_anscheck_prompt``. The
    abstention branch is gated on the ``abstention`` flag, which the caller
    derives from ``"_abs" in question_id`` — identical to upstream.
    """
    if abstention:
        return _TEMPLATE_ABSTENTION.format(question, answer, response)

    if question_type in ("single-session-user", "single-session-assistant", "multi-session"):
        return _TEMPLATE_GENERIC.format(question, answer, response)
    if question_type == "temporal-reasoning":
        return _TEMPLATE_TEMPORAL.format(question, answer, response)
    if question_type == "knowledge-update":
        return _TEMPLATE_KNOWLEDGE_UPDATE.format(question, answer, response)
    if question_type == "single-session-preference":
        return _TEMPLATE_PREFERENCE.format(question, answer, response)
    raise NotImplementedError(f"Unknown question_type: {question_type!r}")


def _extract_label(eval_response: str) -> bool:
    """``"yes" in eval_response.lower()`` — matches upstream exactly."""
    return "yes" in eval_response.lower()


def _is_abstention(question_id: str) -> bool:
    """Upstream convention: ``_abs`` suffix marks abstention questions."""
    return "_abs" in question_id


def _render_history(haystack_sessions: list[list[dict[str, Any]]],
                    haystack_dates: list[str],
                    history_format: str = "json",
                    useronly: bool = False) -> str:
    """Render a question's haystack into upstream's ``History Chats`` block.

    Mirrors the ``orig-session`` / ``oracle-session`` branch of
    ``src/generation/run_generation.py::prepare_prompt`` (the branch used by
    ``bash run_generation.sh ... full-history-session ...``). Turns are
    stripped of ``has_answer`` before serialization.
    """
    if history_format not in ("json", "nl"):
        raise NotImplementedError(f"history_format must be 'json' or 'nl', got {history_format!r}")

    # Pair (date, session) then sort by date — upstream sorts chunks ascending.
    paired = sorted(zip(haystack_dates, haystack_sessions), key=lambda x: x[0])

    chunks = []
    for i, (date, session) in enumerate(paired):
        cleaned = [{k: v for k, v in turn.items() if k != "has_answer"} for turn in session]
        if useronly:
            cleaned = [t for t in cleaned if t.get("role") == "user"]

        if history_format == "json":
            body = "\n" + json.dumps(cleaned)
        else:
            body = "".join(
                f"\n\n{turn.get('role')}: {str(turn.get('content', '')).strip()}"
                for turn in cleaned
            )
        chunks.append(
            f"\n### Session {i + 1}:\nSession Date: {date}\nSession Content:\n{body}\n"
        )
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class LongMemEvalAdapter(ConversationalBenchmarkAdapter):
    """LongMemEval adapter — faithful to xiaowu0162/LongMemEval."""

    name = "longmemeval"
    granularity = TaskGranularity.CONVERSATIONAL

    def __init__(self) -> None:
        super().__init__()
        self._judge_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Task & conversation loading
    # ------------------------------------------------------------------

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        """Yield one :class:`Task` per LongMemEval instance.

        The whole haystack rides along in ``metadata`` — readers that need it
        (long-context baselines, retrieval-augmented systems) can access it;
        memory systems can ignore it and use :meth:`load_conversations`.
        """
        instances = self._load_dataset(params)
        self._judge_config = params.get("judge", {})

        filter_types = set(params.get("filter_question_types", []) or [])
        only_abstention = params.get("only_abstention", False)
        limit = params.get("limit")

        count = 0
        for inst in instances:
            qtype = inst.get("question_type", "")
            if filter_types and qtype in filter_types:
                continue
            qid = inst.get("question_id", "")
            abstention = _is_abstention(qid)
            if only_abstention and not abstention:
                continue
            if limit is not None and count >= limit:
                return

            yield Task(
                task_id=qid,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=inst.get("question", ""),
                metadata={
                    # conversation_id == question_id: each question is its
                    # own haystack, ingested into a fresh memory store.
                    "conversation_id": qid,
                    "question_type": qtype,
                    "is_abstention": abstention,
                    "question_date": inst.get("question_date", ""),
                    "haystack_sessions": inst.get("haystack_sessions", []),
                    "haystack_dates": inst.get("haystack_dates", []),
                    "haystack_session_ids": inst.get("haystack_session_ids", []),
                    "answer_session_ids": inst.get("answer_session_ids", []),
                },
                ground_truth=inst.get("answer", ""),
            )
            count += 1

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Expose each question's haystack as a standalone conversation.

        Conversations are keyed by ``question_id`` (matching the ``Task``'s
        ``conversation_id``) so the engine's per-conversation isolation
        corresponds exactly to upstream's per-question evaluation.

        Each turn carries the upstream ``role`` (``user``/``assistant``) and
        preserves the ``has_answer`` flag when present, for systems that want
        to exploit turn-level evidence labels.
        """
        instances = self._load_dataset(params)
        conversations: list[dict[str, Any]] = []
        for inst in instances:
            qid = inst.get("question_id", "")
            session_ids = inst.get("haystack_session_ids", [])
            dates = inst.get("haystack_dates", [])
            sessions = inst.get("haystack_sessions", [])

            messages: list[dict[str, Any]] = []
            for sess_id, sess_date, sess in zip(session_ids, dates, sessions):
                for turn_idx, turn in enumerate(sess):
                    role = turn.get("role", "")
                    msg: dict[str, Any] = {
                        # Map upstream "role" onto the engine's speaker fields.
                        "speaker_id": role,
                        "speaker_name": role,
                        "content": turn.get("content", ""),
                        "timestamp": sess_date,
                        "session_id": sess_id,
                        "turn_index": turn_idx,
                    }
                    if turn.get("has_answer"):
                        msg["has_answer"] = True
                    messages.append(msg)

            conversations.append({
                "conversation_id": qid,
                "messages": messages,
                "metadata": {
                    "question_type": inst.get("question_type", ""),
                    "question_date": inst.get("question_date", ""),
                    "answer_session_ids": inst.get("answer_session_ids", []),
                },
            })
        return conversations

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """Return the prompt for a reader model.

        Default: just the question, for memory systems that retrieve their
        own context. Override via ``benchmark_params.prompt_style``:

        * ``"question"`` (default) — ``task.prompt`` verbatim
        * ``"question_cot"`` — ``{question}\\nAnswer step by step.``
        * ``"full_history"`` — upstream long-context direct template
        * ``"full_history_cot"`` — upstream long-context step-by-step template

        The ``full_history`` styles embed the haystack using the upstream
        ``History Chats`` block, sorted ascending by date.
        """
        style = (task.metadata.get("prompt_style") or self._judge_config.get("prompt_style")
                 or "question")

        if style == "question":
            return task.prompt
        if style == "question_cot":
            return f"{task.prompt}Answer step by step."

        history_format = task.metadata.get("history_format", "json")
        useronly = bool(task.metadata.get("useronly", False))
        history = _render_history(
            task.metadata.get("haystack_sessions", []),
            task.metadata.get("haystack_dates", []),
            history_format=history_format,
            useronly=useronly,
        )
        question_date = task.metadata.get("question_date", "")

        if style == "full_history":
            template = _ANSWER_TEMPLATE_DIRECT
        elif style == "full_history_cot":
            template = _ANSWER_TEMPLATE_COT
        else:
            raise NotImplementedError(f"Unknown prompt_style: {style!r}")

        return template.format(history=history, question_date=question_date,
                               question=task.prompt)

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    async def judge_answer(
        self,
        task: Task,
        generated_answer: str,
    ) -> dict[str, Any]:
        """Judge a hypothesis against the golden answer via the upstream LLM judge.

        Single API call at ``temperature=0``, ``max_tokens=10``. The label is
        ``"yes" in response.lower()`` — identical to upstream's
        ``evaluate_qa.py``. The result dict matches the shape stored in
        upstream's ``*.eval-results-*`` log files, plus surfacing the
        ``question_type`` and a CrossBench ``pass`` field.
        """
        qtype = task.metadata.get("question_type", "")
        abstention = bool(task.metadata.get("is_abstention",
                                            _is_abstention(task.task_id)))
        question = task.prompt
        golden = str(task.ground_truth or "")

        judge_prompt = _get_anscheck_prompt(
            qtype, question, golden, generated_answer, abstention=abstention,
        )

        judge_alias = self._judge_config.get("model", "gpt-4o")
        judge_model = JUDGE_MODEL_ZOO.get(judge_alias, judge_alias)
        api_key = self._judge_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = self._judge_config.get("base_url") or os.environ.get(
            "LLM_BASE_URL") or None
        organization = self._judge_config.get("organization") or os.environ.get(
            "OPENAI_ORGANIZATION") or None

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            logger.warning("openai SDK not installed; cannot run LongMemEval judge: %s", exc)
            return {
                "pass": False,
                "correct": 0,
                "label": False,
                "judge_error": "openai_not_installed",
                "question_type": qtype,
                "is_abstention": abstention,
                "judge_model": judge_model,
            }

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        client = AsyncOpenAI(**client_kwargs)

        try:
            completion = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                n=1,
                temperature=0,
                max_tokens=10,
            )
            eval_response = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface judge failures in result
            logger.warning("LongMemEval judge call failed: %s", exc)
            return {
                "pass": False,
                "correct": 0,
                "label": False,
                "judge_error": str(exc),
                "question_type": qtype,
                "is_abstention": abstention,
                "judge_model": judge_model,
            }

        label = _extract_label(eval_response)
        return {
            "pass": label,
            "correct": int(label),
            "label": label,
            # Mirrors upstream's per-entry ``autoeval_label`` field.
            "autoeval_label": {"model": judge_model, "label": label},
            "judge_response": eval_response,
            "judge_model": judge_model,
            "question_type": qtype,
            "is_abstention": abstention,
        }

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict[str, Any]:
        return {
            "variant": "oracle",
            "data_path": "data/longmemeval",
            "filter_question_types": [],
            "only_abstention": False,
            "judge": {
                "model": "gpt-4o",  # resolved to gpt-4o-2024-08-06 via JUDGE_MODEL_ZOO
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "base_url": os.environ.get("LLM_BASE_URL", ""),
                "organization": os.environ.get("OPENAI_ORGANIZATION", ""),
            },
        }

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve the benchmark JSON file for the requested variant.

        Resolution order:

        1. Explicit ``file_path`` param — treated as a literal JSON path.
        2. ``{data_path}/{variant_file}`` for each of ``./``, the current
           working directory, and common project locations.
        3. HuggingFace ``xiaowu0162/longmemeval-cleaned`` via
           :func:`huggingface_hub.hf_hub_download` (cached locally).

        The file is a JSON array; we return it as-is (no reshaping) so
        downstream code reads the exact upstream schema.
        """
        file_path = params.get("file_path")
        if file_path:
            return _load_json_array(Path(file_path))

        variant = params.get("variant", "oracle")
        if variant not in _VARIANT_FILES:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANT_FILES)}, got {variant!r}"
            )
        filename = _VARIANT_FILES[variant]

        data_path = params.get("data_path", "data/longmemeval")
        candidate = Path(data_path) / filename
        if candidate.is_file():
            return _load_json_array(candidate)

        # Fall back to HuggingFace hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"LongMemEval dataset not found locally under {data_path!r} and "
                f"huggingface_hub is not installed. Either install it "
                f"(`pip install huggingface_hub`) or download "
                f"https://huggingface.co/datasets/{_HF_REPO_ID}/resolve/main/{filename}"
            ) from exc

        local = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
        return _load_json_array(Path(local))


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    """Read a LongMemEval JSON file.

    Upstream data is a single top-level JSON array of instance objects.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path}, got {type(data).__name__}")
    return data
