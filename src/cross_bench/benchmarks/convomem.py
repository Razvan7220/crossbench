# ruff: noqa: E501
"""ConvoMem adapter — faithful port of SalesforceAIResearch/ConvoMem.

ConvoMem (arxiv 2511.10523) is a conversational-memory benchmark over
75,336 curated Q&A pairs grouped into six evidence categories. Each pair
ships with:

* ``question`` / ``answer``
* ``message_evidences`` — the individual messages that support the answer
* ``conversations`` — the full chat transcripts that contain those messages

The six categories map 1:1 to upstream's ``AnsweringEvaluation`` objects,
which live in
``src/main/scala/com/salesforce/crmmembench/questions/evidence/generation/AnsweringEvaluation.scala``:

=============================  ==========================================
Category (HF path)             Judge evaluator (upstream object)
=============================  ==========================================
``user_evidence``              ``UserFactsAnsweringEvaluation`` — single-
                               vs. multi-evidence prompts
``assistant_facts_evidence``   ``DefaultAnsweringEvaluation``
``changing_evidence``          ``DefaultAnsweringEvaluation``
``preference_evidence``        ``RubricBasedAnsweringEvaluation``
``implicit_connection_evidence`` ``RubricBasedAnsweringEvaluation``
``abstention_evidence``        ``AbstentionAnsweringEvaluation``
=============================  ==========================================

Judging protocol (upstream
``src/main/scala/com/salesforce/crmmembench/evaluation/EvaluationUtils.scala::verifyAnswerCorrectnessWithPrompt``):

* Lowercase + ``strip`` the judge response.
* ``"right"`` and ``"wrong"`` both present → ambiguous, counted as WRONG.
* ``"right"`` only → CORRECT.
* ``"wrong"`` only → WRONG.
* Neither present → retry (up to ``ANSWER_VERIFICATION_MAX_RETRIES = 20``).
  After all retries the judgement is ``None`` (reported as WRONG here).
* Default judge model is ``Gemini.flash``, which upstream resolves to
  ``gemini-2.5-flash``. The equivalent GPT / Claude endpoints are wired
  in upstream but Gemini is the pinned default.

Dataset
    https://huggingface.co/datasets/Salesforce/ConvoMem

    Layout:
    ``core_benchmark/evidence_questions/<category>/<k>_evidence/*.json``

    Each JSON file has a single top-level key ``evidence_items`` — an array
    of objects with ``question``, ``answer``, ``message_evidences`` and
    ``conversations``.

References
    https://github.com/SalesforceAIResearch/ConvoMem
    https://arxiv.org/abs/2511.10523
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
# Upstream constants — DO NOT rename. These are contracts with
# SalesforceAIResearch/ConvoMem's Scala source.
# ---------------------------------------------------------------------------

# Canonical category names, matching the directory names on the HF dataset.
CATEGORIES: tuple[str, ...] = (
    "user_evidence",
    "assistant_facts_evidence",
    "changing_evidence",
    "abstention_evidence",
    "preference_evidence",
    "implicit_connection_evidence",
)

# ``evidence_count`` range per category, taken from the HF dataset README.
# Any k in this range must match a directory ``<k>_evidence/`` on HF.
EVIDENCE_COUNT_RANGES: dict[str, tuple[int, ...]] = {
    "user_evidence": (1, 2, 3, 4, 5, 6),
    "assistant_facts_evidence": (1, 2, 3, 4, 5, 6),
    "changing_evidence": (2, 3, 4, 5, 6),
    "abstention_evidence": (1, 2, 3),
    "preference_evidence": (1, 2),
    "implicit_connection_evidence": (1, 2, 3),
}

# Upstream's default judge. ``Gemini.flash`` in
# ``src/main/scala/com/salesforce/crmmembench/LLM_endpoints/Gemini.scala``
# is pinned to ``gemini-2.5-flash``.
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

# ``Config.Evaluation.ANSWER_VERIFICATION_MAX_RETRIES`` in Utils.scala.
DEFAULT_JUDGE_MAX_RETRIES = 20

_HF_REPO_ID = "Salesforce/ConvoMem"
_HF_BASE_DIR = "core_benchmark/evidence_questions"


# ---------------------------------------------------------------------------
# Judge prompt templates — reproduced BYTE-EXACT from upstream
# ``AnsweringEvaluation.scala``. Any wording or whitespace change here
# breaks faithfulness, so each template is covered by a regression test.
# ---------------------------------------------------------------------------

_DEFAULT_GUIDELINES = """**Crucial Guidelines for your judgment:**

1.  **Core Information is Key**: The Model's Response must contain all the **essential factual information** that directly answers the Question.
2.  **Equivalence Counts**: Phrasing doesn't need to be identical. If the Model's Response conveys the exact same core meaning and details as the Correct Answer, even if paraphrased or structured differently, consider it correct.
3.  **Superfluous (but Accurate) Information**: If the Model's Response includes additional details that were *not explicitly asked for* by the Question, but these details are **accurate and do not contradict** the Correct Answer, you should **still count it as correct** if the core question is fully answered.
4.  **Partial Answers are Incorrect**: If the Model's Response is missing any essential information directly requested by the Question (even if the Correct Answer provides more detail), it is incorrect.
5.  **Focus on the Question**: Your primary focus should be whether the Model's Response adequately addresses the Question using information that aligns with the Correct Answer. Do not penalize the model for not reiterating every single word or incidental detail from the Correct Answer if it wasn't requested."""


_RUBRIC_GUIDELINES = """Guidelines:
- The model's response must fully satisfy ALL criteria in the rubric
- Partial fulfillment of criteria should result in INCORRECT
- The response must demonstrate clear understanding of the context
- Missing any important aspect mentioned in the rubric means INCORRECT"""


_TEMPORAL_GUIDELINES = """Guidelines:
- Time calculations must be accurate within reasonable bounds
- Accept different but equivalent time expressions (3 months = 90 days = 12 weeks)
- The model must show correct temporal reasoning
- Off-by-one errors in day counts are NOT acceptable"""


_ABSTENTION_GUIDELINES = """**Critical Guidelines for Abstention Evaluation:**

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


def _default_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    """``DefaultAnsweringEvaluation.getJudgePromptTemplate`` — byte-exact."""
    return (
        "I will provide you with a **Question**, a **Correct Answer**, and a **Model's Response**. Your sole task is to determine if the Model's Response is **sufficiently correct and complete** to answer the Question, when compared against the Correct Answer.\n"
        "\n"
        f"{_DEFAULT_GUIDELINES}\n"
        "\n"
        "**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n"
        "\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Model Response: {model_answer}\n"
        "\n"
        "Answer (RIGHT/WRONG):"
    )


def _rubric_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    """``RubricBasedAnsweringEvaluation.getJudgePromptTemplate`` — byte-exact."""
    return (
        "You are a judge evaluating if a model's response meets the criteria in a rubric.\n"
        "\n"
        f"Question asked: {question}\n"
        "\n"
        f"Evaluation rubric: {correct_answer}\n"
        "\n"
        f"Model's response: {model_answer}\n"
        "\n"
        f"{_RUBRIC_GUIDELINES}\n"
        "\n"
        "Evaluate whether the model's response satisfies the rubric criteria.\n"
        "\n"
        "If the response satisfies the criteria: RIGHT\n"
        "If the response does not satisfy the criteria: WRONG\n"
        "\n"
        "Respond with only RIGHT or WRONG."
    )


def _temporal_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    """``TemporalAnsweringEvaluation.getJudgePromptTemplate`` — byte-exact."""
    return (
        "You are a judge determining if a model's temporal answer is correct.\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        f"Correct answer: {correct_answer}\n"
        "\n"
        f"Model's answer: {model_answer}\n"
        "\n"
        f"{_TEMPORAL_GUIDELINES}\n"
        "\n"
        "Temporal answers often have legitimate variations. Consider if the model's answer represents the same time period or demonstrates correct temporal reasoning.\n"
        "\n"
        "If the temporal answer is correct: RIGHT\n"
        "If the temporal answer is incorrect: WRONG\n"
        "\n"
        "Respond with only RIGHT or WRONG."
    )


def _user_facts_single_prompt(
    question: str,
    correct_answer: str,
    model_answer: str,
    evidence_messages: list[dict[str, Any]],
) -> str:
    """``UserFactsAnsweringEvaluation`` w/ ``evidenceCount == 1`` — byte-exact.

    Injects the single evidence message's ``text`` (empty string if absent).
    """
    evidence_text = ""
    if evidence_messages:
        evidence_text = str(evidence_messages[0].get("text", ""))
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


def _user_facts_multi_prompt(
    question: str,
    correct_answer: str,
    model_answer: str,
    evidence_messages: list[dict[str, Any]],
) -> str:
    """``UserFactsAnsweringEvaluation`` w/ ``evidenceCount > 1`` — byte-exact.

    Joins evidence messages with ``\\n`` as ``"Evidence Message {i}: {text}"``,
    1-indexed — mirrors upstream's ``zipWithIndex.map { ... }.mkString("\\n")``.
    """
    evidence_block = "\n".join(
        f"Evidence Message {i + 1}: {msg.get('text', '')}"
        for i, msg in enumerate(evidence_messages)
    )
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


def _abstention_prompt(
    question: str,
    correct_answer: str,  # unused — upstream drops this  # noqa: ARG001
    model_answer: str,
) -> str:
    """``AbstentionAnsweringEvaluation.getJudgePromptTemplate`` — byte-exact.

    Upstream deliberately omits ``correctAnswer`` and the evidence messages
    from this template — only the question and model's response are shown
    to the judge.
    """
    return (
        "You are evaluating an ABSTENTION test where the model should recognize it lacks sufficient information to answer the question.\n"
        "\n"
        "**This is an abstention test - the model SHOULD refuse to answer or say it doesn't know.**\n"
        "\n"
        f"{_ABSTENTION_GUIDELINES}\n"
        "\n"
        "**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n"
        "\n"
        f"Question: {question}\n"
        "Expected behavior: The model should indicate it cannot answer due to insufficient information\n"
        f"Model Response: {model_answer}\n"
        "\n"
        "Does the model appropriately abstain from answering or acknowledge insufficient information?\n"
        "\n"
        "Answer (RIGHT/WRONG):"
    )


def _build_judge_prompt(
    category: str,
    evidence_count: int,
    question: str,
    correct_answer: str,
    model_answer: str,
    evidence_messages: list[dict[str, Any]],
) -> str:
    """Route to the right judge template for ``category``.

    Mirrors the ``getAnsweringEvaluation`` override map in each
    ``*EvidenceGenerator.scala``:

    * ``user_evidence`` → ``UserFactsAnsweringEvaluation`` (single / multi)
    * ``abstention_evidence`` → ``AbstentionAnsweringEvaluation``
    * ``preference_evidence`` / ``implicit_connection_evidence``
      → ``RubricBasedAnsweringEvaluation``
    * everything else → ``DefaultAnsweringEvaluation``
    """
    if category == "abstention_evidence":
        return _abstention_prompt(question, correct_answer, model_answer)
    if category in ("preference_evidence", "implicit_connection_evidence"):
        return _rubric_prompt(question, correct_answer, model_answer)
    if category == "user_evidence":
        if evidence_count <= 1:
            return _user_facts_single_prompt(
                question, correct_answer, model_answer, evidence_messages,
            )
        return _user_facts_multi_prompt(
            question, correct_answer, model_answer, evidence_messages,
        )
    # assistant_facts_evidence, changing_evidence, and any future categories
    # that don't override ``getAnsweringEvaluation``.
    return _default_prompt(question, correct_answer, model_answer)


# ---------------------------------------------------------------------------
# Judge response parsing — mirrors
# ``EvaluationUtils.verifyAnswerCorrectnessWithPrompt``.
# ---------------------------------------------------------------------------

def _parse_judge_label(response: str) -> bool | None:
    """Parse a judge response into ``True`` / ``False`` / ``None``.

    Returns:
        * ``True``  — response contains ``"right"`` and not ``"wrong"``.
        * ``False`` — response contains ``"wrong"`` only, OR both
          ``"right"`` and ``"wrong"`` (ambiguous — upstream logs a warning
          and defaults to False).
        * ``None``  — neither token present; caller should retry.
    """
    text = response.strip().lower()
    contains_right = "right" in text
    contains_wrong = "wrong" in text
    if contains_right and contains_wrong:
        logger.warning(
            "Judge response contains both 'right' and 'wrong' (ambiguous): %r", text,
        )
        return False
    if contains_right:
        return True
    if contains_wrong:
        return False
    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class ConvoMemAdapter(ConversationalBenchmarkAdapter):
    """ConvoMem adapter — 1:1 faithful to SalesforceAIResearch/ConvoMem."""

    name = "convomem"
    granularity = TaskGranularity.CONVERSATIONAL

    def __init__(self) -> None:
        super().__init__()
        self._judge_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Task & conversation loading
    # ------------------------------------------------------------------

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        """Yield one :class:`Task` per evidence item in the selected scope.

        Each upstream ``evidence_item`` becomes a ``Task`` whose
        ``conversation_id`` is the same as ``task_id`` so the engine's
        per-conversation isolation corresponds to upstream's per-item
        evaluation.
        """
        self._judge_config = params.get("judge", {})

        filter_categories = set(params.get("filter_categories", []) or [])
        filter_evidence_counts = set(params.get("filter_evidence_counts", []) or [])
        limit = params.get("limit")

        count = 0
        for category, evidence_count, item_idx, item in self._iter_evidence_items(params):
            if filter_categories and category in filter_categories:
                continue
            if filter_evidence_counts and evidence_count not in filter_evidence_counts:
                continue
            if limit is not None and count >= limit:
                return

            qid = self._make_task_id(category, evidence_count, item_idx, item)
            evidence_messages = list(item.get("message_evidences", []) or [])
            conversations = list(item.get("conversations", []) or [])

            yield Task(
                task_id=qid,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=str(item.get("question", "")),
                metadata={
                    "conversation_id": qid,
                    "category": category,
                    "evidence_count": evidence_count,
                    "message_evidences": evidence_messages,
                    "conversations": conversations,
                    "is_abstention": category == "abstention_evidence",
                },
                ground_truth=str(item.get("answer", "")),
            )
            count += 1

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Surface one conversation per evidence item for memory ingestion.

        The ``conversations`` field of each upstream ``evidence_item`` is a
        list of turn-sequences (``[{messages: [{speaker, text}, ...]}, ...]``).
        We flatten all of them in order into a single per-item conversation
        keyed by ``conversation_id == task_id``.
        """
        conversations: list[dict[str, Any]] = []
        for category, evidence_count, item_idx, item in self._iter_evidence_items(params):
            qid = self._make_task_id(category, evidence_count, item_idx, item)
            messages: list[dict[str, Any]] = []
            for conv in item.get("conversations", []) or []:
                for turn_idx, turn in enumerate(conv.get("messages", []) or []):
                    speaker = turn.get("speaker", "") or ""
                    messages.append({
                        # Map upstream ``speaker`` ("User" / "Assistant") onto
                        # the engine's speaker fields, preserving casing.
                        "speaker_id": speaker,
                        "speaker_name": speaker,
                        "content": turn.get("text", ""),
                        "timestamp": "",
                        "turn_index": turn_idx,
                    })
            conversations.append({
                "conversation_id": qid,
                "messages": messages,
                "metadata": {
                    "category": category,
                    "evidence_count": evidence_count,
                },
            })
        return conversations

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """Return just the question text.

        Upstream's ``EvaluationUtils.getModelAnswerPrompt`` wraps the question
        in a conversation-context block, but that's meant for long-context
        baselines that fetch all history themselves. In CrossBench the memory
        system retrieves the relevant context, so we hand the reader only
        the question — matching the ``no-retrieval`` branch used by memory
        systems in the upstream harness.
        """
        return task.prompt

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    async def judge_answer(
        self,
        task: Task,
        generated_answer: str,
    ) -> dict[str, Any]:
        """Judge ``generated_answer`` via the upstream LLM judge protocol.

        Sends a single request to the configured LLM at ``temperature=0``,
        parses ``"right"`` / ``"wrong"`` from the lowered response, and
        retries on invalid responses up to ``DEFAULT_JUDGE_MAX_RETRIES``.
        The judgement dict mirrors upstream's ``verifyAnswerCorrectness``
        semantics plus surfacing the category for downstream metrics.
        """
        category = task.metadata.get("category", "")
        evidence_count = int(task.metadata.get("evidence_count", 0) or 0)
        question = task.prompt
        golden = str(task.ground_truth or "")
        evidence_messages = list(task.metadata.get("message_evidences", []) or [])

        prompt = _build_judge_prompt(
            category=category,
            evidence_count=evidence_count,
            question=question,
            correct_answer=golden,
            model_answer=generated_answer,
            evidence_messages=evidence_messages,
        )

        max_retries = int(
            self._judge_config.get("max_retries", DEFAULT_JUDGE_MAX_RETRIES)
        )
        judge_model = self._judge_config.get("model", DEFAULT_JUDGE_MODEL)
        provider = (self._judge_config.get("provider") or "").strip().lower()

        label: bool | None = None
        last_response = ""
        last_error: str | None = None
        attempts = 0
        for attempts_i in range(max(1, max_retries)):
            attempts = attempts_i + 1
            try:
                raw = await self._call_judge(prompt)
            except Exception as exc:  # noqa: BLE001 - surface judge failures
                last_error = str(exc)
                logger.warning("ConvoMem judge call failed (%s): %s", provider or "auto", exc)
                continue
            last_response = raw
            parsed = _parse_judge_label(raw)
            if parsed is not None:
                label = parsed
                break

        correct = bool(label) if label is not None else False
        return {
            "pass": correct,
            "correct": int(correct),
            "label": correct,
            "judge_label": label,  # True / False / None
            "judge_response": last_response,
            "judge_model": judge_model,
            "judge_attempts": attempts,
            "judge_error": last_error,
            "category": category,
            "evidence_count": evidence_count,
            "is_abstention": category == "abstention_evidence",
        }

    async def _call_judge(self, prompt: str) -> str:
        """Dispatch to the configured judge provider.

        Default provider is inferred from the model name: ``gemini*`` →
        Google, ``claude*`` → Anthropic, anything else (including empty) →
        OpenAI. Override via ``judge.provider`` in the config.
        """
        provider = (self._judge_config.get("provider") or "").strip().lower()
        model = self._judge_config.get("model", DEFAULT_JUDGE_MODEL)
        if not provider:
            if model.startswith("gemini"):
                provider = "google"
            elif model.startswith("claude"):
                provider = "anthropic"
            else:
                provider = "openai"

        if provider == "google":
            return await self._call_judge_google(prompt)
        if provider == "anthropic":
            return await self._call_judge_anthropic(prompt)
        return await self._call_judge_openai(prompt)

    async def _call_judge_google(self, prompt: str) -> str:
        from google import genai  # type: ignore[import-not-found]

        model = self._judge_config.get("model", DEFAULT_JUDGE_MODEL)
        api_key = self._judge_config.get("api_key") or os.environ.get(
            "GOOGLE_AI_API_KEY", ""
        ) or os.environ.get("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model, contents=prompt,
        )
        return (getattr(response, "text", "") or "").strip()

    async def _call_judge_anthropic(self, prompt: str) -> str:
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        model = self._judge_config.get("model", "claude-haiku-4-5")
        api_key = self._judge_config.get("api_key") or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        blocks = response.content or []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                return text.strip()
        return ""

    async def _call_judge_openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        model = self._judge_config.get("model", DEFAULT_JUDGE_MODEL)
        api_key = self._judge_config.get("api_key") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        base_url = self._judge_config.get("base_url") or os.environ.get(
            "LLM_BASE_URL"
        ) or None
        organization = self._judge_config.get("organization") or os.environ.get(
            "OPENAI_ORGANIZATION"
        ) or None
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        client = AsyncOpenAI(**client_kwargs)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict[str, Any]:
        return {
            "data_path": "data/convomem",
            "filter_categories": [],
            "filter_evidence_counts": [],
            "judge": {
                "provider": "google",
                "model": DEFAULT_JUDGE_MODEL,
                "api_key": os.environ.get("GOOGLE_AI_API_KEY", ""),
                "max_retries": DEFAULT_JUDGE_MAX_RETRIES,
            },
        }

    # ------------------------------------------------------------------
    # Dataset traversal
    # ------------------------------------------------------------------

    def _iter_evidence_items(
        self, params: dict[str, Any],
    ) -> Iterator[tuple[str, int, int, dict[str, Any]]]:
        """Iterate ``(category, evidence_count, item_idx, item)`` tuples.

        Walks ``core_benchmark/evidence_questions/<category>/<k>_evidence``
        for each of the six upstream categories, yielding every
        ``evidence_item`` in every JSON file found.
        """
        categories = params.get("categories") or list(CATEGORIES)
        for category in categories:
            if category not in CATEGORIES:
                raise ValueError(
                    f"Unknown category {category!r}; must be one of {list(CATEGORIES)}"
                )
            valid_ks = EVIDENCE_COUNT_RANGES[category]
            evidence_counts = params.get("evidence_counts") or list(valid_ks)
            for k in evidence_counts:
                if k not in valid_ks:
                    # Silently skip combinations upstream doesn't ship (e.g.
                    # ``preference_evidence/3_evidence`` doesn't exist).
                    continue
                for file_path in self._resolve_evidence_files(params, category, k):
                    payload = _load_evidence_file(file_path)
                    for idx, item in enumerate(payload):
                        yield category, k, idx, item

    def _resolve_evidence_files(
        self, params: dict[str, Any], category: str, evidence_count: int,
    ) -> list[Path]:
        """Resolve JSON files for a ``<category>/<k>_evidence`` bucket.

        Resolution order:

        1. ``{data_path}/core_benchmark/evidence_questions/<category>/<k>_evidence/*.json``
        2. ``{data_path}/evidence_questions/<category>/<k>_evidence/*.json``
           (allows users to mirror only the ``evidence_questions`` subtree).

        If ``data_path`` itself exists, it is treated as authoritative — a
        missing subdirectory just means the user downloaded a subset, so we
        return ``[]`` and let the caller skip. We only reach HuggingFace if
        ``data_path`` does not exist at all; in that case we snapshot the
        requested category+k from ``Salesforce/ConvoMem`` and retry locally.
        """
        data_path = Path(params.get("data_path", "data/convomem"))
        subpath = f"{category}/{evidence_count}_evidence"

        def _files_for(base: Path) -> list[Path]:
            primary = base / _HF_BASE_DIR / subpath
            if primary.is_dir():
                return sorted(primary.glob("*.json"))
            alt = base / "evidence_questions" / category / f"{evidence_count}_evidence"
            if alt.is_dir():
                return sorted(alt.glob("*.json"))
            return []

        # If the user has provided a real data_path, honour it exactly — even
        # if some subdirectories are missing (they may have downloaded only a
        # subset). This prevents spurious HF calls during partial-mirror runs.
        if data_path.exists():
            return _files_for(data_path)

        # data_path does not exist — fall back to HuggingFace.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"ConvoMem data for {subpath!r} not found under {data_path!r} and "
                f"huggingface_hub is not installed. Either install it "
                f"(`pip install huggingface_hub`) or download "
                f"https://huggingface.co/datasets/{_HF_REPO_ID}/tree/main/{_HF_BASE_DIR}/{subpath}"
            ) from exc

        allow_patterns = [f"{_HF_BASE_DIR}/{subpath}/*.json"]
        local_root = snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=allow_patterns,
        )
        return _files_for(Path(local_root))

    @staticmethod
    def _make_task_id(
        category: str, evidence_count: int, item_idx: int, item: dict[str, Any],
    ) -> str:
        """Build a stable task id from category/k/index.

        Upstream files don't ship a canonical item id, so we synthesize one
        that's deterministic across runs: ``<category>/<k>_evidence/<i>``.
        If the evidence item happens to carry an ``id`` or ``question_id``
        field we honour that instead.
        """
        explicit = item.get("id") or item.get("question_id")
        if explicit:
            return str(explicit)
        return f"{category}/{evidence_count}_evidence/{item_idx}"


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_evidence_file(path: Path) -> list[dict[str, Any]]:
    """Parse a ConvoMem evidence JSON file into a list of evidence items.

    Upstream files have a single top-level ``evidence_items`` array.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "evidence_items" in data:
        items = data["evidence_items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(
            f"Expected a dict with 'evidence_items' or a JSON array at {path}, "
            f"got {type(data).__name__}"
        )
    if not isinstance(items, list):
        raise ValueError(
            f"'evidence_items' at {path} must be a list, got {type(items).__name__}"
        )
    return items
