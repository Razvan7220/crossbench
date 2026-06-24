# ruff: noqa: E501
"""LoCoMo (original) adapter — 1:1 faithful port of snap-research/locomo.

This is the **paper-faithful** LoCoMo evaluator: Porter-stemmed token F1,
all five reasoning categories scored, byte-exact upstream prompts
(including the upstream typo ``wriiten``). It exists alongside
:mod:`cross_bench.benchmarks.locomo_everos`, which implements the
EverMemOS/EverOS-style variant (LLM-as-judge, 3-run mean$\\pm$std, 30-s
timestamp interpolation, per-system answer prompts, cat 5 excluded).

LoCoMo ("Long Conversational Memory", ACL 2024, Maharana et al., arXiv
2402.17753) is a long-term conversational memory benchmark over 10
human-annotated conversations with ~200 QA pairs each. Questions span the
five reasoning types defined by the paper: **single-hop**, **multi-hop**,
**temporal**, **commonsense / world knowledge**, and **adversarial**.

Upstream stores the integer ``category`` label on every QA item; this
adapter uses the same integer codes everywhere and maps to the paper's
descriptive names via :data:`CATEGORY_NAMES`:

=====  =====================  ===============================================
Code   Name                   Evaluation (``task_eval/evaluation.py``)
=====  =====================  ===============================================
1      ``multi_hop``          multi-answer F1 (comma-split, stemmed)
2      ``temporal``           single-phrase F1 + date hint in prompt
3      ``commonsense``        single-phrase F1 + ``answer.split(';')[0]``
4      ``single_hop``         single-phrase F1
5      ``adversarial``        substring check for "no information available"
                              or "not mentioned" in the response
=====  =====================  ===============================================

Prompt templates and scoring are reproduced byte-for-byte from
``task_eval/gpt_utils.py`` and ``task_eval/evaluation.py`` respectively, and
are exercised by the regression tests in ``tests/benchmarks/test_locomo_original.py``.

Dataset
    The canonical file is ``data/locomo10.json`` in the upstream GitHub repo
    (`snap-research/locomo`_). There is no HuggingFace mirror — this adapter
    falls back to the GitHub raw URL if the file cannot be found locally.

References
    * Paper: https://arxiv.org/abs/2402.17753
    * Code:  https://github.com/snap-research/locomo
    * Data:  https://github.com/snap-research/locomo/blob/main/data/locomo10.json

.. _snap-research/locomo: https://github.com/snap-research/locomo
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cross_bench.benchmarks.base import (
    MEMORY_TOKEN_SCHEMA_KEYS,
    ConversationalBenchmarkAdapter,
)
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream constants — DO NOT rename or renumber. These are contracts with
# snap-research/locomo's task_eval/evaluation.py and task_eval/gpt_utils.py.
# ---------------------------------------------------------------------------

# Integer category codes, matching ``qa[i]["category"]`` in
# ``data/locomo10.json``. Descriptive names follow the paper (Maharana et
# al., ACL 2024, §3 "LoCoMo Evaluation Benchmark").
CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "commonsense",
    4: "single_hop",
    5: "adversarial",
}

# Canonical dataset URL. The file is 2.8 MB as of upstream commit
# 91c1069 (2024-08-13) and hasn't changed since release.
_GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

_DEFAULT_DATA_FILE = "locomo10.json"


# ---------------------------------------------------------------------------
# Prompt templates — reproduced BYTE-EXACT from
# ``task_eval/gpt_utils.py``. Every whitespace / typo (including the
# upstream typo "wriiten") is preserved on purpose and covered by a
# regression test in ``tests/benchmarks/test_locomo_original.py``.
# ---------------------------------------------------------------------------

# ``CONV_START_PROMPT`` in gpt_utils.py, line 51. Note the upstream typo
# "wriiten" — intentional, do not fix.
_CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. The conversation "
    "takes place over multiple days and the date of each conversation is "
    "wriiten at the beginning of the conversation.\n\n"
)

# ``QA_PROMPT`` in gpt_utils.py, lines 25-29. Used for categories 1-4.
_QA_PROMPT = (
    "\n"
    "Based on the above context, write an answer in the form of a short "
    "phrase for the following question. Answer with exact words from the "
    "context whenever possible.\n"
    "\n"
    "Question: {} Short answer:\n"
)

# ``QA_PROMPT_CAT_5`` in gpt_utils.py, lines 31-35. Used for category 5.
_QA_PROMPT_CAT_5 = (
    "\n"
    "Based on the above context, answer the following question.\n"
    "\n"
    "Question: {} Short answer:\n"
)

# Category-2 date hint — appended to the question in
# ``gpt_utils.py::get_gpt_answers``, line 244.
_CAT2_DATE_HINT = " Use DATE of CONVERSATION to answer with an approximate date."

# Category-5 MCQ template — ``gpt_utils.py::get_gpt_answers``, line 246.
_CAT5_MCQ_TEMPLATE = " Select the correct answer: (a) {} (b) {}. "

# The abstention option text presented alongside the (bogus) ground-truth
# answer in the cat-5 MCQ. Grading later scans for the substrings
# ``"no information available"`` / ``"not mentioned"`` in the model's
# response — ``evaluation.py::eval_question_answering``, lines 217-221.
_CAT5_ABSTENTION_OPTION = "Not mentioned in the conversation"


# ---------------------------------------------------------------------------
# F1 / normalization — reproduced from ``task_eval/evaluation.py``.
# ---------------------------------------------------------------------------

# Upstream articles list — NOTE: includes "and", which is non-standard SQuAD
# F1 but matches LoCoMo upstream exactly (evaluation.py, line 80).
#
# Upstream uses the third-party ``regex`` package (``regex.sub``). We match
# that import for two reasons:
#   1. Byte-for-byte fidelity: if upstream ever extends ``remove_articles``
#      to include Unicode tokens, the engines' ``\b`` semantics would diverge
#      (regex defaults to Unicode word chars; stdlib re is ASCII-leaning).
#   2. Upstream's ``evaluation.py`` already imports ``regex`` for its
#      ``SimpleTokenizer`` (Unicode ``\p{L}\p{N}\p{M}`` classes), so using
#      it here mirrors their dependency graph.
# The ``regex`` pin in ``pyproject.toml``'s ``benchmark-locomo`` extras
# mirrors upstream's ``requirements.txt``. We lazy-load so that tests and
# code paths that don't call ``_normalize_answer`` stay importable without
# the extra installed — same pattern as :func:`_porter_stemmer` for nltk.
_ARTICLES_RE: Any | None = None


def _get_articles_re() -> Any:
    """Compile the upstream article-stripping regex using ``regex`` (not ``re``).

    Imports lazily so ``pip install cross-bench`` without the
    ``benchmark-locomo`` extra still loads this module. Raises a clear
    error only when a caller actually tries to normalize an answer.
    """
    global _ARTICLES_RE
    if _ARTICLES_RE is not None:
        return _ARTICLES_RE
    try:
        import regex  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised by CI env check
        raise ImportError(
            "LoCoMo's normalize_answer uses the upstream `regex` package. "
            "Install with `pip install regex>=2022.10.31` or `pip install "
            "cross-bench[benchmark-locomo]`."
        ) from exc
    _ARTICLES_RE = regex.compile(r"\b(a|an|the|and)\b")
    return _ARTICLES_RE


def _normalize_answer(s: str) -> str:
    """Upstream ``evaluation.py::normalize_answer`` — byte-exact.

    Operations, in order:

    1. Drop commas (``s.replace(',', '')``). Redundant because
       :func:`string.punctuation` includes ``,`` and gets stripped below, but
       upstream applies both so we do too.
    2. ``lower()``.
    3. Strip ``string.punctuation`` (all ASCII punctuation).
    4. Remove the articles ``a``, ``an``, ``the``, ``and`` (word-boundary
       match) using the upstream ``regex`` engine (see :func:`_get_articles_re`).
    5. Collapse runs of whitespace.
    """
    s = s.replace(",", "")
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = _get_articles_re().sub(" ", s)
    s = " ".join(s.split())
    return s


def _porter_stemmer() -> Any:
    """Return a Porter-stemmer instance, preferring NLTK.

    Upstream uses ``nltk.stem.PorterStemmer``. If NLTK isn't installed we
    raise a clear error rather than silently producing a less-faithful F1;
    see ``benchmark-locomo`` in ``pyproject.toml``.
    """
    try:
        from nltk.stem import PorterStemmer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised by CI env check
        raise ImportError(
            "LoCoMo's F1 metric uses the Porter stemmer from NLTK. "
            "Install with `pip install nltk>=3.8` or `pip install "
            "cross-bench[benchmark-locomo]`."
        ) from exc
    return PorterStemmer()


def _tokenize_and_stem(text: str, stemmer: Any) -> list[str]:
    """Normalize then Porter-stem — mirrors upstream's comprehension."""
    return [stemmer.stem(w) for w in _normalize_answer(text).split()]


def _f1_score(prediction: str, ground_truth: str, stemmer: Any | None = None) -> float:
    """Single-phrase F1 — ``evaluation.py::f1_score``.

    Porter-stemmed token overlap; returns 0.0 when the token sets are
    disjoint. Division is float throughout (matches upstream's
    ``1.0 * num_same / len(tokens)``).
    """
    stemmer = stemmer or _porter_stemmer()
    pred_tokens = _tokenize_and_stem(prediction, stemmer)
    gt_tokens = _tokenize_and_stem(ground_truth, stemmer)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_multi(prediction: str, ground_truth: str, stemmer: Any | None = None) -> float:
    """Multi-answer F1 — ``evaluation.py::f1``.

    Splits *both* the prediction and the ground truth on commas, then for
    each ground-truth fragment takes the *max* F1 across all prediction
    fragments, and returns the mean across ground-truth fragments. Used
    exclusively for category 1 (multi-hop).
    """
    stemmer = stemmer or _porter_stemmer()
    preds = [p.strip() for p in prediction.split(",")]
    gts = [g.strip() for g in ground_truth.split(",")]
    if not gts:
        return 0.0
    per_gt = [
        max(_f1_score(p, gt, stemmer=stemmer) for p in preds) if preds else 0.0
        for gt in gts
    ]
    return sum(per_gt) / len(per_gt)


_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:final\s*answer|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _extract_final_answer(text: str) -> str:
    """Strip CoT preamble; keep only the trailing ``FINAL ANSWER:`` payload.

    Memory-system answer prompts (mem0, evermemos) require a multi-step CoT
    block before a single-line ``FINAL ANSWER: <terse>``. Upstream LoCoMo's
    F1 scorer is token-overlap over the *entire* prediction, so the verbose
    CoT preamble dominates the denominator and crashes precision (e.g. one
    correct token in a 200-token CoT response yields F1 ~0.01).

    Behaviour
        * If a ``FINAL ANSWER:`` (or ``ANSWER:`` / ``## FINAL ANSWER:``)
          marker exists, return everything after the *last* such marker,
          stripped.
        * Otherwise return ``text`` unchanged — preserves upstream-faithful
          behaviour for raw single-shot completions that never produced a
          marker.
    """
    if not text:
        return text
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return text
    return matches[-1].group(1).strip()


def _adversarial_correct(output: str) -> bool:
    """Category-5 check — ``evaluation.py::eval_question_answering`` cat-5 branch.

    Upstream: ``'no information available' in output.lower() or 'not
    mentioned' in output.lower()``. Either substring triggers a RIGHT label.
    """
    low = output.lower()
    return "no information available" in low or "not mentioned" in low


def _evidence_recall(
    retrieved_context_ids: list[str] | None,
    evidence: list[str],
) -> float:
    """Upstream ``evaluation.py::eval_question_answering`` recall branch — lines 228-237.

    Upstream::

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    Behavior:

    * When no retrieved IDs are provided or ``evidence`` is empty, returns
      ``1.0`` (matches upstream's ``all_recall.append(1)`` default).
    * Session mode — retrieved IDs start with ``S`` (e.g. ``["S1", "S3"]``).
      Session numbers are extracted by stripping the leading ``S``; each
      piece of evidence is an ID like ``"D1:3"`` whose leading ``D`` is
      stripped and split on ``:`` to yield its session number.
    * Dialog mode — retrieved IDs are dialog labels (e.g. ``"D1:1"``). Each
      evidence item is checked for literal membership.

    The engine/middleware populates ``task.metadata["retrieved_context_ids"]``
    before :meth:`LoCoMoOriginalAdapter.judge_answer` is called. When the
    pipeline does not expose retrieval IDs, recall is reported as 1.0 so
    aggregation matches upstream's no-RAG default.
    """
    if not retrieved_context_ids or not evidence:
        return 1.0
    first = retrieved_context_ids[0]
    if isinstance(first, str) and first.startswith("S"):
        sessions = [e[1:] for e in retrieved_context_ids if isinstance(e, str)]
        matches = sum(
            1 for ev in evidence
            if isinstance(ev, str) and ev.split(":")[0][1:] in sessions
        )
    else:
        matches = sum(1 for ev in evidence if ev in retrieved_context_ids)
    return float(matches) / len(evidence)


def _score_qa(
    category: int,
    prediction: str,
    answer: str,
    stemmer: Any | None = None,
) -> float:
    """Dispatch to the correct per-category metric.

    Reproduces the branch logic at
    ``evaluation.py::eval_question_answering``, lines 197-224:

    * cat 3 — ``answer = answer.split(';')[0].strip()`` before grading.
    * cat 1 — :func:`_f1_multi`.
    * cat 2, 3, 4 — :func:`_f1_score`.
    * cat 5 — 1.0 iff :func:`_adversarial_correct`, else 0.0.
    """
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return _f1_multi(prediction, answer, stemmer=stemmer)
    if category in (2, 3, 4):
        return _f1_score(prediction, answer, stemmer=stemmer)
    if category == 5:
        return 1.0 if _adversarial_correct(prediction) else 0.0
    raise ValueError(f"Unknown LoCoMo category: {category!r}")


# ---------------------------------------------------------------------------
# Prompt assembly helpers — byte-exact with ``task_eval/gpt_utils.py``.
# ---------------------------------------------------------------------------

def _format_turn(dialog: dict[str, Any]) -> str:
    """Render one dialog turn — ``gpt_utils.py::get_input_context``.

    Upstream (lines 182-187)::

        turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"' + '\\n'
        if "blip_caption" in dialog:
            turn += ' and shared %s.' % dialog["blip_caption"]
        turn += '\\n'
    """
    speaker = dialog.get("speaker", "")
    text = dialog.get("text", "")
    turn = f'{speaker} said, "{text}"\n'
    if "blip_caption" in dialog:
        turn += f' and shared {dialog["blip_caption"]}.'
    turn += "\n"
    return turn


def _session_keys_sorted(conversation: dict[str, Any]) -> list[int]:
    """Return sorted session numbers present in ``conversation``.

    Mirrors upstream's ``[int(k.split('_')[-1]) for k in data.keys() if
    'session' in k and 'date_time' not in k]`` (gpt_utils.py, line 177).
    """
    nums: list[int] = []
    for key in conversation.keys():
        if not isinstance(key, str):
            continue
        m = re.match(r"^session_(\d+)$", key)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def _build_conversation_context(conversation: dict[str, Any]) -> str:
    """Reproduce ``gpt_utils.py::get_input_context``'s *untruncated* output.

    Upstream iterates sessions in ascending order and uses a prepend
    strategy so it can drop older turns first when a token budget is hit.
    With no budget (our adapter never truncates — memory systems retrieve
    their own context), the result is that sessions appear in **descending**
    chronological order and turns within a session in original (ascending)
    order. This function replicates that exact byte layout so long-context
    baselines (``prompt_style == "full_history"``) see the same prompt the
    upstream harness produces.
    """
    query_conv = ""
    session_nums = _session_keys_sorted(conversation)
    if not session_nums:
        return query_conv
    for i in range(min(session_nums), max(session_nums) + 1):
        sess_key = f"session_{i}"
        if sess_key not in conversation:
            continue
        turns = conversation[sess_key]
        if not isinstance(turns, list):
            continue
        query_conv += "\n\n"
        for dialog in turns[::-1]:
            if not isinstance(dialog, dict):
                continue
            query_conv = _format_turn(dialog) + query_conv
        date_time = conversation.get(f"{sess_key}_date_time", "")
        query_conv = f"DATE: {date_time}\nCONVERSATION:\n" + query_conv
    return query_conv


def _speaker_pair(conversation: dict[str, Any]) -> tuple[str, str]:
    """Derive (speaker_a_name, speaker_b_name) — ``gpt_utils.py`` line 214.

    Upstream: ``speakers_names = list(set([d['speaker'] for d in
    in_data['conversation']['session_1']]))``. Order is nondeterministic in
    upstream (set iteration); we prefer explicit ``speaker_a``/``speaker_b``
    keys when present for stability.
    """
    a = conversation.get("speaker_a")
    b = conversation.get("speaker_b")
    if a and b:
        return str(a), str(b)
    session_1 = conversation.get("session_1") or []
    names: list[str] = []
    for turn in session_1:
        if isinstance(turn, dict):
            name = turn.get("speaker")
            if name and name not in names:
                names.append(str(name))
    if len(names) >= 2:
        return names[0], names[1]
    if len(names) == 1:
        return names[0], ""
    return "", ""


def _cat5_mcq(
    question: str,
    answer: str,
    seed: str,
) -> tuple[str, dict[str, str]]:
    """Deterministic category-5 MCQ conversion.

    Upstream (gpt_utils.py, lines 246-256) uses ``random.random() < 0.5`` to
    decide whether the abstention option is (a) or (b). For reproducibility
    we seed a ``random.Random`` from ``seed`` (typically the task_id) so the
    choice is stable across runs.

    Returns ``(rewritten_question, answer_key)`` where ``answer_key`` maps
    ``{"a": option_a, "b": option_b}``.
    """
    rnd = random.Random(seed)
    if rnd.random() < 0.5:
        q = question + _CAT5_MCQ_TEMPLATE.format(_CAT5_ABSTENTION_OPTION, answer)
        key = {"a": _CAT5_ABSTENTION_OPTION, "b": answer}
    else:
        q = question + _CAT5_MCQ_TEMPLATE.format(answer, _CAT5_ABSTENTION_OPTION)
        key = {"b": _CAT5_ABSTENTION_OPTION, "a": answer}
    return q, key


def _cat5_answer_key(answer: str, seed: str) -> dict[str, str]:
    """Reconstruct the ``_cat5_mcq`` answer key without re-building the prompt.

    ``_cat5_mcq`` is deterministic from ``seed`` (the task_id), so the same
    inputs regenerate the same ``{"a": ..., "b": ...}`` mapping. We expose
    this as a standalone helper so the judge can recover the mapping without
    round-tripping the question through prompt assembly.
    """
    rnd = random.Random(seed)
    if rnd.random() < 0.5:
        return {"a": _CAT5_ABSTENTION_OPTION, "b": answer}
    return {"b": _CAT5_ABSTENTION_OPTION, "a": answer}


def _get_cat_5_answer(model_prediction: str, answer_key: dict[str, str]) -> str:
    """Upstream ``gpt_utils.py::get_cat_5_answer`` — byte-exact port.

    Resolves terse MCQ replies (``"a"`` / ``"b"`` / ``"(a)"`` / ``"(b)"``) to
    the full option text BEFORE the cat-5 substring check runs. Called once
    per cat-5 QA at upstream gpt_utils.py line 292 (batch_size=1) and at
    lines 327, 338, 344 (batch mode).

    Upstream::

        def get_cat_5_answer(model_prediction, answer_key):
            model_prediction = model_prediction.strip().lower()
            if len(model_prediction) == 1:
                if 'a' in model_prediction: return answer_key['a']
                else:                       return answer_key['b']
            elif len(model_prediction) == 3:
                if '(a)' in model_prediction: return answer_key['a']
                else:                          return answer_key['b']
            else:
                return model_prediction

    Note the subtle "in" checks — upstream does NOT require the prediction
    to be exactly "a" or "(a)"; anything of length 1 containing "a" maps to
    option a, anything of length 3 containing "(a)" maps to option a, and
    otherwise the raw prediction is returned. Preserved verbatim.
    """
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if "a" in model_prediction:
            return answer_key["a"]
        return answer_key["b"]
    if len(model_prediction) == 3:
        if "(a)" in model_prediction:
            return answer_key["a"]
        return answer_key["b"]
    return model_prediction


def _rewrite_question(
    category: int,
    question: str,
    answer: str,
    task_id: str,
) -> tuple[str, dict[str, str] | None]:
    """Apply upstream's per-category question rewrites.

    * cat 2 — append :data:`_CAT2_DATE_HINT`.
    * cat 5 — MCQ conversion via :func:`_cat5_mcq`.
    * all others — return the question verbatim.
    """
    if category == 2:
        return question + _CAT2_DATE_HINT, None
    if category == 5:
        q, key = _cat5_mcq(question, answer, seed=task_id)
        return q, key
    return question, None


def _build_qa_prompt(category: int, question: str) -> str:
    """Select the QA prompt template — ``gpt_utils.py::get_gpt_answers`` l.285."""
    template = _QA_PROMPT_CAT_5 if category == 5 else _QA_PROMPT
    return template.format(question)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class LoCoMoOriginalAdapter(ConversationalBenchmarkAdapter):
    """LoCoMo (original) adapter — 1:1 faithful to snap-research/locomo.

    See :mod:`cross_bench.benchmarks.locomo_everos` for the EverMemOS-style
    variant that targets comparability with Mem0 / EverOS leaderboards
    (LLM-as-judge, 3-run mean$\\pm$std, 30-s timestamps, per-system prompts).
    """

    name = "locomo_original"
    granularity = TaskGranularity.CONVERSATIONAL

    def __init__(self) -> None:
        super().__init__()
        self._judge_config: dict[str, Any] = {}
        self._cached_samples: list[dict[str, Any]] | None = None
        self._cached_key: str | None = None

    # ------------------------------------------------------------------
    # Metric schema — declared once, applied to every memory system
    # ------------------------------------------------------------------

    def metrics_schema(self) -> list[str]:
        """The canonical metric set every LoCoMo run reports.

        Locked at the benchmark level so summary.json has the same
        columns regardless of which memory system was stacked. The
        engine's aggregator pre-seeds these with ``None`` and overlays
        averages computed from task-level evaluation dicts where
        present.

        Three groups, all required for cross-memory-system parity:

        * **Grading** — ``pass_rate`` (primary), ``pass_rate_f1``
          (Porter-stemmed F1), ``pass_rate_llm_judge`` (Mem0-paper J
          score, only populated when ``judge.llm_judge.enabled = true``),
          ``avg_f1``, ``avg_evidence_recall``.
        * **Latency** — ``avg_search_latency_ms`` (memory pre_prompt),
          ``avg_inference_latency_ms`` (answer-LLM call),
          ``avg_llm_judge_latency_ms`` (judge call, optional),
          ``avg_total_latency_ms`` (search + inference + judge — the
          end-to-end wall time).
        * **Cost / scale** — ``avg_retrieval_tokens``,
          ``avg_scaffolding_tokens``, ``avg_tokens_per_task``,
          ``total_tasks``.

        Adding a metric here is a public API change: every leaderboard
        consumer must accept the new column. Removing one breaks
        downstream dashboards. Edit deliberately, in lockstep with
        deviations.md.
        """
        return [
            # Grading
            "pass_rate", "pass_rate_f1", "pass_rate_llm_judge",
            "accuracy",
            "avg_f1", "avg_evidence_recall", "avg_evidence_recall_measured",
            "avg_llm_judge_correct", "avg_llm_judge_num_correct",
            "avg_llm_judge_num_runs",
            # Latency (ms)
            "avg_search_latency_ms",
            "avg_inference_latency_ms",
            "avg_llm_judge_latency_ms",
            "avg_total_latency_ms",
            # Cost / scale
            "avg_retrieval_tokens",
            "avg_scaffolding_tokens",
            "total_tokens_in",
            "total_tokens_out",
            "avg_tokens_per_task",
            "total_tasks",
            # Memory-system internal token cost (separate from bench LLM)
            *MEMORY_TOKEN_SCHEMA_KEYS,
            # Run health
            "passed", "failed", "errors", "timeouts",
        ]

    # ------------------------------------------------------------------
    # Task & conversation loading
    # ------------------------------------------------------------------

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        """Yield one :class:`Task` per QA pair.

        Each ``qa`` item becomes a Task with ``conversation_id`` set to the
        sample's ``sample_id`` so the engine's per-conversation isolation
        matches upstream's per-``sample_id`` evaluation.

        Upstream does not assign QA ids; we synthesize ``{sample_id}_q{idx}``
        where ``idx`` is the 0-based index within the sample's ``qa`` list.
        """
        self._judge_config = params.get("judge", {})

        filter_categories = set(params.get("filter_categories", []) or [])
        filter_conv_ids = {
            str(c) for c in (params.get("filter_conversation_ids", []) or [])
        }
        limit = params.get("limit")
        prompt_style = params.get("prompt_style", "question")

        samples = self._load_dataset(params)
        count = 0
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            if filter_conv_ids and sample_id not in filter_conv_ids:
                continue
            conversation = sample.get("conversation", {}) or {}
            qa_list = sample.get("qa", []) or []

            for q_idx, qa in enumerate(qa_list):
                category_raw = qa.get("category", 0)
                try:
                    category = int(category_raw)
                except (TypeError, ValueError):
                    category = 0
                if filter_categories and category in filter_categories:
                    continue
                if limit is not None and count >= limit:
                    return

                question = str(qa.get("question", ""))
                # Upstream stores ``answer`` as str OR int (e.g. ``2022``).
                # Keep int->str conversion faithful to
                # ``evaluation.py::eval_question_answering`` line 199-202.
                answer_raw = qa.get("answer", "")
                answer = "" if answer_raw is None else str(answer_raw)
                evidence = list(qa.get("evidence", []) or [])

                task_id = f"{sample_id}_q{q_idx}"
                yield Task(
                    task_id=task_id,
                    benchmark=self.name,
                    granularity=self.granularity,
                    prompt=question,
                    metadata={
                        # conversation_id == sample_id: the memory store is
                        # shared across all QA pairs for a single sample.
                        "conversation_id": sample_id,
                        "sample_id": sample_id,
                        "qa_index": q_idx,
                        "category": category,
                        "category_name": CATEGORY_NAMES.get(category, "unknown"),
                        "evidence": evidence,
                        "prompt_style": prompt_style,
                        "conversation": conversation,
                        "speaker_a": conversation.get("speaker_a", ""),
                        "speaker_b": conversation.get("speaker_b", ""),
                    },
                    ground_truth=answer,
                )
                count += 1

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Surface one conversation per sample for memory ingestion.

        Walks sessions in ascending chronological order (``session_1``,
        ``session_2``, ...) and flattens each session's turns, preserving
        the upstream ``dia_id`` label and, when present, image-related
        fields (``img_url``, ``blip_caption``, ``query``). No per-message
        timestamp interpolation is performed — upstream has no such concept,
        every turn in a session shares the session's ``session_N_date_time``.
        """
        filter_conv_ids = {
            str(c) for c in (params.get("filter_conversation_ids", []) or [])
        }
        samples = self._load_dataset(params)
        conversations: list[dict[str, Any]] = []
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            if filter_conv_ids and sample_id not in filter_conv_ids:
                continue
            conv = sample.get("conversation", {}) or {}
            session_nums = _session_keys_sorted(conv)

            messages: list[dict[str, Any]] = []
            for s_num in session_nums:
                sess_key = f"session_{s_num}"
                date_time = str(conv.get(f"{sess_key}_date_time", ""))
                turns = conv.get(sess_key) or []
                if not isinstance(turns, list):
                    continue
                for turn_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    speaker = str(turn.get("speaker", ""))
                    msg: dict[str, Any] = {
                        # Map upstream ``speaker`` onto the engine's speaker
                        # fields; upstream uses the literal name ("Caroline",
                        # "Melanie") for both id and display.
                        "speaker_id": speaker,
                        "speaker_name": speaker,
                        "content": f"[{speaker} shared {turn['blip_caption']}] "+ str(turn.get("text", "")) if "blip_caption" in turn else str(turn.get("text", "")),
                        "timestamp": date_time,
                        "session_id": sess_key,
                        "session_index": s_num,
                        "turn_index": turn_idx,
                        "dia_id": str(turn.get("dia_id", "")),
                    }
                    # Image fields, when present. Upstream exposes url /
                    # caption / search query so readers can reconstruct the
                    # multimodal turn; we surface them verbatim.
                    if "img_url" in turn:
                        msg["img_url"] = turn["img_url"]
                    if "blip_caption" in turn:
                        msg["blip_caption"] = turn["blip_caption"]
                    if "query" in turn:
                        msg["img_query"] = turn["query"]
                    messages.append(msg)

            conversations.append({
                "conversation_id": sample_id,
                "messages": messages,
                "metadata": {
                    "sample_id": sample_id,
                    "speaker_a": conv.get("speaker_a", ""),
                    "speaker_b": conv.get("speaker_b", ""),
                    "num_sessions": len(session_nums),
                },
            })
        return conversations

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """Build the reader prompt for ``task``.

        Controlled by ``task.metadata["prompt_style"]`` (which is populated
        from ``params["prompt_style"]`` at ``load_tasks`` time):

        * ``"question"`` (default) — just the per-category-rewritten
          question. Memory systems retrieve their own context, so we don't
          stuff the whole conversation into the prompt.
        * ``"full_history"`` — upstream's byte-exact ``start_prompt +
          query_conv + '\\n\\n' + QA_PROMPT`` assembly for long-context
          baselines.

        In both modes the question is rewritten per upstream's
        :func:`_rewrite_question` rules (cat-2 date hint, cat-5 MCQ).
        """
        style = str(task.metadata.get("prompt_style") or "question")
        category = int(task.metadata.get("category", 0) or 0)
        question = task.prompt
        answer = str(task.ground_truth or "")

        rewritten, _key = _rewrite_question(
            category=category,
            question=question,
            answer=answer,
            task_id=task.task_id,
        )

        if style == "question":
            return rewritten

        if style == "full_history":
            conversation = task.metadata.get("conversation") or {}
            speaker_a = str(task.metadata.get("speaker_a") or "")
            speaker_b = str(task.metadata.get("speaker_b") or "")
            if not speaker_a or not speaker_b:
                speaker_a, speaker_b = _speaker_pair(conversation)
            start = _CONV_START_PROMPT.format(speaker_a, speaker_b)
            context = _build_conversation_context(conversation)
            qa_block = _build_qa_prompt(category, rewritten)
            # Upstream joins with '\n\n' at line 285 of gpt_utils.py:
            #   query = query_conv + '\n\n' + QA_PROMPT.format(...)
            return start + context + "\n\n" + qa_block

        raise NotImplementedError(f"Unknown prompt_style: {style!r}")

    # ------------------------------------------------------------------
    # Judge — upstream uses F1 metrics, not LLM-as-judge.
    # ------------------------------------------------------------------

    async def judge_answer(
        self,
        task: Task,
        generated_answer: str,
    ) -> dict[str, Any]:
        """Score ``generated_answer`` with upstream's F1 / substring check.

        Upstream (``task_eval/evaluate_qa.py``) does not use an LLM judge —
        it runs ``eval_question_answering`` which applies per-category F1
        (or abstention substring check for cat 5). We reproduce that logic
        here so ``pass`` is a thresholded F1 (> 0.5 by default, matching
        upstream's implicit "non-trivial overlap" expectation), while the
        raw F1 is surfaced for aggregation.

        Cat 5 prediction resolution
            Upstream's ``gpt_utils.py::get_cat_5_answer`` maps terse MCQ
            replies (``"a"`` / ``"b"`` / ``"(a)"`` / ``"(b)"``) to the full
            option text BEFORE substring-checking for the abstention phrase.
            We do the same here by regenerating the ``answer_key`` from the
            same ``task_id`` seed used by :func:`_cat5_mcq` at prompt time.

        Evidence recall
            If the engine populates ``task.metadata["retrieved_context_ids"]``
            with the dialog/session labels the memory system surfaced, we
            compute upstream's per-QA recall (``evaluation.py::
            eval_question_answering`` lines 228-237) and return it as
            ``evidence_recall``. When not populated, recall defaults to 1.0
            (matching upstream's ``all_recall.append(1)`` no-retrieval
            fallback).

            The accompanying ``evidence_recall_measured`` flag distinguishes
            a genuine measurement (the engine / middleware supplied IDs,
            possibly empty) from the upstream-faithful default (no key was
            written). Downstream aggregators SHOULD average recall only over
            measured rows — mixing measured recall with unmeasured 1.0s
            silently inflates averages for backends that don't expose
            provenance. See the ``MemoryMiddleware`` class docstring.
        """
        category = int(task.metadata.get("category", 0) or 0)
        golden = str(task.ground_truth or "")
        threshold = float(self._judge_config.get("f1_threshold", 0.5))

        # Strip CoT preamble. Memory-system prompts demand multi-step
        # CoT before a final-line ``FINAL ANSWER: <terse>``; upstream's
        # token-F1 over the entire prediction would otherwise be drowned
        # by the CoT payload. Falls through unchanged when no marker
        # exists (matches upstream's raw-completion behaviour).
        extracted = _extract_final_answer(generated_answer)

        # Cat-5 answer resolution — upstream runs this on the raw model
        # output BEFORE the substring check. Without it, a model reply of
        # "a" / "(a)" (common MCQ shorthand) fails to match "not mentioned"
        # even when option (a) is the abstention text.
        prediction = extracted
        resolved_prediction: str | None = None
        if category == 5:
            answer_key = _cat5_answer_key(golden, seed=task.task_id)
            resolved_prediction = _get_cat_5_answer(prediction, answer_key)
            prediction = resolved_prediction

        try:
            score = _score_qa(
                category=category,
                prediction=prediction,
                answer=golden,
                stemmer=None,
            )
        except ImportError as exc:
            logger.warning("LoCoMo F1 scoring requires nltk: %s", exc)
            return {
                "pass": False,
                "correct": 0,
                "f1": 0.0,
                "category": category,
                "category_name": task.metadata.get("category_name", ""),
                "judge_error": str(exc),
            }

        # Evidence recall — independent from F1. Only meaningful when the
        # engine has populated retrieved_context_ids; otherwise we preserve
        # upstream's no-retrieval default of 1.0. ``measured`` tracks
        # whether the value is real or the upstream-faithful default so
        # downstream aggregators can filter.
        evidence_list = list(task.metadata.get("evidence", []) or [])
        retrieved_ids_raw = task.metadata.get("retrieved_context_ids")
        retrieved_ids = (
            list(retrieved_ids_raw) if isinstance(retrieved_ids_raw, list) else None
        )
        measured = retrieved_ids is not None and bool(evidence_list)
        recall = _evidence_recall(retrieved_ids, evidence_list)

        f1_passed = bool(score >= threshold) if category != 5 else bool(score == 1.0)

        # Optional LLM-as-judge metric (Mem0-paper-style "J" score).
        # Activated by ``judge.llm_judge`` config block. Doesn't replace F1 —
        # both metrics are surfaced so analyses can pick the relevant one.
        # When ``primary_metric`` is ``"llm_judge"`` the judge verdict drives
        # ``pass``/``correct``; otherwise F1 thresholding stays authoritative.
        # We also time the judge call so the engine can add it to the
        # per-task ``total_latency_ms`` (search + inference + judge).
        #
        # Cat-5 (adversarial) is SKIPPED — byte-faithful to upstream Mem0 v2's
        # ``evaluate_llm_judge`` which does ``if int(category) == 5: continue``.
        # LoCoMo stores ``answer=None`` for cat-5 (the trick-question gold lives
        # in ``adversarial_answer``), so ``golden`` is ``""`` here; the judge
        # prompt would render ``Gold answer:`` (blank) + a real prediction,
        # and the LLM would label WRONG every time regardless of whether the
        # model correctly abstained. F1 already grades cat-5 via the
        # substring check (``_adversarial_correct``), so the J score over
        # cat-5 carries no honest signal and would only deflate the headline
        # ``pass_rate_llm_judge``. The aggregator excludes tasks with no
        # ``llm_judge_correct`` key from the average, so leaving the payload
        # empty for cat-5 is enough — no flag plumbing required.
        import time as _time
        llm_judge_cfg = self._judge_config.get("llm_judge") or {}
        llm_judge_passed: bool | None = None
        llm_judge_payload: dict[str, Any] = {}
        if llm_judge_cfg.get("enabled") and category != 5:
            judge_t0 = _time.monotonic()
            try:
                from cross_bench.benchmarks.llm_judge import llm_judge as _llm_judge
                verdict = await _llm_judge(
                    question=str(task.prompt or ""),
                    golden_answer=golden,
                    generated_answer=generated_answer,
                    llm_config=llm_judge_cfg,
                    # mem0 v2's evaluate_llm_judge uses 1 call per question
                    # (no majority vote). Defaulting to 1 keeps our J score
                    # directly comparable to the Mem0 paper number.
                    num_runs=int(llm_judge_cfg.get("num_runs", 1)),
                )
                judge_ms = (_time.monotonic() - judge_t0) * 1000.0
                llm_judge_passed = bool(verdict.get("pass", False))
                llm_judge_payload = {
                    "llm_judge_correct": int(llm_judge_passed),
                    "llm_judge_num_correct": verdict.get("num_correct", 0),
                    "llm_judge_num_runs": verdict.get("num_runs", 0),
                    "llm_judge_latency_ms": round(judge_ms, 1),
                }
            except Exception as exc:
                judge_ms = (_time.monotonic() - judge_t0) * 1000.0
                logger.warning("LoCoMo LLM-judge call failed: %s", exc)
                llm_judge_payload = {
                    "llm_judge_error": str(exc),
                    "llm_judge_latency_ms": round(judge_ms, 1),
                }

        primary = str(self._judge_config.get("primary_metric", "f1")).lower()
        if primary == "llm_judge" and llm_judge_passed is not None:
            passed = llm_judge_passed
        else:
            passed = f1_passed

        result: dict[str, Any] = {
            "pass": passed,
            "correct": int(passed),
            "f1": round(float(score), 3),
            "f1_pass": int(f1_passed),
            "score": round(float(score), 3),
            "category": category,
            "category_name": task.metadata.get("category_name", ""),
            "f1_threshold": threshold,
            "evidence": evidence_list,
            "evidence_recall": round(float(recall), 3),
            "evidence_recall_measured": measured,
            "retrieved_context_ids": (
                list(retrieved_ids) if retrieved_ids is not None else []
            ),
            "is_adversarial": category == 5,
        }
        result.update(llm_judge_payload)
        # Surface the resolved prediction for cat-5 debugging/aggregation —
        # but only when the terse-letter branch of :func:`_get_cat_5_answer`
        # actually mapped the reply to different text. A full-text reply
        # comes back strip/lower-normalized (upstream does this unconditionally)
        # which isn't a semantic resolution, just case folding, so don't
        # surface the key in that case.
        if (
            resolved_prediction is not None
            and resolved_prediction != generated_answer.strip().lower()
        ):
            result["resolved_prediction"] = resolved_prediction
        return result

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict[str, Any]:
        return {
            "data_path": "data/locomo",
            "file_path": "",  # explicit absolute/relative path wins if set
            "filter_categories": [],
            "filter_conversation_ids": [],
            "prompt_style": "question",
            "judge": {
                # Pass/fail threshold on the F1 score. Category 5 is always
                # exact (0.0 or 1.0 from the substring check).
                "f1_threshold": 0.5,
            },
        }

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve ``locomo10.json`` into a list of sample dicts.

        Resolution order:

        1. Explicit ``file_path`` param — treated as a literal JSON path.
        2. ``{data_path}/locomo10.json`` for the ``data_path`` param.
        3. Well-known fallback locations (``./data/locomo/``, ``./locomo/``,
           ``./``).
        4. HTTP download from the canonical GitHub raw URL (cached into
           ``{data_path}/locomo10.json`` on first use).

        The cache key includes ``file_path`` and ``data_path`` so tests can
        redirect to a fixture without polluting previous runs.
        """
        cache_key = f"{params.get('file_path', '')}||{params.get('data_path', '')}"
        if self._cached_samples is not None and self._cached_key == cache_key:
            return self._cached_samples

        # Priority 1: explicit file_path.
        file_path = params.get("file_path") or ""
        if file_path:
            data = _load_json_array(Path(file_path))
            self._cached_samples = data
            self._cached_key = cache_key
            return data

        data_path = Path(params.get("data_path", "data/locomo"))

        # Priority 2+3: local search.
        candidates = [
            data_path / _DEFAULT_DATA_FILE,
            data_path,  # if caller pointed data_path at the file itself
            Path("data/locomo") / _DEFAULT_DATA_FILE,
            Path("locomo") / _DEFAULT_DATA_FILE,
            Path(_DEFAULT_DATA_FILE),
        ]
        for candidate in candidates:
            if candidate.is_file():
                data = _load_json_array(candidate)
                self._cached_samples = data
                self._cached_key = cache_key
                return data

        # Priority 4: download once into data_path and cache on disk.
        target = data_path / _DEFAULT_DATA_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LoCoMo dataset from %s -> %s", _GITHUB_RAW_URL, target)
        try:
            import urllib.request
            urllib.request.urlretrieve(_GITHUB_RAW_URL, target)  # noqa: S310 - trusted URL
        except Exception as exc:  # noqa: BLE001 - surface error with URL context
            raise FileNotFoundError(
                f"LoCoMo dataset not found locally under {data_path!r} and "
                f"download from {_GITHUB_RAW_URL} failed: {exc}. "
                f"Either place the file at {target!s} manually or check "
                f"network access."
            ) from exc

        data = _load_json_array(target)
        self._cached_samples = data
        self._cached_key = cache_key
        return data


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    """Read a LoCoMo JSON file — a top-level JSON array of sample objects."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at {path}, got {type(data).__name__}"
        )
    return data
