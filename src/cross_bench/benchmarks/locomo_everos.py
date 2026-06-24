# ruff: noqa: E501
"""LoCoMo (EverOS-style) adapter — methodology-faithful port of
EverMind-AI/EverOS ``methods/evermemos/evaluation``.

Unlike :mod:`cross_bench.benchmarks.locomo_original` (byte-exact ACL 2024
port), this adapter targets **comparability with Mem0 / EverOS
leaderboards**. It reuses the upstream LoCoMo dataset (``locomo10.json``,
snap-research/locomo) but switches to the evaluation methodology published
by EverMind-AI's EverOS / EverMemOS project:

* **LLM-as-judge** instead of Porter-stemmed F1. The judge prompt is the
  mem0 "Hawaii / shell necklace" 7-rule template (14-day date slop, 50%
  duration tolerance, partial-credit tolerant) that both mem0ai's
  ``memory-benchmarks`` and EverOS's ``methods/evermemos/evaluation``
  ship. See :func:`cross_bench.benchmarks.llm_judge.llm_judge`.
* **3-run mean$\\pm$std** aggregation per question. Each question is
  independently judged ``num_runs`` times; ``accuracy_mean`` and
  ``accuracy_std`` are surfaced alongside the majority-vote ``pass``.
* **30-second-per-turn synthetic timestamps** anchored at
  ``2024-01-01T00:00:00`` — LoCoMo's dataset only ships
  *date-of-session*, not per-turn timestamps. Backends that need event
  time (Zep, MemoryOS) consume the interpolated timestamps.
* **Category 5 excluded by default** via ``filter_categories=[5]``. This
  matches both EverOS's shipped configs and mem0ai's
  ``CATEGORIES_TO_EVALUATE = [1,2,3,4]``.
* **Per-system answer prompts** (mem0 / memos / cot / zep) — the adapter
  selects an answer template based on ``system`` so different memory
  backends get their preferred prompt shape. All four enforce the
  anti-abstention rule.

Dataset loading (JSON resolution, HTTP fallback, sample shape) is reused
from :mod:`cross_bench.benchmarks.locomo_original` for strict
one-dataset-source guarantee.

References
    * Upstream: https://github.com/EverMind-AI/EverOS/tree/main/methods/evermemos/evaluation
    * mem0ai:   https://github.com/mem0ai/memory-benchmarks
    * Paper:    https://arxiv.org/abs/2402.17753
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

from cross_bench.benchmarks.base import ConversationalBenchmarkAdapter
from cross_bench.benchmarks.llm_judge import llm_judge
from cross_bench.benchmarks.locomo_original import (
    _CAT2_DATE_HINT,
    CATEGORY_NAMES,
    _load_json_array,
    _session_keys_sorted,
)
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EverOS-specific constants
# ---------------------------------------------------------------------------

# Seconds between successive turns in the synthetic timeline. EverOS
# hard-codes 30 in ``loaders.py``.
_DEFAULT_TURN_INTERVAL_SECONDS = 30

# Supported per-system answer prompts. Selection is via the ``system``
# param. All four are forks of mem0's CoT template; the per-system
# difference is how retrieved memories are framed for the model.
SYSTEMS: tuple[str, ...] = ("mem0", "memos", "cot", "zep")

# Default judge config. Mirrors EverOS's shipped ``configs/*.yaml``:
# OpenAI-compatible, gpt-4o-mini, 3 runs for mean$\pm$std.
_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_DEFAULT_NUM_RUNS = 3

# Dataset file default — same as locomo_original, shared by both variants.
_DEFAULT_DATA_FILE = "locomo10.json"
_GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


# ---------------------------------------------------------------------------
# Answer prompts — per-system templates, forked from mem0's CoT prompt.
#
# Each template has two placeholders:
#   {memories}  — retrieved / formatted context. Adapter leaves this as a
#                 literal Python format slot so downstream memory-backend
#                 harnesses can do a ``.format(memories=...)`` substitution.
#   {question}  — the (category-rewritten) question text.
#
# Byte-exactness caveat: these prompts are reproduced in spirit from
# EverOS's ``src/prompts/{mem0,memos,cot,zep}.txt`` but may drift by a
# whitespace character or two from upstream's files. The key invariants
# preserved across all four are the 7-rule CoT structure and the
# anti-abstention clause.
# ---------------------------------------------------------------------------

_ANTI_ABSTENTION_RULE = (
    "NEVER say 'not specified', 'not mentioned', 'no record', or 'the "
    "memories don't say' — if ANY memory contains relevant information, "
    "give the best answer."
)

_MEM0_ANSWER_PROMPT = """You are a helpful assistant answering questions about a user based on retrieved memories from prior conversations.

Retrieved memories:
{memories}

Follow this 7-step reasoning process:
1. Understand the question being asked.
2. Identify which memories are relevant to the question.
3. Consider temporal context — when did each memory happen relative to the question's reference time?
4. Draw any inferences needed to bridge what is asked and what the memories state.
5. For dates, accept any format as long as it refers to the same day / month / year. Dates within 14 days of the target are considered correct. Durations within 50% of the target are considered correct.
6. Verify the answer against the memories — does the evidence actually support it?
7. Provide a concise final answer as a short phrase.

""" + _ANTI_ABSTENTION_RULE + """

Question: {question}
Short answer:"""

_MEMOS_ANSWER_PROMPT = """You are a helpful assistant answering questions about a user. You have access to two tiers of memory: working memory (recent) and long-term memory (consolidated).

Working memory:
{memories_working}

Long-term memory:
{memories_longterm}

Follow the 7-step CoT process (understand, identify, consider time, infer, normalize dates/durations, verify, answer). Dates within 14 days of the target are CORRECT. Durations within 50% of the target are CORRECT.

""" + _ANTI_ABSTENTION_RULE + """

Question: {question}
Short answer:"""

_COT_ANSWER_PROMPT = """You are a helpful assistant answering questions about a user based on the full conversation history below.

Conversation history:
{memories}

Follow the 7-step CoT process (understand, identify, consider time, infer, normalize dates/durations, verify, answer). Dates within 14 days of the target are CORRECT. Durations within 50% of the target are CORRECT.

""" + _ANTI_ABSTENTION_RULE + """

Question: {question}
Short answer:"""

_ZEP_ANSWER_PROMPT = """You are a helpful assistant answering questions about a user. You have access to a knowledge graph of facts and episodes extracted from prior conversations.

Facts:
{facts}

Episodes:
{episodes}

Follow the 7-step CoT process (understand, identify, consider time, infer, normalize dates/durations, verify, answer). Dates within 14 days of the target are CORRECT. Durations within 50% of the target are CORRECT.

""" + _ANTI_ABSTENTION_RULE + """

Question: {question}
Short answer:"""


ANSWER_PROMPTS: dict[str, str] = {
    "mem0": _MEM0_ANSWER_PROMPT,
    "memos": _MEMOS_ANSWER_PROMPT,
    "cot": _COT_ANSWER_PROMPT,
    "zep": _ZEP_ANSWER_PROMPT,
}


# ---------------------------------------------------------------------------
# Timestamp interpolation — EverOS's ``loaders.py`` signature move.
# ---------------------------------------------------------------------------

def _parse_baseline(baseline: str) -> datetime:
    """Parse the ISO-8601 timestamp baseline.

    Upstream uses ``datetime(2024, 1, 1, 0, 0, 0)`` — see
    ``methods/evermemos/evaluation/src/core/loaders.py``. We accept either
    the literal upstream baseline or an ISO-8601 override via config.
    """
    try:
        return datetime.fromisoformat(baseline)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp baseline {baseline!r}; expected ISO-8601 "
            f"(e.g. '2024-01-01T00:00:00')"
        ) from exc


def _parse_locomo_session_date_time(value: str) -> datetime | None:
    """Parse LoCoMo's human-readable ``session_N_date_time`` strings.

    LoCoMo stores strings like ``"1:56 pm on 8 May, 2023"`` (with comma)
    or ``"1:00 pm on 1 May 2023"`` (without comma — observed in the
    snap-research mirror as well as some forks). We tolerate both
    variants; the time component is parsed even though EverOS's
    synthetic timeline only uses the calendar date.
    """
    s = (value or "").strip()
    if not s:
        return None

    # Normalize am/pm casing for strptime.
    # Example input: "1:56 pm on 8 May, 2023"
    parts = s.split()
    if len(parts) >= 2 and parts[1].lower() in {"am", "pm"}:
        parts[1] = parts[1].upper()
        s = " ".join(parts)

    # Accept both with- and without-comma variants by trying each format
    # in order. Full-month-name comes first; short-name follows so e.g.
    # "May" matches %B in the first pass.
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%I:%M %p on %d %b, %Y",
        "%I:%M %p on %d %b %Y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _resolve_baseline(
    conversation: dict[str, Any],
    baseline_param: Any,
) -> datetime:
    """Resolve the synthetic-timeline baseline for a single conversation.

    ``baseline_param`` semantics, matching the config contract in
    ``get_default_params``:

    * ``"auto"`` (default) — anchor on the conversation's own
      ``session_1_date_time``. This matches the same trap-fix pattern
      Mastra M38 / Hindsight H46 / MemOS O24 use elsewhere on the
      platform: synthetic per-turn timestamps should derive from the
      source clock, not a fictitious wall-clock.
    * ``None`` — same as ``"auto"`` (treat absent config as auto).
    * ISO-8601 string (e.g. ``"2024-01-01T00:00:00"``) — literal baseline,
      preserves the pre-2026-05-08 behaviour for runs that need to
      reproduce historical numbers.

    Fallback: when ``baseline_param`` is auto but the conversation's
    ``session_1_date_time`` is missing or unparseable, we fall back to
    ``datetime(2024, 1, 1)`` (EverOS's upstream literal) and log a
    warning. Choosing 2024-01-01 over ``datetime.now()`` matters — the
    LLM-extractor / answer-LLM trap from H46 is specifically about not
    silently grounding facts to wall-clock-today.
    """
    if baseline_param is None or baseline_param == "auto":
        date_time = str(conversation.get("session_1_date_time", ""))
        parsed = _parse_locomo_session_date_time(date_time)
        if parsed is not None:
            return parsed
        logger.warning(
            "LoCoMo-EverOS: session_1_date_time %r unparseable; "
            "falling back to datetime(2024, 1, 1) baseline",
            date_time,
        )
        return datetime(2024, 1, 1)
    return _parse_baseline(str(baseline_param))


def interpolate_timestamps(
    conversation: dict[str, Any],
    baseline: datetime,
    interval_seconds: int,
) -> list[tuple[int, int, datetime]]:
    """Assign synthetic per-turn timestamps under EverOS's rules.

    Upstream (``loaders.py``)::

        base_time   = datetime(2024, 1, 1, 0, 0, 0)
        default_interval = 30  # seconds between successive turns
        for session in conversation:
            current_session_time = base_time + per-session offset
            for msg_idx, msg in enumerate(session):
                msg_timestamp = current_session_time + timedelta(
                    seconds=msg_idx * time_interval
                )

    The per-session offset is itself an integer number of days from the
    baseline, so each session starts at midnight on day ``N`` relative to
    the baseline, and turns within the session tick at ``interval_seconds``.

    Returns a list of ``(session_index, turn_index, timestamp)`` tuples in
    ascending session order, ascending turn order.
    """
    session_nums = _session_keys_sorted(conversation)
    interpolated: list[tuple[int, int, datetime]] = []
    for sess_idx, s_num in enumerate(session_nums):
        sess_key = f"session_{s_num}"
        turns = conversation.get(sess_key) or []
        if not isinstance(turns, list):
            continue
        # Each session starts one day after the previous in the synthetic
        # timeline. Upstream uses the integer session index as the offset;
        # we mirror that exactly.
        session_start = baseline + timedelta(days=sess_idx)
        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            ts = session_start + timedelta(seconds=turn_idx * interval_seconds)
            interpolated.append((s_num, turn_idx, ts))
    return interpolated


# ---------------------------------------------------------------------------
# Per-system judge system prompt — biases the judge toward paraphrase
# tolerance. Matches mem0ai's judge system prompt; reused by EverOS.
# ---------------------------------------------------------------------------

_EVEROS_JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "about a user match a gold standard answer. Be generous: partial or "
    "paraphrased answers that touch the same topic as the gold answer are "
    "CORRECT. Dates within 14 days are CORRECT. Durations within 50% are "
    "CORRECT."
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class LoCoMoEverosAdapter(ConversationalBenchmarkAdapter):
    """LoCoMo-EverOS adapter — methodology-faithful to EverMemOS.

    See :mod:`cross_bench.benchmarks.locomo_original` for the ACL 2024
    paper-faithful variant (Porter F1, all five categories, byte-exact
    prompts).
    """

    name = "locomo_everos"
    granularity = TaskGranularity.CONVERSATIONAL

    def __init__(self) -> None:
        super().__init__()
        self._judge_config: dict[str, Any] = {}
        self._num_runs: int = _DEFAULT_NUM_RUNS
        self._system: str = "mem0"
        self._cached_samples: list[dict[str, Any]] | None = None
        self._cached_key: str | None = None

    # ------------------------------------------------------------------
    # Task & conversation loading
    # ------------------------------------------------------------------

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        """Yield one :class:`Task` per QA pair, with EverOS defaults.

        Key differences from :meth:`LoCoMoOriginalAdapter.load_tasks`:

        * Default ``filter_categories = [5]`` — adversarial dropped, matching
          EverOS's ``filter_category: [5]`` config and mem0ai's
          ``CATEGORIES_TO_EVALUATE = [1,2,3,4]``.
        * Default ``system = "mem0"`` and default ``num_runs = 3``.
        * Attaches interpolated per-turn timestamp (30-s cadence from the
          baseline) to each question via its originating session/turn.
        * No cat-5 MCQ conversion (cat 5 excluded by default; if the user
          overrides ``filter_categories`` to include it, the question is
          passed through verbatim — EverOS never exercises this branch).
        """
        self._judge_config = dict(params.get("judge") or {})
        self._num_runs = int(
            params.get("num_runs") or self._judge_config.get("num_runs") or _DEFAULT_NUM_RUNS
        )
        self._system = str(params.get("system") or "mem0").lower()
        if self._system not in SYSTEMS:
            raise ValueError(
                f"Unknown system {self._system!r}; expected one of {SYSTEMS}"
            )

        # Default EverOS behavior: exclude adversarial.
        raw_filter = params.get("filter_categories")
        if raw_filter is None:
            filter_categories = {5}
        else:
            filter_categories = {int(c) for c in raw_filter}

        limit = params.get("limit")

        # Timestamp config — ``baseline`` defaults to ``"auto"`` which
        # resolves per-sample from ``session_1_date_time``. Pass an
        # explicit ISO-8601 string to reproduce the pre-2026-05-08
        # literal-baseline behaviour.
        ts_cfg = params.get("timestamp") or {}
        baseline_raw = ts_cfg.get("baseline", "auto")
        interval = int(ts_cfg.get("interval_seconds") or _DEFAULT_TURN_INTERVAL_SECONDS)

        samples = self._load_dataset(params)
        count = 0
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            conversation = sample.get("conversation", {}) or {}
            qa_list = sample.get("qa", []) or []

            # Resolve the baseline once per sample (so every QA from this
            # sample shares the same synthetic timeline), then precompute
            # the timeline so cat-2 questions have a stable
            # ``reference_timestamp`` to anchor against.
            baseline = _resolve_baseline(conversation, baseline_raw)
            timeline = interpolate_timestamps(conversation, baseline, interval)
            last_ts = timeline[-1][2] if timeline else baseline

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
                        "conversation_id": sample_id,
                        "sample_id": sample_id,
                        "qa_index": q_idx,
                        "category": category,
                        "category_name": CATEGORY_NAMES.get(category, "unknown"),
                        "evidence": evidence,
                        "system": self._system,
                        # EverOS surfaces the last-session synthetic
                        # timestamp as the reference date for cat-2
                        # temporal rewrites. Downstream code can use this
                        # for date-hint materialization.
                        "reference_timestamp": last_ts.isoformat(),
                        "conversation": conversation,
                        "speaker_a": conversation.get("speaker_a", ""),
                        "speaker_b": conversation.get("speaker_b", ""),
                    },
                    ground_truth=answer,
                )
                count += 1

    def load_conversations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Surface one conversation per sample, with per-turn synthetic timestamps.

        Differs from :meth:`LoCoMoOriginalAdapter.load_conversations` in two
        ways:

        * Per-turn ISO-8601 ``timestamp`` is the synthetic 30-s-cadence value
          from :func:`interpolate_timestamps` (instead of the raw
          ``session_N_date_time``). Memory backends that key on event time
          (Zep, MemoryOS) consume these.
        * Per-turn ``session_date_time`` keeps the original LoCoMo session
          date so backends that still want the human-readable date can find
          it.

        Image fields (``img_url``, ``blip_caption``, ``img_query``) are
        preserved verbatim.
        """
        ts_cfg = params.get("timestamp") or {}
        baseline_raw = ts_cfg.get("baseline", "auto")
        interval = int(ts_cfg.get("interval_seconds") or _DEFAULT_TURN_INTERVAL_SECONDS)

        samples = self._load_dataset(params)
        conversations: list[dict[str, Any]] = []
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            conv = sample.get("conversation", {}) or {}

            # Resolve the baseline ONCE per conversation (auto-mode reads
            # ``session_1_date_time``); compute the synthetic timeline
            # ONCE for the whole conversation. The previous loop-inside-
            # loop shape silently overwrote earlier-session timestamps
            # with later sessions' baselines.
            baseline = _resolve_baseline(conv, baseline_raw)
            timeline = interpolate_timestamps(conv, baseline, interval)
            ts_by_key: dict[tuple[int, int], datetime] = {
                (s, t): ts for s, t, ts in timeline
            }

            messages: list[dict[str, Any]] = []
            for s_num in _session_keys_sorted(conv):
                sess_key = f"session_{s_num}"
                date_time = str(conv.get(f"{sess_key}_date_time", ""))
                turns = conv.get(sess_key) or []
                if not isinstance(turns, list):
                    continue
                for turn_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    speaker = str(turn.get("speaker", ""))
                    ts = ts_by_key.get((s_num, turn_idx), baseline)
                    text = str(turn.get("text", ""))
                    if "blip_caption" in turn:
                        content = f"[{speaker} shared {turn['blip_caption']}] {text}"
                    else:
                        content = text
                    msg: dict[str, Any] = {
                        "speaker_id": speaker,
                        "speaker_name": speaker,
                        "content": content,
                        # EverOS's loaders.py replaces the raw
                        # session_N_date_time with the synthetic ISO-8601
                        # timestamp here. Both are surfaced.
                        "timestamp": ts.isoformat(),
                        "session_date_time": date_time,
                        "session_id": sess_key,
                        "session_index": s_num,
                        "turn_index": turn_idx,
                        "dia_id": str(turn.get("dia_id", "")),
                    }
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
                    "num_sessions": len(_session_keys_sorted(conv)),
                    "timestamp_baseline": baseline.isoformat(),
                    "turn_interval_seconds": interval,
                },
            })
        return conversations

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """Return the per-system answer prompt with the question rewritten.

        Unlike the paper-faithful adapter, the returned prompt **contains
        format-slot placeholders** (``{memories}``, or system-specific
        variants like ``{memories_working}``/``{memories_longterm}``/
        ``{facts}``/``{episodes}``). The downstream memory-backend harness
        is responsible for filling those slots; this matches EverOS's
        ``src/core/runners.py`` pipeline where per-system retrievers format
        their memory payload.

        Cat 2 still gets the upstream date hint appended to the question.
        Cat 5 is passed through verbatim (default config excludes cat 5
        entirely — if a caller re-enables it, we deliberately do **not**
        apply the paper's MCQ conversion because EverOS never exercises
        that branch).
        """
        system = str(task.metadata.get("system") or self._system or "mem0").lower()
        if system not in ANSWER_PROMPTS:
            raise ValueError(
                f"Unknown system {system!r}; expected one of {sorted(ANSWER_PROMPTS.keys())}"
            )

        category = int(task.metadata.get("category", 0) or 0)
        question = task.prompt
        if category == 2:
            question = question + _CAT2_DATE_HINT

        template = ANSWER_PROMPTS[system]
        # Downstream (retriever / harness) is expected to substitute the
        # memory-related format slots. We only fill {question} here so the
        # other slots remain available for format(**memory_payload).
        return template.replace("{question}", question)

    # ------------------------------------------------------------------
    # Judge — LLM-as-judge with 3-run mean$\pm$std aggregation.
    # ------------------------------------------------------------------

    async def judge_answer(
        self,
        task: Task,
        generated_answer: str,
    ) -> dict[str, Any]:
        """LLM-as-judge the generated answer against the golden answer.

        Runs :func:`cross_bench.benchmarks.llm_judge.llm_judge` ``num_runs``
        times (default 3) with the EverOS / mem0 judge prompt. Aggregates:

        * ``pass`` — majority vote across runs (inherited from
          :func:`llm_judge`).
        * ``correct`` — integer 0/1 from the majority vote.
        * ``accuracy_mean`` — ``num_correct / num_runs`` (the EverOS
          "mean" value).
        * ``accuracy_std`` — standard deviation of the 0/1 per-run
          outcomes; for a binomial with ``p = accuracy_mean`` and 3 runs
          this has closed form ``sqrt(p*(1-p)*n/(n-1))`` with the
          Bessel-corrected sample variance.
        * ``raw_judgments`` — list of ``"CORRECT"`` / ``"WRONG"`` labels.
        """
        num_runs = int(self._num_runs or _DEFAULT_NUM_RUNS)

        # Pass through the judge config verbatim; llm_judge honours
        # provider / model / api_key / base_url.
        judge_result = await llm_judge(
            question=task.prompt,
            golden_answer=str(task.ground_truth or ""),
            generated_answer=generated_answer,
            llm_config=self._judge_config,
            num_runs=num_runs,
            system_prompt=_EVEROS_JUDGE_SYSTEM_PROMPT,
        )

        labels: list[str] = list(judge_result.get("raw_judgments", []))
        correct_runs: list[int] = [1 if label == "CORRECT" else 0 for label in labels]
        n = len(correct_runs) or 1
        accuracy_mean = sum(correct_runs) / n
        # Bessel-corrected sample std. For n == 1 we return 0.0 (no variance).
        if n > 1:
            variance = sum((x - accuracy_mean) ** 2 for x in correct_runs) / (n - 1)
            accuracy_std = math.sqrt(max(0.0, variance))
        else:
            accuracy_std = 0.0

        category = int(task.metadata.get("category", 0) or 0)
        return {
            "pass": bool(judge_result.get("pass", False)),
            "correct": int(judge_result.get("correct", 0)),
            "score": round(float(accuracy_mean), 3),
            "accuracy_mean": round(float(accuracy_mean), 3),
            "accuracy_std": round(float(accuracy_std), 3),
            "num_correct": int(judge_result.get("num_correct", 0)),
            "num_runs": int(judge_result.get("num_runs", n)),
            "raw_judgments": labels,
            "category": category,
            "category_name": task.metadata.get("category_name", ""),
            "system": task.metadata.get("system", self._system),
            "evidence": task.metadata.get("evidence", []),
        }

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict[str, Any]:
        return {
            "data_path": "data/locomo",
            "file_path": "",  # explicit path wins if set
            # Default: exclude adversarial (cat 5) — matches EverOS's
            # shipped ``filter_category: [5]`` and mem0ai's
            # ``CATEGORIES_TO_EVALUATE = [1,2,3,4]``.
            "filter_categories": [5],
            # One of: mem0, memos, cot, zep.
            "system": "mem0",
            # Synthetic 30-s-per-turn timeline. By default we derive the
            # baseline date from the conversation's session_N_date_time.
            "timestamp": {
                "baseline": "auto",
                "interval_seconds": _DEFAULT_TURN_INTERVAL_SECONDS,
            },
            # 3-run mean$\pm$std — EverOS's default aggregation.
            "num_runs": _DEFAULT_NUM_RUNS,
            "judge": {
                "provider": "openai",
                "model": _DEFAULT_JUDGE_MODEL,
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
            },
        }

    # ------------------------------------------------------------------
    # Dataset loading — identical resolution to ``locomo_original``.
    # ------------------------------------------------------------------

    def _load_dataset(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Same resolution chain as :class:`LoCoMoOriginalAdapter`."""
        from pathlib import Path

        cache_key = f"{params.get('file_path', '')}||{params.get('data_path', '')}"
        if self._cached_samples is not None and self._cached_key == cache_key:
            return self._cached_samples

        file_path = params.get("file_path") or ""
        if file_path:
            data = _load_json_array(Path(file_path))
            self._cached_samples = data
            self._cached_key = cache_key
            return data

        data_path = Path(params.get("data_path", "data/locomo"))
        candidates = [
            data_path / _DEFAULT_DATA_FILE,
            data_path,
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

        target = data_path / _DEFAULT_DATA_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LoCoMo dataset from %s -> %s", _GITHUB_RAW_URL, target)
        try:
            import urllib.request
            urllib.request.urlretrieve(_GITHUB_RAW_URL, target)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
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
