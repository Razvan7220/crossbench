"""RepoBench adapter — repository-level next-line completion (Liu et al., ICLR 2024).

Faithful port of the upstream v1.1 benchmark:
  https://github.com/Leolty/repobench
  https://arxiv.org/abs/2306.03091

Data
----
Per-language datasets on the HuggingFace Hub:
  - ``tianyang/repobench_python_v1.1``
  - ``tianyang/repobench_java_v1.1``

Each dataset exposes three splits (the paper's "settings"):
  - ``cross_file_first``  — masks the line where a cross-file module is used
    for the first time.
  - ``cross_file_random`` — masks a random line using a cross-file module
    (not the first usage).
  - ``in_file``           — masks a random line with no cross-file dependency.

Row schema:
  ``repo_name``          : str
  ``file_path``          : str
  ``context``            : list[{identifier, path, snippet}] — cross-file snippets
  ``import_statement``   : str
  ``cropped_code``       : str — in-file code up to (but excluding) the target line
  ``next_line``          : str — ground-truth target
  ``level``              : "2k"|"4k"|"8k"|"12k"|"16k"|... — context-size bucket
  ``created_at``         : str — UTC timestamp, e.g. "2023-12-28 05:47:18+00:00"
  ``gold_snippet_index`` : int — index into ``context`` of the snippet
                            ``next_line`` actually depends on. Metadata-only;
                            never fed to the model.

Parameters (via ``params``):
  ``language``       : "python" | "java"                 (default "python")
  ``splits``         : list of split names              (default all three)
  ``levels``         : list of level buckets            (default ["2k","4k","8k","12k","16k"])
  ``start_date``     : YYYY-MM-DD inclusive lower bound (default "2023-12-01")
  ``end_date``       : YYYY-MM-DD inclusive upper bound (default "2023-12-31")
  ``max_token_nums`` : int or None — prompt token budget (default 15800)
  ``tokenizer``      : "tiktoken:<enc>" | "char"        (default "tiktoken:cl100k_base")
  ``retrieval_query_lines`` : int — RepoCoder-style query length when memory
                              is active. Default 20, matching RepoCoder's
                              ``window_size``. https://arxiv.org/abs/2303.12570
  ``synthetic_workspace_dir`` : str or None — When set, before each task's
                              memory ingest CrossBench materialises an on-disk
                              tree under ``{dir}/{sanitized_repo_name}/`` (see
                              Memory integration below). Also stamps
                              ``task.metadata["code_workspace"]`` as the resolved
                              ``dir`` for filesystem-backed memory systems.

Prompt construction — matches ``data/utils.py::construct_prompt`` line-for-line::

    {c} Repo Name: {repo_name}
    {c} Path: {snippet.path}
    {snippet.snippet}

    {c} Path: {file_path}
    {import_statement}
    {cropped_code.rstrip()}

where ``{c}`` is ``#`` for Python, ``//`` for Java. When the combined prompt
exceeds ``max_token_nums`` tokens, cross-file prompt lines are dropped from
the end until it fits. Blank-line runs of four or more are collapsed to a
single blank line, per upstream.

Post-processing — matches ``run.py::get_first_line_not_comment``. Strips
leading newlines, skips blank lines, single-line comments, and Python
``\"\"\"``/``'''`` or Java ``/*...*/`` multiline comment blocks; returns the
first real code line (or the raw first line as a fallback).

Metrics — match ``evaluation/metrics.py``:
  - ``exact_match``    : ``pred.split() == gt.split()`` (whitespace-tokenized)
  - ``edit_similarity``: ``rapidfuzz.fuzz.ratio(pred, gt)`` — Levenshtein, 0-100.
    ``rapidfuzz`` is a drop-in compatible replacement for the upstream's
    ``fuzzywuzzy.fuzz.ratio``.
  - ``codebleu``       : ``codebleu.calc_codebleu(...)`` (0-100), emitted only
    when the optional ``codebleu`` package is installed. Argument order
    mirrors upstream ``eval.py`` line 45: refs=generated, hyps=ground_truth
    (a quirk; CodeBLEU's dataflow component is non-symmetric so the
    swap is locked here for byte-faithfulness).

Memory integration
------------------
When a memory middleware is attached, ``ingest_for_memory`` ingests each
cross-file snippet as one ``post_response`` call with
``source_id = "ctx:{idx}"`` (matching ``gold_snippet_index``), and the
cropped target file as ``source_id = "self"``. The LLM sees a stripped
prompt (``format_memory_prompt`` — in-file portion only) plus whatever
the middleware retrieves. The retrieval query is RepoCoder-style:
``import_statement + tail(cropped_code, retrieval_query_lines)``.

When ``params["synthetic_workspace_dir"]`` is a non-empty string, the same
ingest step first writes those snippets and the in-repo target file to
``{synthetic_workspace_dir}/{sanitized_repo_name}/`` so filesystem-backed
memory systems can parse them; the adapter removes that per-repo directory
after each task via ``teardown_task``. Each task also gets
``metadata["code_workspace"]`` set to the **resolved**
``synthetic_workspace_dir`` so CrossBench can pass it to the middleware as the
workspace root (e.g. repo-local ``tmp`` via ``synthetic_workspace_dir = "tmp"``
in ``configs/benchmarks/repobench.toml``).

Simplifications vs. upstream:
  - Upstream passes the model's own tokenizer to ``construct_prompt``.
    We have no model-specific tokenizer here, so we use ``tiktoken``'s
    ``cl100k_base`` encoding as a universal proxy (falls back to a
    character heuristic — 3.5 chars/token — if ``tiktoken`` is not
    installed). This changes exactly which cross-file lines are dropped in
    the truncation edge case, not the metric definitions.
  - Upstream reports per-split weighted averages across the three settings.
    The CrossBench engine's flat mean over all tasks is numerically
    equivalent (``sum(ems) / n`` = ``Σ (em_split·n_split) / Σ n_split``).
    Per-task ``split`` is preserved in metadata; ``category`` is also
    set on the evaluation dict so the engine emits
    ``accuracy_cat_<split>`` per-split breakdowns automatically.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from rapidfuzz.fuzz import ratio as _fuzz_ratio

from cross_bench.benchmarks.base import (
    MEMORY_TOKEN_SCHEMA_KEYS,
    BenchmarkAdapter,
)
from cross_bench.benchmarks.code_workspace import (
    _materialize_synthetic_workspace,
    _teardown_synthetic_workspace,
)
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

if TYPE_CHECKING:
    from cross_bench.systems.base import MemoryMiddleware

logger = logging.getLogger(__name__)

_DATASETS = {
    "python": "tianyang/repobench_python_v1.1",
    "java": "tianyang/repobench_java_v1.1",
}
_SPLITS: tuple[str, ...] = ("cross_file_first", "cross_file_random", "in_file")
_DEFAULT_LEVELS: tuple[str, ...] = ("2k", "4k", "8k", "12k", "16k")
_COMMENT_SYMBOL = {"python": "#", "java": "//"}
_CHARS_PER_TOKEN_FALLBACK = 3.5  # rough proxy when tiktoken is unavailable
_DEFAULT_RETRIEVAL_QUERY_LINES = 20  # RepoCoder window_size default





# Chat-LLM completion directive (deviations.md R16).
#
# Upstream RepoBench prompts are raw code blocks designed for base
# completion models (Codex / CodeT5 / similar). Chat-tuned LLMs
# (gpt-4o, Claude, …) see raw code and respond in chat-mode prose
# ("This file imports ..."), defeating the upstream EM / ES /
# CodeBLEU graders. We prepend a short directive in
# ``format_prompt`` and ``format_memory_prompt`` (the LLM-facing
# entry points) so the bench LLM emits a single line of code. The
# dataset-level prompt strings produced by ``_construct_prompt`` and
# ``_build_in_file_prompt`` stay upstream-faithful (still locked by
# the R7-R10 tests) — the directive lives one layer up.
_COMPLETION_INSTRUCTION_BY_LANG: dict[str, str] = {
    "python": (
        "Complete the next line of Python code from the source file shown "
        "below. Output ONLY that single next line, preserving any leading "
        "whitespace exactly. Do NOT wrap the line in a fenced code block. "
        "Do NOT add explanations, comments, or any other text.\n\n"
    ),
    "java": (
        "Complete the next line of Java code from the source file shown "
        "below. Output ONLY that single next line, preserving any leading "
        "whitespace exactly. Do NOT wrap the line in a fenced code block. "
        "Do NOT add explanations, comments, or any other text.\n\n"
    ),
}


def _completion_instruction(language: str) -> str:
    """Return the LLM directive for the requested language.

    Defaults to the Python directive for unknown languages so we never
    drop the directive entirely (a chat LLM with no directive is the
    failure mode that motivated R16 in the first place).
    """
    return _COMPLETION_INSTRUCTION_BY_LANG.get(
        language, _COMPLETION_INSTRUCTION_BY_LANG["python"]
    )


def _get_token_counter(preference: str) -> Callable[[str], int]:
    """Return a ``text -> token_count`` callable. Falls back to a char heuristic."""
    if preference and preference.startswith("tiktoken:"):
        try:
            import tiktoken

            enc = tiktoken.get_encoding(preference.split(":", 1)[1])
            return lambda s: len(enc.encode(s, disallowed_special=()))
        except Exception:
            pass
    return lambda s: int(len(s) / _CHARS_PER_TOKEN_FALLBACK + 0.5)


def _construct_prompt(
    row: dict[str, Any],
    language: str,
    count_tokens: Callable[[str], int],
    max_token_nums: int | None,
) -> str:
    """Port of ``data/utils.py::construct_prompt``.

    Truncation note: when the prompt exceeds the budget, upstream walks
    cross-file lines backwards and uses ``[:i]`` (exclusive) — meaning
    the line that *pushed* the running surplus below zero is itself
    dropped, leaving a small unused budget. The in-file block is never
    truncated. This is a faithful reproduction of upstream's behavior;
    do not "optimize" the slice to ``[:i+1]``.
    """
    comment = _COMMENT_SYMBOL[language]

    cross_file_prompt = f"{comment} Repo Name: {row['repo_name']}\n"
    for snippet in row.get("context") or []:
        cross_file_prompt += (
            f"{comment} Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
        )

    in_file_prompt = (
        f"{comment} Path: {row['file_path']}\n"
        f"{row.get('import_statement') or ''}\n"
        f"{(row.get('cropped_code') or '').rstrip()}\n"
    )

    if max_token_nums is not None and max_token_nums > 0:
        cross_nums = count_tokens(cross_file_prompt)
        in_file_nums = count_tokens(in_file_prompt)
        exceed = cross_nums + in_file_nums - max_token_nums

        if exceed > 0:
            lines = cross_file_prompt.split("\n")
            i = 0
            for idx in range(len(lines) - 1, -1, -1):
                exceed -= count_tokens(lines[idx])
                i = idx
                if exceed < 0:
                    break
            cross_file_prompt = "\n".join(lines[:i]) + "\n\n"

    prompt = cross_file_prompt + in_file_prompt
    return re.sub(r"\n{4,}", "\n\n", prompt)


def _build_in_file_prompt(
    file_path: str, import_statement: str, cropped_code: str, language: str
) -> str:
    """Memory-mode prompt: just the target-file portion of the upstream layout.

    Same byte format as upstream's ``in_file_prompt`` block (the second
    half of ``construct_prompt``). The cross-file block is dropped — it
    has been ingested into the memory store and will be retrieved on
    demand.
    """
    comment = _COMMENT_SYMBOL.get(language, "#")
    return (
        f"{comment} Path: {file_path}\n"
        f"{import_statement or ''}\n"
        f"{(cropped_code or '').rstrip()}\n"
    )


_FENCE_RE = re.compile(
    # Open fence, optional lang tag, optional newline, body, close fence (or EOS).
    r"```(?:python|py|java)?[ \t]*\n?(.*?)(?:\n[ \t]*```|```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_PREAMBLE_RE = re.compile(
    # "Here is the next line:", "The next line is:", "Next line:", "Answer:"
    # — strip prose up to (and including) the first colon, then any
    # horizontal whitespace, then *optionally* a single newline. Critically,
    # we do NOT use ``\s*`` after the colon: ``\s`` matches ``\n`` too,
    # which would eat the indentation of a multi-line answer
    # (``"...should be:\n    return x"`` → ``"return x"`` instead of
    # ``"    return x"``). RepoBench ground truths preserve leading
    # whitespace; we must too.
    r"^\s*(?:here(?:'s| is)|the next line(?: is| should be)?|next line|answer)"
    r"[^:\n]*:[ \t]*(?:\n)?",
    re.IGNORECASE,
)


def _extract_code_completion(text: str, language: str = "python") -> str:
    """Peel off chat-mode wrappers around a code completion.

    Chat-tuned LLMs (gpt-4o, Claude, …) often wrap their output in:

    - a fenced code block ``` ```python\\n<code>\\n``` ```,
    - a prose preamble such as ``"Here is the next line: <code>"`` or
      ``"The next line should be:\\n<code>"``.

    This helper strips those wrappers so the downstream
    :func:`_get_first_line_not_comment` filter sees actual code rather
    than the model's chat envelope. Returns the original input
    (left-stripped of leading newlines for parity with
    ``_get_first_line_not_comment``) when no wrapper pattern matches —
    base completion models that already return raw code are unaffected.

    Companion to the prompt directive added under deviations.md R16.
    """
    if not text:
        return ""
    s = text

    # 1) Outer fenced code block.
    m = _FENCE_RE.search(s)
    if m is not None:
        inner = m.group(1)
        if inner.strip():
            return inner

    # 2) Prose preamble before the code line.
    m2 = _PREAMBLE_RE.match(s)
    if m2 is not None:
        return s[m2.end():]

    return s


def _format_snippet_for_memory(
    *,
    repo_name: str,
    snippet_path: str,
    identifier: str,
    snippet_code: str,
    language: str,
) -> str:
    """Snippet-as-message content for memory ingestion.

    The format mirrors the cross-file block layout of upstream's
    ``construct_prompt`` so embedding a snippet into memory and later
    retrieving it produces text the LLM has seen in upstream's prompt.
    The identifier line is added on top so identifier-anchored
    retrieval (the model is about to call ``foo.bar()``) has a literal
    string to match against.
    """
    comment = _COMMENT_SYMBOL.get(language, "#")
    parts = [f"{comment} Repo Name: {repo_name}"]
    if identifier:
        parts.append(f"{comment} Identifier: {identifier}")
    parts.append(f"{comment} Path: {snippet_path}")
    parts.append(snippet_code)
    return "\n".join(parts)


def _get_first_line_not_comment(code: str, language: str = "python") -> str:
    """Port of ``run.py::get_first_line_not_comment``.

    Skips leading blank lines, single-line comments (``#``/``//``), and
    multiline comment blocks (``\"\"\"``/``'''`` for Python, ``/* ... */``
    for Java). Returns the raw first line as a fallback.

    Faithful to upstream including its quirks: a single-line docstring
    ``\"\"\"hello\"\"\"`` triggers only the start branch (sets the flag and
    continues); the closing ``\"\"\"`` on the same line is never inspected,
    so the flag stays True and the *next* non-empty code line is also
    skipped. Same pattern for Java single-line block comments. Do not
    "fix" without an explicit deviation note — it is what upstream
    measures.
    """
    if language not in ("python", "java"):
        raise ValueError(f"Unsupported language: {language!r}")

    code = code.lstrip("\n")
    lines = code.split("\n")
    in_multiline = False

    if language == "python":
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not in_multiline and (
                stripped.startswith('"""') or stripped.startswith("'''")
            ):
                in_multiline = True
                continue
            if in_multiline and (
                stripped.endswith('"""') or stripped.endswith("'''")
            ):
                in_multiline = False
                continue
            if in_multiline:
                continue
            if stripped.startswith("#"):
                continue
            return line
    else:  # java
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not in_multiline and stripped.startswith("/*"):
                in_multiline = True
                continue
            if in_multiline and stripped.endswith("*/"):
                in_multiline = False
                continue
            if in_multiline:
                continue
            if stripped.startswith("//"):
                continue
            return line

    return lines[0] if lines else ""


def _codebleu_score(pred: str, gt: str, language: str) -> float | None:
    """CodeBLEU score in ``[0, 1]``, or ``None`` if unavailable.

    **Argument order: upstream-net-faithful** (see deviations.md R14).
    Upstream has two cancelling swaps in the chain from caller to the
    ``codebleu`` library:

    1. ``eval.py:45`` calls
       ``codebleu_score(generated, ground_truth, language)``.
    2. Inside ``codebleu_score`` (``evaluation/metrics.py:120-126``),
       upstream invokes
       ``calc_codebleu(ground_truths, predictions, language, weight,
       tokenizer=None)``.

    Binding through both substitutions, ``calc_codebleu`` receives
    ``(ground_truth, generated, language, ...)``. The ``codebleu``
    PyPI package's signature is
    ``calc_codebleu(references, predictions, lang, weights, tokenizer)``,
    so the net upstream call is
    ``calc_codebleu(references=ground_truth, predictions=generated)`` —
    the conventional non-swapped order.

    A 2026-05-21 re-audit corrected an earlier docstring that had
    flipped the conclusion (treating ``generated`` as references). The
    earlier implementation passed ``calc_codebleu([pred], [gt], ...)``,
    which inverted the order relative to upstream's net effect. The
    inversion was numerically meaningful because CodeBLEU's components
    are all asymmetric in ``(refs, hyps)``: BLEU has a brevity penalty
    and clip-based n-gram precision, weighted-BLEU adds keyword weights
    on the same asymmetric core, AST-match and data-flow-match are
    ratios over the references' subtrees / DFGs. See
    ``tests/benchmarks/test_repobench.py::TestEvaluate::
    test_codebleu_uses_upstream_net_argument_order`` for the binding
    lock.

    Returns ``None`` when:

    * The ``codebleu`` package is not installed (``ImportError``).
    * ``calc_codebleu`` raises any other exception. This is a robustness
      guard (R18), not the upstream-faithful behaviour. Upstream simply
      lets the exception escape (``eval.py`` has no try/except around
      the call). The 2026-05-21 ``repobench-mem0v2-infer-…`` and
      ``repobench-evermemos-…`` runs surfaced why we need it: the
      ``codebleu>=0.6,<1`` ceiling pin landed in commit ``11ed39e``
      made pip actually install ``codebleu`` (the prior ``>=0.6`` floor
      was getting skipped during dependency backtracking), but the
      installed wheel's tree-sitter bindings throw ``TypeError("an
      integer is required")`` on every parse. Without the broader
      catch, ``evaluate()`` raises and the engine wipes the entire
      per-task evaluation dict — ``exact_match`` / ``edit_similarity``
      vanish with ``codebleu``, even though they were already computed.
      Swallowing the CodeBLEU exception preserves EM/ES so the
      leaderboard row stays meaningful. ``avg_codebleu`` simply reads
      as ``null`` in ``summary.json`` for runs where every task hit
      this path.
    """
    try:
        from codebleu import calc_codebleu
    except ImportError:
        return None
    pred = pred.replace("\r", "")
    gt = gt.replace("\r", "")
    # Upstream net (see docstring): calc_codebleu(references=gt, predictions=pred).
    try:
        res = calc_codebleu(
            [gt], [pred], language, [0.25, 0.25, 0.25, 0.25], tokenizer=None
        )
    except Exception as e:  # noqa: BLE001 — log + skip, preserves EM/ES (R18)
        logger.warning(
            "repobench.codebleu_failed lang=%s error_type=%s error=%s",
            language,
            type(e).__name__,
            e,
        )
        return None
    return float(res["codebleu"])


def _parse_bound(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse a YYYY-MM-DD string into a UTC datetime.

    Upstream compares with ``pd.to_datetime('YYYY-MM-DD').tz_localize('UTC')``,
    which yields midnight UTC — so the upper bound at ``end_of_day=False``
    excludes entries later on the same day. We preserve that exact semantics.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_created_at(value: str) -> datetime | None:
    """Parse the dataset's ``created_at`` — e.g. ``2023-12-28 05:47:18+00:00``."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@register_adapter
class RepoBenchAdapter(BenchmarkAdapter):
    name = "repobench"
    granularity = TaskGranularity.CROSS_FILE
    requires_sandbox = False

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        language = params.get("language", "python").lower()
        if language not in _DATASETS:
            raise ValueError(
                f"Unsupported language {language!r}. Options: {sorted(_DATASETS)}"
            )

        splits_param = params.get("splits") or list(_SPLITS)
        splits = [s for s in splits_param if s in _SPLITS]
        if not splits:
            raise ValueError(
                f"No valid splits in {splits_param!r}. Options: {list(_SPLITS)}"
            )

        level_filter = set(params.get("levels") or _DEFAULT_LEVELS)
        start_date = _parse_bound(params.get("start_date", "2023-12-01"))
        end_date = _parse_bound(params.get("end_date", "2023-12-31"))
        max_token_nums = params.get("max_token_nums", 15800)
        tokenizer_pref = params.get("tokenizer", "tiktoken:cl100k_base")
        count_tokens = _get_token_counter(tokenizer_pref)
        limit = params.get("limit")
        retrieval_query_lines = int(
            params.get("retrieval_query_lines", _DEFAULT_RETRIEVAL_QUERY_LINES)
        )

        emitted = 0
        for split_name in splits:
            ds = load_dataset(_DATASETS[language], split=split_name)
            for idx, row in enumerate(ds):
                if limit is not None and emitted >= limit:
                    return

                if row.get("level") not in level_filter:
                    continue

                ts = _parse_created_at(row.get("created_at", ""))
                if ts is None or not (start_date <= ts <= end_date):
                    continue

                prompt = _construct_prompt(
                    row, language, count_tokens, max_token_nums
                )

                # The original ``context`` list, kept verbatim so
                # ``ingest_for_memory`` and ``gold_snippet_index``
                # stay coherent. Upstream's ``context`` is
                # ``list[{identifier, path, snippet}]``.
                memory_snippets = list(row.get("context") or [])

                context_files = [
                    {"path": s.get("path", ""), "content": s.get("snippet", "")}
                    for s in memory_snippets
                ]

                yield Task(
                    task_id=f"repobench-{language}-{split_name}-{idx}",
                    benchmark=self.name,
                    granularity=self.granularity,
                    prompt=prompt,
                    metadata={
                        "language": language,
                        "split": split_name,
                        "level": row.get("level", ""),
                        "repo_name": row.get("repo_name", ""),
                        "file_path": row.get("file_path", ""),
                        "created_at": row.get("created_at", ""),
                        "gold_snippet_index": row.get("gold_snippet_index", -1),
                        # Carried so ``format_memory_prompt`` and
                        # ``get_retrieval_query`` can rebuild the in-file
                        # prompt without parsing ``task.prompt`` again.
                        "cropped_code": row.get("cropped_code", "") or "",
                        "import_statement": row.get("import_statement", "") or "",
                        # The original cross-file snippet list, used by
                        # ``ingest_for_memory``. Kept under a private-ish
                        # key so consumers don't accidentally include it
                        # in the prompt — the no-memory baseline already
                        # has them inside ``prompt``.
                        "memory_snippets": memory_snippets,
                        "retrieval_query_lines": retrieval_query_lines,
                        "synthetic_workspace_dir": params.get("synthetic_workspace_dir"),
                        "all_code": row.get("all_code") or "",
                    },
                    context_files=context_files or None,
                    ground_truth=row.get("next_line", ""),
                )
                emitted += 1

    # ------------------------------------------------------------------
    # Memory integration — RepoCoder-inspired (arXiv:2303.12570).
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """No-memory baseline: upstream prompt prefixed with chat directive.

        ``task.prompt`` is the upstream-faithful ``construct_prompt`` output
        (cross-file + in-file blocks). We prepend the R16 completion
        directive so chat-tuned LLMs emit a single line of code instead
        of chat-mode prose. Base completion models effectively ignore
        the directive because it appears outside any code block and the
        actual code continuation context (the in-file block) comes last.
        """
        meta = task.metadata or {}
        language = meta.get("language", "python")
        return _completion_instruction(language) + task.prompt

    def format_memory_prompt(self, task: Task) -> str:
        """In-file portion only, prefixed with chat directive.

        Cross-file snippets are in the memory store (R4); the LLM sees
        only the in-file block. Same R16 directive as ``format_prompt``
        so memory-mode and no-memory rows are graded on equivalent
        chat-discipline.
        """
        meta = task.metadata or {}
        language = meta.get("language", "python")
        in_file = _build_in_file_prompt(
            file_path=meta.get("file_path", ""),
            import_statement=meta.get("import_statement", ""),
            cropped_code=meta.get("cropped_code", ""),
            language=language,
        )
        return _completion_instruction(language) + in_file

    def get_retrieval_query(self, task: Task) -> str | None:
        """RepoCoder-style query: imports + tail(cropped_code, N).

        Default ``N=20`` (``retrieval_query_lines`` param), matching
        RepoCoder's ``window_size``. The tail of the cropped code is
        the strongest local signal for what the next line will look
        like; the import statement narrows retrieval to symbols
        actually in scope. Returns ``None`` when both are empty so the
        engine falls back to the full LLM prompt as the query.
        """
        meta = task.metadata or {}
        pre_code_prompt = "Find information related to the following code that would be useful when generating the next line of code.. Your job is to search related memories and provide the most relevant information:\n\n"
        cropped = (meta.get("cropped_code") or "").rstrip("\n")
        imports = (meta.get("import_statement") or "").strip()
        n = int(meta.get("retrieval_query_lines", _DEFAULT_RETRIEVAL_QUERY_LINES))
        tail_lines = cropped.split("\n")[-n:] if cropped else []
        tail = "\n".join(tail_lines)
        if imports and tail:
            return f"{imports}\n\n{tail}"
        if imports or tail:
            return imports or tail
        return None

    async def ingest_for_memory(
        self, task: Task, memory: "MemoryMiddleware"
    ) -> None:
        """Snippet-as-message ingestion (literature-review Option A).

        Each cross-file snippet becomes one ``post_response`` call with
        ``source_id = "ctx:{idx}"`` — matching the dataset's
        ``gold_snippet_index`` so the engine's
        ``retrieved_context_ids`` plumbing measures recall directly
        against ground truth.

        The cropped target file is also ingested with
        ``source_id = "self"``: same-file references (e.g. earlier
        helper definitions in the cropped portion) are a real source
        of next-line content, especially in the ``in_file`` split.

        When ``task.metadata["synthetic_workspace_dir"]`` is set,
        filesystem materialisation runs **first** (before any
        ``post_response``), so code indexers can observe the tree.
        """
        _materialize_synthetic_workspace(task)
        meta = task.metadata or {}
        ws_ctx: dict[str, str] = {}
        if meta.get("code_workspace"):
            ws_ctx["code_workspace"] = str(meta["code_workspace"])
        language = meta.get("language", "python")
        repo_name = meta.get("repo_name", "")
        snippets = meta.get("memory_snippets") or []

        for idx, snippet in enumerate(snippets):
            content = _format_snippet_for_memory(
                repo_name=repo_name,
                snippet_path=snippet.get("path", ""),
                identifier=snippet.get("identifier", ""),
                snippet_code=snippet.get("snippet", "") or "",
                language=language,
            )
            await memory.post_response(
                "",
                content,
                {
                    "source_id": f"ctx:{idx}",
                    "path": snippet.get("path", ""),
                    "identifier": snippet.get("identifier", ""),
                    "repo_name": repo_name,
                    "task_id": task.task_id,
                    **ws_ctx,
                },
            )

        # Self-reference ingest (the cropped target file). Comment marker
        # tells the LLM "this is the file you're editing" if the
        # middleware echoes the content verbatim on retrieval.
        cropped = meta.get("cropped_code", "")
        if cropped:
            comment = _COMMENT_SYMBOL.get(language, "#")
            file_path = meta.get("file_path", "")
            self_content = (
                f"{comment} Repo Name: {repo_name}\n"
                f"{comment} Path: {file_path} (current file)\n"
                f"{cropped}"
            )
            await memory.post_response(
                "",
                self_content,
                {
                    "source_id": "self",
                    "path": file_path,
                    "repo_name": repo_name,
                    "task_id": task.task_id,
                    **ws_ctx,
                },
            )

    async def teardown_task(self, task: Task) -> None:
        """Remove the per-task synthetic repo tree when enabled."""
        _teardown_synthetic_workspace(task)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self, task: Task, generated_output: str, sandbox_mgr, sandbox=None
    ) -> dict[str, Any]:
        meta = task.metadata or {}
        language = meta.get("language", "python")
        gt = task.ground_truth or ""
        # R16: strip chat-mode wrappers (fenced code blocks, "Here is the next
        # line:" preambles) before the upstream-faithful
        # ``_get_first_line_not_comment`` filter. A base completion model that
        # already returns raw code is unaffected because no wrapper pattern
        # matches.
        pred_raw = _extract_code_completion(generated_output or "", language=language)
        pred = _get_first_line_not_comment(pred_raw, language=language)

        exact_match = pred.split() == gt.split()
        # Upstream rounds to 5 decimal places (``evaluation/metrics.py``
        # line 25, 51). ``fuzz.ratio`` already returns 0-100; we keep
        # the same scale for direct comparison with upstream JSONL.
        edit_sim = round(float(_fuzz_ratio(pred, gt)), 5)

        # ``category`` = split lets the engine emit
        # ``accuracy_cat_<split>`` per-split breakdowns automatically
        # (see ``RunEngine._aggregate``). Upstream reports these as
        # data-points-weighted means; the engine's flat mean is
        # numerically equivalent (Σ x / N == Σ (mean_split · n_split) /
        # Σ n_split).
        split_name = meta.get("split", "")
        result: dict[str, Any] = {
            "pass": bool(exact_match),
            "exact_match": float(bool(exact_match)),
            "edit_similarity": edit_sim,
            "category": split_name,
            "split": split_name,
            "level": meta.get("level", ""),
        }
        cb = _codebleu_score(pred, gt, language)
        if cb is not None:
            result["codebleu"] = round(cb * 100, 5)
        return result

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""

    def get_default_params(self) -> dict[str, Any]:
        return {
            "language": "python",
            "splits": list(_SPLITS),
            "levels": list(_DEFAULT_LEVELS),
            "start_date": "2023-12-01",
            "end_date": "2023-12-31",
            "max_token_nums": 15800,
            "tokenizer": "tiktoken:cl100k_base",
            "retrieval_query_lines": _DEFAULT_RETRIEVAL_QUERY_LINES,
            "synthetic_workspace_dir": None,
        }

    def metrics_schema(self) -> list[str]:
        """Stable column set for RepoBench summary.json.

        Headline numbers (``pass_rate`` / ``avg_exact_match`` /
        ``avg_edit_similarity`` / ``avg_codebleu``) are the
        upstream-faithful aggregates. Per-split breakdowns
        (``accuracy_cat_<split>``) are emitted automatically by the
        engine via the ``category`` key on each evaluation dict.

        Latency keys are populated by ``execute_single_shot`` when
        memory is active; without memory ``avg_search_latency_ms``
        stays ``None`` ("not measured") and the no-memory baseline's
        ``avg_inference_latency_ms`` is the wall-clock of the LLM call
        alone.
        """
        return [
            # Grading
            "pass_rate", "accuracy",
            "avg_exact_match",
            "avg_edit_similarity",
            "avg_codebleu",
            # Per-split breakdowns
            "accuracy_cat_cross_file_first",
            "accuracy_cat_cross_file_random",
            "accuracy_cat_in_file",
            # Latency
            "avg_search_latency_ms",
            "avg_inference_latency_ms",
            "avg_total_latency_ms",
            # Cost / scale
            "total_tokens_in", "total_tokens_out",
            "avg_tokens_per_task",
            "total_tasks",
            # Memory-system internal token cost (separate from bench LLM)
            *MEMORY_TOKEN_SCHEMA_KEYS,
            # Run health
            "passed", "failed", "errors", "timeouts",
        ]
