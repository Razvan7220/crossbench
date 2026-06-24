"""CrossCodeEval adapter — cross-file line completion (Ding et al., NeurIPS 2023).

Faithful port of the official benchmark:
  https://github.com/amazon-science/cceval
  https://arxiv.org/abs/2310.11248

Data
----
Per-language mirrors on the HuggingFace Hub (the historical
``Fsoft-AIC/CrossCodeEval`` path does not exist):
  - ``ZHENGRAN/cross_code_eval_python``     (2,665 rows)
  - ``ZHENGRAN/cross_code_eval_java``       (2,139 rows)
  - ``ZHENGRAN/cross_code_eval_csharp``     (1,768 rows)
  - ``ZHENGRAN/cross_code_eval_typescript`` (3,356 rows)

Row schema (per the actual HF dataset, verified 2026-05-06):
  ``prompt``                 : str — in-file left context ending at the cursor
  ``right_context``          : str — in-file context *after* the cursor
                               (present in the dataset but **not used** by
                               upstream's reference inference scripts)
  ``groundtruth``            : str — target line(s) to predict
  ``metadata``               : {task_id, repository, file,
                               context_start_lineno, groundtruth_start_lineno,
                               right_context_start_lineno}
  ``crossfile_context_retrieval``       : {text, list[{filename, retrieved_chunk, score}]}
  ``crossfile_context_retrievalwref``   : {text, list[...]}

Both CFC variants ship **exactly 5 pre-retrieved chunks** (top-K=5 from
upstream's ``prompt_builder/augment_with_cfc.py``). The full repo source
is NOT redistributed in the HF mirror — the README says "Please email us
if you need the raw data". So the memory system can only ingest the 5
already-pre-selected chunks per task, not the entire repository.

Retrieval settings
------------------
``retrieval_setting`` selects which pre-computed CFC the model sees:
  - ``"retrieval"``          → ``crossfile_context_retrieval.text``     (RG-1 in the paper)
  - ``"retrieval-with-ref"`` → ``crossfile_context_retrievalwref.text`` (Oracle in the paper)
  - ``"none"``               → no cross-file context (in-file-only baseline)

The "with-ref" variant is upstream's **oracle query** setup
(``augment_with_cfc.py::57-62``): the BM25/UniXCoder query is the last
10 lines of ``prompt + groundtruth`` rather than just ``prompt`` — i.e.
the retriever cheats by knowing the answer. The chunk list is otherwise
constructed identically to ``rg1``. Use the with-ref variant only as an
upper-bound diagnostic.

Prompt assembly — matches ``scripts/vllm_inference.py::prepare_prompt``::

    {cross_file_context_text}
    {prompt}

A single ``\\n`` separator. ``right_context`` is intentionally not
included (upstream's reference scripts ignore it). Token-budget
truncation is left to the LLM layer — the upstream reference uses a
StarCoder/tiktoken tokenizer and a fixed split between CFC and prompt
budgets (12.8k / 3.4k under the modern script defaults; the paper uses
a tighter 512 BPE for CFC at smaller-context CodeGen).

Postprocessing — ``scripts/eval_metric.py`` + ``eval_utils.py``:
  1. ``postprocess_code_lines(prompt, completion, parser, lang)`` —
     language-specific completion truncation:
       - **Java / C# / TypeScript**: ``get_bracket_lang_statement``
         returns the prefix up to (and including) the first ``;``,
         ``{``, or ``}``. We port this byte-faithfully.
       - **Python**: tree-sitter parse-iteration (append one char,
         try ``is_parse_valid``, stop at first valid suffix ending
         in ``\\n``). We replace this with a line-count truncation
         to ``max(1, len(groundtruth.splitlines()))`` lines — the
         tree-sitter dependency is heavy and the line-count
         truncation matches in the common single-line case. See
         ``deviations.md`` for the audit.
  2. ``remove_comments`` — applied to BOTH prediction and groundtruth:
     ``re.sub(r'#.*', '', code)`` then ``re.sub(r'//.*', '', code)``.
     This is the only comment stripping upstream does — it does NOT
     touch ``/* ... */`` blocks, triple-quoted strings, or docstrings
     (literal upstream behavior, preserved here).

Metrics — ``scripts/eval_metric.py`` lines 19-150:
  - ``exact_match``     — list-of-stripped-non-empty-lines equality
                          on post-processed pred and gt.
  - ``edit_similarity`` — ``rapidfuzz.fuzz.ratio(pred.strip(),
                          gt.strip())`` on the WHOLE prediction
                          (not line-by-line). Returns 0-100.
  - ``id_em``           — ``pred_ids == target_ids`` on the
                          ordered identifier lists (NOT set-based).
  - ``id_precision``/``id_recall``/``id_f1`` — set-based TP/FP/FN.

Per-language aggregation: each language is a separate dataset / a
separate run; upstream runs ``eval.py`` once per language, no
per-language weighting in the reference script. CrossBench preserves
the per-task ``language`` and the engine emits per-language breakdowns
via ``evaluation["category"] = language``.

Memory integration
------------------
When a memory middleware is attached:

  - ``ingest_for_memory`` iterates the per-task chunk list (5 chunks
    from ``crossfile_context_retrieval.list`` or
    ``crossfile_context_retrievalwref.list`` depending on
    ``retrieval_setting``). Each chunk is ingested as one
    ``post_response`` call with ``source_id = "ctx:{idx}"`` so the
    engine's ``retrieved_context_ids`` plumbing measures recall
    against per-chunk identity.
  - ``format_memory_prompt`` returns the in-file ``prompt`` only —
    no CFC text, since those chunks now live in memory.
  - ``get_retrieval_query`` returns the last
    ``retrieval_query_lines`` non-empty lines of ``prompt`` (default
    10, matching upstream's ``augment_with_cfc.py:57`` query
    construction for the ``rg1`` setting).

When ``retrieval_setting = "none"`` and memory is enabled, the chunk
list is empty so memory has nothing to retrieve — equivalent to the
no-CFC baseline.
"""

from __future__ import annotations

import keyword
import re
from typing import TYPE_CHECKING, Any, Iterator

from datasets import load_dataset
from rapidfuzz.fuzz import ratio as _fuzz_ratio

from cross_bench.benchmarks.base import (
    MEMORY_TOKEN_SCHEMA_KEYS,
    BenchmarkAdapter,
)
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity
from cross_bench.benchmarks.code_workspace import _materialize_synthetic_workspace, _teardown_synthetic_workspace

if TYPE_CHECKING:
    from cross_bench.systems.base import MemoryMiddleware

_LANG_TO_DATASET = {
    "python": "ZHENGRAN/cross_code_eval_python",
    "java": "ZHENGRAN/cross_code_eval_java",
    "csharp": "ZHENGRAN/cross_code_eval_csharp",
    "typescript": "ZHENGRAN/cross_code_eval_typescript",
}

_RETRIEVAL_FIELD: dict[str, str | None] = {
    "retrieval": "crossfile_context_retrieval",
    "retrieval-with-ref": "crossfile_context_retrievalwref",
    "none": None,
}

# Languages where upstream's ``postprocess_code_lines`` uses
# ``get_bracket_lang_statement`` (first ;, {, } scan). Python is the
# only language that uses tree-sitter parse-iteration upstream.
_BRACKET_LANGS: frozenset[str] = frozenset({"java", "csharp", "typescript"})

# Regexes copied verbatim from upstream ``eval_utils.py``.
_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_STRING_RE = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
_TOKEN_RE = re.compile(r"\w+")
# Upstream ``eval_utils.py:117-120`` — exactly two passes, in this order.
# Does NOT touch /* ... */, triple-quoted strings, or docstrings.
_HASH_COMMENT_RE = re.compile(r"#.*")
_SLASH_COMMENT_RE = re.compile(r"//.*")

_DEFAULT_RETRIEVAL_QUERY_LINES = 10  # upstream ``augment_with_cfc.py:57``

# Chat-LLM completion directive (deviations.md C18 — mirrors RepoBench R16).
#
# Upstream CrossCodeEval prompts are raw code (``cfc_text + '\n' + prompt``)
# designed for base completion models (Codex / CodeT5 / StarCoder etc.). Chat-
# tuned LLMs (gpt-4o, Claude, …) see raw code and respond in chat-mode prose
# ("This file imports ..."), defeating the upstream EM / ES / ID-EM / ID-F1
# graders. We prepend a short directive in ``format_prompt`` and
# ``format_memory_prompt`` (the LLM-facing entry points) so the bench LLM
# emits a single line of code. The dataset-level prompt strings produced by
# ``load_tasks`` and ``_build_in_file_prompt`` stay upstream-faithful — the
# directive lives one layer up.
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
    "csharp": (
        "Complete the next line of C# code from the source file shown "
        "below. Output ONLY that single next line, preserving any leading "
        "whitespace exactly. Do NOT wrap the line in a fenced code block. "
        "Do NOT add explanations, comments, or any other text.\n\n"
    ),
    "typescript": (
        "Complete the next line of TypeScript code from the source file "
        "shown below. Output ONLY that single next line, preserving any "
        "leading whitespace exactly. Do NOT wrap the line in a fenced code "
        "block. Do NOT add explanations, comments, or any other text.\n\n"
    ),
}


def _completion_instruction(language: str) -> str:
    """Return the LLM directive for the requested language.

    Defaults to the Python directive for unknown languages so we never
    drop the directive entirely (a chat LLM with no directive is the
    failure mode that motivated C18 / R16 in the first place).
    """
    return _COMPLETION_INSTRUCTION_BY_LANG.get(
        language, _COMPLETION_INSTRUCTION_BY_LANG["python"]
    )


# Fence and prose-preamble strippers for ``_extract_code_completion``.
# Mirrors RepoBench's ``_FENCE_RE`` / ``_PREAMBLE_RE`` (see R16). Lang tags
# extended for CrossCodeEval's four languages.
_FENCE_RE = re.compile(
    # Open fence, optional lang tag, optional newline, body, close fence
    # (or EOS). ``#`` in ``c#`` is a literal in non-VERBOSE re mode.
    r"```(?:python|py|java|csharp|cs|c#|typescript|ts)?[ \t]*\n?(.*?)"
    r"(?:\n[ \t]*```|```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_PREAMBLE_RE = re.compile(
    # "Here is the next line:", "The next line is:", "Next line:", "Answer:"
    # — strip prose up to (and including) the first colon, then any
    # horizontal whitespace, then *optionally* a single newline. We do NOT
    # use ``\s*`` after the colon: ``\s`` matches ``\n`` too, which would
    # eat the indentation of a multi-line answer
    # (``"...should be:\n    return x"`` → ``"return x"`` instead of
    # ``"    return x"``). Ground truths often preserve leading whitespace;
    # we must too.
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
    :func:`_postprocess_completion` filter sees actual code rather than
    the model's chat envelope. Returns the original input when no
    wrapper pattern matches — base completion models that already
    return raw code are unaffected.

    Companion to the prompt directive added under deviations.md C18.
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


_PYTHON_KEYWORDS = frozenset(k for k in keyword.kwlist if k not in ("True", "False"))
_JAVA_KEYWORDS = frozenset(
    "abstract assert boolean break byte case catch char class const continue "
    "default do double else enum extends final finally float for goto if "
    "implements import instanceof int interface long native new package private "
    "protected public return short static strictfp super switch synchronized "
    "this throw throws transient try void volatile while true false null".split()
)
_CSHARP_KEYWORDS = frozenset(
    "abstract as base bool break byte case catch char checked class const "
    "continue decimal default delegate do double else enum event explicit extern "
    "false finally fixed float for foreach goto if implicit in int interface "
    "internal is lock long namespace new null object operator out override params "
    "private protected public readonly ref return sbyte sealed short sizeof "
    "stackalloc static string struct switch this throw true try typeof uint "
    "ulong unchecked unsafe ushort using virtual void volatile while "
    "async await get set var".split()
)
_TYPESCRIPT_KEYWORDS = frozenset(
    "break case catch class const continue debugger default delete do else enum "
    "export extends false finally for function if import in instanceof new null "
    "return super switch this throw true try typeof var void while with as async "
    "await of yield let any boolean constructor declare module number private "
    "protected public static string type from interface namespace readonly".split()
)
_LANG_KEYWORDS: dict[str, frozenset[str]] = {
    "python": _PYTHON_KEYWORDS,
    "java": _JAVA_KEYWORDS,
    "csharp": _CSHARP_KEYWORDS,
    "typescript": _TYPESCRIPT_KEYWORDS,
}


def _extract_identifiers(source: str, lang: str) -> list[str]:
    """Regex identifier extraction — ``eval_utils.py::extract_identifiers``."""
    stripped = _STRING_RE.sub("", source)
    kw = _LANG_KEYWORDS.get(lang, frozenset())
    return [
        t for t in _TOKEN_RE.findall(stripped)
        if _IDENTIFIER_RE.fullmatch(t) and t not in kw
    ]


def _get_bracket_lang_statement(completion: str) -> str:
    """Port of upstream ``eval_utils.py::get_bracket_lang_statement``
    (lines 96-102).

    Returns the prefix of ``completion`` up to and including the first
    ``;``, ``{``, or ``}``. Used for Java / C# / TypeScript completion
    truncation. If none of those characters appears, returns
    ``completion`` unchanged.

    Note: this is a literal byte scan — it does NOT skip those
    characters when they appear inside strings or comments. Faithful
    to upstream behavior.
    """
    for i, ch in enumerate(completion):
        if ch in (";", "{", "}"):
            return completion[: i + 1]
    return completion


def _truncate_to_target_lines(prediction: str, target: str) -> str:
    """Python-only fallback for ``postprocess_code_lines``.

    Upstream's Python path uses tree-sitter parse-iteration; we use a
    line-count truncation to ``max(1, len(target.splitlines()))``
    lines. For the line-completion task (the common single-line case)
    this matches upstream because there is exactly one statement to
    keep. Multi-line predictions diverge: CrossBench keeps an extra
    line if the model emits one, where upstream's tree-sitter pass
    would have stopped at the first statement boundary.
    """
    n = max(1, len(target.splitlines()))
    return "\n".join(prediction.splitlines()[:n])


def _postprocess_completion(
    prediction: str, target: str, language: str
) -> str:
    """Per-language completion truncation, port of upstream's
    ``postprocess_code_lines`` dispatcher (``eval_utils.py:161-168``).
    """
    if language in _BRACKET_LANGS:
        return _get_bracket_lang_statement(prediction)
    # Python — tree-sitter upstream, line-count here.
    return _truncate_to_target_lines(prediction, target)


def _remove_comments(code: str) -> str:
    """Port of upstream ``eval_utils.py::remove_comments`` (lines 117-120).

    Two regex passes in this order:
        re.sub(r'#.*',  '', code)
        re.sub(r'//.*', '', code)

    Faithful to upstream including its limitations: does NOT handle
    ``/* ... */`` blocks, triple-quoted Python strings, JSDoc, or
    ``#`` / ``//`` characters appearing inside string literals.
    """
    code = _HASH_COMMENT_RE.sub("", code)
    code = _SLASH_COMMENT_RE.sub("", code)
    return code


def _stripped_nonempty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _id_f1(pred_ids: list[str], target_ids: list[str]) -> tuple[int, int, int]:
    """Set-based TP / FP / FN — ``eval_metric.py::compute_id_match``."""
    pred_set = set(pred_ids)
    target_set = set(target_ids)
    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)
    return tp, fp, fn


def _build_in_file_prompt(prompt: str) -> str:
    """Memory-mode LLM prompt — just the in-file ``prompt`` field.

    Upstream's prompt assembly is ``cfc_text + '\\n' + prompt`` when
    CFC is included; without CFC the prompt is just ``prompt``. Memory
    mode falls into the latter case (CFC is in memory now).
    """
    return prompt


def _format_chunk_for_memory(
    *, filename: str, chunk_text: str, score: float | None
) -> str:
    """Snippet-as-message content for memory ingestion.

    Format mirrors what upstream's pre-rendered ``cfc.text`` looks
    like (``# the below code fragment is found in {filename}``-style
    headers from ``augment_with_cfc.py``), so embedding a chunk into
    memory and later retrieving it produces text the LLM has seen in
    upstream's prompt. The score line is kept as a soft hint to the
    answering LLM but does not affect retrieval.
    """
    parts = [f"# the below code fragment is found in {filename}"]
    if score is not None:
        parts.append(f"# retrieval_score={score:.4f}")
    parts.append(chunk_text)
    return "\n".join(parts)


@register_adapter
class CrossCodeEvalAdapter(BenchmarkAdapter):
    name = "crosscodeeval"
    granularity = TaskGranularity.CROSS_FILE
    requires_sandbox = False

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        language = params.get("language", "python").lower()
        setting = params.get("retrieval_setting", "retrieval")
        limit = params.get("limit")
        retrieval_query_lines = int(
            params.get("retrieval_query_lines", _DEFAULT_RETRIEVAL_QUERY_LINES)
        )

        if language not in _LANG_TO_DATASET:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Options: {sorted(_LANG_TO_DATASET)}"
            )
        if setting not in _RETRIEVAL_FIELD:
            raise ValueError(
                f"Unknown retrieval_setting {setting!r}. "
                f"Options: {sorted(_RETRIEVAL_FIELD)}"
            )

        ds = load_dataset(_LANG_TO_DATASET[language], split="train")
        cfc_key = _RETRIEVAL_FIELD[setting]

        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break

            cfc_text = ""
            cfc_chunks: list[dict[str, Any]] = []
            memory_snippets: list[dict[str, Any]] = []
            if cfc_key:
                cfc = row.get(cfc_key) or {}
                if isinstance(cfc, dict):
                    cfc_text = cfc.get("text", "") or ""
                    raw_list = cfc.get("list") or []
                    cfc_chunks = [
                        {
                            "filename": (item or {}).get("filename", ""),
                            # Upstream HF dataset uses ``retrieved_chunk``;
                            # tolerate the alternative ``chunk`` name in
                            # case a future mirror normalizes it.
                            "retrieved_chunk": (
                                (item or {}).get("retrieved_chunk")
                                or (item or {}).get("chunk", "")
                            ),
                            "score": (item or {}).get("score"),
                        }
                        for item in raw_list
                    ]
                    memory_snippets = [
                        {
                            "path": (item or {}).get("filename", ""),
                            "snippet": (item or {}).get("retrieved_chunk", ""),
                        }
                        for item in raw_list
                    ]

            in_file_prompt = row.get("prompt", "") or ""
            assembled = (
                f"{cfc_text}\n{in_file_prompt}" if cfc_text else in_file_prompt
            )

            metadata = dict(row.get("metadata") or {})
            task_id = metadata.get("task_id") or f"{language}_{i}"
            metadata.update({
                "language": language,
                "retrieval_setting": setting,
                "right_context": row.get("right_context", ""),
                "has_crossfile_context": bool(cfc_text),
                # Carried so ``format_memory_prompt`` and
                # ``get_retrieval_query`` can rebuild the in-file
                # prompt without parsing ``task.prompt`` apart again.
                "prompt_no_cfc": in_file_prompt,
                # The per-chunk list, used by ``ingest_for_memory``.
                # Empty when ``retrieval_setting = "none"``.
                "cfc_chunks": cfc_chunks,
                "retrieval_query_lines": retrieval_query_lines,
                # synthetic workspace directory
                "synthetic_workspace_dir": params.get("synthetic_workspace_dir"),
                "repo_name": row.get("metadata", {}).get("repository", ""),
                "memory_snippets": memory_snippets,
                "cropped_code": row.get("cropped_code", ""),
                "all_code": row.get("prompt", ""),
                "file_path": row.get("metadata", {}).get("file", ""),

            })

            yield Task(
                task_id=task_id,
                benchmark=self.name,
                granularity=self.granularity,
                prompt=assembled,
                metadata=metadata,
                ground_truth=row["groundtruth"],
            )

    # ------------------------------------------------------------------
    # Memory integration — RepoCoder-inspired (arXiv:2303.12570),
    # query construction matches upstream's ``augment_with_cfc.py:57``.
    # ------------------------------------------------------------------

    def format_prompt(self, task: Task) -> str:
        """No-memory baseline: upstream prompt prefixed with chat directive.

        ``task.prompt`` is the upstream-faithful ``cfc_text + '\\n' +
        prompt`` assembly (or just ``prompt`` when ``retrieval_setting
        = "none"``). We prepend the C18 completion directive so chat-
        tuned LLMs emit a single line of code instead of chat-mode
        prose. Base completion models effectively ignore the directive
        (it sits outside any code block and the in-file continuation
        comes last).
        """
        meta = task.metadata or {}
        language = meta.get("language", "python")
        return _completion_instruction(language) + task.prompt

    def format_memory_prompt(self, task: Task) -> str:
        """In-file ``prompt`` only, prefixed with chat directive.

        CFC chunks live in the memory store now; the LLM sees only the
        in-file portion. Same C18 directive as ``format_prompt`` so
        memory-mode and no-memory rows are graded on equivalent
        chat-discipline.
        """
        meta = task.metadata or {}
        language = meta.get("language", "python")
        in_file = _build_in_file_prompt(meta.get("prompt_no_cfc") or task.prompt)
        return _completion_instruction(language) + in_file

    def get_retrieval_query(self, task: Task) -> str | None:
        """Last ``retrieval_query_lines`` non-empty lines of ``prompt``.

        Default 10, matching upstream's ``rg1`` query construction
        in ``augment_with_cfc.py:57`` (``query = last 10 non-empty
        lines of prompt``). Comparison against upstream's RG-1
        published numbers is fair on this query shape.
        """
        meta = task.metadata or {}
        prompt = meta.get("prompt_no_cfc") or task.prompt or ""
        n = int(meta.get("retrieval_query_lines", _DEFAULT_RETRIEVAL_QUERY_LINES))
        non_empty = [ln for ln in prompt.split("\n") if ln.strip()]
        if not non_empty:
            return None
        return "\n".join(non_empty[-n:])

    async def ingest_for_memory(
        self, task: Task, memory: "MemoryMiddleware"
    ) -> None:
        """Per-task ingestion of the pre-retrieved CFC chunks.

        Each chunk becomes one ``post_response`` call with
        ``source_id = "ctx:{idx}"`` so the engine's
        ``retrieved_context_ids`` plumbing surfaces a
        per-chunk-identity recall against the dataset. With
        ``retrieval_setting = "none"`` the chunk list is empty and
        this is a no-op.

        Note: the HF dataset only ships top-K=5 pre-retrieved chunks
        (not the full repository), so memory's recall is bounded by
        whatever upstream's BM25/UniXCoder pass already promoted.
        See deviations.md.
        """
        _materialize_synthetic_workspace(task)
        meta = task.metadata or {}
        chunks = meta.get("cfc_chunks") or []
        repository = meta.get("repo_name", "")
        ws_ctx: dict[str, str] = {}
        if meta.get("code_workspace"):
            ws_ctx["code_workspace"] = str(meta["code_workspace"])
        for idx, chunk in enumerate(chunks):
            # Tolerate both ``retrieved_chunk`` (the upstream HF
            # dataset name) and ``chunk`` (a possible future-mirror
            # alias). load_tasks already normalizes via the same
            # fallback, but ingest_for_memory may receive externally
            # constructed task metadata in tests / custom drivers.
            chunk_text = (
                chunk.get("retrieved_chunk")
                or chunk.get("chunk")
                or ""
            )
            if not chunk_text.strip():
                continue
            content = _format_chunk_for_memory(
                filename=chunk.get("filename", ""),
                chunk_text=chunk_text,
                score=chunk.get("score"),
            )
            await memory.post_response(
                "",
                content,
                {
                    "source_id": f"ctx:{idx}",
                    "filename": chunk.get("filename", ""),
                    "score": chunk.get("score"),
                    "repository": repository,
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

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        meta = task.metadata or {}
        language = meta.get("language", "python")
        target = (task.ground_truth or "").rstrip("\n")
        # C18: strip chat-mode wrappers (fenced code blocks, "Here is the
        # next line:" preambles) before the upstream-faithful
        # ``_postprocess_completion`` truncator. A base completion model
        # that already returns raw code is unaffected because no wrapper
        # pattern matches.
        raw_pred = _extract_code_completion(
            generated_output or "", language=language
        )

        # Step 1 — language-specific completion truncation
        # (``postprocess_code_lines``).
        prediction = _postprocess_completion(raw_pred, target, language)

        # Step 2 — strip ``#`` and ``//`` comments from BOTH sides
        # (upstream applies ``remove_comments`` to pred and gt).
        pred_no_comments = _remove_comments(prediction)
        target_no_comments = _remove_comments(target)

        # EM — list-of-stripped-non-empty-lines equality on the
        # post-processed text.
        pred_lines = _stripped_nonempty_lines(pred_no_comments)
        gt_lines = _stripped_nonempty_lines(target_no_comments)
        exact_match = pred_lines == gt_lines

        # ES — fuzz.ratio on whole prediction stripped (NOT line-by-
        # line). Upstream rounds to 2 decimals.
        edit_similarity = round(
            float(_fuzz_ratio(pred_no_comments.strip(), target_no_comments.strip())),
            2,
        )

        # ID-EM — ordered list equality of extracted identifiers.
        # ID-F1 — set-based TP/FP/FN.
        pred_ids = _extract_identifiers(pred_no_comments, language)
        target_ids = _extract_identifiers(target_no_comments, language)
        id_em = pred_ids == target_ids
        tp, fp, fn = _id_f1(pred_ids, target_ids)
        id_precision = tp / (tp + fp) if (tp + fp) else 0.0
        id_recall = tp / (tp + fn) if (tp + fn) else 0.0
        id_f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        return {
            "pass": exact_match,
            "exact_match": float(bool(exact_match)),
            "edit_similarity": edit_similarity,
            "id_em": float(bool(id_em)),
            "id_precision": round(id_precision, 4),
            "id_recall": round(id_recall, 4),
            "id_f1": round(id_f1_score, 4),
            # ``category`` = language so the engine emits
            # ``accuracy_cat_<language>`` per-language breakdowns.
            "category": language,
            "language": language,
            "retrieval_setting": meta.get("retrieval_setting", ""),
        }

    def get_sandbox_image(self, task: Task | None = None) -> str:
        return ""

    def get_default_params(self) -> dict[str, Any]:
        return {
            "language": "python",
            "retrieval_setting": "retrieval",
            "retrieval_query_lines": _DEFAULT_RETRIEVAL_QUERY_LINES,
        }

    def metrics_schema(self) -> list[str]:
        """Stable column set for CrossCodeEval summary.json.

        Headlines (``pass_rate`` / ``avg_exact_match`` /
        ``avg_edit_similarity`` / ``avg_id_em`` / ``avg_id_f1``)
        are upstream's four per-language metrics. Per-language
        breakdowns surface automatically via the engine's
        ``accuracy_cat_<language>`` aggregation when running across
        multiple language splits.

        Latency keys are populated by ``execute_single_shot`` when
        memory is active.
        """
        return [
            # Grading
            "pass_rate", "accuracy",
            "avg_exact_match",
            "avg_edit_similarity",
            "avg_id_em",
            "avg_id_precision",
            "avg_id_recall",
            "avg_id_f1",
            # Per-language (engine emits via category=language).
            "accuracy_cat_python",
            "accuracy_cat_java",
            "accuracy_cat_csharp",
            "accuracy_cat_typescript",
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
