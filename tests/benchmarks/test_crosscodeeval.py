"""CrossCodeEval adapter tests — upstream faithfulness + memory hooks.

References:
  - upstream postprocessing: ``scripts/eval_metric.py`` + ``eval_utils.py``
    https://github.com/amazon-science/cceval/blob/main/scripts/eval/eval_metric.py
  - upstream prompt assembly: ``scripts/vllm_inference.py::prepare_prompt``
  - upstream rg1 query: ``prompt_builder/augment_with_cfc.py:57``
  - paper: Ding et al., NeurIPS 2023, https://arxiv.org/abs/2310.11248
"""

from __future__ import annotations

from typing import Any

import pytest

from pathlib import Path

from cross_bench.benchmarks.crosscodeeval import (
    CrossCodeEvalAdapter,
    _build_in_file_prompt,
    _completion_instruction,
    _extract_code_completion,
    _extract_identifiers,
    _format_chunk_for_memory,
    _get_bracket_lang_statement,
    _postprocess_completion,
    _remove_comments,
    _truncate_to_target_lines,
)
from cross_bench.models import Task, TaskGranularity


@pytest.fixture
def adapter() -> CrossCodeEvalAdapter:
    return CrossCodeEvalAdapter()


def _task(
    gt: str,
    *,
    language: str = "python",
    prompt_no_cfc: str | None = None,
    cfc_chunks: list[dict[str, Any]] | None = None,
    retrieval_setting: str = "retrieval",
    retrieval_query_lines: int = 10,
) -> Task:
    metadata = {
        "language": language,
        "retrieval_setting": retrieval_setting,
        "prompt_no_cfc": prompt_no_cfc if prompt_no_cfc is not None else "",
        "cfc_chunks": cfc_chunks if cfc_chunks is not None else [],
        "retrieval_query_lines": retrieval_query_lines,
    }
    return Task(
        task_id="t",
        benchmark="crosscodeeval",
        granularity=TaskGranularity.CROSS_FILE,
        prompt=prompt_no_cfc if prompt_no_cfc is not None else "",
        metadata=metadata,
        ground_truth=gt,
    )


# ---------------------------------------------------------------------------
# Existing per-evaluate behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_match(adapter):
    res = await adapter.evaluate(
        _task("result = fetch(user_id)"),
        "result = fetch(user_id)",
        None,
        None,
    )
    assert res["pass"] is True
    assert res["exact_match"] == 1.0
    assert res["edit_similarity"] == 100.0
    assert res["id_em"] == 1.0
    assert res["id_f1"] == 1.0


@pytest.mark.asyncio
async def test_near_match(adapter):
    res = await adapter.evaluate(
        _task("result = compute(user_data)"),
        "result = compute(user_dta)",
        None,
        None,
    )
    assert res["exact_match"] == 0.0
    assert res["edit_similarity"] > 80  # Levenshtein ratio on the 0-100 scale
    assert res["id_em"] == 0.0
    assert res["id_f1"] < 1.0


@pytest.mark.asyncio
async def test_whitespace_and_blank_lines_ignored_in_em(adapter):
    res = await adapter.evaluate(_task("foo = bar"), "  foo = bar  \n\n", None, None)
    assert res["exact_match"] == 1.0


@pytest.mark.asyncio
async def test_prediction_truncated_to_target_lines(adapter):
    # Target is one line — extra hallucinated lines must not tank the metric.
    pred = "x = 1\nsome_garbage = other()\nyet_more()"
    res = await adapter.evaluate(_task("x = 1"), pred, None, None)
    assert res["exact_match"] == 1.0
    assert res["edit_similarity"] == 100.0


@pytest.mark.asyncio
async def test_id_metrics_set_based(adapter):
    # Prediction introduces one spurious id and misses one.
    res = await adapter.evaluate(
        _task("result = compute(user_data, config)"),
        "result = derive(user_data, extra)",
        None,
        None,
    )
    # Shared ids (after keyword filter): {result, user_data}; pred-only: {derive, extra};
    # target-only: {compute, config}. TP=2, FP=2, FN=2.
    assert res["id_em"] == 0.0
    assert res["id_precision"] == 0.5
    assert res["id_recall"] == 0.5
    assert res["id_f1"] == 0.5


@pytest.mark.asyncio
async def test_empty_prediction(adapter):
    res = await adapter.evaluate(_task("return x"), "", None, None)
    assert res["pass"] is False
    assert res["exact_match"] == 0.0
    assert res["id_f1"] == 0.0


def test_extract_identifiers_filters_keywords_and_strings():
    ids = _extract_identifiers('x = "hello if else" + y', "python")
    assert ids == ["x", "y"]  # keywords + string contents stripped


def test_extract_identifiers_java_keywords():
    ids = _extract_identifiers("public void handle(Request req)", "java")
    assert "public" not in ids and "void" not in ids
    assert {"handle", "Request", "req"} <= set(ids)


def test_truncate_to_target_lines():
    assert _truncate_to_target_lines("a\nb\nc", "x") == "a"
    assert _truncate_to_target_lines("a\nb\nc", "x\ny") == "a\nb"
    assert _truncate_to_target_lines("", "x") == ""


def test_requires_sandbox_is_false(adapter):
    # Pure text-diff eval — the engine must not spin up a container.
    assert adapter.requires_sandbox is False
    assert adapter.get_sandbox_image() == ""


# ---------------------------------------------------------------------------
# Upstream-faithful postprocessing — bracket-lang + remove_comments
# ---------------------------------------------------------------------------


class TestBracketLangStatement:
    """``eval_utils.py::get_bracket_lang_statement`` (lines 96-102)."""

    def test_stops_at_first_semicolon(self):
        assert _get_bracket_lang_statement("int x = 1; foo();") == "int x = 1;"

    def test_stops_at_first_open_brace(self):
        assert _get_bracket_lang_statement("class Foo { int x;") == "class Foo {"

    def test_stops_at_first_close_brace(self):
        assert _get_bracket_lang_statement("} else {") == "}"

    def test_no_terminator_returns_completion(self):
        assert _get_bracket_lang_statement("just text") == "just text"

    def test_first_terminator_wins(self):
        # Has both `;` and `{`; the first one (in scan order) wins.
        assert _get_bracket_lang_statement("a = 1;\nclass B {") == "a = 1;"
        assert _get_bracket_lang_statement("class B { a = 1;") == "class B {"

    def test_does_not_skip_terminators_inside_strings(self):
        """Faithful to upstream — literal byte scan, no string awareness."""
        assert _get_bracket_lang_statement('s = "x;y"; foo()') == 's = "x;'


class TestRemoveComments:
    """``eval_utils.py::remove_comments`` (lines 117-120). Two regex passes:
    ``re.sub(r'#.*', '', code)`` then ``re.sub(r'//.*', '', code)``."""

    def test_strips_python_hash_comment(self):
        assert _remove_comments("x = 1  # trailing").rstrip() == "x = 1"

    def test_strips_double_slash_comment(self):
        # ``//.*`` consumes "// trailing"; the space before ``//`` is
        # left in place. We compare the rstripped form so trailing
        # whitespace doesn't muddy the assertion.
        assert _remove_comments("int x = 1; // trailing").rstrip() == "int x = 1;"

    def test_strips_both_in_one_pass(self):
        # Hash-comment goes first, then double-slash. Both gone.
        assert _remove_comments("x  # hash\ny = 1 // slash").splitlines() == ["x  ", "y = 1 "]

    def test_does_not_handle_block_comments(self):
        """Faithful to upstream — ``/* ... */`` blocks are NOT touched."""
        assert _remove_comments("/* block */ x = 1") == "/* block */ x = 1"

    def test_does_not_handle_triple_quote_strings(self):
        """Faithful to upstream — Python triple-quote docstrings unchanged."""
        out = _remove_comments('"""docstring"""\nx = 1')
        assert '"""docstring"""' in out

    def test_strips_hash_inside_string(self):
        """Known upstream limitation: ``#.*`` regex doesn't know about
        string boundaries, so a literal ``#`` inside a Python string
        gets stripped along with everything after it on that line."""
        assert _remove_comments('s = "url#frag"\ny = 1').splitlines()[0] == 's = "url'


class TestPostprocessCompletion:
    """``eval_utils.py::postprocess_code_lines`` dispatcher."""

    def test_python_uses_line_count(self):
        # Python falls into ``_truncate_to_target_lines``.
        assert _postprocess_completion("a\nb\nc", "x", "python") == "a"

    def test_java_uses_bracket_scan(self):
        assert _postprocess_completion("int x = 1; foo()", "_", "java") == "int x = 1;"

    def test_csharp_uses_bracket_scan(self):
        assert _postprocess_completion("class Foo { x", "_", "csharp") == "class Foo {"

    def test_typescript_uses_bracket_scan(self):
        assert _postprocess_completion("const x = 1; foo()", "_", "typescript") == "const x = 1;"


@pytest.mark.asyncio
async def test_evaluate_strips_comments_before_em(adapter):
    """``remove_comments`` runs on BOTH pred and gt before line-list
    equality — a trailing ``# foo`` on either side must not break EM."""
    res = await adapter.evaluate(
        _task("x = 1"),
        "x = 1  # this is the next line",
        None,
        None,
    )
    assert res["exact_match"] == 1.0


@pytest.mark.asyncio
async def test_java_completion_truncated_at_first_semicolon(adapter):
    """Java target ``int x = 1;`` against a noisy multi-statement
    completion — postprocess slices at the first ``;`` so EM passes."""
    res = await adapter.evaluate(
        _task("int x = 1;", language="java"),
        "int x = 1; foo(); bar();",
        None,
        None,
    )
    assert res["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# Per-language category propagation + metrics_schema
# ---------------------------------------------------------------------------


class TestCategoryAndSchema:
    @pytest.mark.asyncio
    async def test_category_set_to_language(self, adapter):
        for language in ("python", "java", "csharp", "typescript"):
            res = await adapter.evaluate(
                _task("x", language=language), "x", None, None
            )
            assert res["category"] == language
            assert res["language"] == language

    def test_metrics_schema_includes_per_language(self, adapter):
        keys = adapter.metrics_schema()
        # Headlines.
        assert "pass_rate" in keys
        assert "avg_exact_match" in keys
        assert "avg_edit_similarity" in keys
        assert "avg_id_em" in keys
        assert "avg_id_f1" in keys
        # Per-language (engine emits via category=language).
        assert "accuracy_cat_python" in keys
        assert "accuracy_cat_java" in keys
        assert "accuracy_cat_csharp" in keys
        assert "accuracy_cat_typescript" in keys
        # Latency.
        assert "avg_search_latency_ms" in keys
        assert "avg_inference_latency_ms" in keys
        assert "avg_total_latency_ms" in keys


# ---------------------------------------------------------------------------
# Memory-mode prompt construction
# ---------------------------------------------------------------------------


class TestMemoryPromptHelpers:
    def test_in_file_prompt_returns_prompt_verbatim(self):
        assert _build_in_file_prompt("hello") == "hello"

    def test_format_chunk_includes_filename_header(self):
        out = _format_chunk_for_memory(
            filename="src/util.py", chunk_text="def f(): pass", score=0.85
        )
        assert "# the below code fragment is found in src/util.py" in out
        assert "# retrieval_score=0.8500" in out
        assert "def f(): pass" in out

    def test_format_chunk_omits_score_when_none(self):
        out = _format_chunk_for_memory(
            filename="src/util.py", chunk_text="x", score=None
        )
        assert "retrieval_score" not in out


class TestFormatMemoryPrompt:
    def test_returns_in_file_prompt_only_no_cfc(self, adapter):
        task = _task(
            "x",
            prompt_no_cfc="def foo():\n    return 1\n",
            cfc_chunks=[
                {"filename": "u.py", "retrieved_chunk": "def helper(): pass", "score": 0.9}
            ],
        )
        out = adapter.format_memory_prompt(task)
        # C18: the directive is prepended; the in-file portion remains
        # byte-equal to upstream's no-CFC ``prompt`` field at the tail
        # (no double-counting of CFC).
        assert out.startswith(_completion_instruction("python"))
        assert out.endswith("def foo():\n    return 1\n")
        assert "helper" not in out


class TestGetRetrievalQuery:
    def test_returns_last_n_non_empty_lines(self, adapter):
        prompt = "\n".join(f"line_{i}" for i in range(50))
        task = _task("x", prompt_no_cfc=prompt, retrieval_query_lines=5)
        q = adapter.get_retrieval_query(task)
        assert q is not None
        # Last 5 non-empty lines = line_45..line_49.
        assert "line_45" in q and "line_49" in q
        assert "line_44" not in q

    def test_default_query_lines_is_upstream_rg1_10(self, adapter):
        prompt = "\n".join(f"line_{i}" for i in range(30))
        task = _task("x", prompt_no_cfc=prompt)
        q = adapter.get_retrieval_query(task)
        assert q is not None
        # Default = 10, matching upstream augment_with_cfc.py:57
        assert "line_20" in q and "line_29" in q
        assert "line_19" not in q

    def test_skips_blank_lines(self, adapter):
        # Mix of blank and non-blank — only non-blank count toward the N.
        prompt = "real_a\n\n\nreal_b\n\nreal_c"
        task = _task("x", prompt_no_cfc=prompt, retrieval_query_lines=2)
        q = adapter.get_retrieval_query(task)
        assert q == "real_b\nreal_c"

    def test_returns_none_when_prompt_empty(self, adapter):
        task = _task("x", prompt_no_cfc="")
        assert adapter.get_retrieval_query(task) is None


# ---------------------------------------------------------------------------
# Memory ingestion
# ---------------------------------------------------------------------------


class _RecordingMemory:
    """Captures every ``post_response`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        self.calls.append((prompt, response, dict(context)))

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        return prompt

    async def reset(self) -> None:
        self.calls.clear()


class TestIngestForMemory:
    @pytest.mark.asyncio
    async def test_one_call_per_chunk_with_indexed_source_id(self, adapter):
        chunks = [
            {"filename": f"src/u{i}.py", "retrieved_chunk": f"def f{i}(): pass", "score": 0.5 + i * 0.1}
            for i in range(5)
        ]
        memory = _RecordingMemory()
        task = _task("x", cfc_chunks=chunks)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        assert len(memory.calls) == 5
        for i, (_, content, ctx) in enumerate(memory.calls):
            assert ctx["source_id"] == f"ctx:{i}"
            assert ctx["filename"] == f"src/u{i}.py"
            assert f"def f{i}()" in content

    @pytest.mark.asyncio
    async def test_no_chunks_means_no_calls(self, adapter):
        memory = _RecordingMemory()
        task = _task("x", cfc_chunks=[], retrieval_setting="none")
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert memory.calls == []

    @pytest.mark.asyncio
    async def test_skips_empty_chunk_text(self, adapter):
        chunks = [
            {"filename": "a.py", "retrieved_chunk": "", "score": 0.1},
            {"filename": "b.py", "retrieved_chunk": "def f(): pass", "score": 0.9},
            {"filename": "c.py", "retrieved_chunk": "   ", "score": 0.2},
        ]
        memory = _RecordingMemory()
        task = _task("x", cfc_chunks=chunks)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        # Only the chunk with real text content is ingested.
        assert len(memory.calls) == 1
        assert memory.calls[0][2]["filename"] == "b.py"

    @pytest.mark.asyncio
    async def test_handles_alt_chunk_field_name(self, adapter):
        """Tolerate the alternative ``chunk`` field name in case a future
        dataset mirror normalizes it from ``retrieved_chunk``."""
        chunks = [
            {"filename": "a.py", "chunk": "def f(): pass", "score": 0.9},
        ]
        memory = _RecordingMemory()
        task = _task("x", cfc_chunks=chunks)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert len(memory.calls) == 1
        assert "def f(): pass" in memory.calls[0][1]

    @pytest.mark.asyncio
    async def test_source_id_aligns_with_chunk_index_contract(self, adapter):
        """Per-chunk ``source_id = "ctx:{idx}"`` lets the engine measure
        retrieval recall directly against the dataset's chunk ordering
        — same contract as RepoBench. Lock it."""
        chunks = [
            {"filename": f"f{i}.py", "retrieved_chunk": f"snippet {i}", "score": 0.5}
            for i in range(3)
        ]
        memory = _RecordingMemory()
        task = _task("x", cfc_chunks=chunks)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        for n in range(3):
            assert memory.calls[n][2]["source_id"] == f"ctx:{n}"


# ---------------------------------------------------------------------------
# C18 — chat-LLM completion directive + output postprocessor (mirrors
# RepoBench R16). See ``deviations.md`` for the rationale.
# ---------------------------------------------------------------------------


class TestCompletionInstruction:
    """The directive ``_completion_instruction`` prepends to the LLM prompt
    so chat-tuned models emit a single line of code instead of chat-mode
    prose. See ``deviations.md`` C18.
    """

    def test_python_directive_demands_single_line(self):
        text = _completion_instruction("python")
        assert "Python" in text
        assert "ONLY that single next line" in text
        assert "Do NOT wrap" in text
        # Trailing blank line separates directive from the code block.
        assert text.endswith("\n\n")

    def test_java_directive_demands_single_line(self):
        text = _completion_instruction("java")
        assert "Java" in text
        assert "ONLY that single next line" in text

    def test_csharp_directive_demands_single_line(self):
        text = _completion_instruction("csharp")
        assert "C#" in text
        assert "ONLY that single next line" in text

    def test_typescript_directive_demands_single_line(self):
        text = _completion_instruction("typescript")
        assert "TypeScript" in text
        assert "ONLY that single next line" in text

    def test_unknown_language_defaults_to_python(self):
        # Falls back to Python rather than dropping the directive — a
        # chat LLM with no directive is the failure mode C18 fixes.
        assert _completion_instruction("rust") == _completion_instruction("python")

    def test_format_prompt_prepends_directive(self, adapter):
        task = _task(
            "result = compute()",
            prompt_no_cfc="def f():\n    return ",
        )
        # ``task.prompt`` is the upstream-faithful assembled prompt.
        rendered = adapter.format_prompt(task)
        assert rendered.startswith(_completion_instruction("python"))
        assert rendered.endswith(task.prompt)

    def test_format_memory_prompt_prepends_directive(self, adapter):
        task = _task(
            "x",
            prompt_no_cfc="def foo():\n    return 1\n",
        )
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("python"))
        # In-file portion still ends with the upstream prompt verbatim
        # (no CFC double-count).
        assert rendered.endswith("def foo():\n    return 1\n")

    def test_java_task_gets_java_directive(self, adapter):
        task = _task("x", language="java", prompt_no_cfc="// in-file\n")
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("java"))

    def test_csharp_task_gets_csharp_directive(self, adapter):
        task = _task("x", language="csharp", prompt_no_cfc="// in-file\n")
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("csharp"))

    def test_typescript_task_gets_typescript_directive(self, adapter):
        task = _task("x", language="typescript", prompt_no_cfc="// in-file\n")
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("typescript"))


class TestExtractCodeCompletion:
    """Strip chat-mode wrappers around a code completion. See C18."""

    def test_raw_code_passes_through_unchanged(self):
        # Base completion models that return raw code are unaffected.
        out = _extract_code_completion("    result = compute()\n")
        assert out == "    result = compute()\n"

    def test_python_fence_strips_backticks_and_lang_tag(self):
        out = _extract_code_completion("```python\nresult = compute()\n```")
        assert "```" not in out
        assert "result = compute()" in out

    def test_py_short_fence_tag_supported(self):
        out = _extract_code_completion("```py\nx = 1\n```")
        assert "x = 1" in out
        assert "```" not in out

    def test_bare_fence_no_lang_tag(self):
        out = _extract_code_completion("```\nx = 1\n```")
        assert "x = 1" in out
        assert "```" not in out

    def test_unterminated_fence_still_extracts(self):
        # Model truncated mid-output — we should still recover the code.
        out = _extract_code_completion("```python\nresult = compute()")
        assert "result = compute()" in out

    def test_java_fence_lang_tag(self):
        out = _extract_code_completion("```java\nint x = 1;\n```", "java")
        assert "int x = 1;" in out

    def test_csharp_fence_lang_tag(self):
        out = _extract_code_completion("```csharp\nvar x = 1;\n```", "csharp")
        assert "var x = 1;" in out

    def test_cs_short_fence_tag_supported(self):
        out = _extract_code_completion("```cs\nvar x = 1;\n```", "csharp")
        assert "var x = 1;" in out

    def test_c_hash_fence_tag_supported(self):
        # Some chat LLMs emit ```c# even though it's not a common tag.
        out = _extract_code_completion("```c#\nvar x = 1;\n```", "csharp")
        assert "var x = 1;" in out

    def test_typescript_fence_lang_tag(self):
        out = _extract_code_completion(
            "```typescript\nconst x: number = 1;\n```", "typescript"
        )
        assert "const x: number = 1;" in out

    def test_ts_short_fence_tag_supported(self):
        out = _extract_code_completion(
            "```ts\nconst x: number = 1;\n```", "typescript"
        )
        assert "const x: number = 1;" in out

    def test_prose_preamble_here_is_the_next_line(self):
        out = _extract_code_completion(
            "Here is the next line: result = compute()"
        )
        # Leading horizontal whitespace after the colon is consumed; the
        # code line itself follows.
        assert out.lstrip() == "result = compute()"

    def test_prose_preamble_with_newline_after_colon(self):
        # Multi-line preamble: the answer's indentation is on a new line
        # AFTER the colon. The regex consumes ``:`` + horizontal
        # whitespace + one newline, preserving the indentation that
        # follows.
        out = _extract_code_completion(
            "The next line should be:\n    return helper(x)\n"
        )
        assert "    return helper(x)" in out

    def test_prose_preamble_inline_colon_strips_horizontal_whitespace(self):
        # Inline preamble: spaces between the colon and the code are
        # spacing, not indentation. The regex eats horizontal whitespace
        # after the colon so the code line starts cleanly.
        out = _extract_code_completion("Next line:    z = 1")
        assert out.lstrip() == "z = 1"

    def test_preserves_leading_whitespace_inside_fence(self):
        # Ground truths often have leading whitespace (function-body
        # lines). EM tokenizes on whitespace so leading spaces don't
        # change the EM verdict, but they ARE preserved.
        out = _extract_code_completion("```python\n        return None\n```")
        assert "        return None" in out

    def test_empty_input_returns_empty(self):
        assert _extract_code_completion("") == ""

    def test_chat_mode_prose_essay_does_not_match_fence_or_preamble(self):
        """When the model emits pure chat-mode prose with no fence or
        recognised preamble, we pass the text through. The downstream
        ``_postprocess_completion`` will keep that prose; EM/ES will
        (correctly) fail. This is the failure mode C18 fixes via the
        prompt directive, not via postprocessing — the directive is
        the primary fix; the postprocessor is the belt-and-braces
        backstop for chat preambles we know about.
        """
        prose = "The provided code snippet is a Python script that imports..."
        out = _extract_code_completion(prose)
        assert out == prose


class TestEvaluateUsesCodeCompletionExtractor:
    """End-to-end: when the model wraps its answer in a fence,
    ``evaluate`` recovers the code and grades it correctly.
    """

    @pytest.mark.asyncio
    async def test_fenced_correct_answer_passes_em(self, adapter):
        task = _task("result = compute()")
        result = await adapter.evaluate(
            task, "```python\nresult = compute()\n```", None, None
        )
        assert result["pass"] is True
        assert result["exact_match"] == 1.0

    @pytest.mark.asyncio
    async def test_chat_prose_preamble_correct_answer_passes_em(self, adapter):
        task = _task("z = helper(x, y)")
        result = await adapter.evaluate(
            task, "Here is the next line: z = helper(x, y)", None, None
        )
        assert result["pass"] is True

    @pytest.mark.asyncio
    async def test_raw_correct_answer_still_passes_em(self, adapter):
        # No-regression check: when the model already returns raw code,
        # we shouldn't accidentally break the upstream path.
        task = _task("result = compute()")
        result = await adapter.evaluate(task, "result = compute()", None, None)
        assert result["pass"] is True

    @pytest.mark.asyncio
    async def test_fenced_java_correct_answer_passes_em(self, adapter):
        task = _task("int x = 1;", language="java")
        result = await adapter.evaluate(
            task, "```java\nint x = 1;\n```", None, None
        )
        assert result["pass"] is True


# ---------------------------------------------------------------------------
# Synthetic workspace materialisation (mirrors RepoBench TestSyntheticWorkspace).
# See deviations.md CrossCodeEval section for the Why / What / Where / Impact.
# ---------------------------------------------------------------------------


def _ws_task(
    *,
    file_path: str = "src/module.py",
    repo_name: str = "myorg/MyRepo",
    cfc_chunks: list[dict[str, Any]] | None = None,
    cropped_code: str = "def example(): pass\n",
    import_statement: str = "import os",
    all_code: str = "",
) -> Task:
    if cfc_chunks is None:
        cfc_chunks = [{"filename": "lib/util.py", "retrieved_chunk": "UTIL = 1\n", "score": 0.9}]
    memory_snippets = [
        {"path": c["filename"], "snippet": c.get("retrieved_chunk", "")}
        for c in cfc_chunks
    ]
    metadata: dict[str, Any] = {
        "language": "python",
        "retrieval_setting": "retrieval",
        "prompt_no_cfc": cropped_code,
        "cfc_chunks": cfc_chunks,
        "retrieval_query_lines": 10,
        "file_path": file_path,
        "repo_name": repo_name,
        "memory_snippets": memory_snippets,
        "cropped_code": cropped_code,
        "all_code": all_code,
        "import_statement": import_statement,
    }
    return Task(
        task_id="ws_task",
        benchmark="crosscodeeval",
        granularity=TaskGranularity.CROSS_FILE,
        prompt=cropped_code,
        metadata=metadata,
        ground_truth="x = 1",
    )


class TestSyntheticWorkspace:
    @pytest.mark.asyncio
    async def test_materializes_chunks_and_target_then_teardown(
        self, adapter: CrossCodeEvalAdapter, tmp_path: Any
    ) -> None:
        base = tmp_path / "ws"
        base.mkdir()
        memory = _RecordingMemory()
        task = _ws_task(
            file_path="src/model.py",
            repo_name="myorg/MyProject",
            cfc_chunks=[
                {"filename": "lib/util.py", "retrieved_chunk": "UTIL = 42\n", "score": 0.9}
            ],
            cropped_code="class Model:\n    pass\n",
            import_statement="from lib.util import UTIL",
        )
        task.metadata["synthetic_workspace_dir"] = str(base)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        key = "myorg_MyProject"
        chunk_f = base / key / "lib" / "util.py"
        assert chunk_f.is_file()
        assert chunk_f.read_text(encoding="utf-8") == "UTIL = 42\n"

        main_f = base / key / "src" / "model.py"
        assert main_f.is_file()
        text = main_f.read_text(encoding="utf-8")
        assert "from lib.util import UTIL" in text
        assert "class Model:" in text

        assert task.metadata.get("code_workspace") == str(base.resolve())

        await adapter.teardown_task(task)
        assert not (base / key).exists()
        assert "synthetic_workspace_tear_path" not in task.metadata
        assert "code_workspace" not in task.metadata

    @pytest.mark.asyncio
    async def test_all_code_preferred_when_non_empty(
        self, adapter: CrossCodeEvalAdapter, tmp_path: Any
    ) -> None:
        base = tmp_path / "ws"
        base.mkdir()
        memory = _RecordingMemory()
        task = _ws_task(
            file_path="src/x.py",
            repo_name="r",
            cfc_chunks=[],
            cropped_code="cropped only",
            import_statement="import os",
        )
        task.metadata["synthetic_workspace_dir"] = str(base)
        task.metadata["all_code"] = "full file body\nline 2\n"
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        main_f = base / "r" / "src" / "x.py"
        body = main_f.read_text(encoding="utf-8")
        assert "full file body" in body
        assert "cropped only" not in body

        await adapter.teardown_task(task)

    @pytest.mark.asyncio
    async def test_no_disk_writes_when_synthetic_dir_unset(
        self, adapter: CrossCodeEvalAdapter, tmp_path: Any
    ) -> None:
        memory = _RecordingMemory()
        sub = tmp_path / "monitor"
        sub.mkdir()
        task = _ws_task(
            cfc_chunks=[{"filename": "a.py", "retrieved_chunk": "x = 1", "score": 0.5}],
        )
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert task.metadata.get("synthetic_workspace_tear_path") is None
        assert list(sub.iterdir()) == []

    @pytest.mark.asyncio
    async def test_hostile_filename_in_chunk_is_skipped(
        self, adapter: CrossCodeEvalAdapter, tmp_path: Any
    ) -> None:
        """A chunk whose filename contains ``..`` path traversal must not
        write outside the repo root — the materializer should silently skip
        it rather than escape the workspace boundary."""
        base = tmp_path / "ws"
        base.mkdir()
        memory = _RecordingMemory()
        task = _ws_task(
            cfc_chunks=[
                {"filename": "../escape.py", "retrieved_chunk": "INJECTED = True\n", "score": 0.9}
            ],
        )
        task.metadata["synthetic_workspace_dir"] = str(base)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        # The hostile file must not appear outside the repo subtree.
        assert not (tmp_path / "escape.py").exists()
        assert not (base / "escape.py").exists()
        await adapter.teardown_task(task)
