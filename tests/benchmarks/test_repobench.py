"""RepoBench adapter tests — upstream faithfulness + memory integration.

References:
  - upstream prompt construction: ``data/utils.py::construct_prompt``
    https://github.com/Leolty/repobench/blob/main/data/utils.py
  - upstream post-processing: ``run.py::get_first_line_not_comment``
    https://github.com/Leolty/repobench/blob/main/run.py
  - upstream metrics: ``evaluation/metrics.py``
  - RepoCoder retrieval: arXiv:2303.12570
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from cross_bench.benchmarks.repobench import (
    RepoBenchAdapter,
    _build_in_file_prompt,
    _completion_instruction,
    _construct_prompt,
    _extract_code_completion,
    _format_snippet_for_memory,
    _get_first_line_not_comment,
    _parse_bound,
    _parse_created_at,
)
from cross_bench.models import Task, TaskGranularity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> RepoBenchAdapter:
    return RepoBenchAdapter()


def _row(
    *,
    repo_name: str = "alice/widgets",
    file_path: str = "src/main.py",
    context: list[dict[str, str]] | None = None,
    import_statement: str = "from utils import helper",
    cropped_code: str = "x = 1\ny = 2\n",
    next_line: str = "z = helper(x, y)",
    level: str = "2k",
    created_at: str = "2023-12-15 10:00:00+00:00",
    gold_snippet_index: int = 0,
) -> dict[str, Any]:
    return {
        "repo_name": repo_name,
        "file_path": file_path,
        "context": context if context is not None else [
            {"identifier": "helper", "path": "src/utils.py", "snippet": "def helper(a, b):\n    return a + b"},
        ],
        "import_statement": import_statement,
        "cropped_code": cropped_code,
        "next_line": next_line,
        "level": level,
        "created_at": created_at,
        "gold_snippet_index": gold_snippet_index,
    }


def _task(
    *,
    language: str = "python",
    split: str = "cross_file_first",
    repo_name: str = "alice/widgets",
    file_path: str = "src/main.py",
    cropped_code: str = "x = 1\ny = 2\n",
    import_statement: str = "from utils import helper",
    snippets: list[dict[str, str]] | None = None,
    next_line: str = "z = helper(x, y)",
    retrieval_query_lines: int = 20,
) -> Task:
    if snippets is None:
        snippets = [
            {"identifier": "helper", "path": "src/utils.py", "snippet": "def helper(a, b):\n    return a + b"},
        ]
    return Task(
        task_id=f"repobench-{language}-{split}-0",
        benchmark="repobench",
        granularity=TaskGranularity.CROSS_FILE,
        prompt="<full prompt>",  # not exercised by these tests
        metadata={
            "language": language,
            "split": split,
            "level": "2k",
            "repo_name": repo_name,
            "file_path": file_path,
            "cropped_code": cropped_code,
            "import_statement": import_statement,
            "memory_snippets": snippets,
            "retrieval_query_lines": retrieval_query_lines,
        },
        ground_truth=next_line,
    )


# ---------------------------------------------------------------------------
# _get_first_line_not_comment — upstream-faithful, including known bugs
# ---------------------------------------------------------------------------


class TestFirstLineNotComment:
    """Upstream's ``run.py::get_first_line_not_comment`` (lines 12-79)."""

    def test_python_skips_blank_lines(self) -> None:
        out = _get_first_line_not_comment("\n\n\nresult = compute()", "python")
        assert out == "result = compute()"

    def test_python_skips_hash_comments(self) -> None:
        out = _get_first_line_not_comment(
            "# this is a comment\n# another\nactual_code = 1", "python"
        )
        assert out == "actual_code = 1"

    def test_python_skips_triple_quote_block(self) -> None:
        out = _get_first_line_not_comment(
            '"""\nA module docstring\nspanning multiple lines\n"""\nimport os',
            "python",
        )
        assert out == "import os"

    def test_python_skips_single_quote_triple_block(self) -> None:
        out = _get_first_line_not_comment(
            "'''\nblock\n'''\nresult = 1", "python"
        )
        assert out == "result = 1"

    def test_python_single_line_docstring_quirk_is_preserved(self) -> None:
        """Upstream bug: a one-line ``\"\"\"hello\"\"\"`` triggers only the
        startswith branch, sets ``in_multiline_comment=True``, and ``continue``.
        The closing ``\"\"\"`` on the same line is never inspected, so the
        flag stays True and the *next* code line is also skipped. Locked
        because the official metric depends on this quirk."""
        out = _get_first_line_not_comment(
            '"""one-liner"""\nthis_line_gets_skipped_too = 1\nthis_one_returns = 2',
            "python",
        )
        # Both the docstring line AND the next line are skipped. The
        # third line ends with `2` which doesn't end in triple quotes,
        # so the flag never resets — we keep skipping. We end up
        # returning the fallback `lines[0]` which is the docstring.
        assert out == '"""one-liner"""'

    def test_python_inline_comment_after_code_is_returned(self) -> None:
        """`startswith('#')` only triggers if the line *starts* with #.
        Trailing comments don't make the line a comment line."""
        out = _get_first_line_not_comment("x = 1  # trailing comment", "python")
        assert out == "x = 1  # trailing comment"

    def test_java_skips_double_slash_comments(self) -> None:
        out = _get_first_line_not_comment(
            "// header\n// another\nint x = 1;", "java"
        )
        assert out == "int x = 1;"

    def test_java_skips_block_comment(self) -> None:
        out = _get_first_line_not_comment(
            "/*\n * a block\n */\npublic class Foo {", "java"
        )
        assert out == "public class Foo {"

    def test_fallback_returns_first_line_when_all_comments(self) -> None:
        """Per upstream line 79: when no real code line is found, return
        ``lines[0]`` of the lstripped input."""
        out = _get_first_line_not_comment("# only a comment", "python")
        assert out == "# only a comment"

    def test_unknown_language_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            _get_first_line_not_comment("foo", "rust")


# ---------------------------------------------------------------------------
# _construct_prompt — byte-faithful to upstream
# ---------------------------------------------------------------------------


class TestConstructPrompt:
    """Upstream ``data/utils.py::construct_prompt`` (lines 3-59)."""

    def test_python_layout_matches_upstream(self) -> None:
        row = _row(
            repo_name="alice/widgets",
            file_path="src/main.py",
            context=[
                {
                    "path": "src/utils.py",
                    "identifier": "helper",
                    "snippet": "def helper(a, b):\n    return a + b",
                },
            ],
            import_statement="from utils import helper",
            cropped_code="x = 1\ny = 2\n",
        )
        out = _construct_prompt(row, "python", lambda s: 1, max_token_nums=None)
        # Upstream layout — cross-file block then in-file block.
        # Cross-file: ``# Repo Name: ...\n# Path: ...\n<snippet>\n\n``
        # In-file:    ``# Path: ...\n{imports}\n{cropped.rstrip()}\n``
        assert "# Repo Name: alice/widgets" in out
        assert "# Path: src/utils.py" in out
        assert "def helper(a, b):" in out
        assert "# Path: src/main.py" in out
        assert "from utils import helper" in out
        # Cross-file block precedes in-file block.
        assert out.index("# Path: src/utils.py") < out.index("# Path: src/main.py")
        # In-file block ends with cropped_code rstripped + single trailing \n.
        assert out.endswith("y = 2\n")

    def test_java_uses_double_slash_comment(self) -> None:
        row = _row(
            repo_name="org/proj",
            file_path="Main.java",
            context=[
                {
                    "path": "Util.java",
                    "identifier": "f",
                    "snippet": "int f(int x) { return x; }",
                },
            ],
            import_statement="import java.util.*;",
            cropped_code="int x = 1;\n",
        )
        out = _construct_prompt(row, "java", lambda s: 1, max_token_nums=None)
        assert "// Repo Name: org/proj" in out
        assert "// Path: Util.java" in out
        assert "// Path: Main.java" in out
        # Python comment marker must NOT appear.
        assert "# Repo Name:" not in out

    def test_blank_line_collapse_4plus_to_2(self) -> None:
        """Upstream line 57: ``re.sub(r'\\n{4,}', '\\n\\n', prompt)``.
        Three-in-a-row survives; four+ collapse to two."""
        row = _row(
            cropped_code="line1\n\n\n\n\n\nline2\n",
            context=[],
            import_statement="",
        )
        out = _construct_prompt(row, "python", lambda s: 1, max_token_nums=None)
        # 6 newlines should collapse to 2.
        assert "line1\n\nline2" in out
        # No remnant 3+ newlines anywhere.
        assert not re.search(r"\n{3,}", out)

    def test_truncation_drops_cross_file_lines_from_tail(self) -> None:
        """Upstream lines 41-51: walk cross-file lines backwards, dropping
        until the budget fits. The slice is ``[:i]`` (exclusive) so the
        line that pushes below budget is itself dropped — preserve the
        off-by-one faithfully."""
        # Build many tiny cross-file snippets so the cross-file block
        # has many splittable lines.
        ctx = [
            {"path": f"f{n}.py", "identifier": f"id{n}", "snippet": f"def fn{n}(): pass"}
            for n in range(20)
        ]
        row = _row(context=ctx, cropped_code="z = 0\n", import_statement="")
        # Per-token = per-char counter to keep budget math simple.
        out = _construct_prompt(row, "python", lambda s: len(s), max_token_nums=80)
        # In-file block must be intact (truncation never touches it).
        assert "# Path: src/main.py" in out
        assert "z = 0" in out
        # Some cross-file content is dropped: the full prompt without
        # truncation would be much longer than 80 chars.
        full = _construct_prompt(row, "python", lambda s: len(s), max_token_nums=None)
        assert len(out) < len(full)

    def test_truncation_disabled_when_budget_unbounded(self) -> None:
        ctx = [
            {"path": "f.py", "identifier": "id", "snippet": "x" * 1000},
        ]
        row = _row(context=ctx)
        out = _construct_prompt(row, "python", lambda s: 1, max_token_nums=None)
        assert "x" * 1000 in out

    def test_truncation_does_not_run_when_under_budget(self) -> None:
        row = _row(cropped_code="z = 0\n", context=[])
        out_under = _construct_prompt(row, "python", lambda s: 1, max_token_nums=10000)
        out_unbounded = _construct_prompt(row, "python", lambda s: 1, max_token_nums=None)
        # Identical layout when budget is generous.
        assert out_under == out_unbounded


# ---------------------------------------------------------------------------
# Date / timestamp parsing
# ---------------------------------------------------------------------------


class TestDateParsing:
    """Upstream ``run.py::filter_dataset_by_date_range`` semantics."""

    def test_parse_bound_yields_midnight_utc(self) -> None:
        dt = _parse_bound("2023-12-31")
        assert dt.year == 2023 and dt.month == 12 and dt.day == 31
        assert dt.hour == 0 and dt.minute == 0
        assert dt.tzinfo is not None

    def test_parse_created_at_returns_none_for_empty(self) -> None:
        assert _parse_created_at("") is None

    def test_parse_created_at_returns_none_for_garbage(self) -> None:
        assert _parse_created_at("not-a-date") is None

    def test_parse_created_at_handles_iso_with_offset(self) -> None:
        dt = _parse_created_at("2023-12-28 05:47:18+00:00")
        assert dt is not None
        assert dt.year == 2023 and dt.day == 28


# ---------------------------------------------------------------------------
# Memory-mode prompt construction
# ---------------------------------------------------------------------------


class TestMemoryPromptHelpers:
    def test_in_file_prompt_byte_matches_upstream_layout(self) -> None:
        """``format_memory_prompt`` must produce the SAME bytes upstream's
        in-file block produces — same comment marker, same trailing \\n,
        same single-newline gap between path/imports/cropped."""
        out = _build_in_file_prompt(
            file_path="src/main.py",
            import_statement="from x import y",
            cropped_code="a = 1\nb = 2\n",
            language="python",
        )
        assert out == "# Path: src/main.py\nfrom x import y\na = 1\nb = 2\n"

    def test_in_file_prompt_java(self) -> None:
        out = _build_in_file_prompt(
            file_path="Main.java",
            import_statement="import java.util.*;",
            cropped_code="int x = 1;\n",
            language="java",
        )
        assert out == "// Path: Main.java\nimport java.util.*;\nint x = 1;\n"

    def test_in_file_prompt_handles_empty_imports(self) -> None:
        out = _build_in_file_prompt(
            file_path="src/main.py",
            import_statement="",
            cropped_code="a = 1\n",
            language="python",
        )
        # Empty import line still leaves a blank line between path and
        # cropped (upstream behavior: ``f"...\n{import}\n{cropped}\n"``
        # with empty import = ``\n\n``).
        assert out == "# Path: src/main.py\n\na = 1\n"

    def test_format_snippet_for_memory_includes_repo_path_identifier(self) -> None:
        out = _format_snippet_for_memory(
            repo_name="org/repo",
            snippet_path="lib/util.py",
            identifier="my_helper",
            snippet_code="def my_helper(): pass",
            language="python",
        )
        assert "# Repo Name: org/repo" in out
        assert "# Identifier: my_helper" in out
        assert "# Path: lib/util.py" in out
        assert "def my_helper(): pass" in out

    def test_format_snippet_for_memory_omits_empty_identifier(self) -> None:
        out = _format_snippet_for_memory(
            repo_name="org/repo",
            snippet_path="lib/util.py",
            identifier="",
            snippet_code="x = 1",
            language="python",
        )
        assert "# Identifier:" not in out


# ---------------------------------------------------------------------------
# Memory hooks on the adapter
# ---------------------------------------------------------------------------


class _RecordingMemory:
    """Captures every ``post_response`` call so ingestion contracts can be
    asserted directly, without spinning up a real memory backend."""

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


class TestFormatMemoryPrompt:
    def test_returns_in_file_only_no_cross_file(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(
            file_path="src/m.py",
            import_statement="import os",
            cropped_code="x = 1\n",
        )
        out = adapter.format_memory_prompt(task)
        # R16: directive prepended; remainder is the upstream in-file
        # block byte-for-byte.
        assert out.endswith("# Path: src/m.py\nimport os\nx = 1\n")
        assert out.startswith(_completion_instruction("python"))
        # Cross-file content from snippets must NOT appear.
        assert "def helper" not in out
        assert "Repo Name" not in out  # only present in cross-file blocks


class TestGetRetrievalQuery:
    def test_uses_imports_plus_tail_n_lines(
        self, adapter: RepoBenchAdapter
    ) -> None:
        cropped = "\n".join(f"line_{i}" for i in range(50))
        task = _task(cropped_code=cropped, import_statement="from x import y", retrieval_query_lines=5)
        q = adapter.get_retrieval_query(task)
        assert q is not None
        assert q.startswith("from x import y\n\n")
        # Tail should be the last 5 lines (line_45..line_49).
        assert "line_45" in q and "line_49" in q
        assert "line_44" not in q

    def test_default_query_lines_is_repocoder_20(
        self, adapter: RepoBenchAdapter
    ) -> None:
        cropped = "\n".join(f"line_{i}" for i in range(40))
        task = _task(cropped_code=cropped, import_statement="")
        # Default retrieval_query_lines = 20 (RepoCoder window_size).
        q = adapter.get_retrieval_query(task)
        assert q is not None
        # Last 20 lines = line_20 .. line_39
        assert "line_20" in q and "line_39" in q
        assert "line_19" not in q

    def test_returns_none_when_nothing_to_query(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(cropped_code="", import_statement="")
        assert adapter.get_retrieval_query(task) is None


class TestIngestForMemory:
    @pytest.mark.asyncio
    async def test_one_post_response_per_snippet_with_indexed_source_id(
        self, adapter: RepoBenchAdapter
    ) -> None:
        snippets = [
            {"identifier": "f0", "path": "a.py", "snippet": "def f0(): pass"},
            {"identifier": "f1", "path": "b.py", "snippet": "def f1(): pass"},
            {"identifier": "f2", "path": "c.py", "snippet": "def f2(): pass"},
        ]
        memory = _RecordingMemory()
        task = _task(snippets=snippets, cropped_code="x = 1\n")
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        # 3 snippets + 1 self-ingest = 4 post_response calls.
        assert len(memory.calls) == 4

        # First three calls — snippets in dataset order, with ctx:0 / ctx:1 / ctx:2.
        for i in range(3):
            _, _content, ctx = memory.calls[i]
            assert ctx["source_id"] == f"ctx:{i}"
            assert ctx["path"] == snippets[i]["path"]
            assert ctx["identifier"] == snippets[i]["identifier"]

        # Last call — the self-ingest of the cropped target file.
        _, self_content, self_ctx = memory.calls[3]
        assert self_ctx["source_id"] == "self"
        assert self_ctx["path"] == "src/main.py"  # default file_path from _task
        assert "(current file)" in self_content
        assert "x = 1" in self_content

    @pytest.mark.asyncio
    async def test_source_id_matches_gold_snippet_index_contract(
        self, adapter: RepoBenchAdapter
    ) -> None:
        """The ``ctx:{idx}`` ID scheme is what makes upstream's
        ``gold_snippet_index`` field directly comparable to retrieved
        IDs. If gold is index 2, the relevant retrieved ID is
        ``ctx:2``. Lock the contract."""
        snippets = [
            {"identifier": f"id_{i}", "path": f"p{i}.py", "snippet": f"snippet {i}"}
            for i in range(5)
        ]
        memory = _RecordingMemory()
        task = _task(snippets=snippets)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        # Call N's source_id must be exactly "ctx:N".
        for n in range(5):
            assert memory.calls[n][2]["source_id"] == f"ctx:{n}"

    @pytest.mark.asyncio
    async def test_self_ingest_skipped_when_cropped_code_empty(
        self, adapter: RepoBenchAdapter
    ) -> None:
        memory = _RecordingMemory()
        task = _task(cropped_code="", snippets=[])
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert memory.calls == []  # no snippets, no self-ingest

    @pytest.mark.asyncio
    async def test_post_response_carries_task_id_and_repo_metadata(
        self, adapter: RepoBenchAdapter
    ) -> None:
        memory = _RecordingMemory()
        task = _task(
            repo_name="acme/widget",
            snippets=[
                {"identifier": "h", "path": "u.py", "snippet": "def h(): pass"}
            ],
        )
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        ctx = memory.calls[0][2]
        assert ctx["repo_name"] == "acme/widget"
        assert ctx["task_id"] == task.task_id


class TestSyntheticWorkspace:
    @pytest.mark.asyncio
    async def test_materializes_snippets_and_target_then_teardown(
        self, adapter: RepoBenchAdapter, tmp_path: Any
    ) -> None:
        base = tmp_path / "ws"
        base.mkdir()
        memory = _RecordingMemory()
        task = _task(
            repo_name="myorg/MiniGPT4",
            file_path="minigpt4/common/registry.py",
            snippets=[
                {
                    "identifier": "registry",
                    "path": "minigpt4/common/config.py",
                    "snippet": "A = 1\n",
                },
            ],
            cropped_code="class Registry:\n    pass\n",
            import_statement="from foo import bar",
        )
        task.metadata["synthetic_workspace_dir"] = str(base)
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]

        key = "myorg_MiniGPT4"
        snippet_f = base / key / "minigpt4" / "common" / "config.py"
        assert snippet_f.is_file()
        assert snippet_f.read_text(encoding="utf-8") == "A = 1\n"

        main_f = base / key / "minigpt4" / "common" / "registry.py"
        text = main_f.read_text(encoding="utf-8")
        assert text.startswith("from foo import bar\n\n")
        assert "class Registry" in text

        tear = task.metadata.get("synthetic_workspace_tear_path")
        assert tear and Path(str(tear)).is_dir()
        assert task.metadata.get("code_workspace") == str(base.resolve())

        await adapter.teardown_task(task)
        assert not (base / key).exists()
        assert "synthetic_workspace_tear_path" not in task.metadata
        assert "code_workspace" not in task.metadata

    @pytest.mark.asyncio
    async def test_all_code_preferred_when_non_empty(
        self, adapter: RepoBenchAdapter, tmp_path: Any
    ) -> None:
        base = tmp_path / "ws"
        base.mkdir()
        memory = _RecordingMemory()
        task = _task(
            repo_name="r",
            file_path="src/x.py",
            snippets=[],
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
        self, adapter: RepoBenchAdapter, tmp_path: Any
    ) -> None:
        memory = _RecordingMemory()
        sub = tmp_path / "monitor"
        sub.mkdir()
        task = _task(
            snippets=[{"identifier": "a", "path": "a.py", "snippet": "x"}],
        )
        await adapter.ingest_for_memory(task, memory)  # type: ignore[arg-type]
        assert task.metadata.get("synthetic_workspace_tear_path") is None
        assert list(sub.iterdir()) == []


# ---------------------------------------------------------------------------
# Evaluate — metric values + per-split category propagation
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_exact_match_passes(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(next_line="result = compute(x)")
        res = await adapter.evaluate(task, "result = compute(x)\n", None)
        assert res["pass"] is True
        assert res["exact_match"] == 1.0
        assert res["edit_similarity"] == 100.0
        assert res["category"] == "cross_file_first"
        assert res["split"] == "cross_file_first"

    @pytest.mark.asyncio
    async def test_exact_match_is_whitespace_tokenized(
        self, adapter: RepoBenchAdapter
    ) -> None:
        """Upstream uses ``pred.split() == gt.split()`` — tabs/spaces
        equivalent, multiple internal spaces equivalent."""
        task = _task(next_line="a = b + c")
        res = await adapter.evaluate(task, "a = b + c", None)  # extra spaces
        assert res["exact_match"] == 1.0

    @pytest.mark.asyncio
    async def test_near_match_lowers_edit_similarity(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(next_line="result = compute(user_data)")
        res = await adapter.evaluate(task, "result = compute(user_dta)", None)
        assert res["pass"] is False
        assert res["exact_match"] == 0.0
        # Levenshtein on a 1-character typo over a ~25-char string is ~95+.
        assert res["edit_similarity"] > 90.0

    @pytest.mark.asyncio
    async def test_post_processes_first_real_line(
        self, adapter: RepoBenchAdapter
    ) -> None:
        """The model returns a code block + commentary; the evaluator
        must extract just the first non-comment line."""
        task = _task(next_line="x = 1")
        res = await adapter.evaluate(
            task,
            "# this is the next line\nx = 1\nextra_garbage = 2",
            None,
        )
        assert res["pass"] is True

    @pytest.mark.asyncio
    async def test_category_set_per_split(
        self, adapter: RepoBenchAdapter
    ) -> None:
        for split in ("cross_file_first", "cross_file_random", "in_file"):
            task = _task(split=split, next_line="x")
            res = await adapter.evaluate(task, "x", None)
            assert res["category"] == split
            assert res["split"] == split

    @pytest.mark.asyncio
    async def test_edit_similarity_rounded_to_5_decimal_places(
        self, adapter: RepoBenchAdapter
    ) -> None:
        """Upstream ``evaluation/metrics.py`` line 51 rounds to 5 d.p."""
        task = _task(next_line="abcdefgh")
        res = await adapter.evaluate(task, "abcdefgX", None)
        es = res["edit_similarity"]
        # Round-trip: re-rounding shouldn't change value.
        assert round(es, 5) == es

    @pytest.mark.asyncio
    async def test_codebleu_failure_does_not_kill_em_es(
        self,
        adapter: RepoBenchAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression for the 2026-05-21 ``codebleu>=0.6,<1`` pin: when
        the installed codebleu wheel's tree-sitter bindings throw
        ``TypeError('an integer is required')`` on parse, ``evaluate()``
        must still return valid ``exact_match`` and ``edit_similarity``
        values. Pre-fix, the exception escaped ``_codebleu_score`` and
        the engine's outer try/except wiped the whole evaluation dict
        to ``{"pass": False, "error": "an integer is required"}`` — see
        ``deviations.md`` R14 docstring and the
        ``repobench-mem0v2-infer-20260521-105412`` /
        ``repobench-evermemos-20260521-105136`` runs that surfaced it.

        Test strategy: inject a fake ``codebleu`` module into
        ``sys.modules`` so ``_codebleu_score``'s lazy ``from codebleu
        import calc_codebleu`` finds it. Robust to whether the real
        package is installed locally (it isn't on dev laptops; only
        the AWS orchestrator image carries it post-``11ed39e``).
        """
        import sys
        import types

        fake_codebleu = types.ModuleType("codebleu")

        def _boom(*args: object, **kwargs: object) -> dict[str, float]:
            raise TypeError("an integer is required")

        fake_codebleu.calc_codebleu = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "codebleu", fake_codebleu)

        task = _task(next_line="result = compute(x)")
        res = await adapter.evaluate(task, "result = compute(x)\n", None)

        # Critical invariant: EM and ES still scored, no "error" key.
        assert res["pass"] is True
        assert res["exact_match"] == 1.0
        assert res["edit_similarity"] == 100.0
        assert "error" not in res
        # codebleu key is absent when the score couldn't be computed
        # (matches the no-codebleu-installed code path).
        assert "codebleu" not in res

    @pytest.mark.asyncio
    async def test_codebleu_uses_upstream_net_argument_order(
        self,
        adapter: RepoBenchAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lock the corrected R14 argument order: ``calc_codebleu`` must
        receive ``([gt], [pred], language, ...)``.

        Upstream chain (``Leolty/repobench@e0cfd34c``):

        1. ``eval.py:45`` calls
           ``codebleu_score(generated, ground_truth, language)``.
        2. ``codebleu_score`` (``evaluation/metrics.py:120-126``) calls
           ``calc_codebleu(ground_truths, predictions, ...)`` internally.

        Net result: ``calc_codebleu(references=ground_truth,
        predictions=generated)`` per the
        ``calc_codebleu(references, predictions, lang, ...)`` PyPI
        signature. CrossBench mirrors that net order directly. A
        pre-2026-05-21 implementation inverted this (passing
        ``[pred], [gt]``) based on a docstring that misread the
        binding chain — see deviations.md R14. This test prevents
        that regression from re-landing.
        """
        import sys
        import types

        captured: dict[str, object] = {}

        def _record(
            references: list[str],
            predictions: list[str],
            lang: str,
            weights: list[float] | None = None,
            tokenizer: object | None = None,
        ) -> dict[str, float]:
            captured["references"] = references
            captured["predictions"] = predictions
            captured["lang"] = lang
            captured["weights"] = weights
            captured["tokenizer"] = tokenizer
            return {"codebleu": 0.5}

        fake_codebleu = types.ModuleType("codebleu")
        fake_codebleu.calc_codebleu = _record  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "codebleu", fake_codebleu)

        task = _task(next_line="THE_GROUND_TRUTH")
        await adapter.evaluate(task, "THE_MODEL_PRED\n", None)

        # References = ground truth, Predictions = model output.
        # The body of `evaluate` strips the trailing newline through
        # `_get_first_line_not_comment`, so the model output arrives as
        # `THE_MODEL_PRED`.
        assert captured["references"] == ["THE_GROUND_TRUTH"]
        assert captured["predictions"] == ["THE_MODEL_PRED"]
        # And the rest of the call shape matches upstream's invocation.
        assert captured["lang"] == "python"
        assert captured["weights"] == [0.25, 0.25, 0.25, 0.25]
        assert captured["tokenizer"] is None


# ---------------------------------------------------------------------------
# R16 — chat-LLM completion directive + output postprocessor
# ---------------------------------------------------------------------------


class TestCompletionInstruction:
    """The directive ``_completion_instruction`` prepends to the LLM prompt
    so chat-tuned models emit a single line of code instead of chat-mode
    prose. See ``deviations.md`` R16.
    """

    def test_python_directive_demands_single_line(self) -> None:
        text = _completion_instruction("python")
        assert "Python" in text
        assert "ONLY that single next line" in text
        assert "Do NOT wrap" in text
        # Trailing blank line separates directive from the code block.
        assert text.endswith("\n\n")

    def test_java_directive_demands_single_line(self) -> None:
        text = _completion_instruction("java")
        assert "Java" in text
        assert "ONLY that single next line" in text

    def test_unknown_language_defaults_to_python(self) -> None:
        # Falls back to Python rather than dropping the directive — a
        # chat LLM with no directive is the failure mode R16 fixes.
        assert _completion_instruction("rust") == _completion_instruction("python")

    def test_format_memory_prompt_prepends_directive(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task()
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("python"))
        # The in-file portion still follows the upstream layout
        # (``# Path: ...\n<imports>\n<cropped>\n``) — R4 stays locked.
        assert "# Path: src/main.py" in rendered

    def test_format_prompt_prepends_directive(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task()
        rendered = adapter.format_prompt(task)
        assert rendered.startswith(_completion_instruction("python"))
        # The upstream prompt (``task.prompt``) follows the directive
        # verbatim — the dataset-level prompt isn't rewritten.
        assert rendered.endswith(task.prompt)

    def test_java_task_gets_java_directive(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(language="java", file_path="src/Main.java")
        rendered = adapter.format_memory_prompt(task)
        assert rendered.startswith(_completion_instruction("java"))


class TestExtractCodeCompletion:
    """Strip chat-mode wrappers around a code completion. See R16."""

    def test_raw_code_passes_through_unchanged(self) -> None:
        # Base completion models that return raw code are unaffected.
        out = _extract_code_completion("    result = compute()\n")
        assert out == "    result = compute()\n"

    def test_python_fence_strips_backticks_and_lang_tag(self) -> None:
        out = _extract_code_completion("```python\nresult = compute()\n```")
        assert "```" not in out
        assert _get_first_line_not_comment(out, "python") == "result = compute()"

    def test_py_short_fence_tag_supported(self) -> None:
        out = _extract_code_completion("```py\nx = 1\n```")
        assert _get_first_line_not_comment(out, "python") == "x = 1"

    def test_bare_fence_no_lang_tag(self) -> None:
        out = _extract_code_completion("```\nx = 1\n```")
        assert _get_first_line_not_comment(out, "python") == "x = 1"

    def test_unterminated_fence_still_extracts(self) -> None:
        # Model truncated mid-output — we should still recover the code.
        out = _extract_code_completion("```python\nresult = compute()")
        assert _get_first_line_not_comment(out, "python") == "result = compute()"

    def test_java_fence_lang_tag(self) -> None:
        out = _extract_code_completion("```java\nint x = 1;\n```", "java")
        assert _get_first_line_not_comment(out, "java") == "int x = 1;"

    def test_prose_preamble_here_is_the_next_line(self) -> None:
        out = _extract_code_completion("Here is the next line: result = compute()")
        assert _get_first_line_not_comment(out, "python") == "result = compute()"

    def test_prose_preamble_with_newline_after_colon(self) -> None:
        # Multi-line preamble: the answer's indentation is on a new
        # line AFTER the colon. The regex consumes ``:`` + horizontal
        # whitespace + one newline, preserving the indentation that
        # follows.
        out = _extract_code_completion(
            "The next line should be:\n    return helper(x)\n"
        )
        assert _get_first_line_not_comment(out, "python") == "    return helper(x)"

    def test_prose_preamble_inline_colon_strips_horizontal_whitespace(
        self,
    ) -> None:
        # Inline preamble: spaces between the colon and the code are
        # spacing, not indentation. The regex eats horizontal
        # whitespace after the colon so the code line starts cleanly.
        out = _extract_code_completion("Next line:    z = 1")
        assert _get_first_line_not_comment(out, "python") == "z = 1"

    def test_preserves_leading_whitespace_inside_fence(self) -> None:
        # RepoBench ground truths often have leading whitespace
        # (function-body lines). EM tokenizes on whitespace so leading
        # spaces don't change the EM verdict, but they ARE preserved.
        out = _extract_code_completion("```python\n        return None\n```")
        assert "        return None" in out

    def test_empty_input_returns_empty(self) -> None:
        assert _extract_code_completion("") == ""

    def test_chat_mode_prose_essay_does_not_match_fence_or_preamble(
        self,
    ) -> None:
        """When the model emits pure chat-mode prose with no fence or
        recognised preamble, we pass the text through. The downstream
        ``_get_first_line_not_comment`` will return the first prose
        line; EM/ES will (correctly) fail. This is the failure mode
        R16 fixes via the prompt directive, not via postprocessing —
        the directive is the primary fix; the postprocessor is the
        belt-and-braces backstop for chat preambles we know about.
        """
        prose = "The provided code snippet is a Python script that imports..."
        out = _extract_code_completion(prose)
        assert out == prose


class TestEvaluateUsesCodeCompletionExtractor:
    """End-to-end: when the model wraps its answer in a fence,
    ``evaluate`` recovers the code and grades it correctly.
    """

    @pytest.mark.asyncio
    async def test_fenced_correct_answer_passes_em(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(next_line="result = compute()")
        result = await adapter.evaluate(
            task,
            "```python\nresult = compute()\n```",
            sandbox_mgr=None,
        )
        assert result["pass"] is True
        assert result["exact_match"] == 1.0

    @pytest.mark.asyncio
    async def test_chat_prose_preamble_correct_answer_passes_em(
        self, adapter: RepoBenchAdapter
    ) -> None:
        task = _task(next_line="z = helper(x, y)")
        result = await adapter.evaluate(
            task,
            "Here is the next line: z = helper(x, y)",
            sandbox_mgr=None,
        )
        assert result["pass"] is True

    @pytest.mark.asyncio
    async def test_raw_correct_answer_still_passes_em(
        self, adapter: RepoBenchAdapter
    ) -> None:
        # No-regression check: when the model already returns raw code,
        # we shouldn't accidentally break the upstream path.
        task = _task(next_line="result = compute()")
        result = await adapter.evaluate(task, "result = compute()", sandbox_mgr=None)
        assert result["pass"] is True


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    def test_no_sandbox_required(self, adapter: RepoBenchAdapter) -> None:
        assert adapter.requires_sandbox is False
        assert adapter.get_sandbox_image() == ""

    def test_metrics_schema_includes_per_split_breakdowns(
        self, adapter: RepoBenchAdapter
    ) -> None:
        keys = adapter.metrics_schema()
        # Headlines.
        assert "pass_rate" in keys
        assert "avg_exact_match" in keys
        assert "avg_edit_similarity" in keys
        # Per-split (engine emits via category=split).
        assert "accuracy_cat_cross_file_first" in keys
        assert "accuracy_cat_cross_file_random" in keys
        assert "accuracy_cat_in_file" in keys
        # Latency.
        assert "avg_search_latency_ms" in keys
        assert "avg_inference_latency_ms" in keys
        assert "avg_total_latency_ms" in keys

    def test_default_params_have_repocoder_query_lines(
        self, adapter: RepoBenchAdapter
    ) -> None:
        params = adapter.get_default_params()
        # Default = 20, RepoCoder's window_size.
        assert params["retrieval_query_lines"] == 20

    def test_default_params_full_split_set(
        self, adapter: RepoBenchAdapter
    ) -> None:
        params = adapter.get_default_params()
        assert set(params["splits"]) == {
            "cross_file_first", "cross_file_random", "in_file",
        }
