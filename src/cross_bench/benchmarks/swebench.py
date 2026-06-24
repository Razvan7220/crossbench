"""SWE-bench Verified adapter — delegates evaluation to the swebench library.

Faithful port of the Princeton reference harness
(https://github.com/SWE-bench/SWE-bench). The grading primitives live in the
upstream ``swebench`` package; this adapter does the Docker orchestration
through CrossBench's :class:`SandboxManager` and calls into swebench for the
parts that determine the score.

Data
----
HuggingFace dataset ``princeton-nlp/SWE-bench_Verified`` (500 rows, split
``test``). All 13 columns are preserved on the :class:`Task` — in particular
``test_patch``, ``FAIL_TO_PASS``, and ``PASS_TO_PASS``, which the original
adapter dropped.

Evaluation pipeline
-------------------
Mirrors ``swebench.harness.run_evaluation.run_instance`` step-for-step:

1. Build ``TestSpec`` via :func:`swebench.harness.test_spec.make_test_spec`.
   This yields ``instance_image_key`` (``sweb.eval.{arch}.{id.lower()}:tag``)
   and ``eval_script`` — a bash script that activates the conda env, cd's to
   ``/testbed``, applies ``test_patch`` from a heredoc, emits the
   ``>>>>> Start/End Test Output`` markers around the repo/version-specific
   test command, and resets the working tree.
2. Launch the per-instance image; override ``Sandbox.work_dir`` to
   ``/testbed``.
3. Apply the predicted patch with the upstream 3-command cascade
   (``git apply`` → ``git apply --reject`` → ``patch --fuzz=5``).
4. Write ``spec.eval_script`` to ``/eval.sh`` and execute it.
5. Slice stdout between the markers, parse with
   ``MAP_REPO_TO_PARSER[repo]`` → ``{test_id: status}``.
6. Score with :func:`get_eval_tests_report` + :func:`get_resolution_status`.
   ``resolved`` is True iff the status is ``RESOLVED_FULL`` — i.e. every
   ``FAIL_TO_PASS`` test now passes *and* every ``PASS_TO_PASS`` test still
   passes.

Returned evaluation dict
------------------------
``pass``, ``resolved``, ``patch_successfully_applied``, ``f2p_rate``,
``p2p_rate``, ``tests_status`` (full report), ``exit_code``, ``stdout``,
``stderr``.

Windows development note
------------------------
``swebench.harness`` imports ``prepare_images`` at package load, which in turn
imports the Unix-only ``resource`` module. We stub ``resource`` into
``sys.modules`` before the first import so tests can run on Windows.
Real evaluation requires Linux Docker regardless.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from cross_bench.benchmarks.base import (
    MEMORY_TOKEN_SCHEMA_KEYS,
    BenchmarkAdapter,
)
from cross_bench.benchmarks.registry import register_adapter
from cross_bench.models import Task, TaskGranularity

if TYPE_CHECKING:
    from cross_bench.execution.sandbox import Sandbox, SandboxManager


_DATASET_BY_VARIANT = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    # Scale AI's harder, multi-language benchmark (731 tasks, 4 languages,
    # 11 repos). Uses lowercase ``fail_to_pass``/``pass_to_pass`` fields,
    # ``dockerhub_tag`` image naming via ``jefzda/sweap-images``, and a
    # custom eval harness (see swebench_pro_eval.py + deviations.md S21).
    "pro": "ScaleAI/SWE-bench_Pro",
}

#: Variants that use Pro's custom eval harness + image naming.
#: All other variants use the Princeton ``swebench`` PyPI package
#: + ``make_test_spec``-derived ``sweb.eval.{arch}.{id}:tag`` images.
_PRO_VARIANTS = {"pro"}

# Retrieval mode → HF dataset that ships the paper-faithful augmented
# prompts (BM25 via Pyserini Lucene per upstream's
# ``swebench/inference/make_datasets/bm25_retrieval.py``, Oracle = the
# gold-modified files). The published ``text`` column on each row
# already embeds the retrieved files inside ``<code>...</code>`` per
# ``create_instance.py::prompt_style_2`` at upstream's HEAD; we extract
# only that block so our scaffold owns the surrounding instructions.
# See deviations.md S3 (the retrieval-columns deviation).
_BM25_DATASETS_BY_BUDGET: dict[int, str] = {
    13000: "princeton-nlp/SWE-bench_bm25_13K",
    27000: "princeton-nlp/SWE-bench_bm25_27K",
    40000: "princeton-nlp/SWE-bench_bm25_40K",
    50000: "princeton-nlp/SWE-bench_bm25_50K",
}
_ORACLE_DATASET = "princeton-nlp/SWE-bench_oracle"
_VALID_RETRIEVAL_MODES = ("none", "bm25", "oracle")

_DEFAULT_BM25_BUDGET = 27000

#: Extracts the ``<code>...</code>`` block from upstream's text-formatted
#: row. Matches ``create_instance.py::prompt_style_2`` layout:
#: ``<code>\n{readmes_text}\n{code_text}\n</code>``.
_CODE_BLOCK_RE = re.compile(r"<code>\n(.*?)\n</code>", re.DOTALL)

#: Bare adapter image tag → instance ID extraction. Mirrors
#: ``make_test_spec``'s ``instance_image_key`` shape:
#: ``sweb.eval.<arch>.<instance_id_lower>:<tag>``.
_BARE_IMAGE_RE = re.compile(
    r"^sweb\.eval\.(?P<arch>[^.]+)\.(?P<instance>[^:]+):(?P<tag>.+)$"
)


def _split_bare_image(image: str) -> tuple[str, str, str]:
    """Return ``(arch, instance_id_lower, tag)`` from a bare adapter image.

    Raises ``ValueError`` if the input doesn't match the expected shape
    (cheap defence against passing a non-SWE-bench image through the
    resolver by accident).
    """
    m = _BARE_IMAGE_RE.match(image)
    if not m:
        raise ValueError(
            f"Not a SWE-bench image tag: {image!r}. Expected "
            "'sweb.eval.<arch>.<instance>:<tag>'."
        )
    return m.group("arch"), m.group("instance"), m.group("tag")


def _dockerhub_url_for(image: str) -> str:
    """Translate bare adapter form to Princeton's Docker Hub publishing form.

    Bare: ``sweb.eval.x86_64.astropy__astropy-12907:latest``
    Hub : ``docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest``

    The ``__`` → ``_1776_`` translation is upstream's convention.
    """
    arch, instance, tag = _split_bare_image(image)
    upstream_instance = instance.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.{arch}.{upstream_instance}:{tag}"


def _ecr_url_for(image: str, ecr_repo_uri: str) -> str:
    """Translate bare adapter form to the ECR mirror URL.

    Bare: ``sweb.eval.x86_64.astropy__astropy-12907:latest``
    ECR : ``<ecr_repo_uri>:astropy__astropy-12907``

    ECR mirror uses one repo with one tag per instance (the ``__``
    form, byte-identical to the adapter's ``instance_image_key``).
    Populated by the ``cross-bench-dev-mirror-swebench-images``
    CodeBuild project; see ``buildspec.mirror.yml``.
    """
    _arch, instance, _tag = _split_bare_image(image)
    return f"{ecr_repo_uri}:{instance}"


def _extract_retrieval_block(text: str) -> str:
    """Pull the ``<code>...</code>`` block out of an upstream augmented row.

    Returns the empty string when no block is found — that's an honest
    "retrieval miss" rather than a silent fallback to the issue text.
    """
    if not text:
        return ""
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1) if m else ""


def _load_retrieval_map(
    *, mode: str, bm25_budget: int, instance_ids: set[str]
) -> dict[str, str]:
    """Build ``{instance_id: retrieval_block}`` from the augmented HF dataset.

    Filters by ``instance_ids`` so a Verified-sized run doesn't carry the
    full 2.3k-instance retrieval map in memory.
    """
    if mode == "none":
        return {}
    if mode == "bm25":
        if bm25_budget not in _BM25_DATASETS_BY_BUDGET:
            raise ValueError(
                f"retrieval_bm25_token_budget={bm25_budget} is not one of "
                f"{sorted(_BM25_DATASETS_BY_BUDGET)} (the published budgets)."
            )
        ds_name = _BM25_DATASETS_BY_BUDGET[bm25_budget]
    elif mode == "oracle":
        ds_name = _ORACLE_DATASET
    else:  # pragma: no cover — validated in caller
        raise ValueError(f"Unknown retrieval mode {mode!r}.")

    # Upstream pre-computed retrieval datasets live under ``split='test'``.
    aug_ds = load_dataset(ds_name, split="test")
    out: dict[str, str] = {}
    for row in aug_ds:
        iid = row.get("instance_id", "")
        if iid in instance_ids:
            out[iid] = _extract_retrieval_block(row.get("text", "") or "")
    return out

# Matches ``swebench.harness.run_evaluation.GIT_APPLY_CMDS`` verbatim: tried
# in order, first non-zero exit falls through to the next.
_GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)

_PRED_PATCH_PATH = "/tmp/cross_bench_prediction.patch"
_EVAL_SCRIPT_PATH = "/eval.sh"
_DEFAULT_TEST_TIMEOUT = 1800  # upstream ``swebench.harness.run_evaluation.TIMEOUT``


def _ensure_swebench_importable() -> None:
    """Stub the Unix-only ``resource`` module on Windows so swebench imports.

    Upstream's ``swebench.harness.prepare_images`` imports ``resource`` at
    module level for file-descriptor limits. The stub is a no-op — real
    evaluation always runs inside Linux Docker where the real module exists.
    """
    if "resource" in sys.modules:
        return
    try:
        import resource  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("resource")
        stub.RLIMIT_NOFILE = 7
        stub.getrlimit = lambda _r: (1024, 4096)
        stub.setrlimit = lambda _r, _v: None
        sys.modules["resource"] = stub


@register_adapter
class SWEBenchAdapter(BenchmarkAdapter):
    name = "swebench"
    granularity = TaskGranularity.REPO

    def metrics_schema(self) -> list[str]:
        return [
            "pass_rate", "accuracy",
            "avg_resolved", "avg_patch_successfully_applied",
            "avg_f2p_rate", "avg_p2p_rate", "avg_exit_code",
            "avg_steps_taken", "avg_step_limit_hit",
            "avg_search_latency_ms", "avg_inference_latency_ms", "avg_total_latency_ms",
            "total_tokens_in", "total_tokens_out", "avg_tokens_per_task",
            # Memory-system internal token cost (separate from bench LLM).
            *MEMORY_TOKEN_SCHEMA_KEYS,
            "total_tasks", "passed", "failed", "errors", "timeouts",
        ]

    def load_tasks(self, params: dict[str, Any]) -> Iterator[Task]:
        variant = params.get("variant", "verified")
        if variant not in _DATASET_BY_VARIANT:
            raise ValueError(
                f"Unknown SWE-bench variant {variant!r}. "
                f"Options: {list(_DATASET_BY_VARIANT)}"
            )
        is_pro = variant in _PRO_VARIANTS
        # ``include_hints`` is a run-level switch; stamped per-task so
        # ``format_prompt`` can read it without re-resolving params.
        # Default ``False`` matches the SWE-bench leaderboard's WITHOUT-hints
        # reporting convention. See deviations.md S2.
        include_hints = bool(params.get("include_hints", False))

        # Retrieval mode: ``none`` (default, raw issue text only),
        # ``bm25`` (Pyserini Lucene BM25 over the repo, top-K hits
        # truncated to ``retrieval_bm25_token_budget`` tokens), or
        # ``oracle`` (gold-modified file included verbatim — ceiling
        # column). Paper-faithful via upstream's pre-computed HF
        # datasets. See deviations.md S3.
        retrieval = str(params.get("retrieval", "none"))
        if retrieval not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval mode {retrieval!r}. "
                f"Options: {list(_VALID_RETRIEVAL_MODES)}"
            )
        if is_pro and retrieval != "none":
            # Scale AI doesn't publish BM25/oracle augmented datasets
            # for Pro — only ``retrieval = "none"`` is honest here.
            raise ValueError(
                f"retrieval={retrieval!r} is not supported on variant=pro: "
                "Scale AI doesn't publish augmented BM25/oracle datasets "
                "for SWE-bench Pro. Use retrieval='none'."
            )
        bm25_budget = int(
            params.get("retrieval_bm25_token_budget", _DEFAULT_BM25_BUDGET)
        )

        ds = load_dataset(
            _DATASET_BY_VARIANT[variant], split="test", trust_remote_code=True
        )
        # Build {instance_id: HF dataset row index} before any slicing so
        # the Slack notification can show "task 654/731" style positions
        # regardless of how the run filters / slices the dataset later.
        ds_total = len(ds)
        id_to_dataset_offset = {row["instance_id"]: i for i, row in enumerate(ds)}
        limit = params.get("limit")
        # Materialize the slice we'll actually yield so we can filter the
        # augmented dataset by exactly those instance_ids — keeps the
        # retrieval map's memory footprint proportional to the run, not
        # to the full 2.3k-instance retrieval dataset.
        sliced = list(ds if limit is None else ds.select(range(min(limit, len(ds)))))
        # Preserve caller-supplied order so per-shard task index matches the
        # ``--task-id`` argument order — load-bearing for the Slack progress
        # bar where position N must mean the same task across cells.
        ordered_filter = [str(t) for t in (params.get("filter_task_ids", []) or [])]
        if ordered_filter:
            filter_set = set(ordered_filter)
            by_id = {
                row["instance_id"]: row
                for row in sliced
                if row["instance_id"] in filter_set
            }
            sliced = [by_id[tid] for tid in ordered_filter if tid in by_id]
        instance_ids = {row["instance_id"] for row in sliced}
        retrieval_map = _load_retrieval_map(
            mode=retrieval, bm25_budget=bm25_budget, instance_ids=instance_ids
        )

        for row in sliced:
            metadata: dict[str, Any] = {
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "version": row.get("version", ""),
                "environment_setup_commit": row.get("environment_setup_commit", ""),
                "hints_text": row.get("hints_text", ""),
                "include_hints": include_hints,
                "created_at": row.get("created_at", ""),
                "difficulty": row.get("difficulty", ""),
                "test_patch": row.get("test_patch", ""),
                # Variant tag flows through so ``evaluate`` and
                # ``get_sandbox_image`` can dispatch on it.
                "variant": variant,
                # Retrieval columns (deviations.md S3).
                "retrieval": retrieval,
                "retrieval_bm25_token_budget": bm25_budget if retrieval == "bm25" else 0,
                "retrieval_context": retrieval_map.get(row["instance_id"], ""),
                # Stamped for all variants (Verified may opt in via TOML;
                # Pro TOMLs set requires_network=true — deviations.md S21).
                "requires_network": params.get("requires_network", False),
                # HF dataset position (0-indexed) and total — surfaced in
                # the Slack per-task notification as ``task N/<total>``.
                "dataset_offset": id_to_dataset_offset.get(row["instance_id"]),
                "dataset_total": ds_total,
            }
            if is_pro:
                # Pro ships lowercase ``fail_to_pass``/``pass_to_pass``
                # as stringified Python lists plus extra setup fields the
                # custom eval harness needs (deviations.md S21).
                metadata.update({
                    "fail_to_pass": row.get("fail_to_pass", "[]"),
                    "pass_to_pass": row.get("pass_to_pass", "[]"),
                    "selected_test_files_to_run": row.get("selected_test_files_to_run", "[]"),
                    "before_repo_set_cmd": row.get("before_repo_set_cmd", ""),
                    "dockerhub_tag": row.get("dockerhub_tag", ""),
                    "repo_language": row.get("repo_language", ""),
                    "requirements": row.get("requirements", "") or "",
                    "interface": row.get("interface", "") or "",
                    # Preserve uppercase keys so downstream code that
                    # inspected the Verified-style names still works.
                    "FAIL_TO_PASS": row.get("fail_to_pass", "[]"),
                    "PASS_TO_PASS": row.get("pass_to_pass", "[]"),
                })
            else:
                # Verified / Lite / Full ship uppercase field names.
                metadata.update({
                    "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
                    "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
                })
            yield Task(
                task_id=row["instance_id"],
                benchmark=self.name,
                granularity=self.granularity,
                prompt=row["problem_statement"],
                metadata=metadata,
                ground_truth=row.get("patch", ""),
            )

    def format_prompt(self, task: Task) -> str:
        parts = [
            f"Repository: {task.metadata['repo']}",
            f"Base commit: {task.metadata['base_commit']}",
            "",
            "Issue description:",
            task.prompt,
        ]
        # Hints are gated by the run-level ``include_hints`` toggle (default
        # ``False``). The SWE-bench leaderboard reports WITHOUT hints by
        # convention — opt in only for the ceiling-comparison column.
        if task.metadata.get("include_hints") and task.metadata.get("hints_text"):
            parts.extend(["", "Hints:", task.metadata["hints_text"]])
        # SWE-bench Pro adds two structured fields not present in Verified:
        # ``requirements`` (explicit acceptance criteria) and ``interface``
        # (new APIs the patch must introduce). Scale's published Oct 2025
        # Haiku 4.5 Pro leaderboard run injects these as separate sections
        # below the issue description — verified by diffing one of their
        # captured trajectories at
        # ``s3://scaleapi-results/swe-bench-pro/claude-45haiku-10222025/traj/``
        # against our prior (no-requirements) prompts on the same instance.
        # Omitting them silently truncates ~1.1-1.5 KB of load-bearing spec
        # text, and the agent has no way to recover the missing requirements
        # by exploration — the symptom is patches that apply cleanly,
        # pass-to-pass = 1.00, but fail-to-pass = 0.00. See deviations.md S21.
        requirements = task.metadata.get("requirements", "")
        if requirements:
            parts.extend(["", "Requirements:", requirements])
        interface = task.metadata.get("interface", "")
        if interface:
            parts.extend(["", "New interfaces introduced:", interface])
        # Retrieved files block (deviations.md S3). For ``retrieval ∈
        # {"bm25", "oracle"}``, upstream's pre-computed augmented row
        # supplies a ``<code>...</code>`` block (BM25 Lucene hits or the
        # gold file). We prepend the block so the model sees the
        # retrieved context BEFORE the workflow guidance and submission
        # instructions that the scaffold's instance template appends.
        retrieval_context = task.metadata.get("retrieval_context", "")
        if retrieval_context:
            mode = task.metadata.get("retrieval", "none")
            parts.extend([
                "",
                f"Retrieved files ({mode}):",
                "<code>",
                retrieval_context,
                "</code>",
            ])
        parts.extend([
            "",
            "Please provide a git diff patch that resolves this issue.",
        ])
        return "\n".join(parts)

    async def evaluate(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: SandboxManager,
        sandbox: Sandbox | None = None,
    ) -> dict[str, Any]:
        # Pro uses Scale AI's custom harness, not the Princeton swebench
        # PyPI package. Dispatch on ``variant`` so Verified / Lite / Full
        # continue to flow through the make_test_spec path unchanged.
        if (task.metadata or {}).get("variant") in _PRO_VARIANTS:
            return await self._evaluate_pro(task, generated_output, sandbox_mgr, sandbox)
        _ensure_swebench_importable()
        from swebench.harness.constants import (
            FAIL_ONLY_REPOS,
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            PASS_TO_PASS,
            EvalType,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.test_spec.test_spec import make_test_spec

        instance = _task_to_instance(task)
        spec = make_test_spec(instance)

        if sandbox is None:
            sandbox = await sandbox_mgr.create(spec.instance_image_key, task)
        # SWE-bench instance images lay the repo out at /testbed.
        sandbox.work_dir = "/testbed"

        report_stub: dict[str, Any] = {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": False,
        }

        if not generated_output or not generated_output.strip():
            report_stub["error"] = "empty_prediction"
            return report_stub

        applied = await _apply_patch_cascade(sandbox_mgr, sandbox, generated_output)
        if not applied:
            report_stub["error"] = "patch_apply_failed"
            return report_stub

        timeout = int(task.metadata.get("_timeout_seconds", _DEFAULT_TEST_TIMEOUT))
        await sandbox_mgr.write_file(sandbox, _EVAL_SCRIPT_PATH, spec.eval_script)
        result = await sandbox_mgr.execute(
            sandbox,
            f"bash {_EVAL_SCRIPT_PATH}",
            timeout=timeout,
        )

        combined = result.stdout + "\n" + result.stderr
        start_marker = ">>>>> Start Test Output"
        end_marker = ">>>>> End Test Output"
        if start_marker not in combined or end_marker not in combined:
            return {
                "pass": False,
                "resolved": False,
                "patch_successfully_applied": True,
                "error": "missing_output_markers",
                "exit_code": result.exit_code,
                # Full stdout/stderr — diagnosis blocker if truncated. See
                # deviations.md S5.
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        test_log = combined.split(start_marker, 1)[1].split(end_marker, 1)[0]
        parser = MAP_REPO_TO_PARSER[instance["repo"]]
        status_map = parser(test_log, spec)
        if not status_map:
            # Upstream fallback: some parsers emit status on stderr/stdout interleave.
            status_map = parser(combined, spec)

        eval_ref = {
            KEY_INSTANCE_ID: instance["instance_id"],
            FAIL_TO_PASS: spec.FAIL_TO_PASS,
            PASS_TO_PASS: spec.PASS_TO_PASS,
        }
        eval_type = (
            EvalType.FAIL_ONLY
            if instance["repo"] in FAIL_ONLY_REPOS
            else EvalType.PASS_AND_FAIL
        )
        tests_report = get_eval_tests_report(
            status_map, eval_ref, eval_type=eval_type
        )
        resolved = (
            get_resolution_status(tests_report) == ResolvedStatus.FULL.value
        )

        return {
            "pass": resolved,
            "resolved": resolved,
            "patch_successfully_applied": True,
            "f2p_rate": _section_rate(tests_report, FAIL_TO_PASS),
            "p2p_rate": _section_rate(tests_report, PASS_TO_PASS),
            "tests_status": tests_report,
            "exit_code": result.exit_code,
            # Full stdout/stderr — keeping the entire test log on the
            # evaluation dict lets ``tasks/<id>.json`` carry everything
            # needed to diagnose a failure post-hoc. See deviations.md S5.
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    async def _evaluate_pro(
        self,
        task: Task,
        generated_output: str,
        sandbox_mgr: SandboxManager,
        sandbox: Sandbox | None,
    ) -> dict[str, Any]:
        """Evaluate one SWE-bench Pro task via the Scale AI harness.

        Mirrors upstream ``swe_bench_pro_eval.eval_with_docker`` step-for-step
        but inside CrossBench's ``SandboxManager`` (so prefetch / cleanup /
        observability flow through the existing pipeline). The actual port
        lives in :mod:`cross_bench.benchmarks.swebench_pro_eval` to keep
        the dispatch in this file thin and surgical.
        """
        from cross_bench.benchmarks.swebench_pro_eval import evaluate_pro_task

        sandbox_image = self.get_sandbox_image(task)
        if sandbox is None:
            # entrypoint override: same reason as the scaffold path —
            # Pro images have a custom ENTRYPOINT that exits, breaking
            # ``docker exec``. See deviations.md S21.
            sandbox = await sandbox_mgr.create(
                sandbox_image, task, entrypoint="/bin/bash",
            )
        # Pro images lay the repo at /app (not /testbed). The entryscript
        # cd's there explicitly, so leaving sandbox.work_dir alone is fine —
        # the bash invocation overrides the cwd inside the container.
        timeout = int(task.metadata.get("_timeout_seconds", _DEFAULT_TEST_TIMEOUT))
        return await evaluate_pro_task(
            instance_id=task.task_id,
            metadata=task.metadata or {},
            patch=generated_output,
            sandbox_mgr=sandbox_mgr,
            sandbox=sandbox,
            timeout_seconds=timeout,
        )

    def get_sandbox_image(self, task: Task | None = None) -> str:
        """Per-instance image.

        Verified / Lite / Full: computed via
        :func:`swebench.harness.test_spec.make_test_spec` →
        ``sweb.eval.{arch}.{id}:tag`` (Princeton naming, ``/testbed``
        layout).

        Pro: ``jefzda/sweap-images:<dockerhub_tag>`` from the HF row
        (Scale AI naming, ``/app`` layout). See deviations.md S21.

        Without ``task`` this falls back to the registry-wide tag, which
        is only useful for image management operations (e.g. ``docker
        pull``), not real evaluation.
        """
        if task is None:
            return "swebench/swebench-eval:latest"
        variant = (task.metadata or {}).get("variant", "verified")
        if variant in _PRO_VARIANTS:
            dockerhub_tag = (task.metadata or {}).get("dockerhub_tag", "")
            if not dockerhub_tag:
                raise ValueError(
                    f"Pro task {task.task_id!r} missing 'dockerhub_tag' metadata "
                    "— check load_tasks wiring."
                )
            return f"jefzda/sweap-images:{dockerhub_tag}"
        _ensure_swebench_importable()
        from swebench.harness.test_spec.test_spec import make_test_spec

        spec = make_test_spec(_task_to_instance(task))
        return spec.instance_image_key

    def get_default_params(self) -> dict[str, Any]:
        # ``include_hints`` defaults to False to match the SWE-bench
        # leaderboard's WITHOUT-hints reporting convention (deviations.md S2).
        # ``retrieval`` defaults to "none" — the leaderboard reports three
        # columns; "none" is the baseline (no retrieval). See deviations.md S3.
        # ``image_source`` defaults to ``"dockerhub_1776"`` so local laptop
        # smokes work without ECR credentials. AWS-dispatched runs override
        # via the ``BENCH_SWEBENCH_IMAGE_SOURCE=ecr_mirror`` env var the ECS
        # task definition sets. See deviations.md S4.
        return {
            "variant": "verified",
            "include_hints": False,
            "retrieval": "none",
            "retrieval_bm25_token_budget": _DEFAULT_BM25_BUDGET,
            "image_source": "dockerhub_1776",
            "ecr_repo_uri": "",
            # Sibling ECR for SWE-bench Pro images. Set when
            # ``image_source="ecr_mirror"`` AND running Pro tasks. Empty
            # falls back to Docker Hub for Pro images even in ecr_mirror
            # mode (noisy on AWS, functional). See deviations.md S21.
            "pro_ecr_repo_uri": "",
        }

    def resolve_image_sources(
        self,
        images,  # Iterable[str]
        params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Translate bare ``sweb.eval.x86_64.<id>:latest`` tags to pullable URLs.

        See ``BenchmarkAdapter.resolve_image_sources`` for the contract.

        Behaviour by ``image_source``:

        - ``"dockerhub_1776"`` (default) — pulls from Princeton's
          ``docker.io/swebench/sweb.eval.x86_64.<id_1776>:latest`` and
          re-tags as the bare adapter form. ``_1776_`` is upstream's
          separator quirk.
        - ``"ecr_mirror"`` — pulls from
          ``<ecr_repo_uri>:<id>`` (private mirror populated by the
          ``cross-bench-dev-mirror-swebench-images`` CodeBuild project)
          and re-tags as the bare adapter form. ``ecr_repo_uri`` comes
          from ``params["ecr_repo_uri"]`` or the
          ``BENCH_SWEBENCH_ECR_REPO`` env var.
        - ``"local_build"`` — identity map. Caller is responsible for
          having built the images locally via
          ``swebench.harness.prepare_images.prepare_images(...)``.
        """
        import os

        p = params or {}
        source = str(p.get("image_source") or os.environ.get("BENCH_SWEBENCH_IMAGE_SOURCE", "dockerhub_1776"))

        # Pro images use ``jefzda/sweap-images:<dockerhub_tag>`` naming.
        # In ``dockerhub_1776`` mode they pass through unchanged (already
        # a pullable Docker Hub URL). In ``ecr_mirror`` mode they get
        # translated to ``<pro_ecr>:<dockerhub_tag>`` against the
        # ``swebench-pro`` ECR repo populated by the
        # ``cross-bench-dev-mirror-swebench-pro-images`` CodeBuild
        # project. See deviations.md S21.
        def _is_pro_image(img: str) -> bool:
            return img.startswith("jefzda/sweap-images:")

        def _pro_ecr_url(img: str, pro_ecr: str) -> str:
            tag = img.split(":", 1)[1]
            return f"{pro_ecr}:{tag}"

        if source == "local_build":
            return {img: img for img in images}

        if source == "ecr_mirror":
            ecr_repo = str(p.get("ecr_repo_uri") or os.environ.get("BENCH_SWEBENCH_ECR_REPO", ""))
            pro_ecr = str(p.get("pro_ecr_repo_uri") or os.environ.get("BENCH_SWEBENCH_PRO_ECR_REPO", ""))
            mapping: dict[str, str] = {}
            for img in images:
                if _is_pro_image(img):
                    if not pro_ecr:
                        # Pro image but no Pro ECR configured — fall back
                        # to the Docker Hub URL. Noisy but functional.
                        mapping[img] = img
                    else:
                        mapping[img] = _pro_ecr_url(img, pro_ecr)
                else:
                    if not ecr_repo:
                        raise ValueError(
                            "image_source=ecr_mirror requires `ecr_repo_uri` param "
                            "or BENCH_SWEBENCH_ECR_REPO env var (e.g. "
                            "564175165906.dkr.ecr.us-east-1.amazonaws.com/swebench) "
                            "for non-Pro images"
                        )
                    mapping[img] = _ecr_url_for(img, ecr_repo)
            return mapping

        if source == "dockerhub_1776":
            return {
                img: (img if _is_pro_image(img) else _dockerhub_url_for(img))
                for img in images
            }

        raise ValueError(
            f"Unknown image_source {source!r}. "
            f"Options: dockerhub_1776 / ecr_mirror / local_build"
        )


def _task_to_instance(task: Task) -> dict[str, Any]:
    """Rebuild the ``SWEbenchInstance`` shape expected by the swebench API."""
    md = task.metadata
    return {
        "instance_id": task.task_id,
        "repo": md["repo"],
        "base_commit": md["base_commit"],
        "patch": task.ground_truth or "",
        "test_patch": md.get("test_patch", ""),
        "problem_statement": task.prompt,
        "hints_text": md.get("hints_text", ""),
        "created_at": md.get("created_at", ""),
        "version": md.get("version", ""),
        "FAIL_TO_PASS": md.get("FAIL_TO_PASS", "[]"),
        "PASS_TO_PASS": md.get("PASS_TO_PASS", "[]"),
        "environment_setup_commit": md.get("environment_setup_commit", ""),
    }


async def _apply_patch_cascade(
    sandbox_mgr: SandboxManager, sandbox: Sandbox, patch: str
) -> bool:
    """Try each upstream git/patch command in turn, stopping on first success."""
    await sandbox_mgr.write_file(sandbox, _PRED_PATCH_PATH, patch)
    for cmd in _GIT_APPLY_CMDS:
        result = await sandbox_mgr.execute(
            sandbox,
            f"cd /testbed && {cmd} {_PRED_PATCH_PATH}",
            timeout=60,
        )
        if result.exit_code == 0:
            return True
    return False


def _section_rate(report: dict[str, dict[str, list[str]]], section: str) -> float:
    """Success fraction for a report section (FAIL_TO_PASS or PASS_TO_PASS).

    Matches ``swebench.harness.grading.compute_fail_to_pass`` /
    ``compute_pass_to_pass``: an empty section is 1.0 (vacuous success).
    """
    succ = len(report[section]["success"])
    fail = len(report[section]["failure"])
    total = succ + fail
    return 1.0 if total == 0 else succ / total
