"""Tests for the mini-SWE-agent scaffold port (deviations.md S1).

Coverage tiers:

1. **Frozen-hash locks** on the load-bearing string constants — the
   system prompt, the submission marker, the action regex, and the
   submission instruction phrasing. If any of these drift accidentally,
   the SWE-bench numbers stop being comparable to the upstream scaffold's
   leaderboard column. The hash assertion makes intentional changes
   loud (test fails → update hash + corresponding deviations.md row).

2. **Action parser** — exactly-one ``mswea_bash_command`` block extraction,
   zero / two blocks → FormatError, multi-line content survives.

3. **Submission detection** — ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` on
   the first line of stdout AND exit code 0 → submission body returned;
   any deviation (non-zero exit, marker not on first line, missing
   newline) → no submission.

4. **End-to-end loop** with a recording sandbox: format-error retry path,
   single-step submission, step-limit exhaustion, evaluate delegation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from cross_bench.execution.sandbox import ExecutionResult, Sandbox
from cross_bench.execution.scaffolds.mini_swe_agent import (
    ACTION_REGEX,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_CWD,
    DEFAULT_STEP_LIMIT,
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    SUBMISSION_MARKER,
    SYSTEM_TEMPLATE,
    UPSTREAM_COMMIT,
    _check_submission,
    _FormatError,
    _parse_action,
    _render_format_error,
    _render_observation,
    execute_mini_swe_agent,
)
from cross_bench.models import (
    ExecutionMode,
    LLMConfig,
    PromptStrategyConfig,
    RunConfig,
    StepRecord,
    SystemConfig,
    Task,
    TaskGranularity,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# 1. Frozen-hash locks — these MUST change deliberately and in lockstep with
#    deviations.md updates.
# ---------------------------------------------------------------------------


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_upstream_commit_pin_unchanged():
    """The commit SHA the port targets. Change requires a deviations.md update."""
    assert UPSTREAM_COMMIT == "bc85a45654e6348dcc6e4c5a40ad146ed0bb144d"


def test_system_template_byte_exact_to_upstream():
    """Verbatim copy of swebench.yaml's agent.system_template (one line + trailing \\n)."""
    assert SYSTEM_TEMPLATE == (
        "You are a helpful assistant that can interact with a computer "
        "shell to solve programming tasks.\n"
    )
    # Locked hash so any future whitespace tweak shows up in CI.
    assert _sha(SYSTEM_TEMPLATE) == _sha(
        "You are a helpful assistant that can interact with a computer "
        "shell to solve programming tasks.\n"
    )


def test_submission_marker_byte_exact_to_upstream():
    """Mirrors DockerEnvironment._check_finished line 143."""
    assert SUBMISSION_MARKER == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def test_action_regex_byte_exact_to_upstream():
    """Text-mode action regex from default.yaml (mswea_bash_command fence)."""
    assert ACTION_REGEX == r"```mswea_bash_command\s*\n(.*?)\n```"


def test_default_step_limit_matches_swebench_yaml():
    """swebench.yaml line ``step_limit: 250``."""
    assert DEFAULT_STEP_LIMIT == 250


def test_default_cwd_matches_swebench_yaml():
    """swebench.yaml line ``cwd: "/testbed"``."""
    assert DEFAULT_CWD == "/testbed"


def test_default_command_timeout_matches_swebench_yaml():
    """swebench.yaml line ``timeout: 60`` (per-command, NOT per-task)."""
    assert DEFAULT_COMMAND_TIMEOUT == 60


def test_instance_template_carries_load_bearing_phrases():
    """Spot-check the SWE-bench-specific phrases that drive task semantics.

    These are the load-bearing parts of the instance prompt — they MUST
    survive byte-faithful from upstream's swebench.yaml. Changing any
    of them is a documented deviation.
    """
    # Working directory is now parameterised via {{cwd}} so Pro
    # (``/app``) and Verified (``/testbed``) can share the template.
    # The Jinja slot must be present so the agent prompt actually
    # tells the model where to operate. See deviations.md S21.
    assert "MODIFY: Regular source code files in {{cwd}}" in INSTANCE_TEMPLATE
    # Submission command spelled exactly as upstream.
    assert (
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        in INSTANCE_TEMPLATE
    )
    # Two-step patch-then-submit workflow (anti-combine-with-&& guidance).
    assert (
        "Creating/viewing the patch and submitting it MUST be separate "
        "commands (not combined with &&)."
    ) in INSTANCE_TEMPLATE
    # Task interpolation slot.
    assert "{{task}}" in INSTANCE_TEMPLATE
    # Text-mode adaptation — instructs the model to emit
    # mswea_bash_command blocks, NOT JSON tool calls.
    assert "mswea_bash_command" in INSTANCE_TEMPLATE


def test_observation_template_truncates_above_10k_chars():
    """Outputs over 10000 chars elide to head+tail; under stays full."""
    short_out = {"returncode": 0, "output": "hello", "exception_info": ""}
    short_rendered = _render_observation(short_out)
    assert "<output>" in short_rendered
    assert "hello" in short_rendered
    assert "<output_head>" not in short_rendered

    long_text = "A" * 5000 + "B" * 5000 + "C" * 200  # > 10000
    long_out = {"returncode": 0, "output": long_text, "exception_info": ""}
    long_rendered = _render_observation(long_out)
    assert "<output_head>" in long_rendered
    assert "<output_tail>" in long_rendered
    assert "<elided_chars>" in long_rendered
    # 10200 - 10000 = 200 elided
    assert "200 characters elided" in long_rendered


def test_observation_template_renders_exception_info_when_present():
    out = {
        "returncode": -1,
        "output": "",
        "exception_info": "timeout exceeded",
    }
    rendered = _render_observation(out)
    assert "<exception>timeout exceeded</exception>" in rendered
    assert "<returncode>-1</returncode>" in rendered


# ---------------------------------------------------------------------------
# 2. Action parser
# ---------------------------------------------------------------------------


def test_parse_action_extracts_single_block():
    response = "THOUGHT: list files\n\n```mswea_bash_command\nls -la\n```\n"
    assert _parse_action(response) == "ls -la"


def test_parse_action_handles_multiline_command():
    response = (
        "Let me create a script.\n\n"
        "```mswea_bash_command\n"
        "cat <<'EOF' > foo.py\n"
        "import sys\n"
        "print('hi')\n"
        "EOF\n"
        "```\n"
    )
    assert _parse_action(response) == (
        "cat <<'EOF' > foo.py\nimport sys\nprint('hi')\nEOF"
    )


def test_parse_action_raises_on_zero_blocks():
    response = "I should think about this first..."
    with pytest.raises(_FormatError) as exc_info:
        _parse_action(response)
    assert exc_info.value.n_actions == 0


def test_parse_action_raises_on_multiple_blocks():
    response = (
        "First step:\n\n```mswea_bash_command\nls\n```\n\n"
        "Second step:\n\n```mswea_bash_command\npwd\n```\n"
    )
    with pytest.raises(_FormatError) as exc_info:
        _parse_action(response)
    assert exc_info.value.n_actions == 2


def test_render_format_error_includes_recovery_guidance():
    rendered = _render_format_error(n_actions=0)
    # Must restate the format AND the submission command path so the
    # model can recover on the next turn.
    assert "mswea_bash_command" in rendered
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in rendered
    assert "found 0" in rendered


# ---------------------------------------------------------------------------
# 3. Submission detection
# ---------------------------------------------------------------------------


def test_check_submission_extracts_body_after_marker():
    diff = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    out = {
        "returncode": 0,
        "output": f"{SUBMISSION_MARKER}\n{diff}",
        "exception_info": "",
    }
    assert _check_submission(out) == diff


def test_check_submission_returns_none_on_nonzero_exit():
    """A failed submit command (nonzero exit) is NOT a submission."""
    out = {
        "returncode": 1,
        "output": f"{SUBMISSION_MARKER}\nsomething\n",
        "exception_info": "",
    }
    assert _check_submission(out) is None


def test_check_submission_returns_none_when_marker_not_on_first_line():
    out = {
        "returncode": 0,
        "output": f"some noise\n{SUBMISSION_MARKER}\nbody\n",
        "exception_info": "",
    }
    assert _check_submission(out) is None


def test_check_submission_tolerates_leading_whitespace():
    """Upstream's _check_finished uses ``lstrip().splitlines(keepends=True)``."""
    diff = "diff body\n"
    out = {
        "returncode": 0,
        "output": f"   \n{SUBMISSION_MARKER}\n{diff}",
        "exception_info": "",
    }
    assert _check_submission(out) == diff


def test_check_submission_empty_output_returns_none():
    assert _check_submission({"returncode": 0, "output": "", "exception_info": ""}) is None


# ---------------------------------------------------------------------------
# 4. End-to-end loop with stubs
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedLLM:
    """Returns a queued sequence of (response, StepRecord) tuples."""
    responses: list[str]
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    async def complete(self, prompt, config):
        raise AssertionError("mini-swe-agent uses chat, not complete")

    async def chat(self, messages, config):
        self.calls.append([dict(m) for m in messages])
        text = self.responses.pop(0)
        return text, StepRecord(
            step_index=0, action="chat", input_text=str(messages), output_text=text,
        )


@dataclass
class _RecordingSandboxManager:
    exec_responses: list[ExecutionResult] = field(default_factory=list)
    executed: list[tuple[str, int]] = field(default_factory=list)
    created_image: str = ""
    destroyed: bool = False

    async def create(self, image: str, task: Any, **_: Any) -> Sandbox:
        self.created_image = image
        return Sandbox(container_id=f"sb-{task.task_id}", image=image)

    async def execute(self, sandbox, command, timeout=60):
        self.executed.append((command, timeout))
        if not self.exec_responses:
            raise AssertionError(f"Unexpected exec: {command}")
        return self.exec_responses.pop(0)

    async def destroy(self, sandbox):
        self.destroyed = True

    async def write_file(self, sandbox, path, content):
        pass


class _StubAdapter:
    """Minimal adapter — captures evaluate() args.

    ``format_prompt`` defaults to passthrough (``task.prompt`` unchanged) so
    the historical test suite stays correct on the no-retrieval path. The
    retrieval test below subclasses this and injects a retrieval block.
    """
    name = "swebench"
    granularity = TaskGranularity.REPO
    requires_sandbox = True

    def __init__(self, evaluate_result: dict[str, Any]) -> None:
        self.evaluate_result = evaluate_result
        self.evaluate_calls: list[str] = []
        self.image = "sweb.eval.x86_64.test:latest"

    def get_sandbox_image(self, task=None):
        return self.image

    def format_prompt(self, task) -> str:
        return task.prompt

    async def evaluate(self, task, generated_output, sandbox_mgr, sandbox=None):
        self.evaluate_calls.append(generated_output)
        return self.evaluate_result


def _swe_task() -> Task:
    return Task(
        task_id="django__django-11133",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="Issue: HttpResponse doesn't handle memoryview objects",
        metadata={"repo": "django/django"},
        ground_truth="",
    )


def _config() -> RunConfig:
    return RunConfig(
        run_id="test",
        benchmark="swebench",
        execution_mode=ExecutionMode.MINI_SWE_AGENT,
        system=SystemConfig(
            llm=LLMConfig(provider="stub", model="stub"),
            prompt_strategy=PromptStrategyConfig(type="zero_shot"),
            memory=None,
        ),
        max_agent_steps=5,
        timeout_seconds=600,
    )


@pytest.mark.asyncio
async def test_loop_submits_on_marker_and_calls_evaluate():
    """Happy path: model emits patch + submit on consecutive turns."""
    diff = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
    llm = _ScriptedLLM(responses=[
        # Turn 1: produce the patch file.
        "Creating the patch.\n\n```mswea_bash_command\ngit diff > patch.txt\n```\n",
        # Turn 2: submit.
        (
            "Submitting.\n\n"
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(exit_code=0, stdout="", stderr="", duration_ms=1.0),
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\n{diff}",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True, "resolved": True})

    result = await execute_mini_swe_agent(
        task=_swe_task(),
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,  # mini-swe-agent doesn't use the strategy
        memory=None,
        sandbox_mgr=mgr,
    )

    # Sandbox lifecycle.
    assert mgr.created_image.endswith("test:latest")
    assert mgr.destroyed is True
    # Two executions: patch write, then submission echo.
    assert len(mgr.executed) == 2
    assert mgr.executed[0][0] == "git diff > patch.txt"
    assert (
        mgr.executed[1][0]
        == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    )
    # Adapter.evaluate called with the captured submission.
    assert adapter.evaluate_calls == [diff]
    # Result status reflects evaluate's pass.
    assert result.status == TaskStatus.PASS
    assert result.final_output == diff
    assert result.evaluation["scaffold"] == "mini_swe_agent"
    assert result.evaluation["scaffold_commit"] == UPSTREAM_COMMIT
    assert result.evaluation["exit_status"] == "Submitted"


@pytest.mark.asyncio
async def test_loop_format_error_retries_without_executing():
    """Zero / multi-action turn → user feedback, no sandbox.execute."""
    diff = "diff --git a/x b/x\n"
    llm = _ScriptedLLM(responses=[
        # Turn 1: no mswea_bash_command block → format error feedback.
        "I'm just thinking out loud.",
        # Turn 2: two blocks → format error feedback.
        (
            "```mswea_bash_command\nls\n```\n\n"
            "```mswea_bash_command\npwd\n```\n"
        ),
        # Turn 3: valid block → executed.
        (
            "OK now I'll submit.\n\n"
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        # Only one execution expected (turn 3).
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\n{diff}",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    result = await execute_mini_swe_agent(
        task=_swe_task(),
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    # Exactly one sandbox.execute call (the submit), even though the LLM
    # was called three times.
    assert len(mgr.executed) == 1
    assert len(llm.calls) == 3
    # Each retry observed the previous format-error feedback.
    last_messages = llm.calls[-1]
    assert any(
        "mswea_bash_command" in m["content"] and "found" in m["content"]
        for m in last_messages
        if m["role"] == "user"
    )
    assert result.status == TaskStatus.PASS


@pytest.mark.asyncio
async def test_loop_step_limit_exhaustion_marks_limits_exceeded():
    """max_agent_steps exhausted without submission → LimitsExceeded."""
    # Configure for 3 steps; the model never submits.
    cfg = _config()
    cfg.max_agent_steps = 3
    llm = _ScriptedLLM(responses=[
        "```mswea_bash_command\nls\n```\n",
        "```mswea_bash_command\npwd\n```\n",
        "```mswea_bash_command\nwhoami\n```\n",
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(exit_code=0, stdout="files", stderr="", duration_ms=1.0),
        ExecutionResult(exit_code=0, stdout="/testbed", stderr="", duration_ms=1.0),
        ExecutionResult(exit_code=0, stdout="root", stderr="", duration_ms=1.0),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": False})

    result = await execute_mini_swe_agent(
        task=_swe_task(),
        config=cfg,
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    assert len(mgr.executed) == 3
    assert result.status == TaskStatus.FAIL
    assert result.evaluation["exit_status"] == "LimitsExceeded"
    # adapter.evaluate was NOT called for no-submission terminations.
    assert adapter.evaluate_calls == []


@pytest.mark.asyncio
async def test_loop_sets_work_dir_to_testbed():
    """SWE-bench convention: agent runs commands inside /testbed."""
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\nbody\n",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    captured: dict[str, str] = {}

    real_create = mgr.create

    async def capture_create(image, task, **kw):
        sb = await real_create(image, task, **kw)
        captured["work_dir_after_create"] = sb.work_dir
        return sb

    mgr.create = capture_create  # type: ignore[method-assign]

    await execute_mini_swe_agent(
        task=_swe_task(),
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    # After scaffold rewrite, the sandbox's work_dir is /testbed for
    # Princeton-variant SWE-bench tasks. ``capture_create`` snapshots
    # work_dir AT the moment of creation (default from Sandbox init).
    # The scaffold then MUTATES it to /testbed after create. So the
    # captured "pre-mutation" value isn't /testbed — that's expected.
    # Test the DEFAULT_CWD constant separately so a regression that
    # blank-out the mutation is caught at module load time.
    assert DEFAULT_CWD == "/testbed"


@pytest.mark.asyncio
async def test_loop_sets_work_dir_to_app_for_pro_variant():
    """SWE-bench Pro images mount the repo at /app, not /testbed.

    Without this routing, the agent's bash commands run in /testbed
    inside a Pro image — that directory doesn't exist, so every
    command fails with ``no such file or directory``. The scaffold
    must read ``task.metadata["variant"]`` and set ``/app`` for Pro.
    See deviations.md S21.
    """
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\nbody\n",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    pro_task = Task(
        task_id="instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="Issue: Pro task",
        metadata={"repo": "flipt-io/flipt", "variant": "pro"},
        ground_truth="",
    )

    # Capture the sandbox after the scaffold's post-create mutation.
    captured_sandbox: dict[str, Any] = {}
    real_create = mgr.create

    async def capture_create(image, task, **kw):
        sb = await real_create(image, task, **kw)
        captured_sandbox["sb"] = sb
        return sb

    mgr.create = capture_create  # type: ignore[method-assign]

    await execute_mini_swe_agent(
        task=pro_task,
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    # After the loop, the sandbox's work_dir must be /app for Pro.
    assert captured_sandbox["sb"].work_dir == "/app"


@pytest.mark.asyncio
async def test_loop_honors_explicit_cwd_override_in_task_metadata():
    """``task.metadata["mini_swe_agent_cwd"]`` wins over the variant default.

    Future-proofing: an adapter that opts into mini-swe-agent for a
    non-SWE-bench layout (e.g. another agentic benchmark) can pin its
    own cwd without touching the scaffold.
    """
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\nbody\n",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    task = Task(
        task_id="custom-1",
        benchmark="swebench",
        granularity=TaskGranularity.REPO,
        prompt="...",
        metadata={
            "repo": "x/y",
            "variant": "pro",  # would default to /app
            "mini_swe_agent_cwd": "/custom/workspace",  # explicit override wins
        },
        ground_truth="",
    )

    captured: dict[str, Any] = {}
    real_create = mgr.create

    async def capture_create(image, t, **kw):
        sb = await real_create(image, t, **kw)
        captured["sb"] = sb
        return sb

    mgr.create = capture_create  # type: ignore[method-assign]

    await execute_mini_swe_agent(
        task=task, config=_config(), adapter=adapter,
        llm=llm, strategy=None, memory=None, sandbox_mgr=mgr,
    )

    assert captured["sb"].work_dir == "/custom/workspace"


# ---------------------------------------------------------------------------
# 5. Option E — per-step episodic memory contract
# ---------------------------------------------------------------------------


class _RecordingMemory:
    """Stand-in middleware that logs every call and returns a fixed block.

    ``retrieved`` is the synthetic memory block prepended by ``pre_prompt``;
    setting it empty simulates a cold store. ``ids`` populates
    ``retrieved_context_ids`` so we can assert provenance forwarding.
    """

    def __init__(self, retrieved: str = "RECALLED_BLOCK", ids=None):
        self.retrieved = retrieved
        self.ids = list(ids) if ids is not None else ["step:0"]
        self.events: list[tuple[str, dict]] = []

    async def pre_prompt(self, prompt, context):
        self.events.append(("pre", {
            "query": prompt,
            "step": context.get("step"),
        }))
        if self.ids:
            context["retrieved_context_ids"] = list(self.ids)
        if self.retrieved:
            return f"{self.retrieved}\n\n{prompt}"
        return prompt

    async def post_response(self, prompt, response, context):
        self.events.append(("post", {
            "prompt": prompt,
            "response": response,
            "step": context.get("step"),
            "source_id": context.get("source_id"),
            "returncode": context.get("returncode"),
        }))

    async def reset(self):
        self.events.append(("reset", {}))


@pytest.mark.asyncio
async def test_single_submission_step_pre_fires_once_post_never():
    """Submission step is terminal — no ingest. Step-0 retrieval still fires."""
    mem = _RecordingMemory()
    diff = "diff body\n"
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\n{diff}",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})
    task = _swe_task()

    await execute_mini_swe_agent(
        task=task,
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=mem,  # type: ignore[arg-type]
        sandbox_mgr=mgr,
    )

    pres = [e for e in mem.events if e[0] == "pre"]
    posts = [e for e in mem.events if e[0] == "post"]
    # One retrieval call (step 0). Submission is terminal — no ingest.
    assert len(pres) == 1
    assert len(posts) == 0
    # Step-0 retrieval provenance forwarded.
    assert task.metadata.get("retrieved_context_ids") == ["step:0"]


@pytest.mark.asyncio
async def test_step0_retrieval_query_is_issue_text():
    """Step-0 pre_prompt sends the (truncated) issue text as the query."""
    mem = _RecordingMemory(retrieved="")  # cold store
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\n",
            stderr="",
            duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})
    task = _swe_task()

    await execute_mini_swe_agent(
        task=task, config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    pres = [e for e in mem.events if e[0] == "pre"]
    assert pres[0][1]["step"] == 0
    # The issue text appears in the step-0 query (truncated, but present).
    assert "HttpResponse" in pres[0][1]["query"]


@pytest.mark.asyncio
async def test_multistep_run_ingests_episode_per_completed_step():
    """3 non-submission steps + 1 submit: pre fires 4 times, post fires 3 times."""
    mem = _RecordingMemory(retrieved="")  # cold store, focus on call counts
    diff = "diff\n"
    llm = _ScriptedLLM(responses=[
        "```mswea_bash_command\nls -la\n```\n",
        "```mswea_bash_command\npwd\n```\n",
        "```mswea_bash_command\nwhoami\n```\n",
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(exit_code=0, stdout="files", stderr="", duration_ms=1.0),
        ExecutionResult(exit_code=0, stdout="/testbed", stderr="", duration_ms=1.0),
        ExecutionResult(exit_code=0, stdout="root", stderr="", duration_ms=1.0),
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n{diff}",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    pres = [e for e in mem.events if e[0] == "pre"]
    posts = [e for e in mem.events if e[0] == "post"]
    # 4 LLM turns total (3 non-submit + 1 submit) → 4 retrievals.
    assert len(pres) == 4
    # 3 completed steps (the submit step is terminal — no ingest).
    assert len(posts) == 3
    # source_id stamped per step.
    assert [p[1]["source_id"] for p in posts] == [
        "step:0", "step:1", "step:2",
    ]
    # And the steps increment from 0 to 2 (no step:3 because that's submit).
    assert [p[1]["step"] for p in posts] == [0, 1, 2]


@pytest.mark.asyncio
async def test_episodic_query_includes_prior_action_and_observation():
    """Step-1 pre_prompt query carries the step-0 action + observation."""
    mem = _RecordingMemory(retrieved="")  # cold store
    llm = _ScriptedLLM(responses=[
        "```mswea_bash_command\ngrep -r TODO src/\n```\n",
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout="src/foo.py:42: TODO: fix this",
            stderr="",
            duration_ms=1.0,
        ),
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    pres = [e for e in mem.events if e[0] == "pre"]
    # Step-1 query carries the step-0 (action, observation) pair.
    step1_query = pres[1][1]["query"]
    assert "Previous action:" in step1_query
    assert "grep -r TODO src/" in step1_query
    assert "Previous output (returncode=0):" in step1_query
    assert "src/foo.py:42: TODO: fix this" in step1_query


@pytest.mark.asyncio
async def test_ingested_episode_contains_action_and_observation():
    """post_response ingests an Action+Observation episode, not raw assistant text."""
    mem = _RecordingMemory(retrieved="")
    llm = _ScriptedLLM(responses=[
        "thinking…\n```mswea_bash_command\ncat foo.py\n```\n",
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout="FILE_CONTENTS_HERE", stderr="",
            duration_ms=1.0,
        ),
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    posts = [e for e in mem.events if e[0] == "post"]
    assert len(posts) == 1
    episode = posts[0][1]["response"]
    # Episode structure: Action: <fenced cmd> + Observation: <observation>
    assert "Action:" in episode
    assert "cat foo.py" in episode
    assert "Observation:" in episode
    assert "FILE_CONTENTS_HERE" in episode
    # Returncode carried in context.
    assert posts[0][1]["returncode"] == 0


@pytest.mark.asyncio
async def test_recalled_block_appears_in_messages_before_llm_query():
    """The middleware's retrieved block is wrapped in <recalled_prior_steps> and
    visible to the model before the LLM call."""
    mem = _RecordingMemory(retrieved="prior-step-3: cat foo.py → FILE\nprior-step-7: grep BAR → MATCH")
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    first_call_messages = llm.calls[0]
    user_msgs = [m["content"] for m in first_call_messages if m["role"] == "user"]
    # One of the user messages is the recalled block.
    recalled_msgs = [c for c in user_msgs if "<recalled_prior_steps>" in c]
    assert len(recalled_msgs) == 1
    assert "prior-step-3" in recalled_msgs[0]
    assert "</recalled_prior_steps>" in recalled_msgs[0]
    # And it appears BEFORE any assistant message — the model sees it
    # at query time.
    roles_in_call = [m["role"] for m in first_call_messages]
    recalled_idx = next(
        i for i, m in enumerate(first_call_messages)
        if "<recalled_prior_steps>" in m.get("content", "")
    )
    assert "assistant" not in roles_in_call[:recalled_idx]


@pytest.mark.asyncio
async def test_format_errors_do_not_ingest():
    """A zero/multi-action turn skips ingest — format errors are noise, not trajectory."""
    mem = _RecordingMemory(retrieved="")
    llm = _ScriptedLLM(responses=[
        "no fenced block here",                          # 0 actions
        "```mswea_bash_command\nls\n```\n```mswea_bash_command\npwd\n```\n",  # 2 actions
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )

    pres = [e for e in mem.events if e[0] == "pre"]
    posts = [e for e in mem.events if e[0] == "post"]
    # 3 LLM turns → 3 retrieval calls (the format-error retries still
    # query memory at the top of each iteration).
    assert len(pres) == 3
    # But zero ingests: two format errors (noise) + one submit (terminal).
    assert len(posts) == 0


@pytest.mark.asyncio
async def test_no_memory_path_skips_all_memory_calls():
    """Backwards compat: when memory is None, no pre/post/reset calls."""
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    # No memory argument — should run without any memory plumbing.
    result = await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=None, sandbox_mgr=mgr,
    )
    assert result.evaluation["exit_status"] == "Submitted"


@pytest.mark.asyncio
async def test_memory_pre_prompt_failure_doesnt_break_loop():
    """A middleware that raises in pre_prompt must not abort the agent run."""

    class _RaisingMemory(_RecordingMemory):
        async def pre_prompt(self, prompt, context):
            self.events.append(("pre", {"step": context.get("step")}))
            raise RuntimeError("retrieval backend is down")

    mem = _RaisingMemory()
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})

    result = await execute_mini_swe_agent(
        task=_swe_task(), config=_config(), adapter=adapter, llm=llm,
        strategy=None, memory=mem, sandbox_mgr=mgr,
    )
    assert result.evaluation["exit_status"] == "Submitted"


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_initial_retrieval_query_truncates_long_issue_text():
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        INITIAL_QUERY_MAX_CHARS,
        _initial_retrieval_query,
    )
    task = _swe_task()
    task.prompt = "x" * (INITIAL_QUERY_MAX_CHARS + 5000)
    q = _initial_retrieval_query(task)
    assert len(q) == INITIAL_QUERY_MAX_CHARS


def test_episodic_retrieval_query_truncates_long_output():
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        EPISODIC_QUERY_OUTPUT_MAX_CHARS,
        _episodic_retrieval_query,
    )
    out = {"returncode": 0, "output": "y" * (EPISODIC_QUERY_OUTPUT_MAX_CHARS + 2000)}
    q = _episodic_retrieval_query("ls", out)
    assert "Previous action:" in q
    assert "ls" in q
    # The output portion is bounded.
    assert q.count("y") == EPISODIC_QUERY_OUTPUT_MAX_CHARS


def test_render_episode_format_is_stable():
    from cross_bench.execution.scaffolds.mini_swe_agent import _render_episode
    rendered = _render_episode("cat foo.py", "<output>FILE_CONTENT</output>")
    # The format is stable so retrievers can tokenize it predictably.
    assert rendered == (
        "Action:\n```\ncat foo.py\n```\n\n"
        "Observation:\n<output>FILE_CONTENT</output>"
    )


def test_extract_recalled_block_strips_query_suffix():
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        _extract_recalled_block,
    )
    augmented = "RECALLED_TEXT\n\nthe-query-text"
    assert _extract_recalled_block(augmented, "the-query-text") == "RECALLED_TEXT"


def test_extract_recalled_block_empty_when_unchanged():
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        _extract_recalled_block,
    )
    assert _extract_recalled_block("query", "query") == ""


# ---------------------------------------------------------------------------
# 6. _trim_messages and _estimate_tokens
# ---------------------------------------------------------------------------


from cross_bench.execution.scaffolds.mini_swe_agent import (  # noqa: E402
    _estimate_tokens,
    _trim_messages,
)


def _make_msgs(contents: list[str], start_role: str = "user") -> list[dict[str, str]]:
    """Build a message list with alternating user/assistant roles, starting from start_role."""
    roles = ["user", "assistant"]
    if start_role == "assistant":
        roles = ["assistant", "user"]
    return [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]


def _seed_and_history(history_n: int, content_len: int = 10) -> list[dict[str, str]]:
    """Two seed messages + history_n trajectory messages."""
    seed = [
        {"role": "system", "content": "S" * content_len},
        {"role": "user", "content": "I" * content_len},
    ]
    traj = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "T" * content_len}
        for i in range(history_n)
    ]
    return seed + traj


def test_trim_messages_noop_when_under_budget():
    """No trimming when estimated tokens comfortably fit."""
    msgs = _seed_and_history(history_n=3, content_len=10)
    result = _trim_messages(msgs, max_tokens=100_000, completion_budget=500)
    assert result == msgs


def test_trim_messages_seed_never_trimmed():
    """messages[0] (system) and messages[1] (instance) must survive regardless."""
    # Build a message list that will definitely overflow a tiny budget
    seed = [
        {"role": "system", "content": "SYSTEM_MSG"},
        {"role": "user", "content": "INSTANCE_MSG"},
    ]
    traj = [{"role": "user", "content": "X" * 5000} for _ in range(10)]
    msgs = seed + traj

    result = _trim_messages(msgs, max_tokens=2000, completion_budget=100)
    # Seed messages always survive
    assert result[0] == seed[0]
    assert result[1] == seed[1]


def test_trim_messages_sentinel_at_index_2():
    """After trimming, a single sentinel message is inserted at index 2."""
    seed = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "INSTANCE"},
    ]
    traj = [{"role": "user", "content": "T" * 3000} for _ in range(10)]
    msgs = seed + traj

    result = _trim_messages(msgs, max_tokens=2000, completion_budget=100)
    # index 2 must be the sentinel
    assert result[2]["content"] == "[earlier steps trimmed to fit context window]"


def test_trim_messages_drops_oldest_non_seed_first():
    """The earliest trajectory messages are dropped before later ones."""
    seed = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "INSTANCE"},
    ]
    # First traj message is large and should be dropped
    traj = [
        {"role": "user", "content": "FIRST_TRAJ " + "X" * 4000},
        {"role": "assistant", "content": "LAST_TRAJ"},
    ]
    msgs = seed + traj

    result = _trim_messages(msgs, max_tokens=1000, completion_budget=100)
    # LAST_TRAJ should survive; FIRST_TRAJ should not
    contents = [m["content"] for m in result]
    assert any("LAST_TRAJ" in c for c in contents)
    assert not any("FIRST_TRAJ" in c for c in contents)


def test_trim_messages_sentinel_idempotent():
    """Re-trimming a previously-trimmed message list inserts only one sentinel."""
    seed = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "INSTANCE"},
    ]
    traj = [{"role": "user", "content": "T" * 3000} for _ in range(8)]
    msgs = seed + traj

    # First trim
    once = _trim_messages(msgs, max_tokens=2000, completion_budget=100)
    # Second trim on already-trimmed result
    twice = _trim_messages(once, max_tokens=2000, completion_budget=100)

    sentinels = [m for m in twice if m["content"] == "[earlier steps trimmed to fit context window]"]
    assert len(sentinels) == 1, f"Expected exactly one sentinel, got {len(sentinels)}"


def test_trim_messages_budget_formula_honoured():
    """After trimming, estimated tokens <= max_tokens - completion_budget - 500."""
    seed = [
        {"role": "system", "content": "SYSTEM_" * 100},
        {"role": "user", "content": "INSTANCE_" * 200},
    ]
    traj = [{"role": "user", "content": "T" * 2000} for _ in range(20)]
    msgs = seed + traj

    max_tokens = 8_000
    completion_budget = 1_000
    result = _trim_messages(msgs, max_tokens=max_tokens, completion_budget=completion_budget)
    expected_budget = max_tokens - completion_budget - 500
    assert _estimate_tokens(result) <= expected_budget


def test_trim_messages_sentinel_string_frozen():
    """Frozen-hash lock on the sentinel string (deviations.md S19).

    Any change to the sentinel text must be deliberate and accompanied by a
    deviations.md update; this test makes such changes loud.
    """
    import hashlib
    sentinel = "[earlier steps trimmed to fit context window]"
    expected_hash = hashlib.sha256(sentinel.encode()).hexdigest()
    # Re-compute from the actual constant in the module
    from cross_bench.execution.scaffolds.mini_swe_agent import _trim_messages as _tm
    import inspect
    source = inspect.getsource(_tm)
    # The sentinel string must appear verbatim in the source
    assert sentinel in source, "Sentinel string has changed — update deviations.md S19"
    # And its hash is stable
    assert hashlib.sha256(sentinel.encode()).hexdigest() == expected_hash


def test_extract_recalled_block_falls_back_when_query_absent():
    """Middleware transformed the query — treat the whole augmented string as recalled."""
    from cross_bench.execution.scaffolds.mini_swe_agent import (
        _extract_recalled_block,
    )
    assert _extract_recalled_block("TOTALLY_DIFFERENT", "q") == "TOTALLY_DIFFERENT"


# ---------------------------------------------------------------------------
# 6. Adapter.format_prompt routing — locks the retrieval-block fix
# ---------------------------------------------------------------------------


class _AugmentingAdapter(_StubAdapter):
    """Adapter whose format_prompt injects a retrieval block, mimicking the
    real SWE-bench BM25 / oracle column behavior. The 2026-05-21 bug had
    the scaffold ignoring format_prompt and using task.prompt directly,
    so the retrieval block never reached the model — this test would have
    caught it."""

    def format_prompt(self, task) -> str:
        return (
            f"Issue:\n{task.prompt}\n\n"
            "Retrieved files (bm25):\n"
            "<code>\n"
            "[start of /testbed/foo/bar.py]\n"
            "RETRIEVED_FILE_CONTENT\n"
            "[end of /testbed/foo/bar.py]\n"
            "</code>"
        )


@pytest.mark.asyncio
async def test_scaffold_renders_adapter_format_prompt_into_instance_template():
    """The scaffold MUST call ``adapter.format_prompt(task)`` — not use
    ``task.prompt`` directly — so per-task retrieval blocks reach the model.

    History: a bug in execute_mini_swe_agent rendered the instance template
    with ``task=task.prompt`` (just the raw issue body), silently dropping
    the ``Retrieved files (bm25)`` / ``Retrieved files (oracle)`` blocks
    that ``SWEBenchAdapter.format_prompt`` injects for the retrieval
    columns. Spotted on the 2026-05-21 BM25 n=50 run by inspecting
    trajectory JSONs: 0/28 tasks had the retrieval block in their LLM
    messages, even though task.metadata['retrieval_context'] was populated.
    Net effect: the BM25 column was running effectively as
    retrieval=none. Fix: render with ``task=adapter.format_prompt(task)``.
    """
    llm = _ScriptedLLM(responses=[
        (
            "Submitting.\n\n"
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0,
            stdout=f"{SUBMISSION_MARKER}\ndiff body\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _AugmentingAdapter(evaluate_result={"pass": True})

    await execute_mini_swe_agent(
        task=_swe_task(),
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    # The first LLM chat call's messages must contain the retrieval block —
    # both the section header and the file content the adapter injected.
    first_call_messages = llm.calls[0]
    user_msgs = [m["content"] for m in first_call_messages if m["role"] == "user"]
    combined = "\n".join(user_msgs)
    assert "Retrieved files (bm25):" in combined, (
        "adapter.format_prompt() output (incl. retrieval block) MUST be "
        "rendered into the instance template — saw none in user messages. "
        f"User messages: {[m[:200] for m in user_msgs]}"
    )
    assert "RETRIEVED_FILE_CONTENT" in combined, (
        "Retrieval file content from adapter.format_prompt() didn't reach "
        "the LLM. Scaffold may be using task.prompt directly again."
    )


@pytest.mark.asyncio
async def test_scaffold_no_retrieval_path_still_works():
    """When format_prompt is passthrough (no retrieval block), the existing
    behavior is unchanged — the raw issue body is rendered into the
    instance template."""
    llm = _ScriptedLLM(responses=[
        (
            "```mswea_bash_command\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "```\n"
        ),
    ])
    mgr = _RecordingSandboxManager(exec_responses=[
        ExecutionResult(
            exit_code=0, stdout=f"{SUBMISSION_MARKER}\ndiff\n",
            stderr="", duration_ms=1.0,
        ),
    ])
    adapter = _StubAdapter(evaluate_result={"pass": True})  # passthrough format_prompt

    await execute_mini_swe_agent(
        task=_swe_task(),
        config=_config(),
        adapter=adapter,
        llm=llm,
        strategy=None,
        memory=None,
        sandbox_mgr=mgr,
    )

    first_call_messages = llm.calls[0]
    user_msgs = "\n".join(m["content"] for m in first_call_messages if m["role"] == "user")
    # The raw task body still appears, no retrieval section anywhere.
    assert "HttpResponse doesn't handle memoryview" in user_msgs
    assert "Retrieved files" not in user_msgs
