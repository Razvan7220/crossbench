"""Mini-SWE-agent scaffold port (text-mode) for SWE-bench evaluation.

Byte-faithful port of upstream's ``swebench.yaml`` agent loop, pinned to::

    github.com/SWE-agent/mini-swe-agent
    commit bc85a45654e6348dcc6e4c5a40ad146ed0bb144d   (HEAD of main at audit)

Upstream is MIT-licensed (Copyright 2025 Kilian A. Lieret and Carlos E.
Jimenez). The verbatim prompt strings below originate from
``src/minisweagent/config/benchmarks/swebench.yaml`` and
``src/minisweagent/config/default.yaml`` at that SHA; the agent control
flow mirrors ``src/minisweagent/agents/default.py``.

Why a port, not a dependency
----------------------------
Upstream owns its own ``Model`` (litellm-based) and ``Environment``
(docker / local subprocess) abstractions. CrossBench already has
``LLMDriver`` (with retries / latency / observability already wired
into our existing rows) and ``SandboxManager`` (docker container
lifecycle + network=none + image prefetch). Porting the *loop* —
~150 LOC — and plugging our existing primitives in keeps the
column comparable to mini-swe-agent's leaderboard entry on what
actually matters (prompts + workflow + submission mechanics) while
preserving CrossBench's run-level instrumentation (cost / latency /
memory hooks).

Text-mode adaptation (documented deviation)
-------------------------------------------
Upstream's ``swebench.yaml`` runs in **tool-call mode**
(``parallel_tool_calls: true``, default model
``anthropic/claude-sonnet-4-5-20250929``, uses ``litellm.completion``
with the ``BASH_TOOL`` tool definition). CrossBench's ``LLMDriver.chat``
returns a single text response per turn — no tool-call channel today.
We adapt to upstream's **text-mode** action format
(``mswea_bash_command`` triple-backtick fences, per
``src/minisweagent/models/utils/actions_text.py`` and
``src/minisweagent/config/default.yaml``) for the agent loop wiring,
and we patch the SWE-bench instance prompt to instruct the model to
emit one ``mswea_bash_command`` block per turn instead of one bash
tool call. Submission detection (``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
marker on stdout), step budget (250), workflow guidance, the
``/testbed`` cwd, and the observation template (truncation > 10000
chars to head/tail) are **byte-faithful** to upstream's swebench.yaml.

See ``deviations.md`` § By benchmark → SWE-bench (entry S1).
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import structlog
from jinja2 import StrictUndefined, Template

from cross_bench.models import (
    RunConfig,
    StepRecord,
    Task,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from cross_bench.benchmarks.base import BenchmarkAdapter
    from cross_bench.execution.sandbox import SandboxManager
    from cross_bench.systems.base import LLMDriver, MemoryMiddleware, PromptStrategy

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Upstream pin (audit trail). See module docstring.
# ---------------------------------------------------------------------------

UPSTREAM_REPO = "github.com/SWE-agent/mini-swe-agent"
UPSTREAM_COMMIT = "bc85a45654e6348dcc6e4c5a40ad146ed0bb144d"


# ---------------------------------------------------------------------------
# Constants from upstream ``swebench.yaml`` (verbatim where indicated)
# ---------------------------------------------------------------------------

#: ``agent.system_template`` — verbatim from swebench.yaml.
SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer "
    "shell to solve programming tasks.\n"
)

#: ``agent.step_limit`` — verbatim from swebench.yaml (250).
DEFAULT_STEP_LIMIT = 250

#: ``environment.cwd`` — SWE-bench instance images lay the repo at
#: ``/testbed``. Verbatim from swebench.yaml.
DEFAULT_CWD = "/testbed"

#: ``environment.timeout`` — per-command sandbox timeout. Verbatim
#: from swebench.yaml.
DEFAULT_COMMAND_TIMEOUT = 60

#: ``model.observation_template`` — verbatim from swebench.yaml. Renders
#: the bash command's stdout, with head+tail elision for outputs over
#: 10000 chars (a model-context safeguard).
OBSERVATION_TEMPLATE = (
    "{% if output.exception_info -%}\n"
    "<exception>{{output.exception_info}}</exception>\n"
    "{% endif -%}\n"
    "<returncode>{{output.returncode}}</returncode>\n"
    "{% if output.output | length < 10000 -%}\n"
    "<output>\n"
    "{{ output.output -}}\n"
    "</output>\n"
    "{%- else -%}\n"
    "<warning>\n"
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to "
    "view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you "
    "can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, "
    "you can redirect output to a file and then search in that file.\n"
    "</warning>\n"
    "{%- set elided_chars = output.output | length - 10000 -%}\n"
    "<output_head>\n"
    "{{ output.output[:5000] }}\n"
    "</output_head>\n"
    "<elided_chars>\n"
    "{{ elided_chars }} characters elided\n"
    "</elided_chars>\n"
    "<output_tail>\n"
    "{{ output.output[-5000:] }}\n"
    "</output_tail>\n"
    "{%- endif -%}"
)


#: Action regex — from ``default.yaml`` (text-mode). Matches exactly one
#: ``mswea_bash_command`` triple-backtick block per response. Two or more
#: blocks → format error → retry. See ``actions_text.parse_regex_actions``.
ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"

#: Format-error template emitted when the model doesn't produce exactly
#: one ``mswea_bash_command`` block. Mirrors upstream's text-mode
#: ``format_error_template`` in default.yaml.
FORMAT_ERROR_TEMPLATE = (
    "Format error:\n\n"
    "<error>\n"
    "{{error}}\n"
    "</error>\n\n"
    "Please always provide EXACTLY ONE action in triple backticks "
    "(found {{n_actions}}). Wrap your command in a fenced block tagged "
    "`mswea_bash_command`, like so:\n\n"
    "```mswea_bash_command\n"
    "<your command here>\n"
    "```\n\n"
    "If you want to submit your solution, issue this command (and only "
    "this command) in a single mswea_bash_command block:\n"
    "```mswea_bash_command\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
    "```\n"
)


#: Submission marker — when the FIRST line of a command's stdout matches
#: this exactly AND the command exits 0, we treat the remainder of stdout
#: as the model's submission (a unified diff). Matches
#: ``DockerEnvironment._check_finished`` byte-for-byte at the pinned SHA.
SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


#: Instance template — byte-faithful to ``swebench.yaml`` EXCEPT for the
#: response-format block, which is rewritten to instruct the model to
#: emit one ``mswea_bash_command`` fenced block per turn instead of one
#: bash tool call. Everything else (workflow, important-boundaries,
#: command-execution-rules narrative, submission steps) is preserved
#: verbatim from upstream.
INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide EXACTLY ONE bash command wrapped in a triple-backtick block tagged ``mswea_bash_command``

## Important Boundaries

- MODIFY: Regular source code files in {{cwd}} (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue exactly one command
2. The system executes the command in a subshell
3. You see the result
4. You write your next command

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. Exactly one ``mswea_bash_command`` block with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include EXACTLY ONE ``mswea_bash_command`` block. If you need to run multiple shell operations, chain them with ``&&`` inside a single block.
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with ``MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

```mswea_bash_command
ls -la && find src -name '*.java' | grep -i builder
```
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run ``git diff -- path/to/file1 path/to/file2 > patch.txt`` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show ``--- a/`` and ``+++ b/`` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit (in a single ``mswea_bash_command`` block, with nothing else):

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>
"""


# ---------------------------------------------------------------------------
# Helpers — mirrors ``actions_text.parse_regex_actions`` /
# ``format_observation_messages`` semantically. We don't import upstream
# directly because we want the regex / templates locked here.
# ---------------------------------------------------------------------------


class _FormatError(Exception):
    """Raised when the model output isn't exactly one mswea_bash_command block."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions


def _parse_action(content: str) -> str:
    """Extract the single bash command from a model response.

    Raises ``_FormatError`` if the response contains zero or more than one
    ``mswea_bash_command`` fenced block. Mirrors
    ``actions_text.parse_regex_actions`` at the pinned SHA.
    """
    actions = [a.strip() for a in re.findall(ACTION_REGEX, content, re.DOTALL)]
    if len(actions) != 1:
        raise _FormatError(len(actions))
    return actions[0]


def _render_observation(output: dict[str, Any]) -> str:
    """Render an executed command's output into the upstream observation form.

    ``output`` keys: ``returncode`` (int), ``output`` (str — full stdout),
    ``exception_info`` (str). Matches the dict shape upstream's
    ``DockerEnvironment.execute`` produces.
    """
    return Template(OBSERVATION_TEMPLATE, undefined=StrictUndefined).render(
        output=output
    )


def _render_format_error(n_actions: int) -> str:
    """Render the format-error feedback for non-exactly-one-action responses."""
    return Template(FORMAT_ERROR_TEMPLATE, undefined=StrictUndefined).render(
        n_actions=n_actions,
        error=(
            f"Expected exactly 1 action wrapped in a mswea_bash_command "
            f"block, found {n_actions}."
        ),
    )


def _check_submission(output: dict[str, Any]) -> str | None:
    """Mirror of upstream ``DockerEnvironment._check_finished``.

    Returns the submission body (everything after the first line) when the
    command's stdout starts with the submission marker and exit code is 0;
    returns None otherwise.
    """
    text = output.get("output", "") or ""
    lines = text.lstrip().splitlines(keepends=True)
    if not lines:
        return None
    if lines[0].strip() != SUBMISSION_MARKER:
        return None
    if output.get("returncode") != 0:
        return None
    return "".join(lines[1:])


# ---------------------------------------------------------------------------
# Episodic memory (Option E) — per-step trajectory retrieval
# ---------------------------------------------------------------------------

#: Bound on the step-0 retrieval query (the raw issue text). Issue bodies
#: can hit 256k+ chars on Verified; truncating bounds embedding cost on
#: the first retrieval call. The retriever sees roughly the first ~500
#: tokens of the issue, which is plenty for semantic anchoring against
#: any pre-loaded corpus.
INITIAL_QUERY_MAX_CHARS = 2000

#: Bound on the prior-observation tail embedded in the per-step query.
#: The query is "I just ran X and saw Y; what relevant prior steps do I
#: have?" — recent output is the local signal; deep history would dilute
#: the embedding. Matches the locality assumption RepoCoder makes for
#: code retrieval (arXiv:2303.12570).
EPISODIC_QUERY_OUTPUT_MAX_CHARS = 500


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate: 4 chars ≈ 1 token (cl100k_base proxy)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def _trim_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
    completion_budget: int,
) -> list[dict[str, str]]:
    """Drop oldest non-seed messages until estimated prompt tokens fit.

    Keeps messages[0] (system) and messages[1] (instance) intact — those
    are the seed; everything after index 1 is trajectory and is safe to
    drop oldest-first.  Inserts a single ``[trimmed]`` sentinel at index 2
    so the model knows history was elided.

    The 500-token safety margin accounts for per-message envelope overhead
    (role keys, JSON framing, etc.) that the 4-chars-per-token estimate
    does not capture.
    """
    budget = max_tokens - completion_budget - 500  # 500-token safety margin for envelope overhead
    if _estimate_tokens(messages) <= budget:
        return messages

    seed = messages[:2]
    history = list(messages[2:])
    sentinel = {"role": "user", "content": "[earlier steps trimmed to fit context window]"}

    while history and _estimate_tokens(seed + [sentinel] + history) > budget:
        history.pop(0)

    return seed + [sentinel] + history


def _initial_retrieval_query(task: Task) -> str:
    """Step-0 retrieval query — the truncated issue text."""
    return (task.prompt or "")[:INITIAL_QUERY_MAX_CHARS]


def _episodic_retrieval_query(command: str, output_dict: dict[str, Any]) -> str:
    """Step-N>0 retrieval query — the prior (command, observation) pair.

    Stable shape so memory backends can tokenize either part and match
    against stored episodes. Deliberately short and bounded.
    """
    out = output_dict.get("output", "") or ""
    rc = output_dict.get("returncode", "")
    return (
        f"Previous action:\n{command}\n\n"
        f"Previous output (returncode={rc}):\n"
        f"{out[:EPISODIC_QUERY_OUTPUT_MAX_CHARS]}"
    )


def _render_episode(command: str, observation: str) -> str:
    """Render a (command, observation) pair as one episode for ingestion.

    This is what the memory store sees per completed step: one episode
    per call to ``post_response``. The action is fenced so retrievers
    that tokenize on code-fence boundaries can isolate it; the
    observation follows in plain text.
    """
    return (
        f"Action:\n```\n{command}\n```\n\n"
        f"Observation:\n{observation}"
    )


def _extract_recalled_block(augmented: str, query: str) -> str:
    """Pull the retrieved-memory prefix out of ``augmented``.

    ``MemoryMiddleware.pre_prompt`` returns ``<recalled>\\n\\n<query>``
    when retrieval surfaced something. We extract the prefix so the
    scaffold can wrap it in its own ``<recalled_prior_steps>`` envelope
    instead of leaking the middleware's formatting into the agent's
    message stream.
    """
    if augmented == query:
        return ""
    idx = augmented.find(query)
    if idx > 0:
        return augmented[:idx].strip()
    # Middleware didn't echo the query verbatim (truncated, transformed,
    # …). Treat the whole augmented string as recalled context.
    return augmented.strip()


# ---------------------------------------------------------------------------
# Execution mode entry point
# ---------------------------------------------------------------------------


async def execute_mini_swe_agent(
    task: Task,
    config: RunConfig,
    adapter: "BenchmarkAdapter",
    llm: "LLMDriver",
    strategy: "PromptStrategy",
    memory: "MemoryMiddleware | None",
    sandbox_mgr: "SandboxManager",
) -> TaskResult:
    """Run the mini-SWE-agent loop for one SWE-bench task.

    The loop is upstream's ``DefaultAgent.run``:
      1. Seed messages with [system, instance].
      2. Repeat until exit:
         a. Query model on the full message history.
         b. Extract exactly one ``mswea_bash_command`` block from the
            response (format error → user feedback, no execution).
         c. Execute the command in the sandbox.
         d. Check the command output for the submission marker; if
            present, capture the submission and exit.
         e. Render observation and append as a user message.
      3. Exit on submission, step limit, or unrecoverable error.

    Episodic memory (Option E). When ``memory`` is non-None the scaffold
    runs the per-step trajectory pattern:

    - ``pre_prompt`` fires once per step. The retrieval QUERY is the raw
      issue text on step 0, and the prior ``(action, observation)`` pair
      on step N>0. The retrieved block (if any) is wrapped in
      ``<recalled_prior_steps>...</recalled_prior_steps>`` and appended
      as a user message before the LLM query — making it visible to the
      model but distinct from the live observation stream.
    - ``post_response`` fires once per *completed* step (successful
      action + observation). Format errors and the submission step do
      NOT ingest: format errors are noise, and the submission step is
      terminal. Each ingested episode carries
      ``source_id = f"step:{step_idx}"`` so middlewares that preserve
      provenance can surface stable per-step IDs back via
      ``retrieved_context_ids``.

    Without memory, the loop is byte-identical to upstream's
    ``DefaultAgent.run`` at the pinned SHA: seed [system, instance] →
    query → parse → execute → submission check → observation → repeat.
    """
    # When True: pre_prompt fires only at step 0 (L1 code context injected
    # into the first message); post_response never fires (no episodic ingest).
    # Controlled by benchmark param ``memory_first_step_only``.
    first_step_only: bool = bool(
        config.benchmark_params.get("memory_first_step_only", False)
    )

    # When True: pre_prompt fires every step (L1 retrieved each time);
    # post_response never fires (no episodic ingest at any step).
    # Controlled by benchmark param ``memory_retrieve_only``.
    retrieve_only: bool = bool(
        config.benchmark_params.get("memory_retrieve_only", False)
    )

    # Pre-task memory setup (opt-in). Mirrors execute_single_shot's
    # ingest_for_memory call — runs before the sandbox so any filesystem
    # materialization a middleware needs is in place before retrieval.
    if memory:
        try:
            await adapter.ingest_for_memory(task, memory)
        except Exception as e:
            logger.warning(
                "mini_swe_agent.ingest_for_memory_failed",
                task_id=task.task_id, error=str(e),
            )

    # Sandbox creation — per-instance image via the adapter.
    sandbox_image = config.sandbox_image or adapter.get_sandbox_image(task)
    # Pro images need ``entrypoint="/bin/bash"`` override so
    # ``sleep infinity`` actually holds the container alive — Scale AI's
    # images ship with a custom ENTRYPOINT that runs the test suite and
    # exits. Verified images leave entrypoint unset (their native
    # entrypoint is a long-running shell). See deviations.md S21.
    sandbox_create_kwargs: dict[str, Any] = {}
    if (task.metadata or {}).get("variant") == "pro":
        sandbox_create_kwargs["entrypoint"] = "/bin/bash"
    sandbox = await sandbox_mgr.create(sandbox_image, task, **sandbox_create_kwargs)
    # Per-image working directory.
    #
    # * Princeton SWE-bench (Verified / Lite / Full) images mount the
    #   repo at ``/testbed`` — matches upstream swebench.yaml's
    #   ``environment.cwd`` and the eval_script's heredoc cwd.
    # * Scale AI SWE-bench Pro images mount the repo at ``/app`` —
    #   the Pro eval harness (``swebench_pro_eval._create_entryscript``)
    #   does ``cd /app`` explicitly, but the AGENT LOOP also needs to
    #   start there so bash commands resolve relative paths correctly
    #   ("ls src/", "cat go.mod", etc.).
    # * Adapters that opt into mini-swe-agent for non-SWE-bench tasks
    #   can override via ``task.metadata["mini_swe_agent_cwd"]``.
    variant = (task.metadata or {}).get("variant", "")
    cwd = (task.metadata or {}).get("mini_swe_agent_cwd")
    if not cwd:
        cwd = "/app" if variant == "pro" else DEFAULT_CWD
    sandbox.work_dir = cwd

    steps: list[StepRecord] = []
    submission: str = ""
    exit_status: str = ""
    error_text: str = ""

    try:
        # Render system + instance prompts.
        system_msg = SYSTEM_TEMPLATE.rstrip("\n")  # collapse trailing newline
        # Render the instance template with ``adapter.format_prompt(task)`` —
        # not the raw ``task.prompt``. The adapter's ``format_prompt`` is
        # what injects the per-task retrieval block (e.g. SWE-bench's
        # ``Retrieved files (bm25): <code>...</code>`` from the BM25 /
        # oracle HF datasets — see deviations.md S3) along with the
        # repository / base-commit / hints headers. Using ``task.prompt``
        # directly silently drops everything except the raw issue body —
        # the retrieval column was returning ``retrieval=none``-equivalent
        # numbers because of this bug (caught during the BM25 n=50 run on
        # 2026-05-21 by spot-checking trajectory JSONs).
        instance_msg = Template(
            INSTANCE_TEMPLATE, undefined=StrictUndefined
        ).render(task=adapter.format_prompt(task), cwd=cwd)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instance_msg},
        ]

        step_limit = config.max_agent_steps or DEFAULT_STEP_LIMIT
        cmd_timeout = int(
            task.metadata.get("_command_timeout_seconds", DEFAULT_COMMAND_TIMEOUT)
        )

        # Episodic-memory state — the previous step's action + outcome
        # become the next step's retrieval query (Option E). Initialized
        # ``None`` so step 0 falls back to the issue text.
        last_command: str | None = None
        last_output_dict: dict[str, Any] | None = None

        for step_idx in range(step_limit):
            # ---- 0. Per-step memory retrieval (Option E) ----
            # first_step_only: skip retrieval on all steps after 0.
            if memory and (not first_step_only or step_idx == 0):
                if last_command is None:
                    pp_query = _initial_retrieval_query(task)
                else:
                    pp_query = _episodic_retrieval_query(
                        last_command, last_output_dict or {}
                    )
                retrieval_ctx: dict[str, Any] = {
                    "task_id": task.task_id,
                    "step": step_idx,
                }
                # Forward the materialized repo workspace so a filesystem-
                # backed middleware can scope retrieval to the task's
                # codebase (set by ingest_for_memory when
                # synthetic_workspace_dir is configured).
                code_ws = (task.metadata or {}).get("code_workspace")
                if code_ws:
                    retrieval_ctx["code_workspace"] = code_ws
                try:
                    augmented = await memory.pre_prompt(pp_query, retrieval_ctx)
                except Exception as e:
                    logger.warning(
                        "mini_swe_agent.memory_pre_failed",
                        task_id=task.task_id, step=step_idx, error=str(e),
                    )
                    augmented = pp_query
                recalled = _extract_recalled_block(augmented, pp_query)
                if recalled:
                    messages.append({
                        "role": "user",
                        "content": (
                            "<recalled_prior_steps>\n"
                            f"{recalled}\n"
                            "</recalled_prior_steps>"
                        ),
                    })
                logger.info(
                    "mini_swe_agent.step_memory task_id=%s step=%s "
                    "recalled_chars=%s msg_count=%s est_tokens=%s",
                    task.task_id, step_idx,
                    len(recalled) if recalled else 0,
                    len(messages),
                    _estimate_tokens(messages),
                )
                # Forward retrieval provenance — overwritten each step so
                # ``task.metadata`` carries the last step's hits, which is
                # what most consumers want for per-task recall reporting.
                retrieved_ids = retrieval_ctx.get("retrieved_context_ids")
                if isinstance(retrieved_ids, list):
                    task.metadata["retrieved_context_ids"] = [
                        str(x) for x in retrieved_ids
                    ]

            # ---- 1. Query model ----
            pre_trim_tokens = _estimate_tokens(messages)
            messages = _trim_messages(
                messages,
                max_tokens=config.system.llm.context_window,
                completion_budget=config.system.llm.max_tokens,
            )
            post_trim_tokens = _estimate_tokens(messages)
            if post_trim_tokens < pre_trim_tokens:
                logger.info(
                    "mini_swe_agent.trim task_id=%s step=%s "
                    "before=%s after=%s dropped=%s",
                    task.task_id, step_idx,
                    pre_trim_tokens, post_trim_tokens,
                    pre_trim_tokens - post_trim_tokens,
                )
            response, step = await llm.chat(messages, {})
            step.step_index = step_idx
            steps.append(step)
            messages.append({"role": "assistant", "content": response})

            # ---- 2. Extract the single action ----
            try:
                command = _parse_action(response)
            except _FormatError as fe:
                # Format feedback — append as user message and loop.
                # No ingest: format errors are noise, not trajectory.
                messages.append({
                    "role": "user",
                    "content": _render_format_error(fe.n_actions),
                })
                continue

            # ---- 3. Execute the command in the sandbox ----
            exec_result = await sandbox_mgr.execute(
                sandbox, command, timeout=cmd_timeout
            )
            output_dict = {
                "returncode": exec_result.exit_code,
                # Upstream's docker env merges stdout+stderr into ``output``
                # (subprocess.STDOUT redirect). We mirror that so the
                # observation template renders the same way.
                "output": (exec_result.stdout or "")
                + (("\n" + exec_result.stderr) if exec_result.stderr else ""),
                "exception_info": "Execution timed out"
                if exec_result.timed_out
                else "",
            }

            # Record the bash step for the trajectory.
            steps.append(
                StepRecord(
                    step_index=step_idx,
                    action="bash",
                    input_text=command,
                    output_text=output_dict["output"][:8000],
                    metadata={"returncode": exec_result.exit_code},
                )
            )

            # ---- 4. Submission check (mirrors upstream's _check_finished) ----
            # Submission is terminal — NO ingest. The submission step's
            # value is captured by the SWE-bench evaluator; no future
            # retrieval matters.
            sub = _check_submission(output_dict)
            if sub is not None:
                submission = sub
                exit_status = "Submitted"
                break

            # ---- 5. Render and append the observation ----
            try:
                observation = _render_observation(output_dict)
            except Exception as e:
                # Defensive — Jinja2 rendering shouldn't fail on the
                # known schema, but if it does we don't want to lose the
                # exit-status signal.
                logger.error(
                    "mini_swe_agent.observation_render_failed",
                    error=str(e),
                )
                observation = (
                    f"<returncode>{output_dict['returncode']}</returncode>\n"
                    f"<output>\n{output_dict['output']}\n</output>"
                )
            messages.append({"role": "user", "content": observation})

            # ---- 6. Ingest the completed step as one episode (Option E) ----
            # first_step_only / retrieve_only: no episodic ingest at any step.
            if memory and not first_step_only and not retrieve_only:
                try:
                    store_ctx: dict[str, Any] = {
                        "task_id": task.task_id,
                        "step": step_idx,
                        "source_id": f"step:{step_idx}",
                        "returncode": exec_result.exit_code,
                    }
                    if code_ws:
                        store_ctx["code_workspace"] = code_ws
                    await memory.post_response(
                        "",  # empty prompt — matches the corpus / adapter
                             # ingest convention used elsewhere.
                        _render_episode(command, observation),
                        store_ctx,
                    )
                except Exception as e:
                    logger.warning(
                        "mini_swe_agent.memory_post_failed",
                        task_id=task.task_id, step=step_idx, error=str(e),
                    )

            # Save state for next iteration's per-step retrieval query.
            last_command = command
            last_output_dict = output_dict
        else:
            # Loop exhausted without submission.
            exit_status = "LimitsExceeded"

    except Exception as e:
        logger.error(
            "mini_swe_agent.error", task_id=task.task_id, error=str(e)
        )
        exit_status = type(e).__name__
        error_text = str(e)
    finally:
        await sandbox_mgr.destroy(sandbox)

    # Evaluate the submission against the benchmark adapter. SWE-bench's
    # ``evaluate`` creates a *new* per-instance sandbox via
    # ``make_test_spec`` — it doesn't reuse the agent sandbox (the agent
    # may have mutated /testbed state).
    if exit_status == "Submitted" and submission.strip():
        try:
            evaluation = await adapter.evaluate(
                task, submission, sandbox_mgr, sandbox=None
            )
        except Exception as e:
            logger.error(
                "mini_swe_agent.evaluate_failed",
                task_id=task.task_id,
                error=str(e),
            )
            evaluation = {"pass": False, "error": str(e)}
    else:
        evaluation = {
            "pass": False,
            "resolved": False,
            "patch_successfully_applied": False,
            "error": exit_status or "no_submission",
        }
        if error_text:
            evaluation["error_text"] = error_text

    # Stamp scaffold metadata so summary.json carries which scaffold ran.
    evaluation.setdefault("scaffold", "mini_swe_agent")
    evaluation.setdefault("scaffold_commit", UPSTREAM_COMMIT)
    evaluation.setdefault("exit_status", exit_status)
    evaluation.setdefault("steps_taken", len(steps))
    evaluation.setdefault("step_limit_hit", int(exit_status == "LimitsExceeded"))

    status = TaskStatus.PASS if evaluation.get("pass") else TaskStatus.FAIL
    return TaskResult(
        task_id=task.task_id,
        run_id=config.run_id,
        status=status,
        steps=steps,
        final_output=submission,
        evaluation=evaluation,
    )
