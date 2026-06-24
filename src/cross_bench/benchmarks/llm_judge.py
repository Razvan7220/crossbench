"""LLM-as-judge evaluator for conversational memory benchmarks.

Supports multiple judge providers following MemoryBench's configurable judge system:
  - OpenAI (GPT-4o, GPT-4o-mini) — default
  - Anthropic (Claude)
  - Google (Gemini)
  - Any OpenAI-compatible API via base_url

Uses majority vote across multiple runs for reliability.

Reference:
    https://github.com/EverMind-AI/EverMemOS/blob/main/evaluation/src/evaluators/llm_judge.py
    https://github.com/supermemoryai/memorybench (multi-judge support)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Empty by default — mem0 v2 uses a single user-only message for the
# judge call (no system role). We keep the parameter so callers can
# override for non-faithful experiments, but the default is no system
# instruction so the OpenAI backend stays byte-equivalent to upstream.
_JUDGE_SYSTEM_PROMPT = ""

# Byte-faithful copy of upstream Mem0 v2's ``ACCURACY_PROMPT``
# (``mem0/evaluation/metrics/llm_judge.py`` in the ``mem0ai/mem0``
# repo). Includes the upstream typographic curly quotes around
# ``’CORRECT’`` / ``’WRONG’`` / ``’gold’``, the original trailing
# whitespace on a few lines, and the leading newline that ``"""`` on
# its own line introduces in upstream's source.
#
# The only deviation is the placeholder name: mem0 uses ``{gold_answer}``
# but our public API has shipped ``{golden_answer}`` (so callers know
# to pass it under that key). Renaming a format placeholder doesn't
# change the rendered LLM-facing prompt — the substituted value goes in
# at the same position with the same surrounding context.
_JUDGE_USER_PROMPT = '\nYour task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:\n    (1) a question (posed by one user to another user), \n    (2) a ’gold’ (ground truth) answer, \n    (3) a generated answer\nwhich you will score as CORRECT/WRONG.\n\nThe point of the question is to ask about something one user should know about the other user based on their prior conversations.\nThe gold answer will usually be a concise and short answer that includes the referenced topic, for example:\nQuestion: Do you remember what I got the last time I went to Hawaii?\nGold answer: A shell necklace\nThe generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. \n\nFor time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n\nNow it\'s time for the real question:\nQuestion: {question}\nGold answer: {golden_answer}\nGenerated answer: {generated_answer}\n\nFirst, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. \nDo NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\nJust return the label CORRECT or WRONG in a json format with the key as "label".\n'

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def llm_judge(
    question: str,
    golden_answer: str,
    generated_answer: str,
    llm_config: dict[str, Any] | None = None,
    num_runs: int = 1,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Judge a generated answer against a golden answer using an LLM.

    Args:
        question: The question that was asked.
        golden_answer: The reference answer.
        generated_answer: The answer to evaluate.
        llm_config: LLM configuration with keys:
            provider: "openai" | "anthropic" | "google" (default: "openai")
            model: model name
            api_key: API key
            base_url: API base URL (for OpenAI-compatible endpoints)
        num_runs: Number of independent judge runs for majority vote.
        system_prompt: Override the default judge system prompt.

    Returns:
        Dict with: pass (bool), correct (int 0/1), num_correct (int),
        num_runs (int), raw_judgments (list[str]).
    """
    if llm_config is None:
        llm_config = {}

    provider = llm_config.get("provider", "openai")
    sys_prompt = system_prompt or _JUDGE_SYSTEM_PROMPT

    user_prompt = _JUDGE_USER_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )

    # Select the judge backend
    judge_fn = _JUDGE_BACKENDS.get(provider, _judge_openai)

    judgments: list[str] = []
    for _ in range(num_runs):
        try:
            label = await judge_fn(sys_prompt, user_prompt, llm_config)
            judgments.append(label)
        except Exception as e:
            logger.warning("LLM judge call failed (%s): %s", provider, e)
            judgments.append("ERROR")

    num_correct = sum(1 for j in judgments if j == "CORRECT")
    passed = num_correct > num_runs / 2  # majority vote

    return {
        "pass": passed,
        "correct": int(passed),
        "num_correct": num_correct,
        "num_runs": num_runs,
        "raw_judgments": judgments,
    }


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

async def _judge_openai(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    """OpenAI / OpenAI-compatible judge — byte-faithful to mem0 v2.

    Mirrors ``evaluate_llm_judge`` in upstream
    ``mem0/evaluation/metrics/llm_judge.py``: a single user message
    (no system role), ``temperature=0.0``,
    ``response_format={"type": "json_object"}``, and ``gpt-4o-mini``
    by default. Caller-supplied ``system_prompt`` is honoured when
    non-empty, but the default empty string keeps the request shape
    identical to mem0's so the J score is directly comparable to the
    Mem0 paper number.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return _string_match_from_prompt(user_prompt)

    model = config.get("model", "gpt-4o-mini")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "https://api.openai.com/v1")

    # Upstream sends one user message; only inject a system role when
    # the caller explicitly opts in.
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    return _extract_label(text)


async def _judge_anthropic(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    """Anthropic Claude judge."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.warning("anthropic package not installed, falling back to openai judge")
        return await _judge_openai(system_prompt, user_prompt, config)

    model = config.get("model", "claude-haiku-4-5-20251001")
    api_key = config.get("api_key", "")

    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    text = response.content[0].text if response.content else ""
    return _extract_label(text)


async def _judge_google(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    """Google Gemini judge."""
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai package not installed, falling back to openai judge")
        return await _judge_openai(system_prompt, user_prompt, config)

    model = config.get("model", "gemini-2.0-flash")
    api_key = config.get("api_key", "")

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=genai.types.GenerateContentConfig(temperature=0.0, max_output_tokens=100),
    )
    text = response.text or ""
    return _extract_label(text)


_JUDGE_BACKENDS = {
    "openai": _judge_openai,
    "anthropic": _judge_anthropic,
    "google": _judge_google,
    "generic_openai": _judge_openai,  # same API, different base_url
}


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _extract_label(text: str) -> str:
    """Extract CORRECT/WRONG label from LLM response.

    Handles: raw JSON, markdown code blocks, embedded JSON, plain text.
    """
    # Try JSON parse directly
    try:
        data = json.loads(text.strip())
        return data.get("label", "WRONG").upper()
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("label", "WRONG").upper()
        except json.JSONDecodeError:
            pass

    # Try extracting embedded JSON
    json_match = re.search(r'\{[^{}]*"label"\s*:\s*"(CORRECT|WRONG)"[^{}]*\}', text)
    if json_match:
        return json_match.group(1).upper()

    # Plain text fallback
    text_upper = text.upper()
    if "CORRECT" in text_upper:
        return "CORRECT"
    return "WRONG"


def _string_match_from_prompt(user_prompt: str) -> str:
    """Fallback when no LLM SDK is available — simple string matching.

    Matches the lowercase-``a`` field labels rendered by ``_JUDGE_USER_PROMPT``
    (mem0-faithful: ``Gold answer:`` / ``Generated answer:``). Don't capitalize
    these without also updating the prompt template.
    """
    golden_match = re.search(r"Gold answer:\s*(.*?)(?:\n\n|$)", user_prompt, re.DOTALL)
    gen_match = re.search(r"Generated answer:\s*(.*?)(?:\n\n|$)", user_prompt, re.DOTALL)
    if golden_match and gen_match:
        golden = golden_match.group(1).strip().lower()
        generated = gen_match.group(1).strip().lower()
        if golden in generated or generated in golden or golden == generated:
            return "CORRECT"
    return "WRONG"


def _string_match_fallback(golden: str, generated: str) -> dict[str, Any]:
    """Simple string-based evaluation when LLM judge is unavailable."""
    golden_clean = golden.strip().lower()
    generated_clean = generated.strip().lower()

    correct = (
        golden_clean in generated_clean
        or generated_clean in golden_clean
        or golden_clean == generated_clean
    )
    return {
        "pass": correct,
        "correct": int(correct),
        "num_correct": int(correct),
        "num_runs": 1,
        "raw_judgments": ["STRING_MATCH"],
    }
