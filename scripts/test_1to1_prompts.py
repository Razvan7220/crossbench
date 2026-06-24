"""1:1 LLM prompt comparison: CrossBench vs EverOS.

For each pipeline stage, formats the EXACT prompt that would be sent to the LLM
using the first memcell of conversation 0, then loads the matching EverOS prompt
template and prints a diff.

This does NOT call the LLM — it only shows what WOULD be sent.

Usage:
    python scripts/test_1to1_prompts.py [--conv 0]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from cross_bench.systems.memory.evermemos_embedded import (
    ATOMIC_FACT_PROMPT,
    BATCH_BOUNDARY_DETECTION_PROMPT,
    EPISODE_GENERATION_PROMPT,
    FORESIGHT_GENERATION_PROMPT,
    MULTI_QUERY_GENERATION_PROMPT,
    SUFFICIENCY_CHECK_PROMPT,
    EverMemOSEmbeddedMiddleware,
)

# -- Paths --------------------------------------------------------------
LOCOMO = Path(__file__).resolve().parent.parent / "data" / "locomo" / "locomo10.json"
EVEROS_PROMPTS = Path("C:/Users/ValentinRadu/code/EverOS/methods/evermemos/src/memory_layer/prompts/en")
EVEROS_EVAL_PROMPTS = Path("C:/Users/ValentinRadu/code/EverOS/methods/evermemos/evaluation/src/adapters/evermemos/prompts")
EVEROS_ANSWER_PROMPTS = Path("C:/Users/ValentinRadu/code/EverOS/methods/evermemos/evaluation/src/adapters/evermemos/prompts")


def load_first_memcell_messages(conv_idx: int = 0) -> tuple[list[dict], str]:
    """Load first 20 messages of a conversation to use as sample input."""
    with open(LOCOMO) as f:
        dataset = json.load(f)
    conv = dataset[conv_idx]
    raw = conv.get("conversation", {})

    session_keys = sorted(
        [k for k in raw if re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    messages = []
    for s_key in session_keys:
        ts = raw.get(f"{s_key}_date_time", "")
        base_dt = None
        if ts:
            try:
                from dateutil import parser as _dp
                base_dt = _dp.parse(ts)
            except Exception:
                pass
        for i, msg in enumerate(raw[s_key]):
            if base_dt:
                msg_dt = base_dt + timedelta(minutes=i)
                msg_ts = msg_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                msg_ts = ts
            if isinstance(msg, dict):
                speaker = msg.get("speaker", "")
                messages.append({
                    "speaker_name": speaker,
                    "content": msg.get("text", ""),
                    "timestamp": msg_ts,
                })
        if len(messages) >= 20:
            break

    first_ts = messages[0]["timestamp"] if messages else ""
    return messages[:20], first_ts


def load_everos_prompt(filename: str, search_dirs: list[Path]) -> str | None:
    """Try to load an EverOS prompt file."""
    for d in search_dirs:
        for ext in ["", ".py"]:
            path = d / (filename + ext)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return f.read()
    return None


def extract_prompt_from_py(source: str, var_name: str) -> str | None:
    """Extract a string constant from Python source."""
    # Try triple-quoted string assignment
    patterns = [
        rf'{var_name}\s*=\s*"""(.*?)"""',
        rf"{var_name}\s*=\s*'''(.*?)'''",
        rf'{var_name}\s*=\s*"((?:[^"\\]|\\.)*)"',
    ]
    for pat in patterns:
        m = re.search(pat, source, re.DOTALL)
        if m:
            return m.group(1)
    return None


def print_diff(label: str, cb_text: str, eo_text: str):
    """Print a unified diff between two prompt texts."""
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")

    cb_lines = cb_text.strip().splitlines()
    eo_lines = eo_text.strip().splitlines()

    diff = list(difflib.unified_diff(
        eo_lines, cb_lines,
        fromfile="EverOS", tofile="CrossBench",
        lineterm="",
    ))

    if not diff:
        print("  OK IDENTICAL")
    else:
        # Count differences
        additions = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        deletions = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        print(f"  DIFF DIFFERENT: +{additions} -{deletions} lines")
        print()
        for line in diff[:80]:  # limit output
            print(f"  {line}")
        if len(diff) > 80:
            print(f"  ... ({len(diff) - 80} more diff lines)")


def print_param_comparison(label: str, cb_params: dict, eo_params: dict):
    """Print parameter comparison table."""
    print(f"\n{'-'*78}")
    print(f"  {label} — LLM Call Parameters")
    print(f"{'-'*78}")
    all_keys = sorted(set(list(cb_params.keys()) + list(eo_params.keys())))
    print(f"  {'Parameter':<25} {'CrossBench':>25} {'EverOS':>25}")
    print(f"  {'-'*25} {'-'*25} {'-'*25}")
    for k in all_keys:
        cb_v = str(cb_params.get(k, "N/A"))
        eo_v = str(eo_params.get(k, "N/A"))
        match = "OK" if cb_v == eo_v else "DIFF"
        print(f"  {k:<25} {cb_v:>25} {eo_v:>25}  {match}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, default=0)
    args = parser.parse_args()

    messages, first_ts = load_first_memcell_messages(args.conv)
    if not messages:
        print("No messages found")
        return

    # Format messages for prompts
    formatted_indexed = []
    for i, m in enumerate(messages, start=1):
        formatted_indexed.append(f"[{i}] [{m['timestamp']}] {m['speaker_name']}: {m['content']}")
    formatted_block = "\n".join(formatted_indexed)

    formatted_conv = []
    for m in messages:
        formatted_conv.append(f"[{m['timestamp']}] {m['speaker_name']}: {m['content']}")
    conv_text = "\n".join(formatted_conv)

    # Parse timestamp for extraction prompts
    try:
        from dateutil import parser as _dp
        dt = _dp.parse(first_ts)
        time_formatted = dt.strftime("%B %d, %Y(%A) at %I:%M %p")
    except Exception:
        time_formatted = first_ts

    print("\n" + "#" * 78)
    print("  1:1 LLM PROMPT COMPARISON: CrossBench vs EverOS")
    print("  Using first 20 messages of conversation", args.conv)
    print("#" * 78)

    # --- 1. Boundary Detection ----------------------------------------
    cb_boundary_prompt = BATCH_BOUNDARY_DETECTION_PROMPT.format(messages=formatted_block)

    # Load EverOS boundary prompt
    eo_src = load_everos_prompt("conv_prompts", [EVEROS_PROMPTS])
    eo_boundary_template = None
    if eo_src:
        eo_boundary_template = extract_prompt_from_py(eo_src, "CONV_BATCH_BOUNDARY_DETECTION_PROMPT")

    print_param_comparison("BOUNDARY DETECTION", {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "messages_format": "single user message",
        "retries": "5",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "messages_format": "single user message",
        "retries": "5",
    })

    if eo_boundary_template:
        print_diff("BOUNDARY DETECTION PROMPT", cb_boundary_prompt,
                   eo_boundary_template.replace("{messages}", formatted_block))
    else:
        print("\n  [Could not load EverOS boundary prompt for diff]")
        print(f"\n  CrossBench prompt preview ({len(cb_boundary_prompt)} chars):")
        print(textwrap.indent(cb_boundary_prompt[:500], "    "))

    # --- 2. Episode Extraction ----------------------------------------
    cb_episode_prompt = EPISODE_GENERATION_PROMPT.format(
        conversation_start_time=time_formatted,
        conversation=conv_text,
    )

    eo_src = load_everos_prompt("episode_mem_prompts", [EVEROS_PROMPTS])
    eo_episode_template = None
    if eo_src:
        eo_episode_template = extract_prompt_from_py(eo_src, "GROUP_EPISODE_GENERATION_PROMPT")

    print_param_comparison("EPISODE EXTRACTION", {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "retries": "3",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "retries": "5",
    })

    if eo_episode_template:
        # EverOS uses {conversation_start_time}, {conversation}, {custom_instructions}
        eo_filled = eo_episode_template.replace("{conversation_start_time}", time_formatted)
        eo_filled = eo_filled.replace("{conversation}", conv_text)
        eo_filled = eo_filled.replace("{custom_instructions}",
            "Follow these principles when generating episodic memories:\n"
            "1. Each episode should be a complete, independent story or event\n"
            "2. Preserve all important information including names, time, location, emotions, etc.\n"
            "3. Use declarative language to describe episodes, not dialogue format\n"
            "4. Highlight key information and emotional changes\n"
            "5. Ensure episode content is easy to retrieve later")
        print_diff("EPISODE EXTRACTION PROMPT", cb_episode_prompt, eo_filled)
    else:
        print("\n  [Could not load EverOS episode prompt for diff]")

    # --- 3. Atomic Fact Extraction ------------------------------------
    cb_atomic_prompt = ATOMIC_FACT_PROMPT.format(
        time=time_formatted,
        input_text=conv_text,
    )

    eo_src = load_everos_prompt("atomic_fact_prompts", [EVEROS_PROMPTS])
    eo_atomic_template = None
    if eo_src:
        eo_atomic_template = extract_prompt_from_py(eo_src, "ATOMIC_FACT_PROMPT")

    print_param_comparison("ATOMIC FACT EXTRACTION", {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "retries": "3",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.3",
        "max_tokens": "16384",
        "retries": "5",
    })

    if eo_atomic_template:
        eo_filled = eo_atomic_template.replace("{{TIME}}", time_formatted)
        eo_filled = eo_filled.replace("{{INPUT_TEXT}}", conv_text)
        print_diff("ATOMIC FACT PROMPT", cb_atomic_prompt, eo_filled)
    else:
        print("\n  [Could not load EverOS atomic fact prompt for diff]")

    # --- 4. Sufficiency Check -----------------------------------------
    sample_query = "When did Caroline go to the LGBTQ support group?"
    sample_docs = "[1] [episode] Caroline mentioned the LGBTQ support group..."

    cb_suf_prompt = SUFFICIENCY_CHECK_PROMPT.format(
        query=sample_query, retrieved_docs=sample_docs
    )

    eo_src = load_everos_prompt("sufficiency_check_prompts", [EVEROS_EVAL_PROMPTS])
    eo_suf_template = None
    if eo_src:
        eo_suf_template = extract_prompt_from_py(eo_src, "SUFFICIENCY_CHECK_PROMPT")

    print_param_comparison("SUFFICIENCY CHECK", {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "16384",
        "retries": "1",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "500",
        "retries": "1",
    })

    if eo_suf_template:
        eo_filled = eo_suf_template.replace("{query}", sample_query)
        eo_filled = eo_filled.replace("{retrieved_docs}", sample_docs)
        print_diff("SUFFICIENCY CHECK PROMPT", cb_suf_prompt, eo_filled)
    else:
        print("\n  [Could not load EverOS sufficiency prompt for diff]")

    # --- 5. Multi-Query Generation ------------------------------------
    cb_mq_prompt = MULTI_QUERY_GENERATION_PROMPT.format(
        original_query=sample_query,
        key_info="LGBTQ support group",
        missing_info="exact date",
        retrieved_docs=sample_docs,
    )

    eo_src = load_everos_prompt("multi_query_prompts", [EVEROS_EVAL_PROMPTS])
    eo_mq_template = None
    if eo_src:
        eo_mq_template = extract_prompt_from_py(eo_src, "MULTI_QUERY_GENERATION_PROMPT")

    print_param_comparison("MULTI-QUERY GENERATION", {
        "model": "gpt-4.1-mini",
        "temperature": "0.4",
        "max_tokens": "16384",
        "retries": "1",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.4",
        "max_tokens": "300",
        "retries": "1",
    })

    if eo_mq_template:
        eo_filled = eo_mq_template.replace("{original_query}", sample_query)
        eo_filled = eo_filled.replace("{key_info}", "LGBTQ support group")
        eo_filled = eo_filled.replace("{missing_info}", "exact date")
        eo_filled = eo_filled.replace("{retrieved_docs}", sample_docs)
        print_diff("MULTI-QUERY GENERATION PROMPT", cb_mq_prompt, eo_filled)
    else:
        print("\n  [Could not load EverOS multi-query prompt for diff]")

    # --- 6. Answer Generation -----------------------------------------
    from cross_bench.evaluation.provider_prompts import EVERMEMOS_ANSWER_PROMPT
    cb_answer = EVERMEMOS_ANSWER_PROMPT.format(
        context="[episodes here]", question=sample_query
    )

    eo_src = load_everos_prompt("answer_prompts", [EVEROS_ANSWER_PROMPTS])
    eo_answer_template = None
    if eo_src:
        eo_answer_template = extract_prompt_from_py(eo_src, "ANSWER_PROMPT")

    print_param_comparison("ANSWER GENERATION", {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "16384",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "16384",
    })

    if eo_answer_template:
        eo_filled = eo_answer_template.replace("{context}", "[episodes here]")
        eo_filled = eo_filled.replace("{question}", sample_query)
        print_diff("ANSWER GENERATION PROMPT", cb_answer.strip(), eo_filled)
    else:
        print("\n  [Could not load EverOS answer prompt for diff]")

    # --- 7. Judge -----------------------------------------------------
    print_param_comparison("JUDGE", {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "100",
        "num_runs": "3",
        "system_msg": "yes",
    }, {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "max_tokens": "SDK default",
        "num_runs": "3",
        "system_msg": "yes",
    })

    # --- Summary ------------------------------------------------------
    print(f"\n{'='*78}")
    print("  SUMMARY OF DIFFERENCES")
    print(f"{'='*78}")
    print("""
  Key findings:
  1. Boundary detection: EverOS prompt has CRITICAL LANGUAGE RULE headers/footers
     and examples — CrossBench prompt is stripped down.
  2. Episode extraction: EverOS has separate {custom_instructions} placeholder;
     CrossBench inlines them. EverOS retries 5x, CrossBench 3x.
  3. Atomic fact extraction: EverOS has CRITICAL LANGUAGE RULE headers/footers;
     CrossBench omits them. EverOS uses {{TIME}}/{{INPUT_TEXT}} (str.replace),
     CrossBench uses {time}/{input_text} (.format). EverOS retries 5x, CB 3x.
  4. Sufficiency check: EverOS uses max_tokens=500, CrossBench uses 16384.
     Both now use temperature=0.0.
  5. Multi-query generation: EverOS uses max_tokens=300, CrossBench uses 16384.
  6. Answer generation: Prompts should be identical. Both use temp=0.0.
  7. Judge: Both use gpt-4.1-mini, temp=0.0, 3 runs. EverOS doesn't set max_tokens.
     CrossBench sets max_tokens=100.
""")


if __name__ == "__main__":
    main()
