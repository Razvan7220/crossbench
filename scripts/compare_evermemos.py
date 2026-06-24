"""Side-by-side comparison: CrossBench evermemos_embedded vs EverOS original.

Runs both implementations on LoCoMo and compares:
  1. Memory extraction (memcells, atomic facts, episodes)
  2. Retrieval results for sample questions
  3. Final accuracy numbers (LLM judge)

Usage:
  python scripts/compare_evermemos.py [--conv 0] [--limit 5] [--full]

  --conv N    Run on conversation N only (default: 0)
  --limit N   Limit to N questions per conversation (default: 5, 0=all)
  --full      Run all 10 conversations, all questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure cross_bench is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cross_bench.models import MemoryConfig
from cross_bench.systems.memory.evermemos_embedded import EverMemOSEmbeddedMiddleware
from cross_bench.benchmarks.llm_judge import llm_judge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare")

# Paths
LOCOMO_DATA = Path(__file__).resolve().parent.parent / "data" / "locomo" / "locomo10.json"
EVEROS_RESULTS = Path("C:/Users/ValentinRadu/code/EverOS/methods/evermemos/evaluation/results/locomo-evermemos")

CATEGORY_NAMES = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain", 5: "adversarial"}


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_locomo():
    """Load LoCoMo dataset (raw format)."""
    with open(LOCOMO_DATA) as f:
        return json.load(f)


def parse_conversation_messages(conv: dict) -> list[dict]:
    """Parse a LoCoMo conversation into flat message list."""
    import re
    raw = conv.get("conversation", {})
    if not isinstance(raw, dict):
        return []

    speaker_a = raw.get("speaker_a", "Speaker A")
    speaker_b = raw.get("speaker_b", "Speaker B")

    session_keys = sorted(
        [k for k in raw if re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    messages = []
    for s_key in session_keys:
        ts = raw.get(f"{s_key}_date_time", "")

        # Parse session timestamp → generate per-message ISO timestamps
        # at 30-second intervals (matching EverOS loaders.py default_interval=30)
        base_dt = None
        if ts:
            try:
                from dateutil import parser as _dateutil_parser
                base_dt = _dateutil_parser.parse(ts)
            except Exception:
                pass

        for msg_idx, msg in enumerate(raw[s_key]):
            if base_dt is not None:
                from datetime import timedelta as _td
                msg_dt = base_dt + _td(seconds=30 * msg_idx)
                msg_ts = msg_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                msg_ts = ts

            if isinstance(msg, dict):
                speaker = msg.get("speaker", msg.get("node_id", ""))
                content = msg.get("text", msg.get("content", ""))
                # Handle image messages (matches EverOS loaders.py)
                if msg.get("img_url"):
                    blip_caption = msg.get("blip_caption", "an image")
                    content = f"[{speaker} shared an image: {blip_caption}] {content}"
                messages.append({
                    "speaker_name": speaker,
                    "content": content,
                    "timestamp": msg_ts,
                })
    return messages


def load_everos_memcells(conv_idx: int) -> list[dict]:
    """Load EverOS cached memcell results for a conversation."""
    path = EVEROS_RESULTS / "memcells" / f"memcell_list_conv_{conv_idx}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# CrossBench middleware runner
# ─────────────────────────────────────────────────────────────────────

def create_crossbench_middleware() -> EverMemOSEmbeddedMiddleware:
    """Create CrossBench evermemos_embedded middleware with production config."""
    config = MemoryConfig(
        provider="evermemos_embedded",
        enabled=True,
        params={
            "llm_model": "gpt-4.1-mini",
            "llm_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "llm_base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            "embedding_model": "text-embedding-3-small",
            "retrieve_method": "agentic",
            "retrieval_limit": "10",
            "user_id": "bench_user",
            "hard_token_limit": "65536",
            "hard_message_limit": "500",
            "similarity_threshold": "0.65",
            "max_time_gap_seconds": "604800",
            "enable_foresight": "false",
            "enable_event_log": "true",
            "enable_clustering": "true",
        },
    )
    return EverMemOSEmbeddedMiddleware(config)


async def ingest_conversation(mw: EverMemOSEmbeddedMiddleware, messages: list[dict]) -> float:
    """Ingest all messages into the CrossBench middleware. Returns elapsed ms."""
    t0 = time.monotonic()
    for msg in messages:
        speaker = msg.get("speaker_name", "unknown")
        content = msg.get("content", "")
        # Pass raw content — speaker is stored via context["speaker_id"]
        # and _format_messages() adds "speaker:" prefix for LLM prompts
        await mw.post_response("", content, {
            "timestamp": msg.get("timestamp", ""),
            "speaker_id": speaker,
        })
    elapsed = (time.monotonic() - t0) * 1000
    return elapsed


async def retrieve_and_answer(
    mw: EverMemOSEmbeddedMiddleware,
    question: str,
    golden_answer: str,
) -> dict:
    """Search memory, generate answer, judge it."""
    # Retrieve
    t0 = time.monotonic()
    augmented = await mw.pre_prompt(question, {})
    search_ms = (time.monotonic() - t0) * 1000

    # Extract context
    if augmented != question:
        idx = augmented.find(question)
        context = augmented[:idx].strip() if idx > 0 else ""
    else:
        context = ""

    # Generate answer using the same LLM
    from cross_bench.evaluation.provider_prompts import format_answer_prompt
    answer_prompt = format_answer_prompt(
        provider="evermemos_embedded",
        question=question,
        context=context,
    )

    answer_text = await mw._llm_generate(answer_prompt, temperature=0.0)

    # Extract FINAL ANSWER from CoT (matches EverOS stage4_response.py)
    if "FINAL ANSWER:" in answer_text:
        parts = answer_text.split("FINAL ANSWER:")
        if len(parts) > 1:
            answer_text = parts[-1].strip()

    # Judge
    judge_config = {
        "model": "gpt-4.1-mini",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    }
    judgment = await llm_judge(
        question=question,
        golden_answer=str(golden_answer),
        generated_answer=answer_text,
        llm_config=judge_config,
        num_runs=3,
    )

    return {
        "question": question,
        "golden_answer": str(golden_answer),
        "generated_answer": answer_text[:500],
        "context_length": len(context),
        "search_ms": round(search_ms, 1),
        "correct": judgment["correct"],
        "raw_judgments": judgment["raw_judgments"],
    }


# ─────────────────────────────────────────────────────────────────────
# Comparison logic
# ─────────────────────────────────────────────────────────────────────

def compare_extraction(
    cb_mw: EverMemOSEmbeddedMiddleware,
    everos_memcells: list[dict],
    conv_idx: int,
):
    """Compare extraction results between CrossBench and EverOS."""
    print(f"\n{'='*70}")
    print(f" EXTRACTION COMPARISON — Conversation {conv_idx}")
    print(f"{'='*70}")

    # CrossBench stats
    cb_memcells = len(cb_mw._memcells)
    cb_episodes = len(cb_mw._episodes)
    cb_event_logs = len(cb_mw._event_logs)
    cb_foresights = len(cb_mw._foresights)
    cb_total_facts = sum(len(el.atomic_facts) for el in cb_mw._event_logs)
    cb_clusters = len(cb_mw._cluster_state.centroids)

    # EverOS stats
    eo_memcells = len(everos_memcells)
    eo_total_facts = sum(
        len(m.get("atomic_fact", {}).get("atomic_fact", []))
        for m in everos_memcells
    )
    eo_total_msgs = sum(len(m.get("original_data", [])) for m in everos_memcells)

    # EverOS clusters
    cluster_path = EVEROS_RESULTS / "memcells" / "clusters" / f"conv_{conv_idx}" / f"mem_scene_conv_{conv_idx}.json"
    eo_clusters = 0
    if cluster_path.exists():
        with open(cluster_path, encoding="utf-8") as f:
            scene = json.load(f)
            eo_clusters = len(scene.get("memscene_info", {}))

    print(f"\n{'Metric':<30} {'CrossBench':>12} {'EverOS':>12} {'Delta':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'MemCells':<30} {cb_memcells:>12} {eo_memcells:>12} {cb_memcells - eo_memcells:>+12}")
    print(f"{'Episodes':<30} {cb_episodes:>12} {'N/A':>12} {'':>12}")
    print(f"{'Event Log entries':<30} {cb_event_logs:>12} {eo_memcells:>12} {cb_event_logs - eo_memcells:>+12}")
    print(f"{'Total atomic facts':<30} {cb_total_facts:>12} {eo_total_facts:>12} {cb_total_facts - eo_total_facts:>+12}")
    print(f"{'Avg facts/memcell':<30} {cb_total_facts/max(cb_memcells,1):>12.1f} {eo_total_facts/max(eo_memcells,1):>12.1f}")
    print(f"{'Clusters (MemScenes)':<30} {cb_clusters:>12} {eo_clusters:>12} {cb_clusters - eo_clusters:>+12}")
    print(f"{'Foresights':<30} {cb_foresights:>12} {'N/A':>12}")

    # Sample fact comparison
    print(f"\n--- Sample atomic facts (first memcell) ---")
    if cb_mw._event_logs:
        print(f"\nCrossBench (first {min(5, len(cb_mw._event_logs[0].atomic_facts))} facts):")
        for f in cb_mw._event_logs[0].atomic_facts[:5]:
            print(f"  - {f}")

    if everos_memcells:
        eo_facts = everos_memcells[0].get("atomic_fact", {}).get("atomic_fact", [])
        print(f"\nEverOS (first {min(5, len(eo_facts))} facts):")
        for f in eo_facts[:5]:
            print(f"  - {f}")


def print_results_table(results: list[dict], title: str):
    """Print a summary table of QA results."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0

    print(f"\nOverall: {correct}/{total} = {accuracy:.1%}")

    # Per-category
    by_cat: dict[str, list[int]] = {}
    for r in results:
        cat = r.get("category_name", "unknown")
        by_cat.setdefault(cat, []).append(r["correct"])

    if by_cat:
        print(f"\n{'Category':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10}")
        for cat in sorted(by_cat.keys()):
            outcomes = by_cat[cat]
            c = sum(outcomes)
            t = len(outcomes)
            print(f"{cat:<20} {c:>8} {t:>8} {c/t:>10.1%}")

    # Timing
    avg_search = sum(r["search_ms"] for r in results) / total if total else 0
    avg_ctx = sum(r["context_length"] for r in results) / total if total else 0
    print(f"\nAvg search latency: {avg_search:.0f} ms")
    print(f"Avg context length: {avg_ctx:.0f} chars")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

async def run_single_conversation(conv_idx: int, limit: int | None = None):
    """Run comparison on a single conversation."""
    dataset = load_locomo()
    conv = dataset[conv_idx]

    messages = parse_conversation_messages(conv)
    qa_pairs = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]

    if limit and limit > 0:
        qa_pairs = qa_pairs[:limit]

    logger.info(f"Conv {conv_idx}: {len(messages)} messages, {len(qa_pairs)} questions")

    # Load EverOS baseline
    everos_memcells = load_everos_memcells(conv_idx)

    # Create CrossBench middleware
    mw = create_crossbench_middleware()

    # Phase 1: Ingest
    logger.info("Ingesting messages into CrossBench evermemos_embedded...")
    ingest_ms = await ingest_conversation(mw, messages)
    logger.info(f"Ingestion complete: {ingest_ms:.0f} ms")

    # Flush remaining accumulator via batch boundary detection
    if mw._accumulator:
        await mw._flush_accumulator(flush=True)

    # Phase 2: Compare extraction
    compare_extraction(mw, everos_memcells, conv_idx)

    # Phase 3: Retrieve + Answer + Judge
    logger.info(f"Running {len(qa_pairs)} questions through retrieval + answer + judge...")
    results = []
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        golden = qa["answer"]
        category = qa.get("category", 0)

        logger.info(f"  Q{i+1}/{len(qa_pairs)}: {question[:60]}...")

        result = await retrieve_and_answer(mw, question, str(golden))
        result["category"] = category
        result["category_name"] = CATEGORY_NAMES.get(category, "unknown")
        results.append(result)

        status = "CORRECT" if result["correct"] else "WRONG"
        logger.info(f"    -> {status} (search: {result['search_ms']:.0f}ms, ctx: {result['context_length']} chars)")

    print_results_table(results, f"CROSSBENCH ACCURACY — Conversation {conv_idx}")

    # Print a few detailed examples
    print(f"\n--- Sample Q&A (first 3) ---")
    for r in results[:3]:
        print(f"\nQ: {r['question']}")
        print(f"Golden: {r['golden_answer']}")
        print(f"Generated: {r['generated_answer'][:200]}...")
        print(f"Judgment: {'CORRECT' if r['correct'] else 'WRONG'} ({r['raw_judgments']})")

    return results


async def run_full(limit_per_conv: int | None = None):
    """Run on all 10 conversations."""
    dataset = load_locomo()
    all_results = []

    for conv_idx in range(len(dataset)):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Conversation {conv_idx}")
        logger.info(f"{'#'*70}")

        results = await run_single_conversation(conv_idx, limit=limit_per_conv)
        all_results.extend(results)

    print(f"\n{'='*70}")
    print(f" FINAL AGGREGATE RESULTS — All Conversations")
    print(f"{'='*70}")
    print_results_table(all_results, "CROSSBENCH OVERALL — LoCoMo (all 10 convs)")

    # Save results
    out_path = Path("artifacts/compare_evermemos_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


async def main():
    parser = argparse.ArgumentParser(description="Compare evermemos implementations")
    parser.add_argument("--conv", type=int, default=0, help="Conversation index (default: 0)")
    parser.add_argument("--limit", type=int, default=5, help="Max questions per conv (0=all)")
    parser.add_argument("--full", action="store_true", help="Run all 10 conversations")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    if args.full:
        await run_full(limit_per_conv=args.limit if args.limit > 0 else None)
    else:
        await run_single_conversation(args.conv, limit=args.limit if args.limit > 0 else None)


if __name__ == "__main__":
    asyncio.run(main())
