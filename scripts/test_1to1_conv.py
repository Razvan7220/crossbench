"""1:1 pipeline comparison: CrossBench vs EverOS for a single LoCoMo conversation.

Runs ONE conversation through CrossBench evermemos_embedded, dumps every
intermediate artefact (messages, boundaries, memcells, episodes, atomic facts,
clusters, BM25 tokens, retrieval results), then loads the matching EverOS
cached artefacts and prints a side-by-side diff for each pipeline stage.

Usage:
    python scripts/test_1to1_conv.py [--conv 0] [--questions 3]

Outputs:  artifacts/1to1_conv{N}/  with JSON dumps for every stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure cross_bench is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from cross_bench.models import MemoryConfig
from cross_bench.systems.memory.evermemos_embedded import EverMemOSEmbeddedMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_1to1")

# ─── Paths ────────────────────────────────────────────────────────────
LOCOMO_DATA = Path(__file__).resolve().parent.parent / "data" / "locomo" / "locomo10.json"
EVEROS_RESULTS = Path("C:/Users/ValentinRadu/code/EverOS/methods/evermemos/evaluation/results/locomo-evermemos")

CATEGORY_NAMES = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain", 5: "adversarial"}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_locomo():
    with open(LOCOMO_DATA) as f:
        return json.load(f)


def parse_conversation_messages(conv: dict) -> list[dict]:
    """Parse raw LoCoMo → flat message list with per-message ISO timestamps."""
    raw = conv.get("conversation", {})
    if not isinstance(raw, dict):
        return []

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

        for msg_idx, msg in enumerate(raw[s_key]):
            if base_dt is not None:
                msg_dt = base_dt + timedelta(seconds=30 * msg_idx)
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
    path = EVEROS_RESULTS / "memcells" / f"memcell_list_conv_{conv_idx}.json"
    if not path.exists():
        logger.warning("EverOS memcells not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_everos_clusters(conv_idx: int) -> dict:
    path = EVEROS_RESULTS / "memcells" / "clusters" / f"conv_{conv_idx}" / f"mem_scene_conv_{conv_idx}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# CROSSBENCH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════

def create_middleware() -> EverMemOSEmbeddedMiddleware:
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


# ═══════════════════════════════════════════════════════════════════════
# STAGE DUMPERS — capture CrossBench intermediate state
# ═══════════════════════════════════════════════════════════════════════

def dump_input_messages(messages: list[dict], out_dir: Path):
    """Stage 0: Dump the input messages fed to CrossBench."""
    path = out_dir / "stage0_input_messages.json"
    with open(path, "w") as f:
        json.dump(messages, f, indent=2, default=str)
    logger.info("  Dumped %d input messages → %s", len(messages), path.name)
    return messages


def dump_memcells(mw: EverMemOSEmbeddedMiddleware, out_dir: Path) -> list[dict]:
    """Stage 1a: Dump memcells with message counts and boundaries."""
    memcells = []
    for i, mc in enumerate(mw._memcells):
        memcells.append({
            "index": i,
            "num_messages": len(mc.messages),
            "first_msg": mc.messages[0].get("content", "")[:100] if mc.messages else "",
            "last_msg": mc.messages[-1].get("content", "")[:100] if mc.messages else "",
            "first_ts": mc.messages[0].get("timestamp", "") if mc.messages else "",
            "last_ts": mc.messages[-1].get("timestamp", "") if mc.messages else "",
            "first_speaker": mc.messages[0].get("speaker_name", "") if mc.messages else "",
            "messages": [
                {
                    "speaker": m.get("speaker_name", ""),
                    "content": m.get("content", "")[:200],
                    "timestamp": m.get("timestamp", ""),
                }
                for m in mc.messages
            ],
        })

    path = out_dir / "stage1a_memcells.json"
    with open(path, "w") as f:
        json.dump(memcells, f, indent=2, default=str)
    logger.info("  Dumped %d memcells → %s", len(memcells), path.name)
    return memcells


def dump_episodes(mw: EverMemOSEmbeddedMiddleware, out_dir: Path) -> list[dict]:
    """Stage 1b: Dump episode memories."""
    episodes = []
    for i, ep in enumerate(mw._episodes):
        episodes.append({
            "index": i,
            "title": ep.title,
            "content": ep.content,
            "content_length": len(ep.content),
            "has_embedding": ep.embedding is not None and len(ep.embedding) > 0,
            "embedding_dim": len(ep.embedding) if ep.embedding else 0,
            "timestamp": ep.timestamp,
            "memcell_id": ep.memcell_id,
        })

    path = out_dir / "stage1b_episodes.json"
    with open(path, "w") as f:
        json.dump(episodes, f, indent=2, default=str)
    logger.info("  Dumped %d episodes → %s", len(episodes), path.name)
    return episodes


def dump_atomic_facts(mw: EverMemOSEmbeddedMiddleware, out_dir: Path) -> list[dict]:
    """Stage 1c: Dump atomic facts (event log entries)."""
    event_logs = []
    for i, el in enumerate(mw._event_logs):
        event_logs.append({
            "index": i,
            "num_facts": len(el.atomic_facts),
            "facts": el.atomic_facts,
            "has_embeddings": len(el.fact_embeddings) > 0,
            "num_embeddings": len(el.fact_embeddings),
            "embedding_dim": len(el.fact_embeddings[0]) if el.fact_embeddings else 0,
            "timestamp": el.timestamp,
            "memcell_id": el.memcell_id,
        })

    path = out_dir / "stage1c_atomic_facts.json"
    with open(path, "w") as f:
        json.dump(event_logs, f, indent=2, default=str)
    logger.info("  Dumped %d event log entries, %d total facts → %s",
                len(event_logs), sum(e["num_facts"] for e in event_logs), path.name)
    return event_logs


def dump_clusters(mw: EverMemOSEmbeddedMiddleware, out_dir: Path) -> dict:
    """Stage 1d: Dump cluster state."""
    cs = mw._cluster_state
    cluster_info = {
        "num_clusters": len(cs.centroids),
        "cluster_names": list(cs.centroids.keys()),
        "cluster_sizes": {k: cs.counts.get(k, 0) for k in cs.centroids},
        "memcell_to_cluster": dict(cs.memcell_to_cluster),
        "last_timestamps": {k: str(v) for k, v in cs.last_ts.items()},
    }

    path = out_dir / "stage1d_clusters.json"
    with open(path, "w") as f:
        json.dump(cluster_info, f, indent=2, default=str)
    logger.info("  Dumped %d clusters → %s", cluster_info["num_clusters"], path.name)
    return cluster_info


def dump_bm25_index(mw: EverMemOSEmbeddedMiddleware, out_dir: Path) -> dict:
    """Stage 2: Dump BM25 index stats and sample tokens."""
    info = {
        "num_docs": len(mw._bm25_docs),
        "has_index": mw._bm25_index is not None,
        "sample_docs": [d[:200] for d in mw._bm25_docs[:5]],
    }
    # Get tokenized forms if available
    if mw._bm25_index:
        idx = mw._bm25_index
        if hasattr(idx, '_corpus') and idx._corpus:
            info["sample_tokenized"] = [list(doc)[:20] for doc in idx._corpus[:3]]
        info["avg_doc_len"] = getattr(idx, '_avgdl', None)

    path = out_dir / "stage2_bm25_index.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("  Dumped BM25 index (%d docs) → %s", info["num_docs"], path.name)
    return info


async def dump_retrieval(
    mw: EverMemOSEmbeddedMiddleware,
    question: str,
    out_dir: Path,
    q_idx: int,
) -> dict:
    """Stage 3: Dump retrieval results for one question."""
    t0 = time.monotonic()
    augmented = await mw.pre_prompt(question, {})
    search_ms = (time.monotonic() - t0) * 1000

    # Extract context
    if augmented != question:
        idx = augmented.find(question)
        context = augmented[:idx].strip() if idx > 0 else ""
    else:
        context = ""

    result = {
        "question": question,
        "search_ms": round(search_ms, 1),
        "context_length": len(context),
        "context": context,
        "augmented_prompt_length": len(augmented),
    }

    path = out_dir / f"stage3_retrieval_q{q_idx}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("  Dumped retrieval for Q%d → %s (ctx=%d chars, %dms)",
                q_idx, path.name, len(context), round(search_ms))
    return result


async def dump_answer_and_judge(
    mw: EverMemOSEmbeddedMiddleware,
    question: str,
    golden_answer: str,
    context: str,
    out_dir: Path,
    q_idx: int,
) -> dict:
    """Stage 4+5: Generate answer and judge it."""
    from cross_bench.evaluation.provider_prompts import format_answer_prompt
    from cross_bench.benchmarks.llm_judge import llm_judge

    # Stage 4: Answer generation
    answer_prompt = format_answer_prompt(
        provider="evermemos_embedded",
        question=question,
        context=context,
    )

    t0 = time.monotonic()
    # Match EverOS: temperature=0.0 for answer generation
    answer_text = await mw._llm_generate(answer_prompt, temperature=0.0)
    answer_ms = (time.monotonic() - t0) * 1000

    # Extract FINAL ANSWER from CoT (matches EverOS stage4_response.py)
    if "FINAL ANSWER:" in answer_text:
        parts = answer_text.split("FINAL ANSWER:")
        if len(parts) > 1:
            answer_text = parts[-1].strip()

    # Stage 5: Judge (use gpt-4.1-mini to match EverOS)
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

    result = {
        "question": question,
        "golden_answer": str(golden_answer),
        "answer_prompt": answer_prompt[:500] + "..." if len(answer_prompt) > 500 else answer_prompt,
        "answer_prompt_length": len(answer_prompt),
        "generated_answer": answer_text,
        "answer_ms": round(answer_ms, 1),
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "correct": judgment["correct"],
        "raw_judgments": judgment["raw_judgments"],
        "judge_model": "gpt-4.1-mini",
    }

    path = out_dir / f"stage4_5_answer_judge_q{q_idx}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    status = "CORRECT" if result["correct"] else "WRONG"
    logger.info("  Q%d: %s (answer=%dms, judge=%s)", q_idx, status,
                round(answer_ms), judgment["raw_judgments"])
    return result


# ═══════════════════════════════════════════════════════════════════════
# EVEROS ARTEFACT LOADING — for side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════

def load_everos_artefacts(conv_idx: int) -> dict:
    """Load all EverOS cached artefacts for comparison."""
    artefacts = {}

    # Memcells
    memcells = load_everos_memcells(conv_idx)
    artefacts["memcells"] = {
        "count": len(memcells),
        "message_counts": [len(m.get("original_data", [])) for m in memcells],
        "total_messages": sum(len(m.get("original_data", [])) for m in memcells),
    }

    # Episodes (embedded in memcells)
    artefacts["episodes"] = {
        "count": sum(1 for m in memcells if m.get("episode")),
        "titles": [m.get("subject", "N/A") for m in memcells if m.get("episode")],
        "avg_length": (
            sum(len(m.get("episode", "")) for m in memcells) / max(len(memcells), 1)
        ),
    }

    # Atomic facts
    fact_counts = []
    all_facts = []
    for m in memcells:
        af = m.get("atomic_fact", {})
        facts = af.get("atomic_fact", []) if isinstance(af, dict) else []
        # Facts can be strings or dicts
        fact_strings = []
        for f in facts:
            if isinstance(f, str):
                fact_strings.append(f)
            elif isinstance(f, dict):
                fact_strings.append(f.get("fact", f.get("content", str(f))))
        fact_counts.append(len(fact_strings))
        all_facts.extend(fact_strings)

    artefacts["atomic_facts"] = {
        "total": len(all_facts),
        "per_memcell": fact_counts,
        "avg_per_memcell": len(all_facts) / max(len(memcells), 1),
        "sample_facts": all_facts[:10],
    }

    # Clusters
    clusters = load_everos_clusters(conv_idx)
    scene_info = clusters.get("memscene_info", {})
    artefacts["clusters"] = {
        "count": len(scene_info),
        "names": list(scene_info.keys()),
        "sizes": {k: len(v.get("event_ids", [])) for k, v in scene_info.items()},
    }

    return artefacts


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON PRINTER
# ═══════════════════════════════════════════════════════════════════════

def print_comparison(cb: dict, eo: dict, out_dir: Path):
    """Print side-by-side comparison of all stages."""
    sep = "=" * 78

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"\n{sep}")
    p("  1:1 PIPELINE COMPARISON")
    p(sep)

    # Stage 1a: MemCells
    p(f"\n{'─'*78}")
    p("  STAGE 1a: MEMCELL EXTRACTION (Boundary Detection)")
    p(f"{'─'*78}")
    p(f"  {'Metric':<35} {'CrossBench':>18} {'EverOS':>18}")
    p(f"  {'-'*35} {'-'*18} {'-'*18}")
    p(f"  {'MemCell count':<35} {cb['memcells']['count']:>18} {eo['memcells']['count']:>18}")
    p(f"  {'Total messages covered':<35} {cb['memcells']['total_messages']:>18} {eo['memcells']['total_messages']:>18}")
    p(f"  {'Avg msgs/memcell':<35} {cb['memcells']['avg_msgs']:>18.1f} {eo['memcells']['total_messages']/max(eo['memcells']['count'],1):>18.1f}")

    # Per-memcell message counts
    cb_mc = cb["memcells"]["message_counts"]
    eo_mc = eo["memcells"]["message_counts"]
    max_show = max(len(cb_mc), len(eo_mc))
    p(f"\n  Per-memcell message counts (CB vs EO):")
    for i in range(min(max_show, 30)):
        cb_v = str(cb_mc[i]) if i < len(cb_mc) else "-"
        eo_v = str(eo_mc[i]) if i < len(eo_mc) else "-"
        match = "OK" if cb_v == eo_v else "DIFF"
        p(f"    MC{i:02d}: CB={cb_v:>4}  EO={eo_v:>4}  {match}")

    # Stage 1b: Episodes
    p(f"\n{'─'*78}")
    p("  STAGE 1b: EPISODE EXTRACTION")
    p(f"{'─'*78}")
    p(f"  {'Metric':<35} {'CrossBench':>18} {'EverOS':>18}")
    p(f"  {'-'*35} {'-'*18} {'-'*18}")
    p(f"  {'Episode count':<35} {cb['episodes']['count']:>18} {eo['episodes']['count']:>18}")
    p(f"  {'Avg content length':<35} {cb['episodes']['avg_length']:>18.0f} {eo['episodes']['avg_length']:>18.0f}")

    p(f"\n  Episode titles comparison:")
    for i in range(min(len(cb["episodes"]["titles"]), len(eo["episodes"]["titles"]), 10)):
        p(f"    [{i}] CB: {cb['episodes']['titles'][i][:60]}")
        p(f"        EO: {eo['episodes']['titles'][i][:60]}")

    # Stage 1c: Atomic Facts
    p(f"\n{'─'*78}")
    p("  STAGE 1c: ATOMIC FACT EXTRACTION")
    p(f"{'─'*78}")
    p(f"  {'Metric':<35} {'CrossBench':>18} {'EverOS':>18}")
    p(f"  {'-'*35} {'-'*18} {'-'*18}")
    p(f"  {'Total facts':<35} {cb['atomic_facts']['total']:>18} {eo['atomic_facts']['total']:>18}")
    p(f"  {'Avg facts/memcell':<35} {cb['atomic_facts']['avg_per_memcell']:>18.1f} {eo['atomic_facts']['avg_per_memcell']:>18.1f}")

    p(f"\n  Per-memcell fact counts:")
    cb_fc = cb["atomic_facts"]["per_memcell"]
    eo_fc = eo["atomic_facts"]["per_memcell"]
    for i in range(min(max(len(cb_fc), len(eo_fc)), 30)):
        cb_v = str(cb_fc[i]) if i < len(cb_fc) else "-"
        eo_v = str(eo_fc[i]) if i < len(eo_fc) else "-"
        p(f"    MC{i:02d}: CB={cb_v:>4}  EO={eo_v:>4}")

    p(f"\n  Sample facts comparison (first memcell):")
    cb_sf = cb["atomic_facts"]["sample_facts"][:5]
    eo_sf = eo["atomic_facts"]["sample_facts"][:5]
    for i in range(max(len(cb_sf), len(eo_sf))):
        if i < len(cb_sf):
            p(f"    CB[{i}]: {cb_sf[i][:80]}")
        if i < len(eo_sf):
            p(f"    EO[{i}]: {eo_sf[i][:80]}")
        p()

    # Stage 1d: Clusters
    p(f"\n{'─'*78}")
    p("  STAGE 1d: CLUSTERING (MemScenes)")
    p(f"{'─'*78}")
    p(f"  {'Metric':<35} {'CrossBench':>18} {'EverOS':>18}")
    p(f"  {'-'*35} {'-'*18} {'-'*18}")
    p(f"  {'Cluster count':<35} {cb['clusters']['count']:>18} {eo['clusters']['count']:>18}")

    p(f"\n  Cluster sizes:")
    cb_cs = cb["clusters"]["sizes"]
    eo_cs = eo["clusters"]["sizes"]
    all_keys = sorted(set(list(cb_cs.keys()) + list(eo_cs.keys())))
    for k in all_keys[:15]:
        p(f"    {k}: CB={cb_cs.get(k, '-'):>4}  EO={eo_cs.get(k, '-'):>4}")

    # Stage 2: BM25 Index
    p(f"\n{'─'*78}")
    p("  STAGE 2: BM25 INDEX")
    p(f"{'─'*78}")
    p(f"  {'Metric':<35} {'CrossBench':>18}")
    p(f"  {'-'*35} {'-'*18}")
    p(f"  {'Indexed docs':<35} {cb['bm25']['num_docs']:>18}")
    p(f"  {'Has index':<35} {str(cb['bm25']['has_index']):>18}")

    # Stage 3+4+5: Retrieval & QA
    if cb.get("qa_results"):
        p(f"\n{'─'*78}")
        p("  STAGE 3+4+5: RETRIEVAL → ANSWER → JUDGE")
        p(f"{'─'*78}")
        total = len(cb["qa_results"])
        correct = sum(1 for r in cb["qa_results"] if r["correct"])
        p(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
        p()
        for r in cb["qa_results"]:
            status = "CORRECT" if r["correct"] else "WRONG  "
            p(f"  [{status}] Q: {r['question'][:65]}")
            p(f"           Golden: {r['golden_answer'][:65]}")
            p(f"           Answer: {r['generated_answer'][:65]}...")
            p(f"           Context: {r['context_length']} chars | Search: {r['search_ms']}ms | Judge: {r['raw_judgments']}")
            p()

    p(sep)

    # Save full comparison
    path = out_dir / "comparison_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Full comparison report → %s", path)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="1:1 pipeline comparison")
    parser.add_argument("--conv", type=int, default=0, help="Conversation index")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions to test")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    conv_idx = args.conv
    out_dir = Path(f"artifacts/1to1_conv{conv_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running 1:1 comparison for conversation %d", conv_idx)
    logger.info("Output directory: %s", out_dir)

    # ── Load dataset ──────────────────────────────────────────────────
    dataset = load_locomo()
    conv = dataset[conv_idx]
    messages = parse_conversation_messages(conv)
    qa_pairs = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
    qa_pairs = qa_pairs[:args.questions]

    logger.info("Conversation %d: %d messages, %d questions", conv_idx, len(messages), len(qa_pairs))

    # ── Stage 0: Dump input ───────────────────────────────────────────
    logger.info("Stage 0: Input messages")
    dump_input_messages(messages, out_dir)

    # ── Create middleware ─────────────────────────────────────────────
    mw = create_middleware()

    # ── Stage 1: Ingest ───────────────────────────────────────────────
    logger.info("Stage 1: Ingesting messages...")
    t0 = time.monotonic()
    for msg in messages:
        speaker = msg.get("speaker_name", "unknown")
        content = msg.get("content", "")
        await mw.post_response("", content, {
            "timestamp": msg.get("timestamp", ""),
            "speaker_id": speaker,
        })
    ingest_ms = (time.monotonic() - t0) * 1000
    logger.info("  Accumulation: %.0fms", ingest_ms)

    # Flush
    logger.info("  Flushing accumulator (boundary detection + extraction)...")
    t0 = time.monotonic()
    if mw._accumulator:
        await mw._flush_accumulator(flush=True)
    flush_ms = (time.monotonic() - t0) * 1000
    logger.info("  Flush: %.0fms", flush_ms)

    # ── Dump CrossBench artefacts ─────────────────────────────────────
    logger.info("Dumping CrossBench artefacts...")
    cb_memcells = dump_memcells(mw, out_dir)
    cb_episodes = dump_episodes(mw, out_dir)
    cb_facts = dump_atomic_facts(mw, out_dir)
    cb_clusters = dump_clusters(mw, out_dir)
    cb_bm25 = dump_bm25_index(mw, out_dir)

    # ── Stage 3+4+5: Retrieval + Answer + Judge ───────────────────────
    logger.info("Stage 3+4+5: Retrieval → Answer → Judge")
    qa_results = []
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        golden = qa["answer"]

        logger.info("  Q%d: %s", i, question[:60])
        retrieval = await dump_retrieval(mw, question, out_dir, i)
        result = await dump_answer_and_judge(
            mw, question, str(golden), retrieval["context"], out_dir, i
        )
        result["search_ms"] = retrieval["search_ms"]
        result["context_length"] = retrieval["context_length"]
        qa_results.append(result)

    # ── Load EverOS artefacts ─────────────────────────────────────────
    logger.info("Loading EverOS cached artefacts...")
    eo = load_everos_artefacts(conv_idx)

    # ── Build CrossBench summary ──────────────────────────────────────
    cb = {
        "memcells": {
            "count": len(cb_memcells),
            "message_counts": [m["num_messages"] for m in cb_memcells],
            "total_messages": sum(m["num_messages"] for m in cb_memcells),
            "avg_msgs": sum(m["num_messages"] for m in cb_memcells) / max(len(cb_memcells), 1),
        },
        "episodes": {
            "count": len(cb_episodes),
            "titles": [e["title"] for e in cb_episodes],
            "avg_length": sum(e["content_length"] for e in cb_episodes) / max(len(cb_episodes), 1),
        },
        "atomic_facts": {
            "total": sum(e["num_facts"] for e in cb_facts),
            "per_memcell": [e["num_facts"] for e in cb_facts],
            "avg_per_memcell": sum(e["num_facts"] for e in cb_facts) / max(len(cb_facts), 1),
            "sample_facts": cb_facts[0]["facts"][:5] if cb_facts else [],
        },
        "clusters": {
            "count": cb_clusters["num_clusters"],
            "sizes": cb_clusters["cluster_sizes"],
        },
        "bm25": cb_bm25,
        "qa_results": qa_results,
    }

    # ── Print comparison ──────────────────────────────────────────────
    print_comparison(cb, eo, out_dir)

    # Save full dump
    summary_path = out_dir / "full_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"crossbench": cb, "everos": eo}, f, indent=2, default=str)
    logger.info("Full summary → %s", summary_path)


if __name__ == "__main__":
    asyncio.run(main())
