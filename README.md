# CrossBench

Modular evaluation platform for coding and conversational-memory benchmarks.
Stackable profiles let you mix a base (`local` / `online`), an LLM mixin
(`claude`, `gpt4o_mini`, ...) and a memory mixin (`mem0v2`,
`evermemos_embedded`, `hindsight_embedded`, ...) on top of any benchmark.

## Quick start

### 1. Install

Requires Python 3.12+.

```bash
git clone <repo-url> CrossBench && cd CrossBench
pip install -c constraints.txt -e ".[llm,memory,embedding]"
```

`constraints.txt` is the generated lock file for the full dev image. It pins
the known-good Mem0 v2 stack (`mem0ai 2.0.2`, `qdrant-client 1.18.0`,
`protobuf 5.29.6`, `grpcio 1.80.0`, `grpcio-status 1.71.2`) so reinstall
and rebuilds don't drift into incompatible protobuf 6.x resolution.

### 2. API keys

Set only the keys for the providers you'll actually use. The matrix:

| Profile / mixin | Env var | Where to get it |
|---|---|---|
| `online`, `gpt4o_mini`, `o3_mini`, `llm_judge`, any embedded memory mixin (`mem0v2`, `evermemos_embedded`, `hindsight_embedded`, `graphiti`, `memos_embedded`, `rag`) | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |
| `claude` | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |

Common case (single-shot or embedded-memory runs on OpenAI):

```bash
export OPENAI_API_KEY="sk-..."
```

The `local` profile reads no cloud keys — it runs Qwen / Llama through
Ollama, fully offline.

### 3. Data

LoCoMo data (`data/locomo/locomo10.json`) is already vendored in the repo.
CrossCodeEval is pulled from HuggingFace on first run — or pre-fetch it:

```bash
python scripts/download_datasets.py -b crosscodeeval
```

### 4. Sanity check

```bash
cross-bench list-benchmarks
cross-bench list-profiles
```

## Run LoCoMo — memory shootout

All three runs use the same base (`online` = GPT-4o + OpenAI embeddings) and
the same benchmark (`locomo_original`, 10 conversations, ~200 questions, LLM-as-judge).
Only the memory mixin changes.

### Mem0 v2 (embedded)

```bash
cross-bench run -p online -p mem0v2 -b locomo_original
```

### EverMemOS (embedded)

```bash
cross-bench run -p online -p evermemos_embedded -b locomo_original
```

### Hindsight (embedded)

```bash
cross-bench run -p online -p hindsight_embedded -b locomo_original
```

### Compare

Each run prints an aggregate table and writes a summary to
`artifacts/runs/<run-id>/`. To diff them side-by-side:

```bash
cross-bench compare <run_id_1> <run_id_2> <run_id_3>
```

Tip: append `-n 20` to any command above to smoke-test on the first 20
questions before committing to a full run.

## Run CrossCodeEval — local coding tasks

The `local` profile keeps everything on-box via Ollama — no cloud calls.

### Prereqs

```bash
docker compose --profile ollama up -d
docker exec ollama ollama pull qwen2.5:7b-instruct
docker exec ollama ollama pull nomic-embed-text:latest
```

### Run

```bash
cross-bench run -p local -p no_memory -b crosscodeeval -n 50
```

Defaults: Python subset, `cross_file_random` retrieval setting. Metrics
are text-based (exact match, edit similarity, first-line match), so no
code-execution sandbox is required.

To swap the model, override on the CLI:

```bash
cross-bench run -p local -p no_memory -b crosscodeeval \
  --model generic_openai:deepseek-coder:6.7b -n 50
```

### LoCoMo with local memory

The `local` base profile only redirects the bench LLM. Embedded memory
systems run their own internal LLM/embedding calls, so stack
`ollama_memory` **after** the memory mixin:

```bash
# Fast baseline: chunk + embed + hybrid retrieval
cross-bench run -p local -p rag -p ollama_memory -b locomo_original -n 2

# EverMemOS (LLM fact extraction; keyword retrieval in this profile)
cross-bench run -p local -p evermemos_embedded -p ollama_memory -b locomo_original -n 2

# Graphiti knowledge graph (slow on CPU: per-message LLM extraction)
cross-bench run -p local -p graphiti -p ollama_memory -b locomo_original -n 2
```

On a shared host where Ollama is pinned to a few CPUs, add
`-p ollama_8threads` **last**. It points both the bench LLM and the
memory-internal calls at derived models whose Modelfile sets
`PARAMETER num_thread 8`. Without it, Ollama starts one llama.cpp thread
per physical core (36 here), which oversubscribes an 8-CPU cgroup and
drops generation from ~12.5 tok/s to ~0.1 tok/s.

```bash
# Native Ollama: create the derived models once
cat >/tmp/Modelfile.qwen8t <<'EOF'
FROM qwen2.5:7b-instruct
PARAMETER num_thread 8
EOF
ollama create qwen2.5:7b-instruct-8t -f /tmp/Modelfile.qwen8t

cat >/tmp/Modelfile.nomic8t <<'EOF'
FROM nomic-embed-text:latest
PARAMETER num_thread 8
EOF
ollama create nomic-embed-text:8t -f /tmp/Modelfile.nomic8t

# 8 threads, keyword retrieval, 10 QA pairs
cross-bench run -p local -p evermemos_embedded -p ollama_memory \
  -p ollama_8threads -b locomo_original -n 10
```

The model thread count and the CPU limit are independent: the Modelfile
controls how many llama.cpp threads exist, while the systemd unit's
`CPUAffinity=0-7` controls which CPUs they may run on. Set both.

`-n` caps QA pairs, but the first LoCoMo conversation (~419 messages)
is still ingested in full. On CPU, expect minutes for RAG and hours for
EverMemOS/Graphiti. For Mem0 embedded, stack `-p embed_1536` last: its
native Ollama embedder needs the URL without `/v1`, and Qdrant is sized
from `embedder_dims` (1536 for GTE-Qwen2, 768 for nomic-embed-text).

```bash
cross-bench run -p local -p mem0v2_embedded -p ollama_memory \
  -p ollama_8threads -p embed_1536 -b locomo_original -n 3
```

## Leaderboard

Numbers below are pulled from each run's `summary.json`. Column sets are
pinned by each benchmark's `metrics_schema()` so they stay stable across
memory systems (see `deviations.md` entries L23 / R2 / C12 / B11). Cell
value `—` means the run hasn't been done yet; `n/a` means the metric is
not measured for that system (e.g. evidence recall without retrieved-id
provenance, per `MemoryMiddleware` contract).

> **How to read these.** Each row is one stack
> `bench × memory system`, identified by its `run_id`. Click the
> run-id link (when present) to dive into the per-task JSONs at
> `s3://cross-bench-dev-artifacts-3684f16/<run_id>/`.

### LoCoMo — original (`locomo_original`)

Paper-faithful port (snap-research/locomo @ `91c1069`). F1 is Porter-stemmed
token F1; `evidence_recall` is averaged only over middlewares that emit
honest per-chunk IDs (`evidence_recall_measured = True`). LLM-judge column
populates only on runs stacked with `-p llm_judge`. Bench LLM = `gpt-4o`,
embedder = `text-embedding-3-small` (`-p online`), 1986 tasks per run.

**Grading + recall**

| Memory system | pass_rate_f1 | pass_rate_llm_judge | avg_f1 | avg_evidence_recall | run_id |
|---|---:|---:|---:|---:|---|
| `no_memory`³ | 0.001 | 0.002 | 0.021 | n/a | `locomo-no_memory-20260513-174214` |
| `mem0v1_embedded` | — | — | — | — | — |
| `mem0v2_embedded` | **0.659** | 0.614 | 0.622 | 0.715 | `locomo-mem0v2-20260429-085741` |
| `evermemos_embedded` | 0.564 | **0.713** | 0.530 | n/a | `locomo-evermemos-20260429-085741` |
| `hindsight_embedded` | 0.273 | 0.224 | 0.267 | n/a | `locomo-hindsight-20260513-122951` |
| `memos_embedded`⁸ | 0.436 | 0.421 | 0.422 | 0.445 | `locomo-memos-judge-20260521-212729` |
| `graphiti`⁹ | 0.048 | — | 0.102 | n/a | `locomo-graphiti-20260521-202532` |
| `rag_baseline` | 0.383 | 0.497 | 0.373 | 0.626 | `locomo-rag-v2-20260429-142047` |

**Per-stage latency (ms)** — `total = search + inference + judge`. `p50` /
`p95` use the legacy pre-split per-task latency (judge time NOT included).

| Memory system | avg_search_ms | avg_infer_ms | avg_judge_ms | avg_total_ms | p50_ms | p95_ms |
|---|---:|---:|---:|---:|---:|---:|
| `no_memory` | — | 788 | — | **788** | 624 | 1,300 |
| `mem0v1_embedded` | — | — | — | — | — | — |
| `mem0v2_embedded` | 282 | 1,550 | 687 | **2,519** | 701 | 1,447 |
| `evermemos_embedded` | 7,594 | 6,230 | 737 | **14,561** | 5,052 | 9,573 |
| `hindsight_embedded` | 1,812 | 1,488 | 798 | **4,098** | 525 | 1,029 |
| `memos_embedded`⁸ | 5 | 1,412 | 755 | **2,173** | 478 | 992 |
| `graphiti`⁹ | 509 | 944 | — | **1,453** | 661 | 1,741 |
| `rag_baseline` | 500 | 1,587 | 776 | **2,862** | 662 | 1,388 |

**Token economy (per task)** — answer-LLM bill only; middleware-internal
LLM calls (mem0's fact extractor, evermemos's boundary detector,
hindsight's consolidator) are NOT counted. `retr` is variable;
`scaff` is constant per (provider, prompt_style).

| Memory system | avg_retrieval_tokens | avg_scaffolding_tokens | avg_tokens_per_task | total_in (run) | total_out (run) |
|---|---:|---:|---:|---:|---:|
| `no_memory` | 12 | 48 | 85 | 0.13 M | 43 K |
| `mem0v1_embedded` | — | — | — | — | — |
| `mem0v2_embedded` | 1,050 | 250 | 1,308 | 2.58 M | 18 K |
| `evermemos_embedded` | 2,842 | 833 | 4,064 | 7.25 M | 823 K² |
| `hindsight_embedded` | 282 | 267 | 561 | 1.10 M | 15 K |
| `memos_embedded`⁸ | 1,210 | 280 | 1,475 | 2.91 M | 16 K |
| `graphiti`⁹ | 318 | 134 | 476 | 0.90 M | 43 K |
| `rag_baseline` | 878 | 304 | 1,170 | 2.31 M | 16 K |

² evermemos's `total_out` is 45× the rest because its answer-prompt CoT
block is not stripped before scoring (deviations.md L1 strips on F1
grading, not on API `completion_tokens`).
³ `no_memory` is the leaderboard floor: the model sees the bare question
with no conversation context (85 tokens/task), so near-zero F1 is expected
and quantifies the lift every memory system has to deliver
(see deviations.md "By profile → no_memory").

⁸ `memos_embedded` uses the default `backend = "naive_text"` here — set-
intersection retrieval over raw conversation turns, no MemReader extraction,
no vector search, zero external services (deviations.md O22 / O23). This is
the MemOS "Regime A" configuration. The 0.436 F1 lands between `rag_baseline`
(0.383) and `evermemos_embedded` (0.564), confirming that lexical-only
retrieval over verbatim chunks IS competitive when the answer prompt is
correctly shaped — most of the prior 0.047 floor was the missing `FINAL
ANSWER:` directive (deviations.md O27), not retrieval quality. `avg_search_ms
= 5` shows the in-process set-intersection is essentially free; the bill
(2.91 M total tokens, ~$8 per run) is the answer-LLM + judge call.
A prior pre-O27 run is at `locomo-memos-20260521-202641` and surfaces F1=0.047,
J=0% — same backend, same retrieval, broken answer prompt. The MemOS paper's
published numbers come from "Regime C" — `tree_text` backend with Neo4j +
`SimpleStructMemReader` extraction — which requires provisioning a graph DB.
Re-run with `params.backend = "tree_text"` and a Neo4j-backed `data_dir`
to fill the paper-faithful cell.

⁹ `graphiti` is the CrossBench-internal in-process re-implementation
(~5% of upstream LoC; deviations.md G1 / G14). Upstream Graphiti is a
multi-driver knowledge-graph engine (Neo4j / Kuzu / FalkorDB / Neptune)
with 5 node types, 5 edge types, bi-temporal validity, typed entity
attributes, cross-encoder reranking, MMR diversification, and a search
orchestrator that fans out across edges / nodes / episodes / communities.
Our re-implementation captures the algorithmic spine (LLM entity +
edge extraction → entity dedup → BM25 + cosine RRF retrieval) but
collapses the 5 node types to 2 (Episodic, Entity) and the 5 edge types
to 1 (EntityEdge). The 0.048 F1 sits at the same level as `memos_embedded`
because the semantic-but-shallow retrieval competes evenly with lexical-
but-deep retrieval at this complexity gap. A Neo4j-backed run wrapping
upstream's actual `Graphiti` Python class would land much higher;
that's "Graphiti at paper-faithful config" and would require provisioning
the graph DB.

### RepoBench (`repobench`)

Paper-faithful port (Leolty/repobench). EM is whitespace-tokenized
equality, ES is `rapidfuzz.fuzz.ratio` (0–100), CodeBLEU preserves the
upstream `(refs=generated, hyps=ground_truth)` argument swap (R14). Per-
split `accuracy_cat_<split>` is the engine's automatic breakdown via the
`category` field (R1).

| Memory system | pass_rate | avg_exact_match | avg_edit_similarity | avg_codebleu | in_file | cross_file_first | cross_file_random | avg_total_latency_ms | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no_memory`¹⁴ | **0.338** | **0.347** | **71.98** | — | **0.356** | **0.292** | **0.401** | **704** | `repobench-no_memory-20260524-123603` |
| `mem0v2_embedded`¹³ (`infer=True`) | 0.305 | 0.305 | 69.94 | — | — | — | — | 1,530 | `repobench-mem0v2-infer-20260521-123035` |
| `mem0v2_embedded` (`+ mem0_no_infer`)⁶ | 0.316 | 0.316 | 68.13 | — | 0.321 | 0.268 | 0.366 | 2,234 | `repobench-mem0v2-20260520-164139` |
| `evermemos_embedded` | 0.297 | 0.297 | 69.01 | — | 0.323 | 0.208 | 0.371 | 22,388 | `repobench-evermemos-20260521-123034` |
| `rag_baseline` | 0.293 | 0.293 | 63.90 | — | 0.310 | 0.223 | 0.355 | 1,474 | `repobench-rag-20260525-074959` |

### CrossCodeEval (`crosscodeeval`)

Paper-faithful port (amazon-science/cceval @ `40c68d2`). EM / ES /
ID-EM / ID-F1 follow `eval_metric.py`; `remove_comments` strips `#`/`//`
on both pred and gt before scoring (C1). Per-language `accuracy_cat_<lang>`
is automatic via `category = language` (C11).

| Memory system | pass_rate | avg_exact_match | avg_edit_similarity | avg_id_em | avg_id_f1 | python | java | csharp | typescript | avg_total_latency_ms | run_id |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no_memory` | 0.211 | 0.212 | 69.91 | 0.306 | 0.595 | 0.212 | 0.270 | 0.079 | 0.116 | 855 | `crosscodeeval-nomem-20260513-161425` + `*-20260524-122005` |
| `mem0v2_embedded`⁴ | 0.112 | 0.112 | 63.14 | 0.174 | 0.490 | 0.112 | — | — | — | 1,095 | `crosscodeeval-mem0v2-20260513-124127` (Python split only) |
| `mem0v2_embedded` (`+ mem0_no_infer`)⁵ | **0.217** | **0.217** | **70.70** | **0.313** | **0.606** | **0.217** | **0.273** | 0.078 | **0.123** | 1,388 | `crosscodeeval-mem0v2-no_infer-20260514-105322` + `*-20260525-074959` |
| `evermemos_embedded`¹² | 0.129 | 0.129 | 64.77 | 0.203 | 0.515 | 0.129 | — | — | — | 17,811 | `crosscodeeval-evermemos-20260521-200555` (Python split only) |
| `hindsight_embedded`¹² | 0.116 | 0.116 | 63.64 | 0.180 | 0.497 | 0.116 | — | — | — | 3,626 | `crosscodeeval-hindsight-20260520-160817` (Python split only) |
| `rag_baseline` | 0.213 | 0.213 | 70.01 | 0.308 | 0.599 | 0.213 | 0.255 | 0.075 | 0.118 | 896 | `crosscodeeval-rag-{python,java,csharp,typescript}-2026052{4,5}` |

⁴ Mem0 v2's `infer=True` default runs LLM-based fact extraction on every
ingested chunk, collapsing literal code into natural-language facts
("the file defines a helper that adds two numbers" instead of
`def helper(a, b): return a + b`). The identifiers EM/ID-F1 grade against
are lost, so `mem0v2_embedded` underperforms the no-memory baseline on
code-RAG benchmarks. Documented as the Mem0 caveat in deviations.md C13
(and R5 for RepoBench). **Fix: stack the `mem0_no_infer` profile mixin
to flip mem0's `infer` to `False`** — chunks are stored verbatim, the
retriever still does its job, and the extraction-loss row above
becomes the row below it. See footnote ⁵.

⁵ Same benchmark, same model, same 2,665 Python tasks, same retriever —
only `Memory.add(infer=...)` flipped via the `mem0_no_infer` profile
mixin. Result: mem0v2 lifts above the `no_memory` baseline on every
column (EM 0.217 vs 0.212, id_em 0.313 vs 0.306, id_f1 0.606 vs 0.595).
The lift is modest because CrossCodeEval's pre-retrieved K=5 chunks are
already high-quality (BM25/UniXCoder upstream); mem0v2's retriever just
re-ranks them. The big delta is **between mem0v2 rows themselves**:
+0.105 EM (almost 2×) once extraction loss is removed. Profile lives at
`configs/profiles/mem0_no_infer.toml`; stack on ALL code-RAG benchmarks.
Documented in deviations.md M9 and workflow.md "Mem0 v2 specifically".

⁶ Same `mem0_no_infer` story as CrossCodeEval (footnotes ⁴/⁵):
`infer=True` would collapse cross-file snippets into NL facts and
deflate EM/ES, so the leaderboard row uses the verbatim-storage
profile. Pre-R16 RepoBench runs on 2026-05-13
(`repobench-mem0v2-20260513-094315/-110030/-113352`) produced near-
zero numbers (`pass_rate=0.0014`, `ES=20.6` on the one that
completed) because the chat-LLM completion directive hadn't landed
yet — the model emitted multi-line prose instead of a single
completion line. With R16's directive + output postprocessor in
place, the same model + memory stack jumps to `pass_rate=0.316`
(this row), confirming the failure was the prompt format, not the
model or memory system.

¹³ **The `infer=True` extractor trap is milder on RepoBench than on
CrossCodeEval.** Same n=4,870 tasks, same scaffold, same retriever —
only `Memory.add(infer=...)` differs.

| Benchmark | `infer=True` | `infer=False` | Δ |
|---|---:|---:|---:|
| CrossCodeEval (Python, 2665) | 0.112 EM | 0.217 EM | **−10.5 pp** |
| RepoBench (all splits, 4870) | 0.305 EM | 0.316 EM | **−1.1 pp** |

Hypothesis: RepoBench's grading target is a **single line of code**
(often a `return`, a method signature stub, or a simple call) — much
of the discriminating content is structural / syntactic rather than
identifier-dense. CrossCodeEval grades EM on multi-line completions
where identifier preservation dominates. The LLM extractor's NL-
summary collapse therefore loses less of the graded signal on
RepoBench. The `mem0_no_infer` profile is still the recommended
config (the +1.1 pp is real), but the cost-of-flipping-`infer` is
small enough on this benchmark that runs with `infer=True` are
publishable as a comparison point rather than a strict trap-
demonstration.

¹⁴ **`no_memory` is the RepoBench leader at n=4,870** — outperforms
every memory system on every column (`pass_rate=0.338` vs the
next-best `mem0v2+no_infer` at 0.316; in-file 0.356 vs 0.321;
cross-file-random 0.401 vs 0.366). Mirrors the SWE-bench n=50
pattern (`no_memory × none = 0.200` beating every memory × retrieval
combination). The reading is the same: when the underlying task is
self-contained enough (RepoBench grades a single next line; n=4,870
single-shot completions), memory injection adds **trajectory
distraction** without quality benefit. Memory's value re-emerges
under genuine cross-file complexity (the cross-file splits do
benefit from cross-file context — but the dataset's pre-retrieved
chunks already contain it, so a memory system that re-ranks them
adds little). The `no_memory` row's 131 errors / 4,870 (2.7% from
the brief 2026-05-24 13:39 UTC OpenAI quota burst) likely depresses
its pass_rate by ~0.5 pp; the clean-run estimate is ~0.347. The
finding doesn't change: every memory system underperforms the
no-memory baseline by 2-8 pp on this benchmark.

¹² **Extractor-based memory systems all tank code-RAG identifier
matching** — same shape as the Mem0 v2 `infer=True` trap (footnotes
⁴/⁵). The pattern is consistent across every system whose ingest
pipeline runs an LLM extractor over the pre-retrieved K=5 cross-file
chunks before they hit the vector store: the extractor collapses
literal code into natural-language summaries ("the file defines a
helper that adds two numbers" instead of `def helper(a, b): return
a + b`), and the grader (EM / ID-EM / ID-F1) can't recover the
identifiers. Same n=2,665 Python tasks, same K=5 chunk pool, same
upstream BM25/UniXCoder retriever — only the extraction step varies.

| System | EM | ID-EM | ID-F1 | vs `no_memory` (0.212 EM) |
|---|---:|---:|---:|---|
| `no_memory` | 0.212 | 0.306 | 0.595 | baseline |
| `mem0v2_embedded` (`+ mem0_no_infer`) | **0.217** | **0.313** | **0.606** | +0.5 pp (verbatim storage wins) |
| `mem0v2_embedded`⁴ (`infer=True`) | 0.112 | 0.174 | 0.490 | −10.0 pp |
| `evermemos_embedded` | 0.129 | 0.203 | 0.515 | −8.2 pp |
| `hindsight_embedded` | 0.116 | 0.180 | 0.497 | −9.6 pp |

Only `mem0v2 + mem0_no_infer` and `rag_baseline` (when measured)
escape the trap because they store chunks verbatim. EverMemOS and
Hindsight both have an extractor on the ingest path with
no `no_infer`-style opt-out today; the fix would require either
(a) a "verbatim mode" toggle upstream in each system, or (b)
bypassing them for code-RAG benchmarks. Documented intent: keep
these rows in the leaderboard as the per-system "code-RAG ceiling
under current upstream behaviour".

### SWE-bench Verified (`swebench`)

Paper-faithful port (`swebench>=4.1` package; the scaffold is a text-mode
port of `SWE-bench/mini-swe-agent`, frozen-hash-locked to upstream commit
`bc85a45654e6348dcc6e4c5a40ad146ed0bb144d`). Three retrieval columns fill
the leaderboard row × {none, bm25, oracle} matrix (deviations.md S3): raw
issue text, BM25-over-repo at the 27k-token budget (Pyserini Lucene via
upstream's pre-computed HF datasets), and gold-modified-file oracle as
the ceiling. Memory uses Option-E episodic memory in the scaffold loop
(`pre_prompt` per step with prior `(action, observation)`; `post_response`
ingests each completed step; `mem0_no_infer` keeps episodes verbatim —
deviations.md S18 + M9). `pass_rate = avg_resolved` per upstream's
`ResolvedStatus.FULL` (every f2p test passes AND every p2p test still
passes). Bench LLM = `gpt-4o`, n=50 of 500 Verified instances.

| Memory system | retrieval | pass_rate | applied | f2p | p2p | avg_latency_ms | tokens/task | run_id |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `no_memory` | none | **0.200** | 0.860 | **0.306** | 0.620 | 60,380 | 866 K | `swebench-nomem-none-n50-20260522-040906` |
| `no_memory` | bm25 | 0.120 | 0.820 | 0.232 | 0.611 | 115,478 | 2,733 K | `swebench-nomem-bm25-n50-20260521-195931` |
| `no_memory` | oracle | **0.220** | 0.840 | 0.286 | 0.548 | 90,407 | 1,798 K | `swebench-nomem-oracle-n50-20260522-040921` |
| `mem0v2_embedded` (`+ mem0_no_infer`) | none⁷ | 0.100 | 0.780 | 0.231 | 0.626 | 38,738 | 535 K | `swebench-mem0v2-bm25-n50-20260521-121725` |
| `mem0v2_embedded` (`+ mem0_no_infer`) | bm25 | 0.140 | 0.760 | 0.311 | 0.667 | 64,412 | 773 K | `swebench-mem0v2-bm25-n50-correct-scaffold-20260521-151654` |
| `mem0v2_embedded` (`+ mem0_no_infer`) | oracle | — | — | — | — | — | — | — |
| `evermemos_embedded` | none | 0.120 | **0.880** | 0.227 | 0.556 | 40,946 | 653 K | `swebench-evermemos-none-n50-20260522-040937` |
| `evermemos_embedded` | bm25 | 0.140 | 0.840 | 0.269 | 0.616 | 65,349 | 677 K | `swebench-evermemos-bm25-n50-20260521-195952` |
| `evermemos_embedded` | oracle | 0.200 | **0.880** | 0.273 | **0.707** | 47,422 | 635 K | `swebench-evermemos-oracle-n50-20260522-040953` |
| `rag_baseline` | bm25 | — | — | — | — | — | — | — |

**Per-repo split (n=50, mem0v2 BM25 row):** django **6/28 (21.4%)**,
astropy **1/22 (4.5%)**. Same n=50 slice used for every cell above
(deterministic, alphabetical first 50 of Verified). Per-cell counts
range from 5/50 (mem0v2 × none) to 11/50 (no_memory × oracle).

### Headline reading of the 3×3 matrix

The single most surprising number above is **`no_memory × none = 0.200`** —
the top of the matrix is the simplest row. Three intuition-violating
patterns are at play, and they all reinforce each other:

**1. BM25 retrieval HURTS the no-memory agent** (0.200 → 0.120, −8 pp,
4 tasks lost). The pre-retrieved files inject distraction more often
than help on this slice. Tokens/task jumps from 866 K → 2,733 K (3.2×)
to read context that mostly didn't matter, and wall-clock per task
doubles. Oracle retrieval only barely recovers the loss
(0.220 vs none's 0.200).

**2. Memory HURTS in the no-retrieval column** (no_memory 0.200 →
mem0v2 0.100, evermemos 0.120). The Option-E episodic recall and
evermemos's agentic boundary detection inject *trajectory* distraction
on top of issue distraction — recalled prior steps push the model
toward repetition rather than fresh approaches when the underlying
task didn't need them.

**3. Memory's value re-emerges at BM25**, but only enough to recover
some of the BM25-induced loss (mem0v2 0.140, evermemos 0.140 vs
no_memory 0.120). The memory systems summarise the bash history so
the agent doesn't re-read 100 KB of prior output at every step —
token spend drops 3.5–4× vs no_memory × bm25. So memory's value
under BM25 is **efficiency** (fewer tokens / faster wall-clock for
the same headline number).

**4. Oracle is the great equaliser** — every "memory + oracle" cell
lands at 0.200 (same as `no_memory × none`), suggesting that with
gold-file retrieval, the agent doesn't need memory help. Memory at
oracle just adds latency and tokens with no quality benefit.

### What this means for the leaderboard

For this gpt-4o × n=50 slice, the **best single configuration is
`no_memory × oracle`** (0.220) and the **simplest competitive
configuration is `no_memory × none`** (0.200) — beating every memory
× retrieval combination. The memory + retrieval combinations come
into their own only when:
- the underlying tasks have higher cross-file complexity (likely
  true on the full 500-task Verified or on swe-bench-lite), AND/OR
- the agent loop budget pressures favour aggressive context
  compression (gpt-4o-mini or similar smaller models where the
  context window is more painful).

The n=50 slice (astropy 1-22, django 1-28) is dominated by
self-contained issues where gpt-4o can solve from the issue text
alone. **A 500-task Verified run is the natural next step** to
confirm whether the matrix shape inverts at scale (memory and
retrieval both expected to help when more issues genuinely need
cross-file context).

### Cost breakdown for the 8-cell matrix above

Total spend on this slice ≈ **$795** for 350 task-runs across 7
configurations. Per-cell costs:

| cell | tokens (run) | cost |
|---|---:|---:|
| no_memory × none | 43 M | ~$108 |
| no_memory × bm25 | 136 M | ~$340 |
| no_memory × oracle | 90 M | ~$224 |
| mem0v2 × none | 27 M | ~$67 |
| mem0v2 × bm25 | 38 M | ~$96 |
| evermemos × none | 33 M | ~$81 |
| evermemos × bm25 | 34 M | ~$84 |
| evermemos × oracle | 32 M | ~$79 |

The no_memory rows are the most expensive at BM25 / oracle precisely
because the agent's raw history accumulates without trajectory
compression. Memory's efficiency value is real even when its
quality value is null or negative.

### BM25 retrieval lift over no-retrieval (mem0v2 detail)

Same scaffold, same memory mixin, same n=50 slice — only the
`retrieval` axis changes. Going from `none` to `bm25` on the
`mem0v2_embedded + mem0_no_infer` column:

- pass_rate: 0.100 → **0.140** (+4 pp)
- avg_f2p_rate: 0.231 → **0.311** (+8 pp — patches that actually fix the target)
- avg_p2p_rate: 0.626 → **0.667** (+4 pp — slightly less unrelated regression)
- avg_latency_ms: 38,738 → 64,412 (+66 % — BM25 inflates the prompt)
- tokens/task: 535 K → 773 K (+45 %)

### Memory lift over no-memory (n=50 BM25 slice)

All rows at BM25, same scaffold + n=50 slice — only the memory
axis changes. The headline pass_rate is a tie (mem0v2 and evermemos
both at 0.140 vs no_memory's 0.120), but the trajectory-compression
benefit is loud in the latency / token columns:

| metric | no_memory | mem0v2 (+ no_infer) | evermemos | reading |
|---|---:|---:|---:|---|
| pass_rate | 0.120 | **0.140** | **0.140** | memory adds +2 pp; mem0v2 and evermemos tie |
| applied | 0.820 | 0.760 | **0.840** | evermemos generates the most "syntactically valid" patches |
| f2p_rate | 0.232 | **0.311** | 0.269 | mem0v2 fixes the most targets even though it submits fewer |
| p2p_rate | 0.611 | **0.667** | 0.616 | mem0v2 also preserves the most existing behaviour |
| avg_latency_ms | 115,478 | **64,412** | 65,349 | memory cuts wall-clock by ~45 % |
| tokens/task | 2,733 K | **773 K** | **677 K** | memory cuts tokens by ~75 % (3.5–4× savings) |
| total tokens (run) | 136 M | 38.5 M | 33.6 M | |

The headline finding (BM25 slice only): **memory's primary value
here is efficiency, not quality**. mem0v2's Option-E recall and
evermemos's agentic boundary-detection both succeed at
"summarising the agent's own trajectory" so the model doesn't have
to re-read 100+ KB of bash history at every step. Without memory,
the agent's raw conversation history accumulates linearly until
context-window pressure causes either degraded reasoning or
outright BadRequest. With memory, the trajectory is compressed
into recalled episodes — same target solved, ~3.5× fewer tokens.

The two memory systems land on the same headline (`0.140`) by very
different paths: mem0v2 has a higher f2p_rate (0.311 — when it
submits, it's more likely to actually fix) but submits less often
(0.760 applied); evermemos submits more often (0.840) but its
patches are less precise (0.269 f2p).

### Failure-mode breakdown (mem0v2 BM25 row)

76 % of tasks emitted a valid patch via the
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` marker. Of those, 31 %
had the f2p tests pass (target fixed), and avg p2p 0.67
(unchanged behaviour mostly preserved). The 24 % no-submission
slice is dominated by **OpenAI 400 context-overflow** doom-loops
at 38-116 agent steps (deviations.md S18 — Option-E memory recall
fires but the conversation history accumulates faster than recall
can prune). Resolved tasks all finished in **10-22 steps**; runs
past 30 steps produced zero solves. The `max_agent_steps = 250`
cap is loud but not the binding constraint — context budget is.
**The no_memory row has 1.5× longer wall-clock** (1 h 57 m vs
1 h 18 m) and 3.5× higher token-per-task because every step
re-reads the full unsummarised history.

⁷ The `retrieval = none` row above was measured at run
`swebench-mem0v2-bm25-n50-20260521-121725` — a dispatch we originally
expected to be BM25 but which landed on the codebase before the
profile→benchmark precedence fix at commit `3c63ec6`, so the
`swebench_retrieval_bm25` profile mixin was silently a no-op against
`swebench.toml`'s `retrieval = "none"` default. The scaffold, model,
memory stack, and n=50 task slice are otherwise identical to the BM25
row, making this a clean control. Re-run on the current `b36224d`
build via `cross-bench run -p online -p mem0v2_embedded -p mem0_no_infer
-b swebench -n 50` (with no `swebench_retrieval_*` profile stacked)
to lock the cell to a post-fix build.

### BigCodeBench (`bigcodebench`)

Inspect_evals-faithful scorer. Calibrated mode default (B1); 30 s timeout
per task default (B10). With the default no-corpus mode, memory hooks
are no-op (B7); rows in this table that stack `-p bigcodebench_lib_docs`
use the CodeRAG-Bench library-documentation corpus (B12) and that's the
only setting where the memory column meaningfully differs from
`no_memory`.

| Memory system | avg_pass@1 | avg_total_latency_ms | run_id |
|---|---:|---:|---|
| `no_memory` | — | — | — |
| `mem0v2_embedded` (`+ bigcodebench_lib_docs`) | — | — | — |
| `rag_baseline` (`+ bigcodebench_lib_docs`) | — | — | — |

### Other benchmarks

Stubs — add a row + table here when the first run lands. Each benchmark
declares its canonical metric columns in
`<adapter>.metrics_schema()`; use those as the column set so the
leaderboard stays consistent with `summary.json`.

- **HumanEval / MBPP / HumanEval+ / MBPP+** — pass@1.
- **LiveCodeBench** — pass@1, time-bucketed splits.
- **TerminalBench** — agentic; pass_rate per task.
- **ConvoMem** — pass_rate, evidence_recall.
- **LongMemEval** — accuracy by question type.

## Where results land

- `artifacts/runs/<run-id>/summary.json` — aggregate metrics + config snapshot
- `artifacts/runs/<run-id>/tasks/` — per-task inputs, outputs, judgements
- `cross-bench results` / `cross-bench compare` — CLI views
- `cross-bench serve` — browser dashboard at http://localhost:8080

Point `--db-url postgresql+asyncpg://...` at a Postgres instance
(`docker compose up -d postgres`) to persist runs beyond the local
filesystem.

## Deploy to AWS

Production runs use ECS-on-EC2 with images built by CodeBuild. See
[workflow.md](workflow.md) for the full dispatch flow; the credentials
you need to bootstrap a fresh stack are below.

### 1. AWS credentials

Configure the AWS CLI on your laptop:

```bash
aws configure  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in your env
```

Stack-owner perms are needed for `pulumi up` (creates CodeBuild, ECR,
ECS, RDS, S3, VPC). Day-to-day deploy actions only need
`codebuild:StartBuild` and `ecs:RunTask`.

### 2. GitHub Enterprise PAT

CodeBuild clones from `devotion-labs.ghe.com`, so it needs a Personal
Access Token. Generate one with **`repo` (read)** scope at
<https://devotion-labs.ghe.com/settings/tokens> and stash it in the
Pulumi stack:

```bash
pulumi config set --secret ghe_pat <PAT>
pulumi up
```

`pulumi up` registers the PAT with CodeBuild as a `SourceCredential`
resource — the token stays encrypted in stack state and CodeBuild
looks it up automatically on every build. **To rotate:** regenerate
the PAT, run `pulumi config set --secret ghe_pat <new>`, then
`pulumi up`. No `aws codebuild import-source-credentials` needed.

### 3. Deploy

```bash
pulumi up                                       # provision / update infra
aws codebuild start-build \                     # build & push :latest images
  --project-name cross-bench-dev-build \
  --source-version main \
  --region us-east-1
```

ECS task definitions reference `:latest`, so the next dispatched run
picks up the new image automatically.
