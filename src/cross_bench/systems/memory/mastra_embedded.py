"""Mastra memory — standalone embedded implementation (no TypeScript server).

Re-implements Mastra's multi-layered memory architecture in pure Python:

  - **Message History** — raw conversation messages stored per-thread, scoped
    to a resource (user). Last N messages retrieved chronologically.
  - **Semantic Recall** — pure vector cosine search over past messages, with
    optional message-range expansion (1 before/after each hit by default).
    Matches upstream `Memory.recall()`'s vector-only retrieval — no BM25,
    no fusion, no reranking. See deviations.md M3 / M9.
  - **Working Memory** — a persistent Markdown scratchpad. Note the
    architectural divergence (deviations.md M11): upstream wires WM into
    the agent loop as a tool-call; the `MemoryMiddleware` contract has
    no agent-tool channel, so we run an LLM call after each
    `post_response`. The WM template content is byte-faithful to upstream.
  - **Observational Memory** — observer extracts structured observations,
    reflector compresses with 5 escalating levels (0=no guidance through
    4=extreme). Default activation thresholds match upstream
    (`messageTokens=30000`, `observationTokens=40000`); for short
    benchmarks like LoCoMo the observer rarely fires, which is the
    upstream-faithful behaviour.

Audited against upstream `mastra-ai/mastra` at commit `636a66db`
(2026-05-07). Engine source lives in `packages/memory/src/index.ts`
and `packages/core/src/memory/`. See deviations.md "mastra_embedded"
section for the line-by-line audit.

Reference implementation (TypeScript):
    https://github.com/mastra-ai/mastra/tree/main/packages/memory/src

This Python port uses:
    - sentence-transformers for embeddings (local, no API)
    - OpenAI-compatible LLM for working-memory updates, observation, reflection
    - tiktoken `cl100k_base` for token counts (matches upstream)
    - In-memory storage (no database required)
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts (adapted from Mastra's TypeScript sources)
# ---------------------------------------------------------------------------

WORKING_MEMORY_UPDATE_PROMPT = """\
You are a memory manager for an AI assistant. Your job is to maintain a \
structured scratchpad of key information about the user.

Below is the current working memory (Markdown) and the latest exchange.
Return the COMPLETE updated working memory. Include ALL sections — even \
empty ones — so nothing is accidentally dropped.

Rules:
- Preserve every existing fact unless the new exchange explicitly \
contradicts or supersedes it.
- Add new facts discovered in the exchange.
- Do NOT remove information just because it was not mentioned this turn.
- Be concise: bullet points, not paragraphs.
- Return ONLY the Markdown — no preamble, no code fences.

Current working memory:
{working_memory}

Latest exchange:
User: {user_message}
Assistant: {assistant_message}

Updated working memory:"""

# Byte-exact copy of upstream `Memory.defaultWorkingMemoryTemplate`
# (`packages/memory/src/index.ts:1580-1591`). Preserves the trailing
# space after each `**:` colon — see deviations.md M10. Locked: do
# NOT trim the trailing whitespace; downstream eq-checks rely on it.
WORKING_MEMORY_TEMPLATE = (
    "\n# User Information"
    "\n- **First Name**: "
    "\n- **Last Name**: "
    "\n- **Location**: "
    "\n- **Occupation**: "
    "\n- **Interests**: "
    "\n- **Goals**: "
    "\n- **Events**: "
    "\n- **Facts**: "
    "\n- **Projects**: "
    "\n"
)

OBSERVER_SYSTEM_PROMPT = """\
You are an observation agent. Your task is to extract key observations from \
conversation messages that an AI assistant should remember about this user.

Your observations will be the ONLY information the assistant has about past \
interactions with this user.

Rules:
- Distinguish user assertions ("I live in NYC") from questions ("What's the weather in NYC?")
- Include timestamps and referenced dates when present
- Use precise verbs: "purchased", "scheduled", "completed" — not vague terms
- Assign priority:
  [HIGH] explicit facts, unresolved goals, strong preferences
  [MED]  project details, learned information, context
  [LOW]  minor details, small talk
  [DONE] completed / resolved tasks

Output format (one observation per line):
[priority] YYYY-MM-DD | observation text

Messages to observe:
{messages}

Observations:"""

_REFLECTOR_BASE_PROMPT = """\
You are a memory reflector. Your task is to take accumulated observations \
and produce reflections that preserve ALL important information.

Rules:
- Preserve dates and temporal context
- User assertions take precedence over questions
- Keep [DONE] markers as completion signals
- Never fabricate information not present in the source observations

OUTPUT FORMAT — HARD CONSTRAINT:
Your output MUST keep the per-observation line format from the input:

    [priority] YYYY-MM-DD | observation text

One observation per line. Every line starts with a bracketed
priority (`[HIGH]`, `[MED]`, `[LOW]`, `[DONE]`), followed by a
`YYYY-MM-DD` date, then ` | `, then the observation text. **Never
collapse multiple observations into a paragraph. Never drop the
date prefix. Never emit prose, bullet points, headers, or any
shape other than this one.** A reflection that loses the date
prefix is worse than no reflection at all — downstream answering
relies on per-observation dates for temporal questions.

To compress, MERGE observations:
- Two observations on the same date about the same topic → one line
  with the higher priority of the two.
- Multiple observations on consecutive dates about the same fact →
  one line, keep the EARLIEST date.
- Redundant or fully superseded observations → drop entirely.
- `[DONE]` entries → keep them as completion markers.

Compression is achieved by reducing the **count** of lines, never
by reformatting individual lines.
{guidance}
Current observations ({token_count} tokens):
{observations}

Reflections (per-line `[priority] YYYY-MM-DD | text` format only):"""

# Compression guidance per level. Mirrors upstream's
# `COMPRESSION_GUIDANCE` dict in
# `packages/memory/src/processors/observational-memory/reflector-agent.ts`
# — five levels (0-4), where level 0 emits no extra guidance (first
# attempt, "is this already short enough?") and 1-4 escalate the
# compression demand. We don't byte-copy the upstream level-1-4 text
# (it's many paragraphs each); we keep the level shape and rely on
# the model to interpret the level number plus a short instruction.
#
# Every level reiterates the OUTPUT FORMAT constraint because the
# 2026-05-11 LoCoMo run showed gpt-4o-mini dropping the per-line
# `[priority] YYYY-MM-DD | text` shape under Level 4 pressure even
# though the base prompt asked for it. The repetition is defensive
# — LLMs follow the LAST instruction they saw more reliably than
# the first.
COMPRESSION_GUIDANCE: dict[int, str] = {
    0: "",
    1: "\nCOMPRESSION REQUIRED: previous reflection was the same size or "
       "larger. Merge older observations on the same date or about the "
       "same topic into single lines. Keep the per-line "
       "`[priority] YYYY-MM-DD | text` format on EVERY surviving line.\n",
    2: "\nAGGRESSIVE COMPRESSION REQUIRED: merge more heavily across "
       "topics on the same date. Drop fully superseded observations. "
       "Every output line MUST start with `[priority] YYYY-MM-DD | `.\n",
    3: "\nCRITICAL COMPRESSION REQUIRED: keep only the most recent "
       "observations in full detail; merge each older topic into a "
       "single date-bucketed line. The per-line "
       "`[priority] YYYY-MM-DD | text` format is non-negotiable — "
       "dropping the date prefix invalidates the output.\n",
    4: "\nEXTREME COMPRESSION: keep one line per topic, anchored to "
       "the EARLIEST date that topic was first observed. Preserve "
       "[DONE] markers. Output remains one observation per line in "
       "`[priority] YYYY-MM-DD | text` format. Do NOT rewrite into "
       "paragraphs, bullets, or headers.\n",
}
MAX_COMPRESSION_LEVEL = 4

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single conversation message."""

    id: str
    role: str  # "user" | "assistant"
    text: str
    thread_id: str
    resource_id: str
    created_at: float = field(default_factory=time.time)
    # Original timestamp from the source conversation (e.g. LoCoMo's
    # per-message ``timestamp`` field). When set, observer/recent-history
    # rendering uses this verbatim instead of ``strftime(created_at)`` so
    # observations are anchored to the conversation's clock, not the
    # ingestion-wall clock. Without this, LoCoMo temporal questions
    # (cat 2) fail almost universally because observations get stamped
    # with ``datetime.now()`` and the model answers e.g. "May 9, 2026"
    # for an event that happened in 2023.
    timestamp_str: str = ""
    embedding: list[float] | None = None
    observed: bool = False  # has the observer processed this?


@dataclass
class Thread:
    """A conversation thread."""

    id: str
    resource_id: str
    messages: list[Message] = field(default_factory=list)
    working_memory: str = ""
    observations: str = ""
    observation_tokens: int = 0


# ---------------------------------------------------------------------------
# Token counter — tiktoken `cl100k_base` to match upstream
# (`packages/memory/src/processors/observational-memory/token-counter.ts`).
# Falls back to chars/4 heuristic if tiktoken can't load (offline / install
# issue). See deviations.md M30.
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODING: Any | None = None


def _count_tokens(text: str) -> int:
    """Count tokens with tiktoken `cl100k_base`. Cached encoding."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken

            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODING = False  # signal: fallback path
    if _TIKTOKEN_ENCODING is False:
        return max(1, len(text) // 4)
    return len(_TIKTOKEN_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Mastra Embedded Middleware
# ---------------------------------------------------------------------------


@register_memory("mastra_embedded")
class MastraEmbeddedMiddleware(MemoryMiddleware):
    """Mastra memory system — fully embedded, no external server.

    Implements all four memory layers:
      1. Message History — recent messages in chronological order
      2. Semantic Recall — pure vector cosine search with messageRange
         expansion (matches upstream `Memory.recall()`)
      3. Working Memory — Markdown scratchpad updated via LLM call after
         each `post_response` (architectural deviation from upstream's
         agent-tool-call paradigm — see deviations.md M11)
      4. Observational Memory — observer extracts, reflector compresses
         through 5 escalating levels (0=no guidance through 4=extreme)

    Config params (all in ``MemoryConfig.params``):
        llm_model:            LLM for working memory + observation (default gpt-4o-mini)
        llm_api_key:          API key for the LLM
        llm_base_url:         Base URL (default https://api.openai.com/v1)
        embedding_model:      Sentence-transformer model, or the model name
                              when ``embedding_provider`` is set (default
                              all-MiniLM-L6-v2)
        embedding_provider:   ``sentence_transformers`` (default),
                              ``ollama``, ``openai`` or ``openai_compatible``
        embedding_base_url:   OpenAI-compatible endpoint (``/v1``) for the
                              non-SentenceTransformer providers
        embedding_api_key:    API key for the embedding endpoint
        semantic_top_k:       Top-K for semantic recall (default 4, matches upstream)
        message_range_before: Messages before each vector hit (default 1, matches upstream)
        message_range_after:  Messages after each vector hit (default 1, matches upstream)
        last_messages:        Recent messages to include (default 10, matches upstream)
        working_memory:       Enable working memory (default True)
        observational_memory: Enable observer/reflector (default True)
        observe_threshold:    Tokens of unobserved content to trigger observer
                              (default 30000, matches upstream
                              `OBSERVATIONAL_MEMORY_DEFAULTS.observation.messageTokens`)
        reflect_threshold:    Observation token count to trigger reflector
                              (default 40000, matches upstream
                              `OBSERVATIONAL_MEMORY_DEFAULTS.reflection.observationTokens`)
        resource_id:          User/resource ID (default bench-user)
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        p = config.params or {}

        # LLM config
        self._llm_model = p.get("llm_model", "gpt-4o-mini")
        self._llm_api_key = p.get("llm_api_key", "")
        self._llm_base_url = p.get(
            "llm_base_url", "https://api.openai.com/v1"
        ).rstrip("/")

        # Embedding
        self._embedding_model_name = p.get("embedding_model", "all-MiniLM-L6-v2")
        self._embedding_provider = p.get(
            "embedding_provider", "sentence_transformers"
        )
        self._embedding_base_url = p.get(
            "embedding_base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self._embedding_api_key = (
            p.get("embedding_api_key", "") or self._llm_api_key
        )
        self._embedder = None  # lazy

        # Retrieval settings — defaults match upstream
        # `packages/memory/src/index.ts:204-205`.
        self._semantic_top_k = int(p.get("semantic_top_k", 4))
        self._message_range_before = int(p.get("message_range_before", 1))
        self._message_range_after = int(p.get("message_range_after", 1))
        self._last_messages = int(p.get("last_messages", 10))
        self._enable_working_memory = p.get("working_memory", True)
        self._enable_observations = p.get("observational_memory", True)
        # OM thresholds match upstream
        # `packages/memory/src/processors/observational-memory/constants.ts:7,26`
        # (`messageTokens: 30_000`, `observationTokens: 40_000`).
        # For short benchmarks the observer rarely fires — that's
        # upstream-faithful behaviour. Override per-benchmark if needed.
        self._observe_threshold = int(p.get("observe_threshold", 30000))
        self._reflect_threshold = int(p.get("reflect_threshold", 40000))

        # Identity
        self._resource_id = p.get("resource_id", "bench-user")

        # State
        self._threads: dict[str, Thread] = {}
        self._current_thread_id: str = ""

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_provider in {"ollama", "openai", "openai_compatible"}:
            from openai import OpenAI

            base_url = self._embedding_base_url
            if self._embedding_provider == "ollama" and not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            client = OpenAI(
                api_key=self._embedding_api_key or "ollama-local",
                base_url=base_url,
            )
            resp = client.embeddings.create(
                model=self._embedding_model_name, input=texts
            )
            return [item.embedding for item in resp.data]
        model = self._get_embedder()
        return model.encode(texts, normalize_embeddings=True).tolist()

    def _embed_one(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _ensure_thread(self) -> Thread:
        if not self._current_thread_id:
            self._current_thread_id = str(uuid.uuid4())
        if self._current_thread_id not in self._threads:
            t = Thread(
                id=self._current_thread_id,
                resource_id=self._resource_id,
                working_memory=WORKING_MEMORY_TEMPLATE,
            )
            self._threads[self._current_thread_id] = t
        return self._threads[self._current_thread_id]

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _llm_call(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Synchronous LLM call via OpenAI-compatible API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self._llm_api_key or "unused",
            base_url=self._llm_base_url,
        )
        resp = client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        raw_usage = getattr(resp, "usage", None)
        usage = None
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
            }
        self._record_internal_llm(
            prompt=system + "\n" + user, response=content, usage=usage,
        )
        return content

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ------------------------------------------------------------------
    # Semantic recall: pure vector cosine + messageRange expansion
    # (matches upstream `Memory.recall()` at
    # `packages/memory/src/index.ts:346-556`). No BM25, no fusion, no
    # reranking — upstream's recall is vector-only. See deviations.md
    # M3, M9.
    # ------------------------------------------------------------------

    def _semantic_recall(
        self, query: str, thread: Thread
    ) -> tuple[list[Message], set[str]]:
        """Vector cosine search across all messages for this resource.

        Returns ``(expanded_msgs, hit_ids)`` where ``expanded_msgs`` is
        the top-K vector hits plus their before/after neighbours within
        each message's own thread (chronological, deduped). ``hit_ids``
        is the message IDs included so callers can dedup against
        recent-history blocks.
        """
        # Gather all messages for this resource (resource-scope default,
        # matching upstream `index.ts:441-443`).
        all_msgs: list[Message] = []
        for t in self._threads.values():
            if t.resource_id == thread.resource_id:
                all_msgs.extend(t.messages)

        if not all_msgs:
            return [], set()

        # Lazy-embed any unembedded messages.
        to_embed = [m for m in all_msgs if m.embedding is None]
        if to_embed:
            texts = [m.text for m in to_embed]
            embeddings = self._embed(texts)
            for m, emb in zip(to_embed, embeddings):
                m.embedding = emb

        # Pure vector cosine ranking.
        q_emb = self._embed_one(query)
        scored: list[tuple[Message, float]] = [
            (m, self._cosine(q_emb, m.embedding))  # type: ignore[arg-type]
            for m in all_msgs
            if m.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_hits = [m for m, _ in scored[: self._semantic_top_k]]

        # MessageRange expansion: for each hit, pull surrounding messages
        # from the SAME thread (matches upstream's `include` parameter
        # with `withPreviousMessages` / `withNextMessages` at
        # `index.ts:514-528`).
        seen_ids: set[str] = set()
        expanded: list[Message] = []
        for hit in top_hits:
            for m in self._expand_message_range(
                hit, self._message_range_before, self._message_range_after
            ):
                if m.id in seen_ids:
                    continue
                seen_ids.add(m.id)
                expanded.append(m)
        expanded.sort(key=lambda m: m.created_at)
        return expanded, seen_ids

    def _expand_message_range(
        self, msg: Message, before: int, after: int
    ) -> list[Message]:
        """Return ``msg`` plus its ``before`` previous and ``after`` next
        siblings within ``msg``'s own thread (clamped to thread bounds).
        """
        thread = self._threads.get(msg.thread_id)
        if thread is None:
            return [msg]
        try:
            idx = next(
                i for i, m in enumerate(thread.messages) if m.id == msg.id
            )
        except StopIteration:
            return [msg]
        start = max(0, idx - before)
        end = min(len(thread.messages), idx + after + 1)
        return thread.messages[start:end]

    # ------------------------------------------------------------------
    # Working memory update
    # ------------------------------------------------------------------

    def _update_working_memory(
        self, thread: Thread, user_msg: str, assistant_msg: str
    ) -> None:
        """Ask the LLM to update the working memory scratchpad."""
        prompt = WORKING_MEMORY_UPDATE_PROMPT.format(
            working_memory=thread.working_memory,
            user_message=user_msg[:2000],
            assistant_message=assistant_msg[:2000],
        )
        try:
            updated = self._llm_call(
                "You are a structured memory manager.",
                prompt,
                temperature=0.0,
            )
            # Safety check: don't overwrite with empty template
            if updated.strip() and updated.strip() != WORKING_MEMORY_TEMPLATE.strip():
                thread.working_memory = updated.strip()
            elif not thread.working_memory.strip():
                thread.working_memory = updated.strip()
        except Exception as e:
            logger.warning("Working memory update failed: %s", e)

    # ------------------------------------------------------------------
    # Observational memory: observer + reflector
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Token count via tiktoken `cl100k_base` (matches upstream)."""
        return _count_tokens(text)

    def _run_observer(self, thread: Thread) -> None:
        """Extract observations from unobserved messages.

        Activates when unobserved content exceeds ``observe_threshold``
        tokens (default 30000, matches upstream
        `OBSERVATIONAL_MEMORY_DEFAULTS.observation.messageTokens`).
        """
        unobserved = [m for m in thread.messages if not m.observed]
        if not unobserved:
            return

        # Token-budget gate: only run when unobserved content reaches
        # the threshold. Upstream is async chunk-aware; we're sync.
        unobserved_tokens = sum(self._count_tokens(m.text) for m in unobserved)
        if unobserved_tokens < self._observe_threshold:
            return

        msg_text = "\n".join(
            f"[{m.role}] ({m.timestamp_str or time.strftime('%Y-%m-%d %H:%M', time.localtime(m.created_at))}) {m.text[:500]}"
            for m in unobserved
        )

        try:
            observations = self._llm_call(
                OBSERVER_SYSTEM_PROMPT.format(messages=msg_text),
                "Extract observations from the messages above.",
                temperature=0.3,
            )
            if observations.strip():
                if thread.observations:
                    thread.observations += "\n" + observations.strip()
                else:
                    thread.observations = observations.strip()
                thread.observation_tokens = self._count_tokens(
                    thread.observations
                )
                # Mark as observed
                for m in unobserved:
                    m.observed = True
        except Exception as e:
            logger.warning("Observer failed: %s", e)

    def _run_reflector(self, thread: Thread) -> None:
        """Compress observations when they exceed the reflection threshold.

        Iterates upstream's 5 compression levels (0-4): level 0 emits
        no extra guidance (first attempt — "is this already short
        enough?"); levels 1-4 escalate the compression demand. Accepts
        the first reflection that is strictly smaller than the source.
        """
        if thread.observation_tokens < self._reflect_threshold:
            return

        original_tokens = thread.observation_tokens

        for level in range(0, MAX_COMPRESSION_LEVEL + 1):
            try:
                prompt = _REFLECTOR_BASE_PROMPT.format(
                    guidance=COMPRESSION_GUIDANCE.get(level, ""),
                    token_count=thread.observation_tokens,
                    observations=thread.observations,
                )
                compressed = self._llm_call(
                    prompt,
                    "Produce reflections from the observations above.",
                    temperature=0.0,
                )
                if not compressed.strip():
                    continue
                new_tokens = self._count_tokens(compressed)
                # Accept the first strictly-smaller result. Mirrors
                # upstream's level-1 retry trigger ("previous reflection
                # was the same size or larger") at
                # `reflector-agent.ts:154-170`.
                if new_tokens < original_tokens:
                    thread.observations = compressed.strip()
                    thread.observation_tokens = new_tokens
                    logger.debug(
                        "Reflector compressed observations: %d -> %d tokens (level %d)",
                        original_tokens,
                        new_tokens,
                        level,
                    )
                    return
            except Exception as e:
                logger.warning("Reflector failed at level %d: %s", level, e)

    # ------------------------------------------------------------------
    # MemoryMiddleware interface
    # ------------------------------------------------------------------

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Retrieve memories from all four layers and prepend to prompt.

        Layout matches upstream's `Memory.recall()` + `getContext()`
        composition (`packages/memory/src/index.ts:346-556, 1350-...`):
        observations + working memory go into a system-message-like
        block, then a SINGLE chronological message list is built by
        merging semantic-recall hits (with messageRange expansion) and
        recent-history, deduped by message id. We can't send a
        separate `messages` array to the LLM through the
        `MemoryMiddleware` contract, so the merged list is rendered as
        text.
        """
        thread = self._ensure_thread()
        parts: list[str] = []

        # 1. Observational memory (if enabled and available)
        if self._enable_observations and thread.observations:
            parts.append(f"[Observations]\n{thread.observations[:3000]}")

        # 2. Working memory (if enabled and non-empty)
        if self._enable_working_memory and thread.working_memory:
            wm = thread.working_memory.strip()
            if wm and wm != WORKING_MEMORY_TEMPLATE.strip():
                parts.append(f"[Working Memory]\n{wm}")

        # 3+4. Single chronological message list — semantic recall hits
        # (with their before/after neighbours) merged with the most
        # recent N messages, deduped by id. Mirrors upstream's
        # `listMessages({ include: [...] })` call shape at
        # `index.ts:507-530`.
        if thread.messages:
            sem_msgs: list[Message] = []
            sem_ids: set[str] = set()
            try:
                sem_msgs, sem_ids = self._semantic_recall(prompt, thread)
            except Exception as e:
                logger.debug("Semantic recall failed: %s", e)

            recent: list[Message] = []
            if self._last_messages:
                recent = thread.messages[-self._last_messages :]

            # Merge + dedup + chronological sort.
            merged: dict[str, Message] = {}
            for m in sem_msgs:
                merged[m.id] = m
            for m in recent:
                if m.id not in merged:
                    merged[m.id] = m
            ordered = sorted(merged.values(), key=lambda m: m.created_at)

            if ordered:
                lines = [
                    (
                        f"[{m.timestamp_str}] {m.role}: {m.text[:500]}"
                        if m.timestamp_str
                        else f"{m.role}: {m.text[:500]}"
                    )
                    for m in ordered
                ]
                # Use the upstream's [Semantic Recall] header tag when
                # any vector hits made it in; otherwise [Recent
                # History]. This is informational only — both paths
                # produce the same merged list.
                header = (
                    "[Semantic Recall + Recent History]"
                    if sem_ids
                    else "[Recent History]"
                )
                parts.append(f"{header}\n" + "\n".join(lines))

        if parts:
            memory_text = "\n\n---\n\n".join(parts)
            return f"Retrieved memories:\n{memory_text}\n\n{prompt}"
        return prompt

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Store messages and update working memory + observations."""
        thread = self._ensure_thread()

        # Capture the source-conversation timestamp from the engine-supplied
        # context (e.g. LoCoMo's per-message ``timestamp``). Passed through
        # verbatim — the observer/recent-history rendering uses whatever
        # format upstream provides, without trying to parse it.
        ts_str = str(context.get("timestamp") or "")

        # Store user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            role="user",
            text=prompt[:4000],
            thread_id=thread.id,
            resource_id=thread.resource_id,
            timestamp_str=ts_str,
        )
        thread.messages.append(user_msg)

        # Store assistant message
        asst_msg = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            text=response[:4000],
            thread_id=thread.id,
            resource_id=thread.resource_id,
            timestamp_str=ts_str,
        )
        thread.messages.append(asst_msg)

        # Update working memory (LLM call)
        if self._enable_working_memory:
            try:
                self._update_working_memory(thread, prompt[:2000], response[:2000])
            except Exception as e:
                logger.debug("Working memory update error: %s", e)

        # Run observer if enough unobserved content
        if self._enable_observations:
            try:
                self._run_observer(thread)
                self._run_reflector(thread)
            except Exception as e:
                logger.debug("Observation pipeline error: %s", e)

    async def reset(self) -> None:
        """Clear all threads and state for a fresh session."""
        self._threads.clear()
        self._current_thread_id = ""
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_context(self, parts: list[str]) -> str:
        return "\n\n---\n\n".join(parts)
