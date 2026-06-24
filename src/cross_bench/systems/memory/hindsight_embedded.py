"""Hindsight Embedded — standalone implementation of the Hindsight memory system.

Implements the full Hindsight memory lifecycle without PostgreSQL or any external
server.  The three core operations mirror human cognition:

  1. **Retain** — ingest text, use LLM to extract atomic facts with five
     dimensions (what, when, where, who, why), resolve entities, build a
     multi-link graph (temporal, semantic, entity, causal).

  2. **Recall** — 4-way parallel retrieval (semantic vector, BM25 keyword,
     graph expansion, temporal proximity) fused via Reciprocal Rank Fusion
     (k=60).

  3. **Reflect** — optional agentic multi-step reasoning that queries
     observations first, then raw facts, then synthesises an answer.

  4. **Consolidation** — background synthesis of raw facts into higher-level
     *observations* (the third fact type), with staleness tracking.

Memory is organised into three fact types:
  - **world** — facts about the environment and other people
  - **experience** — the agent's own events and interactions
  - **observation** — consolidated knowledge synthesised from multiple facts

Based on https://github.com/vectorize-io/hindsight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryUnit:
    """Atomic fact — the core storage unit."""
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    fact_text: str = ""
    fact_type: str = "world"           # world | experience | observation
    embedding: list[float] = field(default_factory=list)
    occurred_at: datetime | None = None
    mentioned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: str = ""                  # surrounding text from source chunk
    entity_ids: list[str] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)  # for observations: which facts they consolidate


@dataclass
class Entity:
    """Resolved named entity with co-occurrence tracking."""
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    canonical_name: str = ""
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = 0
    cooccurs_with: dict[str, int] = field(default_factory=dict)  # entity_uuid → count


@dataclass
class MemoryLink:
    """Graph edge between two memory units."""
    from_id: str = ""
    to_id: str = ""
    link_type: str = ""                # temporal | semantic | entity | causes | caused_by
    weight: float = 0.0
    entity_id: str | None = None       # for entity-type links


# ═══════════════════════════════════════════════════════════════════════════
# LLM prompts
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACT_FACTS_SYSTEM = (
    "You are an expert memory curator that extracts atomic facts from text. "
    "Each fact should capture one discrete piece of information."
)

_EXTRACT_FACTS_USER = """\
Extract atomic facts from the following text. For each fact, identify:
- fact_text: a single, self-contained factual statement
- fact_type: "world" (about the environment/others) or "experience" (the speaker's own events)
- who: entities/people mentioned (list of names — use NAMES, not pronouns)
- when: temporal reference if any (resolved to YYYY-MM-DD using the reference time below)
- cause_of: index of another fact this one causes (or null)

## REFERENCE TIME (anchor for temporal expressions)

{reference_time}

Resolve every relative time expression in the text against THIS reference
time, NOT against today's date. Examples (assuming reference_time =
2023-05-07):
- "yesterday" → "2023-05-06"
- "last Saturday" (and ref is Sunday) → "2023-04-29"
- "next Monday" → "2023-05-08"
- "two weeks ago" → "2023-04-23"
- "this morning" → "2023-05-07"

If no temporal expression is present, set `when` to null.

## ATOMIC FACTS — focus

Extract specific, durable facts about:
- Personal information (name, age, location, occupation, family)
- Preferences and opinions (with clear attribution to the speaker)
- Significant events (with dates — past, present, planned, completed)
- Plans, goals, commitments (with target dates if mentioned)
- Expertise, skills, hobbies
- Relationships and roles
- State changes (started/finished/changed something)

## RULES

1. **Atomicity.** One fact per object. "Caroline works at Google and lives
   in NYC" → TWO facts, not one.
2. **Coreference resolution.** Replace pronouns with the named entity when
   it's unambiguous from context. "She moved to NYC" with prior context
   that "Caroline" is the speaker → "Caroline moved to NYC", who=["Caroline"].
3. **Preserve unusual phrasing.** If the speaker uses a specific phrase
   (a brand name, slang, technical term), keep it verbatim in `fact_text`.
   Don't normalize "ChatGPT" to "an AI assistant".
4. **Use precise action verbs.** "purchased" not "got"; "scheduled" not
   "talked about"; "completed" not "did".
5. **Preserve details.** Numbers, quantities, identifiers, frequencies,
   role titles. "I have 2 sisters" → fact_text="The speaker has 2 sisters"
   (not "The speaker has sisters").
6. **USER ASSERTIONS vs QUESTIONS.** Only extract assertions. Skip
   questions the user asked unless they reveal an underlying fact (e.g.
   "What's the weather in NYC?" is a question, not a fact; but "I'm
   in NYC, what's the weather?" gives you `fact_text="The speaker is
   in NYC"`).
7. **STATE CHANGES.** When a fact supersedes earlier information, capture
   it. "I used to be a teacher, now I'm a software engineer" → TWO
   facts, both with appropriate `when`.
8. **Speaker attribution.** For "experience" facts, the speaker is the
   subject. For "world" facts, name the actor explicitly.

## SKIP

- Greetings, filler ("oh", "wow", "yeah"), generic pleasantries
- Process chatter ("let me check", "give me a sec")
- Acknowledgements without new info ("got it", "thanks")
- Hypotheticals without commitment ("I might one day...")

## WORKED EXAMPLES

(Reference time = 2023-05-07)

Text: "I went to a LGBTQ support group yesterday and it was so powerful."
Output: [
  {{"fact_text": "The speaker attended a LGBTQ support group on 2023-05-06.",
    "fact_type": "experience", "who": ["the speaker"],
    "when": "2023-05-06", "cause_of": null}},
  {{"fact_text": "The speaker found the LGBTQ support group to be powerful.",
    "fact_type": "experience", "who": ["the speaker"],
    "when": "2023-05-06", "cause_of": 0}}
]

Text: "Caroline finally got that promotion to senior engineer last week
at Acme — she's been working towards it for two years."
Output: [
  {{"fact_text": "Caroline was promoted to senior engineer at Acme around 2023-04-30.",
    "fact_type": "world", "who": ["Caroline", "Acme"],
    "when": "2023-04-30", "cause_of": null}},
  {{"fact_text": "Caroline worked towards her promotion to senior engineer for two years.",
    "fact_type": "world", "who": ["Caroline"],
    "when": null, "cause_of": null}}
]

Text: "How was your weekend?"
Output: []
(question with no factual assertion — skip)

## TASK

Now extract facts from this text. Return ONLY a JSON array of fact objects
(no preamble, no explanation, no code fences):

{text}"""

_CONSOLIDATE_SYSTEM = (
    "You are a memory consolidation expert. You synthesise raw facts into "
    "concise, durable observations. One facet per observation. "
    "Never merge disparate topics."
)

_CONSOLIDATE_USER = """\
Given these raw facts and any existing observations, produce consolidated observations.

Raw facts:
{raw_facts}

Existing observations (update or keep as-is):
{existing_observations}

For each observation, specify an action:
- CREATE: new observation from novel facts
- UPDATE: extend existing observation with new evidence (provide the observation id)
- DELETE: remove superseded observations (provide the observation id)

Rules:
- One facet per observation (e.g. "dietary preferences" separate from "work schedule")
- Resolve vague references with concrete values when available
- Preserve significant historical events even if superseded
- State changes: update concisely with dates

Return ONLY a JSON array:
[{{"action": "CREATE|UPDATE|DELETE", "id": "existing_id or null", "text": "observation text", "source_fact_ids": ["id1", "id2"]}}, ...]"""

_REFLECT_SYSTEM = (
    "You are a reasoning agent with access to structured memory. "
    "Use ONLY information from retrieved results. Never fabricate facts. "
    "Synthesise and infer rather than repeating raw text."
)

_REFLECT_USER = """\
Answer this query using the retrieved memory results below.

Query: {query}

Observations (consolidated knowledge):
{observations}

Raw facts (ground truth):
{raw_facts}

Synthesise an answer based strictly on the evidence above. Be concise and direct.
If the evidence is insufficient, say so honestly."""


# ═══════════════════════════════════════════════════════════════════════════
# Main middleware
# ═══════════════════════════════════════════════════════════════════════════

@register_memory("hindsight_embedded")
class HindsightEmbeddedMiddleware(MemoryMiddleware):
    """Standalone Hindsight memory with graph-linked facts, 4-way retrieval, and consolidation.

    Config params (in ``MemoryConfig.params``):
        llm_model:              Model for extraction (default gpt-4o-mini)
        llm_api_key:            API key (defaults to OPENAI_API_KEY env var)
        llm_base_url:           Base URL (default https://api.openai.com/v1)
        embedding_model:        Embedding model (default text-embedding-3-small)
        retrieval_limit:        Max results to return (default 50)
        recall_max_tokens:      Cap recall output at this many tokens
                                 (default 3000). Facts are added until the
                                 cumulative token count crosses the cap.
                                 Mirrors upstream's
                                 ``recall_async(max_tokens=4096)`` token
                                 budget — see deviations.md H21/H22.
        user_id:                Namespace (default bench_user)
        use_reflect:            Use agentic reflect for retrieval (default false)
        enable_consolidation:   Run observation consolidation (default true)
        consolidation_batch:    Facts per consolidation LLM call (default 8)
        semantic_threshold:     Min cosine for semantic links (default 0.7)
        temporal_window_hours:  Window for temporal links (default 24)
        entity_match_threshold: Min score for entity resolution (default 0.6)
        recency_boost_weight:   Boost weight for recency in combined
                                 scoring (default 0.2). Set to 0 to
                                 disable. See deviations.md H19.
        recency_half_life_days: Decay half-life for recency boost in days
                                 (default 30). Older facts boost less.
    """

    def __init__(self, config: MemoryConfig):
        p = config.params or {}

        import os
        self._llm_model = p.get("llm_model", "gpt-4o-mini")
        self._llm_api_key = p.get("llm_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        self._llm_base_url = p.get("llm_base_url", "https://api.openai.com/v1")
        self._embedding_model = p.get("embedding_model", "text-embedding-3-small")
        self._retrieval_limit = int(p.get("retrieval_limit", 50))
        self._recall_max_tokens = int(p.get("recall_max_tokens", 3000))
        self._user_id = p.get("user_id", "bench_user")
        self._use_reflect = str(p.get("use_reflect", "false")).lower() == "true"
        self._enable_consolidation = str(p.get("enable_consolidation", "true")).lower() == "true"
        self._consolidation_batch = int(p.get("consolidation_batch", 8))
        self._semantic_threshold = float(p.get("semantic_threshold", 0.7))
        self._temporal_window_hours = float(p.get("temporal_window_hours", 24))
        self._entity_match_threshold = float(p.get("entity_match_threshold", 0.6))
        self._recency_boost_weight = float(p.get("recency_boost_weight", 0.2))
        self._recency_half_life_days = float(p.get("recency_half_life_days", 30))

        # ── Storage ────────────────────────────────────────────────────
        self._units: dict[str, MemoryUnit] = {}             # uuid → MemoryUnit
        self._entities: dict[str, Entity] = {}               # uuid → Entity
        self._entity_names: dict[str, str] = {}              # lower(name) → uuid
        self._links: list[MemoryLink] = []
        self._unconsolidated: list[str] = []                 # unit UUIDs pending consolidation

        # ── Indices ────────────────────────────────────────────────────
        self._bm25_index: _BM25Index | None = None

        # ── LLM clients ───────────────────────────────────────────────
        self._llm_client = None

    # ------------------------------------------------------------------
    # LLM / embedding helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._llm_client is None:
            from openai import AsyncOpenAI
            self._llm_client = AsyncOpenAI(
                api_key=self._llm_api_key, base_url=self._llm_base_url
            )
        return self._llm_client

    async def _llm(self, user_prompt: str, system: str = "", temperature: float = 0.0) -> str:
        try:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user_prompt})
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self._llm_model, messages=msgs,
                temperature=temperature, max_tokens=2000,
            )
            content = resp.choices[0].message.content or ""
            usage = None
            raw_usage = getattr(resp, "usage", None)
            if raw_usage is not None:
                usage = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                    "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                }
            prompt_for_count = (system + "\n" + user_prompt) if system else user_prompt
            self._record_internal_llm(
                prompt=prompt_for_count, response=content, usage=usage,
            )
            return content
        except Exception as e:
            logger.warning("Hindsight LLM call failed: %s", e)
            return ""

    async def _embed(self, text: str) -> list[float]:
        if not text:
            return []
        try:
            client = self._get_client()
            resp = await client.embeddings.create(
                model=self._embedding_model, input=[text[:8000]]
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.debug("Embedding failed (%s), using BoW fallback", e)
            return _bow_vector(text)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            client = self._get_client()
            resp = await client.embeddings.create(
                model=self._embedding_model, input=[t[:8000] for t in texts]
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            logger.debug("Batch embedding failed (%s), using BoW fallback", e)
            return [_bow_vector(t) for t in texts]

    # ------------------------------------------------------------------
    # MemoryMiddleware interface
    # ------------------------------------------------------------------

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """RETAIN — ingest text through the full Hindsight pipeline."""
        text = response.strip()
        if not text:
            return

        # Anchor ``ref_time`` to the source conversation's per-message
        # timestamp (LoCoMo: "1:56 pm on 8 May, 2023") so the LLM fact
        # extractor stamps facts with the conversation's clock, not
        # today's. ``_parse_date`` handles ISO, the LoCoMo natural-
        # language format, and naive ``datetime`` inputs — always
        # returning aware UTC. Falls back to ``datetime.now(timezone
        # .utc)`` only when the caller supplied no timestamp at all.
        ref_time = _parse_date(context.get("timestamp")) or datetime.now(
            timezone.utc
        )

        # Step 1: LLM fact extraction
        raw_facts = await self._extract_facts(text, ref_time)
        if not raw_facts:
            return

        # Step 2: Embed all facts in batch
        fact_texts = [f["fact_text"] for f in raw_facts]
        embeddings = await self._embed_batch(fact_texts)

        # Step 3: Entity resolution + unit creation
        new_units: list[MemoryUnit] = []
        for i, raw in enumerate(raw_facts):
            # Parse temporal info
            occurred_at = _parse_date(raw.get("when")) or ref_time

            # Resolve entities. The expanded H48 extraction prompt
            # encourages the LLM to omit ``who`` when there's no
            # explicit speaker attribution. ``dict.get("who", [])``
            # only returns ``[]`` when the key is MISSING — when the
            # LLM emits ``"who": null`` explicitly, the default is
            # bypassed and we get ``None``. Iterating ``None`` raises
            # ``TypeError: 'NoneType' object is not iterable``,
            # crashing the whole shard. Same trap if the LLM emits
            # a string instead of a list. Normalize defensively:
            # null / non-list → empty list; bare string → singleton list.
            raw_who = raw.get("who")
            if isinstance(raw_who, list):
                entity_names = raw_who
            elif isinstance(raw_who, str) and raw_who.strip():
                entity_names = [raw_who]
            else:
                entity_names = []
            entity_ids = []
            for name in entity_names:
                if not name or not isinstance(name, str):
                    continue
                eid = await self._resolve_entity(name.strip(), ref_time)
                if eid:
                    entity_ids.append(eid)

            unit = MemoryUnit(
                fact_text=raw["fact_text"],
                # ``or`` handles both missing AND explicit-null values;
                # ``raw.get(..., "world")`` would return None when the
                # LLM emits ``"fact_type": null`` explicitly.
                fact_type=raw.get("fact_type") or "world",
                embedding=embeddings[i] if i < len(embeddings) else [],
                occurred_at=occurred_at,
                mentioned_at=ref_time,
                context=text[:500],
                entity_ids=entity_ids,
            )
            self._units[unit.uuid] = unit
            new_units.append(unit)
            self._unconsolidated.append(unit.uuid)

        # Step 4: Build graph links
        await self._build_links(new_units, raw_facts)

        # Step 5: Update entity co-occurrences
        self._update_cooccurrences(new_units)

        # Step 6: Rebuild BM25 index
        self._rebuild_bm25()

        # Step 7: Background consolidation
        if self._enable_consolidation and len(self._unconsolidated) >= self._consolidation_batch:
            asyncio.ensure_future(self._consolidate())

        logger.debug("Hindsight: retained %d facts, %d total units",
                      len(new_units), len(self._units))

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """RECALL or REFLECT — retrieve relevant memories."""
        # Flush any pending consolidation
        if self._enable_consolidation and self._unconsolidated:
            await self._consolidate()

        if self._use_reflect:
            return await self._reflect(prompt)

        results = await self._recall(prompt, self._retrieval_limit)
        if not results:
            return prompt

        # Apply token budget — add facts in score order until the
        # cumulative token count crosses ``recall_max_tokens``. Mirrors
        # upstream's ``recall_async(max_tokens=4096)`` semantics
        # (see deviations.md H21/H22). The byte fallback (chars/4) is
        # tolerant of missing tiktoken; the answering LLM ultimately
        # tokenises its own context with its own tokeniser.
        lines: list[str] = []
        cumulative_tokens = 0
        for unit, score in results:
            prefix = f"[{unit.fact_type}]"
            date_str = ""
            if unit.occurred_at:
                date_str = f", date={unit.occurred_at.strftime('%Y-%m-%d')}"
            line = f"{prefix} {unit.fact_text} (score: {score:.3f}{date_str})"
            # Approximate token count: chars/4 is the standard cheap
            # proxy; it matches tiktoken's `cl100k_base` to within ~5%
            # on English text, which is all we need for a stop-loss
            # cap on context size.
            cumulative_tokens += max(1, len(line) // 4)
            lines.append(line)
            if cumulative_tokens >= self._recall_max_tokens:
                break

        memory_text = "\n".join(lines)
        return f"Retrieved memories:\n{memory_text}\n\n{prompt}"

    async def reset(self) -> None:
        """Clear all memory state."""
        self._units.clear()
        self._entities.clear()
        self._entity_names.clear()
        self._links.clear()
        self._unconsolidated.clear()
        self._bm25_index = None
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # RETAIN: Fact extraction
    # ------------------------------------------------------------------

    async def _extract_facts(self, text: str, ref_time: datetime) -> list[dict]:
        """LLM-based atomic fact extraction with five dimensions."""
        prompt = _EXTRACT_FACTS_USER.format(
            text=text, reference_time=ref_time.isoformat()
        )
        for attempt in range(3):
            resp = await self._llm(prompt, system=_EXTRACT_FACTS_SYSTEM)
            parsed = _parse_json_array(resp)
            if parsed:
                return [f for f in parsed if isinstance(f, dict) and f.get("fact_text")]
        return []

    # ------------------------------------------------------------------
    # RETAIN: Entity resolution
    # ------------------------------------------------------------------

    async def _resolve_entity(self, name: str, ref_time: datetime) -> str | None:
        """Resolve entity name to existing or new entity, using multi-signal scoring."""
        if not name:
            return None

        key = name.lower().strip()
        # Fast path: exact match
        if key in self._entity_names:
            eid = self._entity_names[key]
            ent = self._entities[eid]
            ent.last_seen = ref_time
            ent.mention_count += 1
            return eid

        # Score against existing entities
        best_score = 0.0
        best_uuid = None

        for eid, ent in self._entities.items():
            # 1. Name similarity (0 - 0.5)
            name_sim = SequenceMatcher(None, key, ent.canonical_name.lower()).ratio() * 0.5

            # 2. Temporal proximity (0 - 0.2): full credit within 7 days
            time_diff = abs((ref_time - ent.last_seen).total_seconds()) / 86400.0
            temporal = 0.2 * max(0.0, 1.0 - time_diff / 7.0)

            score = name_sim + temporal
            if score > best_score:
                best_score = score
                best_uuid = eid

        if best_score >= self._entity_match_threshold and best_uuid:
            ent = self._entities[best_uuid]
            ent.last_seen = ref_time
            ent.mention_count += 1
            # Add alias
            self._entity_names[key] = best_uuid
            return best_uuid

        # Create new entity
        entity = Entity(
            canonical_name=name,
            first_seen=ref_time,
            last_seen=ref_time,
            mention_count=1,
        )
        self._entities[entity.uuid] = entity
        self._entity_names[key] = entity.uuid
        return entity.uuid

    # ------------------------------------------------------------------
    # RETAIN: Graph link building
    # ------------------------------------------------------------------

    async def _build_links(self, new_units: list[MemoryUnit], raw_facts: list[dict]) -> None:
        """Build temporal, semantic, entity, and causal links."""
        all_existing = [u for u in self._units.values() if u.uuid not in {n.uuid for n in new_units}]

        for unit in new_units:
            # Temporal links: connect to facts within time window
            if unit.occurred_at:
                for other in all_existing:
                    if other.occurred_at:
                        hours_diff = abs((unit.occurred_at - other.occurred_at).total_seconds()) / 3600.0
                        if hours_diff <= self._temporal_window_hours:
                            weight = max(0.3, 1.0 - hours_diff / self._temporal_window_hours)
                            self._links.append(MemoryLink(
                                from_id=unit.uuid, to_id=other.uuid,
                                link_type="temporal", weight=weight,
                            ))

            # Semantic links: connect to most similar existing facts
            if unit.embedding:
                scored = []
                for other in all_existing:
                    if other.embedding:
                        sim = _cosine(unit.embedding, other.embedding)
                        if sim >= self._semantic_threshold:
                            scored.append((other, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                for other, sim in scored[:5]:
                    self._links.append(MemoryLink(
                        from_id=unit.uuid, to_id=other.uuid,
                        link_type="semantic", weight=sim,
                    ))

            # Entity links: connect facts sharing entities
            for eid in unit.entity_ids:
                for other in all_existing:
                    if eid in other.entity_ids:
                        self._links.append(MemoryLink(
                            from_id=unit.uuid, to_id=other.uuid,
                            link_type="entity", weight=0.8,
                            entity_id=eid,
                        ))

        # Causal links: from LLM-extracted cause_of indices
        for i, raw in enumerate(raw_facts):
            cause_of = raw.get("cause_of")
            if cause_of is not None and isinstance(cause_of, int) and 0 <= cause_of < len(new_units):
                if i < len(new_units) and cause_of != i:
                    self._links.append(MemoryLink(
                        from_id=new_units[i].uuid,
                        to_id=new_units[cause_of].uuid,
                        link_type="causes", weight=0.9,
                    ))

    def _update_cooccurrences(self, new_units: list[MemoryUnit]) -> None:
        """Track entity co-occurrence pairs."""
        for unit in new_units:
            eids = unit.entity_ids
            for i, eid_a in enumerate(eids):
                for eid_b in eids[i + 1:]:
                    if eid_a in self._entities:
                        self._entities[eid_a].cooccurs_with[eid_b] = \
                            self._entities[eid_a].cooccurs_with.get(eid_b, 0) + 1
                    if eid_b in self._entities:
                        self._entities[eid_b].cooccurs_with[eid_a] = \
                            self._entities[eid_b].cooccurs_with.get(eid_a, 0) + 1

    # ------------------------------------------------------------------
    # RECALL: 4-way parallel retrieval + RRF fusion
    # ------------------------------------------------------------------

    async def _recall(self, query: str, top_k: int) -> list[tuple[MemoryUnit, float]]:
        """4-way parallel retrieval fused via Reciprocal Rank Fusion."""
        units = list(self._units.values())
        if not units:
            return []

        query_emb = await self._embed(query)

        # Run 4 strategies in parallel
        sem_task = asyncio.ensure_future(self._search_semantic(query_emb, units))
        bm25_task = asyncio.ensure_future(self._search_bm25(query, units))
        graph_task = asyncio.ensure_future(self._search_graph(query_emb, units))
        temporal_task = asyncio.ensure_future(self._search_temporal(query, units))

        sem_results, bm25_results, graph_results, temporal_results = await asyncio.gather(
            sem_task, bm25_task, graph_task, temporal_task,
            return_exceptions=True,
        )

        # Collect valid ranked lists
        ranked_lists: list[list[str]] = []
        for result in [sem_results, bm25_results, graph_results, temporal_results]:
            if isinstance(result, list):
                ranked_lists.append([uid for uid, _ in result])

        # RRF fusion (k=60)
        rrf_scores: dict[str, float] = {}
        k = 60
        for ranked in ranked_lists:
            for rank, uid in enumerate(ranked, start=1):
                rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        # Combined scoring: multiply RRF score by a recency boost
        # derived from ``mentioned_at`` (when the fact entered the
        # store) with an exponential half-life decay. Mirrors
        # upstream's ``apply_combined_scoring(ce * recency_boost ×
        # temporal_boost)`` shape — see deviations.md H19. We omit
        # the temporal_boost (vs question-date) for now because we
        # don't yet plumb a per-question reference date through
        # recall; the recency boost alone gives the freshness signal.
        # ``recency_boost_weight=0`` disables the boost entirely.
        if self._recency_boost_weight > 0:
            now = datetime.now(timezone.utc)
            half_life = max(
                1.0, self._recency_half_life_days * 86400.0
            )  # seconds
            for uid in list(rrf_scores.keys()):
                u = self._units.get(uid)
                if u is None or u.mentioned_at is None:
                    continue
                age_sec = max(0.0, (now - u.mentioned_at).total_seconds())
                # decay = 2^(-age/half_life) ∈ (0, 1]
                decay = math.pow(0.5, age_sec / half_life)
                boost = 1.0 + self._recency_boost_weight * decay
                rrf_scores[uid] *= boost

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._units[uid], score) for uid, score in fused if uid in self._units]

    async def _search_semantic(
        self, query_emb: list[float], units: list[MemoryUnit]
    ) -> list[tuple[str, float]]:
        """Vector cosine similarity search."""
        if not query_emb:
            return []
        scored = []
        for u in units:
            if u.embedding:
                sim = _cosine(query_emb, u.embedding)
                scored.append((u.uuid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:50]

    async def _search_bm25(
        self, query: str, units: list[MemoryUnit]
    ) -> list[tuple[str, float]]:
        """BM25 keyword search."""
        if not self._bm25_index:
            return []
        scores = self._bm25_index.score(query)
        scored = [(units[i].uuid, scores[i]) for i in range(len(units)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:50]

    async def _search_graph(
        self, query_emb: list[float], units: list[MemoryUnit]
    ) -> list[tuple[str, float]]:
        """Graph expansion from semantic seeds through multi-link traversal."""
        if not query_emb or not self._links:
            return []

        # Find top-5 semantic seeds
        seeds = []
        for u in units:
            if u.embedding:
                sim = _cosine(query_emb, u.embedding)
                seeds.append((u.uuid, sim))
        seeds.sort(key=lambda x: x[1], reverse=True)
        seed_ids = {uid for uid, _ in seeds[:5]}

        if not seed_ids:
            return []

        # BFS expansion (max 3 hops, 10 neighbors per node)
        visited: dict[str, float] = {}
        frontier = {uid: score for uid, score in seeds[:5]}

        for depth in range(3):
            next_frontier: dict[str, float] = {}
            for uid, parent_score in frontier.items():
                if uid in visited:
                    continue
                visited[uid] = parent_score

                # Find links from this node
                neighbors: list[tuple[str, float]] = []
                for link in self._links:
                    if link.from_id == uid:
                        # Boost causal links (Hindsight's special treatment)
                        boost = 1.0 if link.link_type != "causes" else 1.5
                        neighbors.append((link.to_id, link.weight * boost))
                    elif link.to_id == uid:
                        boost = 1.0 if link.link_type != "caused_by" else 1.5
                        neighbors.append((link.from_id, link.weight * boost))

                neighbors.sort(key=lambda x: x[1], reverse=True)
                for nid, w in neighbors[:10]:
                    if nid not in visited:
                        propagated = parent_score * w * 0.7  # decay
                        if nid in next_frontier:
                            next_frontier[nid] = max(next_frontier[nid], propagated)
                        else:
                            next_frontier[nid] = propagated

            frontier = next_frontier
            if not frontier:
                break

        # Add remaining frontier
        for uid, score in frontier.items():
            if uid not in visited:
                visited[uid] = score

        # Remove seeds (they're already in semantic results), return graph-expanded nodes
        result = [(uid, score) for uid, score in visited.items() if uid not in seed_ids]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:50]

    async def _search_temporal(
        self, query: str, units: list[MemoryUnit]
    ) -> list[tuple[str, float]]:
        """Temporal proximity search — rank by date relevance."""
        # Try to extract a date reference from the query
        ref_date = _extract_query_date(query)
        if not ref_date:
            # Fallback: prefer more recent facts
            ref_date = datetime.now(timezone.utc)

        scored = []
        for u in units:
            if u.occurred_at:
                days_diff = abs((ref_date - u.occurred_at).total_seconds()) / 86400.0
                # Linear decay over 365 days
                score = max(0.0, 1.0 - days_diff / 365.0)
                if score > 0:
                    scored.append((u.uuid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:50]

    # ------------------------------------------------------------------
    # REFLECT: Agentic multi-step reasoning
    # ------------------------------------------------------------------

    async def _reflect(self, query: str) -> str:
        """Agentic reflection: observations → raw facts → synthesise."""
        # Retrieve observations first (consolidated knowledge)
        observations = [u for u in self._units.values() if u.fact_type == "observation"]
        raw_units = [u for u in self._units.values() if u.fact_type != "observation"]

        obs_text = "(none)"
        if observations:
            query_emb = await self._embed(query)
            obs_scored = []
            for u in observations:
                if u.embedding:
                    sim = _cosine(query_emb, u.embedding)
                    obs_scored.append((u, sim))
            obs_scored.sort(key=lambda x: x[1], reverse=True)
            top_obs = obs_scored[:5]
            if top_obs:
                obs_text = "\n".join(
                    f"- {u.fact_text} (score: {s:.3f})" for u, s in top_obs
                )

        # Retrieve raw facts
        raw_results = await self._recall(query, 10)
        raw_text = "(none)"
        if raw_results:
            raw_text = "\n".join(
                f"- [{u.fact_type}] {u.fact_text}" for u, _ in raw_results
            )

        # Synthesise
        prompt = _REFLECT_USER.format(
            query=query, observations=obs_text, raw_facts=raw_text,
        )
        answer = await self._llm(prompt, system=_REFLECT_SYSTEM)
        if answer.strip():
            return f"Retrieved memories:\n[Reflection] {answer.strip()}\n\n{query}"
        return query

    # ------------------------------------------------------------------
    # CONSOLIDATION: Background synthesis into observations
    # ------------------------------------------------------------------

    async def _consolidate(self) -> None:
        """Synthesise unconsolidated facts into observation units."""
        if not self._unconsolidated:
            return

        batch_ids = self._unconsolidated[:self._consolidation_batch]

        # Get the raw facts
        raw_facts = []
        for uid in batch_ids:
            if uid in self._units:
                u = self._units[uid]
                if u.fact_type != "observation":
                    raw_facts.append({"id": u.uuid, "text": u.fact_text})

        if not raw_facts:
            self._unconsolidated = self._unconsolidated[len(batch_ids):]
            return

        # Get existing observations
        existing_obs = []
        for u in self._units.values():
            if u.fact_type == "observation":
                existing_obs.append({"id": u.uuid, "text": u.fact_text})

        prompt = _CONSOLIDATE_USER.format(
            raw_facts=json.dumps(raw_facts[:self._consolidation_batch], indent=2),
            existing_observations=json.dumps(existing_obs[:20], indent=2) if existing_obs else "(none)",
        )

        resp = await self._llm(prompt, system=_CONSOLIDATE_SYSTEM)
        actions = _parse_json_array(resp) or []

        for action in actions:
            if not isinstance(action, dict):
                continue
            act = action.get("action", "").upper()
            text = action.get("text", "").strip()
            target_id = action.get("id")
            source_ids = action.get("source_fact_ids", [])

            if act == "CREATE" and text:
                emb = await self._embed(text)
                obs = MemoryUnit(
                    fact_text=text,
                    fact_type="observation",
                    embedding=emb,
                    mentioned_at=datetime.now(timezone.utc),
                    source_unit_ids=source_ids,
                )
                self._units[obs.uuid] = obs

            elif act == "UPDATE" and target_id and target_id in self._units and text:
                old = self._units[target_id]
                old.fact_text = text
                old.embedding = await self._embed(text)
                old.mentioned_at = datetime.now(timezone.utc)
                old.source_unit_ids = list(set(old.source_unit_ids + source_ids))

            elif act == "DELETE" and target_id and target_id in self._units:
                del self._units[target_id]

        self._unconsolidated = self._unconsolidated[len(batch_ids):]
        self._rebuild_bm25()
        logger.debug("Hindsight: consolidated %d facts into observations", len(raw_facts))

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        units = list(self._units.values())
        if units:
            self._bm25_index = _BM25Index([u.fact_text for u in units])
        else:
            self._bm25_index = None

    async def is_indexed(self) -> bool:
        return len(self._unconsolidated) == 0


# ═══════════════════════════════════════════════════════════════════════════
# BM25 (Okapi)
# ═══════════════════════════════════════════════════════════════════════════

class _BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        self._docs = [_tok(d) for d in docs]
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        self._n = len(self._docs)
        self._df: Counter = Counter()
        for doc in self._docs:
            self._df.update(set(doc))

    def score(self, query: str) -> list[float]:
        qtoks = _tok(query)
        out = []
        for doc in self._docs:
            s = 0.0
            dl = len(doc)
            tf = Counter(doc)
            for t in qtoks:
                if t not in tf:
                    continue
                f = tf[t]
                df = self._df.get(t, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                s += idf * f * (self._k1 + 1) / (f + self._k1 * (1 - self._b + self._b * dl / self._avgdl))
            out.append(s)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _bow_vector(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for w in _tok(text):
        vec[hash(w) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


def _parse_json_array(text: str) -> list | None:
    if not text:
        return None
    for pat in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group(1))
                if isinstance(r, list):
                    return r
            except json.JSONDecodeError:
                pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group())
            if isinstance(r, list):
                return r
        except json.JSONDecodeError:
            pass
    return None


def _parse_date(val: Any) -> datetime | None:
    """Try to parse a date from various formats.

    Always returns a tz-aware UTC ``datetime`` when non-None. Naive
    inputs (e.g. ``datetime.fromisoformat("2023-05-08")`` or a
    caller-supplied naive ``datetime``) are assumed UTC and stamped
    with ``timezone.utc``.

    This matters because the call sites compare
    ``MemoryUnit.occurred_at`` against ``datetime.now(timezone.utc)``
    (recency / temporal-search) and against each other (temporal-link
    construction in ``_build_links``). A single naive return value
    crashes those subtractions with ``TypeError: can't subtract
    offset-naive and offset-aware datetimes`` — the bug that took
    down every LoCoMo conv-* shard on the 2026-05-10 dev-AWS run.
    """
    if not val or val == "null":
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        s = str(val).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    # Try YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(val))
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass
    # Try LoCoMo natural-language format: "1:56 pm on 8 May, 2023".
    # This is the per-message ``timestamp`` field shape that LoCoMo
    # ships in ``locomo10.json`` (verified from a working mem0v2 task
    # on 2026-05-12). ``datetime.fromisoformat`` does not accept this
    # — without a dedicated branch, ``post_response`` falls back to
    # ``datetime.now(timezone.utc)`` and every extracted fact gets
    # stamped with today's date instead of the conversation's clock.
    # That's the 2026-05-11 Hindsight 3% pass-rate root cause.
    m = re.match(
        r"(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)\s+on\s+(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December),?\s*(\d{4})",
        str(val).strip(),
    )
    if m:
        try:
            hour = int(m.group(1)) % 12
            if m.group(3).lower() == "pm":
                hour += 12
            minute = int(m.group(2))
            day = int(m.group(4))
            months = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
            }
            month = months[m.group(5).lower()]
            year = int(m.group(6))
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        except (ValueError, KeyError):
            pass
    return None


def _extract_query_date(query: str) -> datetime | None:
    """Try to extract a date reference from a query string."""
    # Look for explicit dates
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", query)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass
    # Look for relative references
    q_lower = query.lower()
    now = datetime.now(timezone.utc)
    if "yesterday" in q_lower:
        return now - timedelta(days=1)
    if "last week" in q_lower:
        return now - timedelta(days=7)
    if "last month" in q_lower:
        return now - timedelta(days=30)
    return None
