"""Graphiti Embedded — standalone implementation of the Graphiti knowledge-graph
memory system (the open-source successor to Zep Community Edition).

Implements the full Graphiti memory lifecycle without requiring Neo4j or any
external server. The "secret sauce" compared to plain RAG:

  1. **Temporally-aware knowledge graph** — extracted facts (EntityEdges) carry
     ``valid_at`` / ``invalid_at`` timestamps. When a new fact contradicts an
     existing one the old edge is *invalidated* rather than deleted — full
     fact history is preserved and queryable.

  2. **Entity deduplication** — exact string normalization then LLM-based
     deduplication prevents the graph from filling with near-duplicates
     ("Sam" and "Sam Johnson" → merged).

  3. **Contradiction detection** — each new fact is checked against existing
     facts for the same entity pair. Contradicting facts trigger
     ``invalid_at`` timestamps on the old edge (e.g., job title change).

  4. **Episode provenance chain** — every EntityEdge records which EpisodicNode
     UUIDs produced it, enabling source tracing.

  5. **Entity summaries** — incremental LLM-generated summaries per entity
     node (dense, 2-6 sentence factual descriptions).

  6. **Hybrid retrieval with RRF** — BM25 (fulltext over facts + entity names)
     and vector cosine similarity run in parallel; fused via Reciprocal Rank
     Fusion (k=60). Only *valid* (non-invalidated) edges are searched.

Based on https://github.com/getzep/graphiti (Apache 2.0).
Paper: Graphiti — A Temporally-Aware Knowledge Graph for LLM Memory Systems.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures — mirror upstream graphiti_core/{nodes,edges}.py field-by-
# field. Audited 2026-05-07 against getzep/graphiti@56cf7b36.
# ═══════════════════════════════════════════════════════════════════════════


class EpisodeType(str, enum.Enum):
    """Mirror of upstream ``graphiti_core.nodes.EpisodeType``.

    Attributes match the upstream enum values byte-for-byte:
      - ``message``     — "actor: content" formatted dialogue
      - ``json``        — structured JSON episode
      - ``text``        — plain-text episode
      - ``fact_triple`` — pre-extracted fact triple shortcut
    """

    message = "message"
    json = "json"
    text = "text"
    fact_triple = "fact_triple"


@dataclass
class EpisodicNode:
    """Raw message stored verbatim — the base layer of the graph.

    Mirrors upstream ``graphiti_core.nodes.EpisodicNode`` field shape
    (modulo the in-process ``entity_edge_ids`` rename of upstream's
    ``entity_edges``):
        uuid, name, group_id (here ``user_id`` analogue), labels,
        created_at, source: EpisodeType, source_description, content,
        valid_at, entity_edges, episode_metadata
    """

    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    content: str = ""
    source: EpisodeType = EpisodeType.message
    source_description: str = ""  # e.g. "speaker: text"
    valid_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: list[str] = field(default_factory=list)
    entity_edge_ids: list[str] = field(default_factory=list)
    episode_metadata: dict[str, Any] | None = None


@dataclass
class EntityNode:
    """Extracted named entity with an LLM-maintained summary.

    Mirrors upstream ``graphiti_core.nodes.EntityNode`` field shape:
        uuid, name, group_id (covered by middleware-level ``user_id``),
        labels, created_at, name_embedding, summary, attributes.
    """

    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""  # canonical name
    name_embedding: list[float] = field(default_factory=list)
    summary: str = ""
    labels: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EntityEdge:
    """Fact triple with bi-temporal validity.

    Mirrors upstream ``graphiti_core.edges.EntityEdge`` field shape,
    including the **two distinct invalidation timestamps** from
    `edges.py:271-279`:

      - ``expired_at``: datetime when the *system retracted* this fact
        (a NEW PIECE OF INFORMATION contradicted it). Set during
        contradiction-resolution.
      - ``invalid_at``: datetime when the fact *stopped being true in
        the world*. Set when the fact text itself encodes a date when
        the truth-value flipped.
      - ``valid_at``: datetime when the fact became true.
      - ``reference_time``: timestamp from the episode that produced
        the edge.

    An edge is *active* when ``expired_at`` is None — i.e. the system
    has not retracted it. This lines up with upstream's default search
    filter which excludes retracted edges via
    ``WHERE e.expired_at IS NULL``.

    Note: we keep ``relation_type`` as the field name for historical
    compatibility with stored data; upstream calls this field ``name``
    on the edge and uses lowercase relation strings, while we use
    SCREAMING_SNAKE_CASE — see deviations.md G15.
    """

    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_uuid: str = ""  # EntityNode.uuid
    target_uuid: str = ""  # EntityNode.uuid
    relation_type: str = ""  # SCREAMING_SNAKE_CASE; upstream uses ``name`` field, lowercase
    fact: str = ""  # natural-language fact sentence
    fact_embedding: list[float] = field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None  # world-truth end (extracted from text)
    expired_at: datetime | None = None  # system retraction (set on contradiction)
    reference_time: datetime | None = None  # episode timestamp that produced the edge
    episodes: list[str] = field(default_factory=list)  # provenance: EpisodicNode UUIDs
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_active(self) -> bool:
        """Return True if the system has not retracted this edge.

        Matches upstream's default search filter ``e.expired_at IS NULL``.
        For backwards compatibility with the previous field semantics,
        we ALSO treat the edge as inactive when ``invalid_at`` is set —
        callers that want strict upstream semantics (active = expired_at
        IS NULL only) should use ``edge.expired_at is None`` directly.
        """
        return self.expired_at is None and self.invalid_at is None


# ═══════════════════════════════════════════════════════════════════════════
# LLM prompts — adapted verbatim from graphiti_core/prompts/
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACT_ENTITIES_SYSTEM = (
    "You are an entity extraction specialist for conversational messages. "
    "NEVER extract abstract concepts, feelings, or generic words."
)

_EXTRACT_ENTITIES_USER = """\
NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time, way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified (photo, pic, picture, image, video, post, story)
- Generic event/activity nouns unless uniquely identified (event, game, meeting, class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company, team, office)
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per message.
- Bare relational/kinship terms (dad, mom, mother, father, sister, brother, husband, wife, spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor, roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). Qualify with possessor: "Nisha's dad" not "dad", "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Your task is to extract entity nodes that are EXPLICITLY mentioned in the CURRENT MESSAGE.
Pronoun references such as he/she/they should be disambiguated to the names of the referenced entities.
Only extract distinct entities from the CURRENT MESSAGE.

<PREVIOUS_MESSAGES>
{previous_messages}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{current_message}
</CURRENT_MESSAGE>

1. Speaker Extraction: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node. If the speaker is mentioned again in the message, treat both mentions as a SINGLE entity.
2. Entity Identification: Extract named entities and specific, concrete things. Ask: "Could this have its own Wikipedia article, OR is it specific enough to distinguish from other items of the same category within this conversation?"
3. Specificity: Always use the MOST SPECIFIC form mentioned. "road cycling" not "cycling", "wool coat" not "coat", "dog leash" not "leash".
4. Exclusions: Do NOT extract entities representing relationships, actions, dates, or temporal information.

<EXAMPLE>
Message: "Jordan: We just moved to Denver last month. My spouse started a new role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center."
Extract: ["Jordan", "Denver", "Lockheed Martin", "Belmont Arts Center", "ceramics"]
Skip: "spouse" (generic — extract only if named), "new role", "last month", "we"
</EXAMPLE>

<EXAMPLE>
Message: "Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park."
Extract: ["Nisha", "Nisha's dad", "Riverside Park"]
Skip: "dad" (qualify as "Nisha's dad"), "dogs" (no specific identity), "next week"
</EXAMPLE>

<EXAMPLE>
Message: "Mary: I forgot Trigger's leash so I couldn't take him on a dog walk. After that I went road cycling in my new wool coat."
Extract: ["Mary", "Trigger", "dog leash", "road cycling", "wool coat"]
Skip: "leash" (use "dog leash"), "cycling" (use "road cycling"), "coat" (use "wool coat"), "dog walk" (activity)
</EXAMPLE>

Return ONLY a JSON array of entity names:
["Entity A", "Entity B", ...]"""

_EXTRACT_EDGES_SYSTEM = (
    "You are an expert fact extractor that extracts fact triples from text. "
    "1. Extracted fact triples should also include relevant date information. "
    "2. Treat REFERENCE_TIME as the time the CURRENT MESSAGE was sent. "
    "All temporal information should be extracted relative to this time."
)

_EXTRACT_EDGES_USER = """\
<PREVIOUS_MESSAGES>
{previous_messages}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{current_message}
</CURRENT_MESSAGE>

<ENTITIES>
{entities}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- Involve TWO DISTINCT entities from the ENTITIES list
- Are clearly stated or unambiguously implied in the CURRENT MESSAGE
- Can be represented as an edge in a knowledge graph

RULES:
1. source_entity and target_entity MUST be names from the ENTITIES list above
2. source_entity and target_entity must be DIFFERENT entities
3. relation_type uses SCREAMING_SNAKE_CASE (e.g. WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)
4. fact is a natural-language description of the relationship
5. valid_at / invalid_at use ISO 8601 UTC format or null
6. Never duplicate semantically identical facts
7. Never extract single-entity states — always find the second entity

Return ONLY a JSON array:
[{{"source": "Entity A", "target": "Entity B", "relation": "RELATION_TYPE", "fact": "natural language fact", "valid_at": "ISO8601 or null", "invalid_at": null}}, ...]"""

_DEDUPE_NODE_SYSTEM = (
    "You are an entity deduplication assistant. "
    "NEVER fabricate entity names or mark distinct entities as duplicates."
)

_DEDUPE_NODE_USER = """\
<EXISTING_ENTITIES>
{existing_entities}
</EXISTING_ENTITIES>

<NEW_ENTITY>
{new_entity}
</NEW_ENTITY>

Entities should only be considered duplicates if they refer to the same real-world object or concept.
NEVER mark entities as duplicates if they are related but distinct, or have similar names but are separate instances.

Return the index (0-based) of the matching EXISTING ENTITY, or -1 if no match.
Return ONLY a JSON integer: 0, 1, 2, ... or -1"""

_RESOLVE_EDGE_SYSTEM = (
    "You are a fact deduplication assistant. "
    "NEVER mark facts with key differences as duplicates."
)

_RESOLVE_EDGE_USER = """\
<EXISTING_FACTS>
{existing_facts}
</EXISTING_FACTS>

<NEW_FACT>
{new_fact}
</NEW_FACT>

1. DUPLICATE: Does the NEW FACT represent identical information to any EXISTING FACT?
   If yes, return the 0-based index of the duplicate.
2. CONTRADICTION: Does the NEW FACT contradict any EXISTING FACT (same relationship, different value)?
   If yes, return the 0-based index of the contradicted fact.
   A fact that is BOTH duplicate and contradicted = it was updated/superseded.
   Different numeric values or dates = contradiction, NOT duplicate.

Return ONLY a JSON object:
{{"duplicate": -1, "contradicted": [-1]}}
(-1 means none. contradicted is a list of indices.)"""

_ENTITY_SUMMARY_SYSTEM = (
    "You maintain detailed, information-dense entity memories from conversation text. "
    "Use ONLY explicitly stated facts. NEVER infer or hallucinate."
)

_ENTITY_SUMMARY_USER = """\
<ENTITY>
{entity_name}
</ENTITY>

<EXISTING_SUMMARY>
{existing_summary}
</EXISTING_SUMMARY>

<NEW_FACTS>
{new_facts}
</NEW_FACTS>

Update the entity summary to incorporate the NEW FACTS.
Rules:
- 2-6 dense sentences maximum, third-person, no "mentioned/stated/described" filler
- Preserve all accurate existing information unless contradicted by new facts
- Use ONLY facts from the provided text — never infer or hallucinate

Return ONLY the updated summary text (no JSON, no quotes)."""


# ═══════════════════════════════════════════════════════════════════════════
# Main middleware
# ═══════════════════════════════════════════════════════════════════════════

@register_memory("graphiti_embedded")
@register_memory("zep_embedded")
class GraphitiEmbeddedMiddleware(MemoryMiddleware):
    """Standalone Graphiti knowledge-graph memory (no Neo4j required).

    Config params (in ``MemoryConfig.params``):
        llm_model:           Model for extraction/dedup (default gpt-4o-mini)
        llm_api_key:         API key (defaults to OPENAI_API_KEY env var)
        llm_base_url:        Base URL (default https://api.openai.com/v1)
        embedding_model:     Embedding model (default text-embedding-3-small)
        retrieval_limit:     Max results to return (default 10)
        user_id:             Namespace for memory isolation (default bench_user)
        context_window:      Recent episodes to include as LLM context (default 5)
        enable_dedup:        Run entity + edge deduplication LLM calls (default true)
        enable_summaries:    Maintain per-entity LLM summaries (default true)
        search_active_only:  Only return non-invalidated edges (default true)
    """

    def __init__(self, config: MemoryConfig):
        p = config.params or {}

        import os
        self._llm_model = p.get("llm_model", "gpt-4o-mini")
        self._llm_api_key = p.get("llm_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        self._llm_base_url = p.get("llm_base_url", "https://api.openai.com/v1")
        self._embedding_model = p.get("embedding_model", "text-embedding-3-small")
        self._retrieval_limit = int(p.get("retrieval_limit", 10))
        self._user_id = p.get("user_id", "bench_user")
        self._context_window = int(p.get("context_window", 5))
        self._enable_dedup = str(p.get("enable_dedup", "true")).lower() == "true"
        self._enable_summaries = str(p.get("enable_summaries", "true")).lower() == "true"
        self._search_active_only = str(p.get("search_active_only", "true")).lower() == "true"

        # ── Graph storage ──────────────────────────────────────────────
        self._episodes: list[EpisodicNode] = []
        self._entities: dict[str, EntityNode] = {}          # uuid → EntityNode
        self._entity_names: dict[str, str] = {}             # lower(name) → uuid
        self._edges: list[EntityEdge] = []

        # ── Derived indices (rebuilt on writes) ────────────────────────
        self._bm25_facts: _BM25Index | None = None
        self._bm25_names: _BM25Index | None = None

        # ── LLM / embed clients (lazy) ─────────────────────────────────
        self._llm_client = None
        self._embed_client = None

    # ------------------------------------------------------------------
    # MemoryMiddleware interface
    # ------------------------------------------------------------------

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Ingest a message through the full Graphiti pipeline."""
        text = response.strip()
        if not text:
            return

        speaker = context.get("speaker_id", self._user_id)
        ts_raw = context.get("timestamp", "")
        try:
            valid_at = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            valid_at = datetime.now(timezone.utc)

        # Step 1 — create episodic node. ``source = EpisodeType.message``
        # is the upstream default (see ``add_episode`` signature in
        # ``graphiti.py:933-951``); we don't currently dispatch on
        # episode type but carry the field so downstream can.
        episode = EpisodicNode(
            name=f"{speaker} @ {valid_at.isoformat()}",
            content=text,
            source=EpisodeType.message,
            source_description=f"{speaker}: {text}",
            valid_at=valid_at,
        )

        # Step 2 — gather recent context for LLM calls
        recent = self._episodes[-self._context_window:]
        prev_text = "\n".join(
            f"[{e.valid_at.isoformat()}] {e.source_description}" for e in recent
        )

        # Step 3 — extract entities
        extracted_names = await self._extract_entities(text, prev_text)
        if not extracted_names:
            self._episodes.append(episode)
            return

        # Step 4 — resolve / deduplicate entity nodes
        entity_nodes = await self._resolve_entities(extracted_names)

        # Step 5 — extract edges (fact triples)
        raw_edges = await self._extract_edges(
            text, prev_text, entity_nodes, valid_at
        )

        # Step 6 — resolve edges (dedup + contradiction)
        new_edges, invalidated_ids = await self._resolve_edges(
            raw_edges, entity_nodes, valid_at
        )

        # Step 7 — persist
        # Contradiction = system retraction. Per upstream
        # ``edges.py:271-279``, that's ``expired_at``, NOT
        # ``invalid_at``. ``invalid_at`` is reserved for the case
        # where the FACT TEXT itself encodes a date when the
        # truth-value flipped (e.g. "John worked at Acme until March
        # 2024"). See deviations.md G6.
        for eid in invalidated_ids:
            for e in self._edges:
                if e.uuid == eid:
                    e.expired_at = valid_at
                    break

        for edge in new_edges:
            edge.episodes.append(episode.uuid)
            edge.reference_time = valid_at
            self._edges.append(edge)
            episode.entity_edge_ids.append(edge.uuid)

        self._episodes.append(episode)

        # Step 8 — update entity summaries (async, non-blocking)
        if self._enable_summaries and new_edges:
            asyncio.ensure_future(
                self._update_summaries(entity_nodes, new_edges)
            )

        # Rebuild BM25 indices
        self._rebuild_indices()

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Search the knowledge graph and prepend relevant context."""
        results = await self._search(prompt, self._retrieval_limit)
        if not results:
            return prompt

        lines = []
        for text, score, rtype in results:
            validity = ""
            lines.append(f"[{rtype}] {text} (score: {score:.3f}){validity}")

        memory_text = "\n".join(lines)
        return f"Retrieved memories:\n{memory_text}\n\n{prompt}"

    async def reset(self) -> None:
        """Clear all memory state."""
        self._episodes.clear()
        self._entities.clear()
        self._entity_names.clear()
        self._edges.clear()
        self._bm25_facts = None
        self._bm25_names = None
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # Step 3: Entity extraction
    # ------------------------------------------------------------------

    async def _extract_entities(self, current: str, previous: str) -> list[str]:
        """LLM call to extract named entities from the current message."""
        prompt = _EXTRACT_ENTITIES_USER.format(
            previous_messages=previous or "(none)",
            current_message=current,
        )
        resp = await self._llm(prompt, system=_EXTRACT_ENTITIES_SYSTEM)
        parsed = _parse_json_array(resp)
        if isinstance(parsed, list):
            return [str(n).strip() for n in parsed if n and str(n).strip()]
        return []

    # ------------------------------------------------------------------
    # Step 4: Entity resolution / deduplication
    # ------------------------------------------------------------------

    async def _resolve_entities(self, names: list[str]) -> list[EntityNode]:
        """Resolve extracted names to existing or new EntityNodes."""
        nodes: list[EntityNode] = []
        existing_list = list(self._entities.values())

        for name in names:
            # Fast path: exact (case-insensitive) match
            key = name.lower().strip()
            if key in self._entity_names:
                nodes.append(self._entities[self._entity_names[key]])
                continue

            # LLM deduplication against existing entities
            matched_uuid: str | None = None
            if self._enable_dedup and existing_list:
                existing_text = "\n".join(
                    f"{i}. {e.name}: {e.summary[:100]}"
                    for i, e in enumerate(existing_list)
                )
                prompt = _DEDUPE_NODE_USER.format(
                    existing_entities=existing_text,
                    new_entity=name,
                )
                resp = await self._llm(prompt, system=_DEDUPE_NODE_SYSTEM)
                idx = _parse_int(resp, default=-1)
                if 0 <= idx < len(existing_list):
                    matched_uuid = existing_list[idx].uuid

            if matched_uuid:
                nodes.append(self._entities[matched_uuid])
            else:
                # Create new entity
                emb = await self._embed(name)
                node = EntityNode(name=name, name_embedding=emb)
                self._entities[node.uuid] = node
                self._entity_names[node.name.lower().strip()] = node.uuid
                existing_list.append(node)
                nodes.append(node)

        return nodes

    # ------------------------------------------------------------------
    # Step 5: Edge extraction
    # ------------------------------------------------------------------

    async def _extract_edges(
        self,
        current: str,
        previous: str,
        entity_nodes: list[EntityNode],
        reference_time: datetime,
    ) -> list[dict[str, Any]]:
        """LLM call to extract fact triples."""
        entities_text = "\n".join(f"- {n.name}" for n in entity_nodes)
        prompt = _EXTRACT_EDGES_USER.format(
            previous_messages=previous or "(none)",
            current_message=current,
            entities=entities_text,
            reference_time=reference_time.isoformat(),
        )
        resp = await self._llm(prompt, system=_EXTRACT_EDGES_SYSTEM)
        parsed = _parse_json_array(resp)
        if not isinstance(parsed, list):
            return []
        return [e for e in parsed if isinstance(e, dict)]

    # ------------------------------------------------------------------
    # Step 6: Edge resolution (dedup + contradiction detection)
    # ------------------------------------------------------------------

    async def _resolve_edges(
        self,
        raw_edges: list[dict[str, Any]],
        entity_nodes: list[EntityNode],
        reference_time: datetime,
    ) -> tuple[list[EntityEdge], list[str]]:
        """Convert raw extracted edges to EntityEdge objects, dedup and detect contradictions.

        Returns (new_edges_to_persist, uuids_to_invalidate).
        """
        name_to_uuid = {n.name.lower(): n.uuid for n in entity_nodes}
        # Also include all known entities for fallback lookup
        name_to_uuid.update(
            {e.name.lower(): e.uuid for e in self._entities.values()}
        )

        new_edges: list[EntityEdge] = []
        to_invalidate: list[str] = []

        for raw in raw_edges:
            src_name = str(raw.get("source", "")).strip()
            tgt_name = str(raw.get("target", "")).strip()
            relation = str(raw.get("relation", "")).strip().upper().replace(" ", "_")
            fact = str(raw.get("fact", "")).strip()

            if not src_name or not tgt_name or not fact:
                continue
            if src_name.lower() == tgt_name.lower():
                continue  # no self-loops

            src_uuid = name_to_uuid.get(src_name.lower())
            tgt_uuid = name_to_uuid.get(tgt_name.lower())
            if not src_uuid or not tgt_uuid:
                continue  # entities not in graph — skip

            # Parse validity dates
            valid_at = _parse_iso(raw.get("valid_at")) or reference_time
            invalid_at = _parse_iso(raw.get("invalid_at"))

            # Find existing edges between same pair for dedup/contradiction check.
            # ``is_active`` excludes both system-retracted (``expired_at``) and
            # world-invalidated (``invalid_at``) edges, matching upstream's
            # default search filter.
            existing_pair = [
                e for e in self._edges
                if (
                    (e.source_uuid == src_uuid and e.target_uuid == tgt_uuid)
                    or (e.source_uuid == tgt_uuid and e.target_uuid == src_uuid)
                )
                and e.is_active
            ]

            if self._enable_dedup and existing_pair:
                existing_text = "\n".join(
                    f"{i}. [{e.relation_type}] {e.fact}"
                    for i, e in enumerate(existing_pair)
                )
                prompt = _RESOLVE_EDGE_USER.format(
                    existing_facts=existing_text,
                    new_fact=f"[{relation}] {fact}",
                )
                resp = await self._llm(prompt, system=_RESOLVE_EDGE_SYSTEM)
                resolution = _parse_json_obj(resp) or {}
                dup_idx = resolution.get("duplicate", -1)
                contra_idxs = resolution.get("contradicted", [])

                # If exact duplicate — skip
                if isinstance(dup_idx, int) and 0 <= dup_idx < len(existing_pair):
                    continue

                # Invalidate contradicted edges
                if isinstance(contra_idxs, list):
                    for idx in contra_idxs:
                        if isinstance(idx, int) and 0 <= idx < len(existing_pair):
                            to_invalidate.append(existing_pair[idx].uuid)

            # Create new edge
            fact_emb = await self._embed(fact)
            edge = EntityEdge(
                source_uuid=src_uuid,
                target_uuid=tgt_uuid,
                relation_type=relation,
                fact=fact,
                fact_embedding=fact_emb,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            new_edges.append(edge)

        return new_edges, to_invalidate

    # ------------------------------------------------------------------
    # Step 8: Entity summary updates
    # ------------------------------------------------------------------

    async def _update_summaries(
        self, nodes: list[EntityNode], new_edges: list[EntityEdge]
    ) -> None:
        """Update entity summaries with new facts (non-blocking background task)."""
        node_uuids = {n.uuid for n in nodes}

        for node in nodes:
            relevant = [
                e.fact for e in new_edges
                if e.source_uuid == node.uuid or e.target_uuid == node.uuid
            ]
            if not relevant:
                continue

            prompt = _ENTITY_SUMMARY_USER.format(
                entity_name=node.name,
                existing_summary=node.summary or "(none yet)",
                new_facts="\n".join(f"- {f}" for f in relevant),
            )
            updated = await self._llm(prompt, system=_ENTITY_SUMMARY_SYSTEM)
            if updated.strip():
                node.summary = updated.strip()[:800]

    # ------------------------------------------------------------------
    # Retrieval — hybrid BM25 + vector + RRF
    # ------------------------------------------------------------------

    async def _search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float, str]]:
        """Hybrid search: BM25 over facts + vector cosine, fused via RRF."""
        # Search across edges (facts) and entity summaries
        all_docs = self._all_searchable_docs()
        if not all_docs:
            return []

        texts, embeddings, rtypes = zip(*all_docs) if all_docs else ([], [], [])
        texts = list(texts)
        embeddings = list(embeddings)
        rtypes = list(rtypes)

        # BM25 scores
        bm25_scores = [0.0] * len(texts)
        if self._bm25_facts:
            bm25_scores = self._bm25_facts.score(query)

        # Vector scores
        query_emb = await self._embed(query)
        vec_scores = [
            _cosine(query_emb, emb) if emb and query_emb else 0.0
            for emb in embeddings
        ]

        # RRF fusion (k=60)
        k = 60
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        vec_ranked = sorted(enumerate(vec_scores), key=lambda x: x[1], reverse=True)

        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(vec_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            (texts[idx], score, rtypes[idx])
            for idx, score in fused
            if idx < len(texts)
        ]

    def _all_searchable_docs(self) -> list[tuple[str, list[float], str]]:
        """Build flat (text, embedding, type) list from active edges + entity summaries.

        Default search excludes system-retracted (``expired_at``) and
        world-invalidated (``invalid_at``) edges via ``edge.is_active``,
        which matches upstream's default search filter that requires
        ``e.expired_at IS NULL``.
        """
        docs: list[tuple[str, list[float], str]] = []

        for edge in self._edges:
            # Skip retracted/invalidated edges in normal mode.
            if self._search_active_only and not edge.is_active:
                continue
            src = self._entities.get(edge.source_uuid)
            tgt = self._entities.get(edge.target_uuid)
            src_name = src.name if src else "?"
            tgt_name = tgt.name if tgt else "?"
            text = (
                f"{src_name} {edge.relation_type.replace('_', ' ').lower()} "
                f"{tgt_name}: {edge.fact}"
            )
            docs.append((text, edge.fact_embedding, "fact"))

        for node in self._entities.values():
            if node.summary:
                docs.append((
                    f"{node.name}: {node.summary}",
                    node.name_embedding,
                    "entity",
                ))

        return docs

    def _rebuild_indices(self) -> None:
        """Rebuild BM25 index over all searchable documents."""
        docs = self._all_searchable_docs()
        if docs:
            texts = [t for t, _, _ in docs]
            self._bm25_facts = _BM25Index(texts)

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

    async def _llm(
        self, user_prompt: str, system: str = "", temperature: float = 0.0
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
            )
            content = resp.choices[0].message.content or ""
            raw_usage = getattr(resp, "usage", None)
            usage = None
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
            logger.warning("Graphiti LLM call failed: %s", e)
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


# ═══════════════════════════════════════════════════════════════════════════
# BM25 (Okapi)
# ═══════════════════════════════════════════════════════════════════════════

class _BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
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
                num = f * (self._k1 + 1)
                den = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                s += idf * num / den
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


def _parse_json_obj(text: str) -> dict | None:
    if not text:
        return None
    for pat in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group(1))
                if isinstance(r, dict):
                    return r
            except json.JSONDecodeError:
                pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group())
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None


def _parse_int(text: str, default: int = -1) -> int:
    if not text:
        return default
    m = re.search(r"-?\d+", text.strip())
    return int(m.group()) if m else default


def _parse_iso(val: Any) -> datetime | None:
    if not val or val == "null":
        return None
    try:
        s = str(val).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
