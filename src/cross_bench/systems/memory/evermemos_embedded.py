"""EverMemOS Embedded — standalone implementation of the EverMemOS memory system.

Implements the full EverMemOS memory lifecycle without requiring an external server:

  1. **Episodic Trace Formation** — LLM-based boundary detection segments
     conversations into MemCells.  Three parallel extractors produce:
       - *Episodes* (narrative summaries)
       - *Foresights* (future-oriented predictions with temporal validity)
       - *Event Logs* (atomic searchable facts)

  2. **Semantic Consolidation** — incremental centroid-based clustering groups
     MemCells into thematic *MemScenes*.

  3. **Reconstructive Recollection** — hybrid retrieval (BM25 + vector + RRF)
     with optional agentic multi-round refinement via LLM sufficiency checks.

Based on https://github.com/EverMind-AI/EverMemOS (arXiv:2601.02163).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from cross_bench.models import MemoryConfig
from cross_bench.systems.base import MemoryMiddleware
from cross_bench.systems.registry import register_memory

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    import nltk
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemCell:
    """Atomic boundary-defined memory unit — a conversation segment."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    summary: str = ""
    user_id: str = ""


@dataclass
class EpisodeMemory:
    """Narrative summary of a conversation segment."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    timestamp: str = ""
    user_id: str = ""
    memcell_id: str = ""


@dataclass
class Foresight:
    """Future-oriented prediction with temporal validity window."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    evidence: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_days: int | None = None
    embedding: list[float] = field(default_factory=list)
    timestamp: str = ""
    user_id: str = ""
    memcell_id: str = ""


@dataclass
class EventLogEntry:
    """Atomic facts extracted from a conversation segment."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    time_str: str = ""
    atomic_facts: list[str] = field(default_factory=list)
    fact_embeddings: list[list[float]] = field(default_factory=list)
    timestamp: str = ""
    user_id: str = ""
    memcell_id: str = ""


@dataclass
class ClusterState:
    """Incremental clustering state for MemScenes."""
    centroids: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    last_ts: dict[str, float] = field(default_factory=dict)
    next_idx: int = 0
    memcell_to_cluster: dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Prompts (from EverMemOS / EverOS prompt templates)
# ═══════════════════════════════════════════════════════════════════════════

# All prompts below are byte-exact copies of upstream EverCore prompt
# templates at commit facd7b69 (HEAD of EverMind-AI/EverOS main on
# 2026-04-27). Locked to upstream by tests/systems/test_evermemos_prompts.py;
# any drift surfaces as a test failure on next run.

BATCH_BOUNDARY_DETECTION_PROMPT = '\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n\nYou are an episodic memory boundary detection expert. Your task is to find all natural "episode boundaries" in a continuous conversation and split it into meaningful, independently memorable segments (MemCells). Your core principle is **"default to merging, split cautiously"**.\n\n### Input Format\nThe following is a complete chronological conversation log. Each message is prefixed with a 1-based index and timestamp:\n\n```\n{messages}\n```\n\n### When to split\n\nAdd a boundary (by message number) only when **clear signals** appear:\n- **Cross-day split (highest priority):** Adjacent messages have different calendar dates — MUST split at the date boundary.\n- **Substantive topic change:** Conversation shifts from one concrete topic to a completely unrelated one (e.g., project architecture → weekend plans).\n- **Task completion + new topic:** A closing message ("migration done") belongs to its task\'s episode; split only when the **next** message opens a genuinely unrelated topic.\n- **Long gap + new topic:** Time gap > 4 hours AND new messages have no clear connection to prior conversation.\n\n**Do NOT split for:**\n- Greetings, farewells ("bye", "thanks") — keep with the main episode\n- Transition phrases ("by the way", "oh also") — usually continue the current episode\n- Brief pauses (< 4 hours) followed by the same topic\n\n### `should_wait`\nSet to `true` when the **last segment** has insufficient information to determine its episode context:\n- **Non-text messages:** Only media placeholders (`[image]`, `[video]`, `[file]`) with no accompanying text\n- **Intent-free short replies:** Minimal responses like "ok", "sure", "got it", "😂"\n- **System or non-conversational messages:** System notifications (join/leave group, payment reminders, etc.) cannot themselves determine episode boundaries — wait for the next human message to decide\n- **Ambiguous intermediate state:** Gap of 30 min–4 hours with content that is neither clearly continuing nor clearly starting a new topic\n\n### Decision Principles\n- **Merge by default:** When in doubt, do not split; only split on clear signals\n- **Content over form:** Greetings and farewells belong to the episode they serve, not their own\n- **Process continuity:** Consecutive actions toward the same goal (e.g., create group → post first instruction) form one episode\n- **System messages don\'t trigger splits:** The episode context of a system message is determined by the next human message that follows it\n\n### Examples\n\n**Example 1 — one boundary:**\nInput messages:\n```\n[1] [2024-03-10 09:00:00+00:00] Alice: Can you help me debug the login issue?\n[2] [2024-03-10 09:01:00+00:00] Bob: Sure, let me check the logs.\n[3] [2024-03-10 09:05:00+00:00] Bob: Found it — a null pointer in AuthService line 42.\n[4] [2024-03-10 09:06:00+00:00] Alice: Fixed, thanks!\n[5] [2024-03-11 10:00:00+00:00] Alice: Hey, are you free for lunch today?\n[6] [2024-03-11 10:01:00+00:00] Bob: Sure, 12:30?\n```\nOutput:\n```json\n{{\n    "reasoning": "Messages 1-4 are a complete bug-fix episode; message 5 starts a new day with an unrelated lunch topic.",\n    "boundaries": [4],\n    "should_wait": false\n}}\n```\n\n**Example 2 — no boundary:**\nInput messages:\n```\n[1] [2024-03-10 14:00:00+08:00] Alice: What\'s the status of the Q2 roadmap?\n[2] [2024-03-10 14:02:00+08:00] Bob: About 60% done. Need to finalize the API specs.\n[3] [2024-03-10 14:10:00+08:00] Alice: OK, let\'s review the specs tomorrow.\n```\nOutput:\n```json\n{{\n    "reasoning": "All messages are part of the same Q2 roadmap discussion with no topic change.",\n    "boundaries": [],\n    "should_wait": false\n}}\n```\n\n### Output Format\nReturn strictly in the following JSON format:\n```json\n{{\n    "reasoning": "<one sentence explaining all boundary decisions>",\n    "boundaries": [<1-indexed message numbers after which to split>],\n    "should_wait": <boolean, whether the last segment has insufficient information>\n}}\n```\n\n**`boundaries: []` means all messages belong to the same episode — no split.**\n\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n'

DEFAULT_CUSTOM_INSTRUCTIONS = '\nFollow these principles when generating episodic memories:\n1. Each episode should be a complete, independent story or event\n2. Preserve all important information including names, time, location, emotions, etc.\n3. Use declarative language to describe episodes, not dialogue format\n4. Highlight key information and emotional changes\n5. Ensure episode content is easy to retrieve later\n'

EPISODE_GENERATION_PROMPT = '\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output (title and content) MUST be in Chinese. If in English, output in English. This is mandatory.\n\nYou are an episodic memory generation expert. Please convert the following conversation content into an episodic memory.\n\nConversation start time: {conversation_start_time}\nConversation content:\n{conversation}\n\nCustom instructions:\n{custom_instructions}\n\nIMPORTANT TIME HANDLING:\n- Use the provided "Conversation start time" as the exact time when this conversation/episode began\n- When the conversation mentions relative times (e.g., "yesterday", "last week"), preserve both the original relative expression AND calculate the absolute date\n- Format time references as: "original relative time (absolute date)" - e.g., "last week (May 7, 2023)"\n- This dual format supports both absolute and relative time-based questions\n- All absolute time calculations should be based on the provided start time\n\nPlease generate a structured episodic memory and return only a JSON object containing the following two fields:\n{{\n    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",\n    "content": "A concise factual record of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account focusing on observable actions and direct statements. Remove redundant expressions and verbose descriptions while preserving all facts, entities (names, dates, locations), and specific details. Keep the content concise without losing key information. Use the provided conversation start time as the base time for this episode."\n}}\n\nRequirements:\n1. The title should be specific and easy to search (including key topics/activities).\n2. The content must include all important information from the conversation while being concise.\n3. Convert the dialogue format into a narrative description.\n4. Maintain chronological order and causal relationships.\n5. Use third-person unless explicitly first-person.\n6. Include specific details that aid keyword search, especially concrete activities, places, and objects.\n7. For time references, use the dual format: "relative time (absolute date)" to support different question types.\n8. When describing decisions or actions, naturally include the reasoning or motivation behind them, but avoid repetitive explanations.\n9. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.\n10. CONCISENESS AND REDUNDANCY REMOVAL:\n    - Remove redundant expressions and verbose descriptions\n    - Avoid repeating the same information in different ways\n    - Eliminate unnecessary filler words and phrases\n    - Keep sentences direct and to the point\n    - Preserve all facts, entities (names, dates, locations), and specific details\n    - Maintain the core meaning and important information\n    - Aim for content length similar to or shorter than the original conversation\n11. CRITICAL DETAIL PRESERVATION:\n   - Person Names: Always include full names of people mentioned (e.g., "went to yoga with Amy\'s colleague, Rob" not just "went to yoga with a colleague")\n   - Special Nouns & Entities: Preserve all proper nouns, brand names, place names, organization names exactly as mentioned\n   - Item Names: Include specific product names, book titles, movie names, restaurant names, etc.\n   - Quantities & Numbers: Record exact numbers, amounts, prices, percentages, dates, times (e.g., "ordered 3 pizzas" not "ordered pizzas")\n   - Specific Activities: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")\n   - Time Points: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")\n12. FREQUENCY INFORMATION:\n   - Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")\n   - Note patterns of behavior (e.g., "mentioned calling mom three times during the conversation")\n   - Include habitual actions (e.g., "usually has coffee at 8 AM before work")\n   - Document repetition counts (e.g., "asked about the project status twice")\n\n\nExample:\nIf the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about Caroline planning to go hiking:\n{{\n    "title": "Caroline\'s Mount Rainier Hiking Plan March 14, 2024: Weekend Adventure Planning Session",\n    "content": "On March 14, 2024 at 3:00 PM UTC, Caroline expressed interest in hiking this weekend (March 16-17, 2024) and sought advice. She wanted to see the sunrise at Mount Rainier. When asked about gear by Melanie, Caroline received suggestions: hiking boots, warm clothing, flashlight, water, and high-energy food. Caroline decided to leave early Saturday morning (March 16, 2024) to catch the sunrise and planned to invite friends. She was excited about the trip."\n}}\n\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output (title and content) MUST be in Chinese. If in English, output in English. This is mandatory.\n\nReturn only the JSON object, do not add any other text:\n'

FORESIGHT_GENERATION_PROMPT = '\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n\nYou are an advanced personal foresight analysis agent. Your task is to predict the specific impacts that a user\'s latest MemCell event might have on their future personal behaviors, habits, decisions, and lifestyle.\n\n## Task Objectives:\n1. **Personal-Level Association**: Analyze the event\'s potential impact on the user\'s future behavior, thinking patterns, life habits, or decision preferences from the personal perspective.\n2. **Associative Prediction, Not Summary**: Based on event content, predict potential personal changes rather than repeating or summarizing the original content.\n3. **Scenario Style Matching**: Predictions must match the scenario style of the event:\n   - Life scenarios (e.g., health, family, leisure, learning) → Use casual language, focus on personal habits, emotional states, lifestyle, personal growth, etc.\n   - Work scenarios (e.g., career development, skill improvement, work style) → Use professional language, focus on career planning, capability enhancement, work habits, professional development, etc.\n4. **Personal Behavior-Oriented**: Each association should reflect the user\'s "potential changes" or "behavioral tendencies," focusing on individual-level future development.\n5. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and personal status.\n6. **Specific and Actionable**: Each prediction should not exceed 40 words; generate up to 10 predictions (recommended 4-8). Content must be specific and verifiable.\n7. **Prefer user_name**: Prefer using user_name when provided; otherwise use user_id (e.g., user_1). Avoid using generic terms like "the user."\n8. **Semantic Grounding**: Predictions must remain semantically related to the input; store grounded supporting facts in evidence so the system can trace back the source.\n## Output Format:\nReturn results as a JSON array, each association includes time information and evidence:\n[\n  {{\n    "content": "XiaoMing will avoid hot/spicy food for the next week",\n    "evidence": "Doctor advice: keep oral hygiene; avoid hot/spicy food for a week",\n    "start_time": "2025-10-21",\n    "end_time": "2025-10-28",\n    "duration_days": 7\n  }},\n  ...\n]\n\n## Example Input (Life Scenario):\n- user_id: xiaoming-001\n- user_name: XiaoMing\n- conversation:\n```text\n[2025-10-21T14:05:00Z] XiaoMing: The extraction went fine, but it\'s still sore.\n[2025-10-21T14:06:10Z] Doctor: Keep oral hygiene, avoid hot/spicy food for a week, and follow up if swelling worsens.\n[2025-10-21T14:07:30Z] XiaoMing: Got it, I\'ll follow the instructions and watch for symptoms.\n```\n\n## Example Output (Life Scenario):\n[\n  {{\n    "content": "XiaoMing will avoid hot/spicy food for the next week",\n    "evidence": "Doctor advice: avoid hot/spicy food; keep oral hygiene for a week",\n    "start_time": "2025-10-21",\n    "end_time": "2025-10-28",\n    "duration_days": 7\n  }},\n  {{\n    "content": "XiaoMing will pay more attention to oral hygiene this week",\n    "evidence": "Doctor advice: keep oral hygiene for a week",\n    "start_time": "2025-10-21",\n    "end_time": "2025-10-28",\n    "duration_days": 7\n  }},\n  {{\n    "content": "If swelling/pain worsens, XiaoMing will seek a follow-up soon",\n    "evidence": "Doctor: follow up if swelling worsens",\n    "start_time": "2025-10-21",\n    "end_time": "2025-11-04",\n    "duration_days": 14\n  }},\n  {{\n    "content": "XiaoMing will avoid hard chewing for the next few days",\n    "evidence": "XiaoMing said it is still sore after the extraction",\n    "start_time": "2025-10-21",\n    "end_time": "2025-10-25",\n    "duration_days": 4\n  }}\n  ...\n]\n\n## Example Input (Work Scenario):\n- user_id: LiHua-001\n- user_name: LiHua\n- conversation:\n```text\n[2025-10-21T10:00:00Z] Trainer: Today we\'ll cover agile planning and sprint rituals.\n[2025-10-21T11:15:20Z] LiHua: The daily standup structure is clearer—I can apply it to my team.\n[2025-10-23T16:40:05Z] Trainer: Review metrics and improve collaboration after each sprint.\n```\n\n## Example Output (Work Scenario):\n[\n  {{\n    "content": "LiHua will trial a more structured standup in the team over the next two weeks",\n    "evidence": "LiHua: the daily standup structure is clearer and can be applied to the team",\n    "start_time": "2025-10-21",\n    "end_time": "2025-11-04",\n    "duration_days": 14\n  }},\n  {{\n    "content": "LiHua will try to introduce sprint rituals in the next month",\n    "evidence": "Training covered sprint rituals; LiHua intends to apply learnings",\n    "start_time": "2025-10-21",\n    "end_time": "2025-11-21",\n    "duration_days": 31\n  }},\n  {{\n    "content": "After the next iteration, LiHua will try to review metrics and do a retrospective",\n    "evidence": "Trainer: review metrics and improve collaboration after each sprint",\n    "start_time": "2025-10-21",\n    "end_time": "2025-11-21",\n    "duration_days": 31\n  }},\n  {{\n    "content": "LiHua will pay more attention to concrete collaboration improvement actions this month",\n    "evidence": "Trainer emphasized improving collaboration after each sprint",\n    "start_time": "2025-10-21",\n    "end_time": "2025-11-21",\n    "duration_days": 31\n  }}\n  ...\n]\n\n## Important Notes:\n- **Personal-Oriented**: Focus on "personal-level future changes," content can cover life, learning, work, emotions, habits, and other personal development areas.\n- **Associative Innovation**: Don\'t repeat original content; generate personal behavioral, habitual, or decision-making changes that the event might trigger.\n- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.\n- **Time Inference**: Reasonably infer time ranges based on event type, personal status, and common sense - don\'t rigidly apply fixed times.\n- **Content Practicality**: Content must be specific, reasonable, practical, and usable by the system for personal foresight modeling.\n- **Semantic Retrieval Friendly**: content should be the prediction result (e.g., "will choose soft food"), evidence stores the original fact (e.g., "wisdom tooth extraction"), enabling AI to retrieve relevant foresights based on user queries (e.g., "recommend food") and trace back reasons.\n- **Time Information Extraction Rules:**\n  - start_time: Extract the specific date when the event occurred from the MemCell\'s timestamp field, format: YYYY-MM-DD\n  - end_time: Extract the specific end time from the original content. If there\'s an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense\n  - duration_days: Extract duration from the original content. If there\'s explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense\n  - evidence: Provide a short grounded summary (1–2 sentences) of the key supporting facts (can merge multiple lines from the transcript/summary); do not introduce new facts; keep it concise (≤40 words)\n  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null\n\n## Input (Markdown):\nYou will receive the following Markdown structure:\n- user_id: {USER_ID}\n- user_name: {USER_NAME}\n- conversation:\n```text\n{CONVERSATION_TEXT}\n```\n\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n\n## Please generate 4-8 (up to 10) associations that may impact the user\'s future life and decisions based on the above content:\n\n'

ATOMIC_FACT_PROMPT = '\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n\nYou are an expert information extraction analyst and information architect.  \nYour task is to analyze the given raw conversation transcript (called "CONVERSATION_TEXT") and produce atomic facts optimized for factual retrieval.\n\n---\n\n### INPUT\n- CONVERSATION_TEXT: The raw conversation transcript.\n- TIME: The start time of the conversation, e.g., "March 10, 2024(Sunday) at 2:00 PM UTC".\n\n---\n\n### OUTPUT\nReturn **only** one valid JSON object, with the following exact structure:\n\n{\n  "atomic_facts": {\n    "time": "<THE EXACT TIME STRING FROM INPUT TIME>",\n    "atomic_fact": [\n      "<Atomic fact sentence 1>",\n      "<Atomic fact sentence 2>",\n      ...\n    ]\n  }\n}\n\n---\n\n### EXTRACTION RULES\n\n#### 1. Atomicity\n* Each entry in `"atomic_fact"` must express **exactly one coherent unit of meaning** — an action, emotion, reason, plan, decision, or statement.\n* If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.\n* Each atomic_fact must be **independent and retrievable on its own**.\n\n#### 2. Time & Date Handling (CRITICAL)\n* The `"time"` field (at the top level) represents the conversation start time.\n* **All generated timestamps must be in UTC.**\n* **Preserve** explicit dates verbatim.\n* **Resolve** relative or vague times (e.g., "yesterday", "last week") relative to `TIME`, and **append the resolved absolute date in parentheses** (e.g., "yesterday (March 9, 2024)").\n\n#### 3. Content Preservation & Attribution\n* **Base facts strictly on the conversation.** Do not infer information not present.\n* **Explicit Attribution**: Always state WHO said or did what.\n  - GOOD: "John said he liked the movie."\n  - BAD: "The movie was liked." (Who liked it?)\n* Resolve pronouns (he/she/it) to specific names where possible.\n\n#### 4. Expression Format\n* Write each atomic_fact as a **single, complete English sentence** in **third-person** form.\n* Do **not** simplify, paraphrase, or merge logically distinct ideas.\n\n#### 5. Retrieval Clarity & Filtering\n* **Filter out**: Greetings, phatic communication ("Okay", "Cool"), and low-value chatter unless it conveys specific emotional or factual content.\n* **Keep**: Events, decisions, plans, preferences, specific opinions, factual statements.\n\n#### 6. Output Requirements\n* Output **only** the JSON object — no additional explanation, markdown, or commentary.\n* Ensure valid JSON.\n\n---\n\n### QUALITY CHECKS\nBefore returning the final output, verify that:\n1. Every meaningful fact is captured.\n2. Attribution is correct (who said what).\n3. Timestamps are UTC and relative times are resolved.\n4. Redundancy is minimized.\n\n---\n\nNow analyze the provided conversation content and start time carefully, apply all rules above, and return **only the JSON object** in the specified format.\n\nConversation start time: {{TIME}}\nConversation content:\n{{INPUT_TEXT}}\n\n**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.\n'

SUFFICIENCY_CHECK_PROMPT = 'You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user\'s query.\n--------------------------\nUser Query:\n{query}\n\nRetrieved Documents:\n{retrieved_docs}\n--------------------------\n\n### Instructions:\n\n1. **Analyze the Query Structure**  \n   - Identify key entities AND determine if the query requires temporal reasoning.\n   - If the query involves time (e.g., "before", "after", "since", "during", "from X to Y", "how long"), you MUST decompose it into:\n       * start_time_needed (if any)\n       * end_time_needed (if any)\n       * temporal_relation_needed (ordering, duration, interval)\n\n2. **Scan Documents for Coverage**  \n   - Look for explicit facts addressing *each* required component:\n       * required entities  \n       * start time  \n       * end time  \n       * temporal relations (ordering or duration)\n\n3. **Extract Key Information**  \n   - List specific resolved entities or facts found in the documents.\n   - If time expressions exist, normalize them (e.g., "two weeks ago", "before she moved").\n\n4. **Identify Missing Information**  \n   - For temporal queries:  \n        * missing start time  \n        * missing end time  \n        * missing ordering facts  \n        * missing duration  \n   - Use resolved names to be specific (e.g., "Start time of Alice moving", "Whether Bob visited before Alice moved").\n\n5. **Judgment**  \n   - **Sufficient**: All required components (entities + temporal boundaries + relations) appear explicitly.  \n   - **Insufficient**: ANY required part is missing.\n\n### Output Format (strict JSON):\n{{\n  "is_sufficient": true or false,\n  "reasoning": "1-2 sentence explanation.",\n  "key_information_found": ["List of resolved entities/facts"],\n  "missing_information": ["Specific missing components, using resolved entity names"]\n}}\n\nNow evaluate:'

MULTI_QUERY_GENERATION_PROMPT = 'You are an expert at query reformulation for long-term conversational retrieval.\nYour goal is to generate multiple complementary search queries that recover BOTH:\n- the starting point of a time interval\n- the ending point of a time interval\n- all temporally-linked events in between\n\nYou MUST explicitly expand temporal references (e.g., "last week", "before moving", \n"when they first met") into alternative expressions.\n\n--------------------------\nOriginal Query:\n{original_query}\n\nKey Information Found:\n{key_info}\n\nMissing Information:\n{missing_info}\n\nRetrieved Documents:\n{retrieved_docs}\n--------------------------\n\n### Temporal Reasoning Strategy (MANDATORY)\nWhen the question involves time or order:\n1. **Boundary Decomposition**  \n   Generate queries that separately target:\n   - the earliest relevant event ("start boundary")\n   - the latest relevant event ("end boundary")\n\n2. **Temporal Expression Expansion**  \n   Rewrite relative time expressions into multiple equivalent forms:\n   - absolute dates (if deducible)\n   - session numbers\n   - “before/after X”\n   - duration phrasing (“two weeks earlier”, “shortly after”)\n\n3. **Interval Reconstruction**  \n   Include a declarative query that resembles a hypothetical answer containing BOTH\n   the start and end time anchors.\n\n### Standard Query Requirements\n1. Generate 2-3 diverse queries.\n2. Query 1 MUST be a specific **Question**.\n3. Query 2 MUST be a **Declarative Statement or Hypothetical Answer (HyDE)**.\n4. Query diversity MUST include different temporal forms (before/after/during).\n5. MUST use Key Info to resolve pronouns IF provided.\n6. No invented facts.  \n7. Keep queries < 25 words, same language as original.\n\n### Output Format (STRICT JSON):\n{{\n  "queries": [\n    "Refined query 1",\n    "Refined query 2",\n    "Refined query 3 (optional)"\n  ],\n  "reasoning": "Brief explanation of how temporal boundaries and expressions were expanded."\n}}\n\nNow generate:\n'

REFINED_QUERY_PROMPT = 'You are an expert at query reformulation for information retrieval.\n\n**Task**: Generate a refined query that targets the missing information in the retrieved results.\n\n**Original Query**:\n{original_query}\n\n**Retrieved Documents** (insufficient):\n{retrieved_docs}\n\n**Missing Information**:\n{missing_info}\n\n**Instructions**:\n1. Keep the core intent of the original query unchanged.\n2. Add specific keywords or rephrase to target the missing information.\n3. Make the query more specific and focused.\n4. The refined query should be a direct question that seeks to extract the missing facts.\n5. Do NOT change the query\'s meaning or make it too broad.\n6. Keep it concise (1-2 sentences maximum).\n\n**Examples**:\n\nExample 1:\nOriginal Query: "What does Alice like?"\nMissing Info: ["Alice\'s specific interests or hobbies"]\nRefined Query: "What are Alice\'s hobbies and interests?"\n\nExample 2:\nOriginal Query: "Tell me about the meeting"\nMissing Info: ["meeting date", "location", "participants"]\nRefined Query: "When and where was the meeting held, and who attended?"\n\nExample 3:\nOriginal Query: "Bob\'s project"\nMissing Info: ["project name", "status", "purpose"]\nRefined Query: "What is the name, current status, and purpose of Bob\'s project?"\n\nNow generate the refined query (output only the refined query, no additional text):\nOriginal Query: {original_query}\nMissing Info: {missing_info}\n\nRefined Query:\n'


# ═══════════════════════════════════════════════════════════════════════════
# Main middleware
# ═══════════════════════════════════════════════════════════════════════════

@register_memory("evermemos_embedded")
class EverMemOSEmbeddedMiddleware(MemoryMiddleware):
    """Standalone EverMemOS implementation with full memory lifecycle.

    Config params (all in ``MemoryConfig.params``):
        llm_model:          Model for extraction LLM calls (default gpt-4.1-mini)
        llm_api_key:        API key (defaults to OPENAI_API_KEY env var)
        llm_base_url:       Base URL (default https://api.openai.com/v1)
        embedding_model:    Embedding model (default text-embedding-3-small)
        retrieve_method:    keyword | vector | hybrid | rrf | agentic (default agentic)
        retrieval_limit:    Max memories to return (default 10)
        user_id:            User ID for memory scoping (default bench_user)
        hard_token_limit:   Force-split boundary at N tokens (default 65536)
        hard_message_limit: Force-split boundary at N messages (default 500)
        similarity_threshold: Cluster cosine threshold (default 0.65)
        max_time_gap_seconds: Max seconds between clustered items (default 604800)
        enable_foresight:   Enable foresight extraction (default false)
        enable_event_log:   Enable event log extraction (default true)
        enable_clustering:  Enable MemScene clustering (default true)
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        p = config.params or {}

        # LLM config
        self._llm_model = p.get("llm_model", "gpt-4.1-mini")
        self._llm_api_key = p.get("llm_api_key", "")
        self._llm_base_url = p.get("llm_base_url", "https://api.openai.com/v1")
        self._embedding_model = p.get("embedding_model", "text-embedding-3-small")

        # Retrieval config
        self._retrieve_method = p.get("retrieve_method", "agentic")
        self._retrieval_limit = int(p.get("retrieval_limit", 10))
        self._user_id = p.get("user_id", "bench_user")

        # Boundary detection config
        self._hard_token_limit = int(p.get("hard_token_limit", 65536))
        self._hard_message_limit = int(p.get("hard_message_limit", 500))

        # Clustering config
        self._similarity_threshold = float(p.get("similarity_threshold", 0.65))
        self._max_time_gap = float(p.get("max_time_gap_seconds", 604800))

        # Feature flags
        self._enable_foresight = str(p.get("enable_foresight", "false")).lower() == "true"
        self._enable_event_log = str(p.get("enable_event_log", "true")).lower() == "true"
        self._enable_clustering = str(p.get("enable_clustering", "true")).lower() == "true"

        # Internal state — memory stores
        self._accumulator: list[dict[str, Any]] = []  # current session messages
        self._memcells: list[MemCell] = []
        self._episodes: list[EpisodeMemory] = []
        self._foresights: list[Foresight] = []
        self._event_logs: list[EventLogEntry] = []
        self._cluster_state = ClusterState()

        # LLM / embedding clients (lazy init)
        self._llm_client = None
        self._embed_client = None

        # BM25 index (rebuilt on new memories — MemCell level)
        self._bm25_index: _BM25Index | None = None
        self._bm25_docs: list[str] = []
        self._bm25_event_ids: list[str] = []

        # Processing state
        self._processing = False

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _get_llm_client(self):
        if self._llm_client is None:
            import os
            from openai import AsyncOpenAI
            api_key = self._llm_api_key or os.environ.get("OPENAI_API_KEY", "")
            self._llm_client = AsyncOpenAI(
                api_key=api_key, base_url=self._llm_base_url
            )
        return self._llm_client

    def _get_embed_client(self):
        if self._embed_client is None:
            import os
            from openai import AsyncOpenAI
            api_key = self._llm_api_key or os.environ.get("OPENAI_API_KEY", "")
            self._embed_client = AsyncOpenAI(
                api_key=api_key, base_url=self._llm_base_url
            )
        return self._embed_client

    async def _llm_generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Call LLM for extraction tasks."""
        client = self._get_llm_client()
        try:
            resp = await client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=16384,
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
                prompt=prompt, response=content, usage=usage,
            )
            return content
        except Exception as e:
            logger.warning("EverMemOS LLM call failed: %s", e)
            return ""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings via OpenAI API, fallback to bag-of-words."""
        if not texts:
            return []
        try:
            client = self._get_embed_client()
            resp = await client.embeddings.create(
                model=self._embedding_model, input=texts
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            logger.debug("Embedding API unavailable (%s), using BoW fallback", e)
            return [_bow_vector(t) for t in texts]

    async def _get_embedding(self, text: str) -> list[float]:
        results = await self._get_embeddings([text])
        return results[0] if results else []

    # ------------------------------------------------------------------
    # MemoryMiddleware interface
    # ------------------------------------------------------------------

    async def post_response(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> None:
        """Ingest a message into the memory system (online per-message).

        Matches EverOS's online boundary detection: after every new message
        (starting from message #3), run boundary detection on the full
        accumulator + new message.  This produces finer-grained MemCell
        splits than a single batch call at flush time.

        See EverOS ``stage1_memcells_extraction.py``
        ``memcell_extraction_from_conversation()`` for the reference loop.
        """
        text = response.strip()
        if not text:
            return

        msg = {
            "content": text,
            "speaker_name": context.get("speaker_id", self._user_id),
            "timestamp": context.get("timestamp", datetime.utcnow().isoformat()),
        }

        # EverOS skips boundary detection for the first 2 messages
        # (history_raw_data_list == [] or len == 1 → just append & continue)
        if len(self._accumulator) < 2:
            self._accumulator.append(msg)
            return

        # Run online boundary detection: history = accumulator, new = [msg]
        all_msgs = self._accumulator + [msg]
        result_memcells = await self._online_boundary_detect(all_msgs)

        if not result_memcells:
            # No boundary found — accumulate
            self._accumulator.append(msg)
        else:
            # Consume boundary-produced MemCells, keep unconsumed tail
            consumed = sum(len(mc.messages) for mc in result_memcells)
            self._accumulator = all_msgs[consumed:]

            # Process each MemCell (episode + atomic fact extraction)
            for memcell in result_memcells:
                self._memcells.append(memcell)
                await self._process_memcell(memcell)

            logger.info(
                "EverMemOS: online extracted %d MemCell(s), remaining=%d",
                len(result_memcells), len(self._accumulator),
            )

    async def _online_boundary_detect(
        self, all_msgs: list[dict[str, Any]]
    ) -> list[MemCell]:
        """Run boundary detection on accumulated messages (online mode).

        Implements the same 3-phase algorithm as EverOS
        ``ConvMemCellExtractor.extract_memcell()``:
          Phase 1: Force-split while messages exceed hard limits
          Phase 2: LLM batch boundary detection on remaining messages
                   (creates MemCells for segments before each boundary)

        Does NOT flush the tail — that's done by ``_flush_accumulator``.
        Returns list of extracted MemCells (may be empty = no boundary).
        """
        result_memcells: list[MemCell] = []

        # Phase 1: force-split loop
        while len(all_msgs) > 1:
            token_count = sum(
                len((m.get("speaker_name", "") + ": " + m.get("content", "")).split())
                for m in all_msgs
            )
            if (token_count < self._hard_token_limit
                    and len(all_msgs) < self._hard_message_limit):
                break
            # Find split point
            split_at = min(self._hard_message_limit - 1, len(all_msgs) - 1)
            while split_at > 1:
                chunk_tokens = sum(
                    len((m.get("speaker_name", "") + ": " + m.get("content", "")).split())
                    for m in all_msgs[:split_at]
                )
                if chunk_tokens < self._hard_token_limit:
                    break
                split_at = split_at // 2

            memcell = MemCell(
                messages=all_msgs[:split_at],
                timestamp=all_msgs[split_at - 1].get("timestamp", ""),
                user_id=self._user_id,
            )
            result_memcells.append(memcell)
            all_msgs = all_msgs[split_at:]

        # Phase 2: LLM boundary detection
        if all_msgs:
            boundaries = await self._detect_boundaries_batch(all_msgs)
            prev = 0
            for boundary_idx in boundaries:
                segment = all_msgs[prev:boundary_idx]
                if segment:
                    memcell = MemCell(
                        messages=segment,
                        timestamp=segment[-1].get("timestamp", ""),
                        user_id=self._user_id,
                    )
                    result_memcells.append(memcell)
                prev = boundary_idx

        return result_memcells

    async def _flush_accumulator(self, flush: bool = True) -> None:
        """Flush remaining accumulated messages into a final MemCell.

        Called at query time (pre_prompt) to ensure all ingested messages
        have been processed.  In the online model, most MemCells are
        created during ``post_response``; this only handles the trailing
        segment that hasn't hit a boundary yet.
        """
        if not self._accumulator:
            return

        if flush:
            msgs = list(self._accumulator)
            self._accumulator.clear()

            memcell = MemCell(
                messages=msgs,
                timestamp=msgs[-1].get("timestamp", ""),
                user_id=self._user_id,
            )
            self._memcells.append(memcell)
            await self._process_memcell(memcell)

            logger.info(
                "EverMemOS: flushed tail MemCell (%d msgs)", len(msgs),
            )

    async def pre_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Search memories and prepend relevant context.

        Implements EverMemOS retrieval with configurable strategy:
        - keyword: BM25 only
        - vector: cosine similarity only
        - hybrid: BM25 + vector with RRF fusion
        - rrf: same as hybrid (explicit RRF)
        - agentic: multi-round LLM-guided retrieval
        """
        # Flush any remaining accumulated messages (batch boundary detection)
        if self._accumulator:
            await self._flush_accumulator(flush=True)

        results = await self._retrieve(prompt, self._retrieval_limit)
        if not results:
            return prompt

        # Build context from top-k MemCells (matches EverOS stage4 format)
        event_ids = [eid for eid, _ in results]
        memory_text = self._format_context_for_answer(event_ids, top_k=self._retrieval_limit)

        return f"{memory_text}\n\n{prompt}"

    async def reset(self) -> None:
        """Clear all memory state."""
        self._accumulator.clear()
        self._memcells.clear()
        self._episodes.clear()
        self._foresights.clear()
        self._event_logs.clear()
        self._cluster_state = ClusterState()
        self._bm25_index = None
        self._bm25_docs.clear()
        self._bm25_event_ids.clear()
        self._reset_internal_counters()

    # ------------------------------------------------------------------
    # Phase 1: Episodic Trace Formation — Batch Boundary Detection
    # ------------------------------------------------------------------

    async def _detect_boundaries_batch(
        self, messages: list[dict[str, Any]]
    ) -> list[int]:
        """Run batch boundary detection on a list of messages.

        Uses a single LLM call to find ALL natural episode boundaries,
        matching EverOS's ConvMemCellExtractor algorithm.

        Returns a sorted list of boundary indices (exclusive end indices
        into the messages list, 0-based).
        """
        if len(messages) < 2:
            return []

        # Format messages with 1-based indices and timestamps
        formatted = self._format_messages_indexed(messages)

        prompt = BATCH_BOUNDARY_DETECTION_PROMPT.format(messages=formatted)

        for attempt in range(5):
            resp = await self._llm_generate(prompt, temperature=0.3)
            parsed = _parse_json(resp)
            if parsed and "boundaries" in parsed:
                raw_boundaries = parsed["boundaries"]
                if not isinstance(raw_boundaries, list):
                    continue
                # Convert 1-indexed boundaries to 0-based exclusive end indices
                # Boundary N means "split after message N" → segment ends at index N
                valid = []
                for b in raw_boundaries:
                    if isinstance(b, (int, float)):
                        b = int(b)
                        if 1 <= b < len(messages):
                            valid.append(b)  # already 0-based exclusive end
                return sorted(set(valid))

        return []

    @staticmethod
    def _format_messages_indexed(messages: list[dict[str, Any]]) -> str:
        """Format messages with 1-based indices and timestamps for batch boundary detection.

        Matches EverOS ``_format_messages_with_indices()`` output:
        ``[N] [YYYY-MM-DD HH:MM:SS+TZ] sender_name: content``

        Preserves the original timestamp string instead of round-tripping
        through epoch floats (which loses timezone info).
        """
        lines = []
        for i, msg in enumerate(messages, start=1):
            speaker = msg.get("speaker_name", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Preserve the original timestamp string, matching EverOS which
            # uses datetime.fromisoformat() → .isoformat(sep=" ").
            # For LoCoMo, timestamps are stored as "YYYY-MM-DD HH:MM:SS"
            # strings, so we can pass them through directly.
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.isoformat(sep=" ", timespec="seconds")
                    elif isinstance(timestamp, str):
                        # Parse and re-format to match EverOS's isoformat output
                        dt = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        time_str = dt.isoformat(sep=" ", timespec="seconds")
                except (ValueError, AttributeError, TypeError):
                    time_str = str(timestamp)

            if time_str:
                lines.append(f"[{i}] [{time_str}] {speaker}: {content}")
            else:
                lines.append(f"[{i}] {speaker}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 1: Parallel Extraction (Episode + Foresight + AtomicFacts)
    # ------------------------------------------------------------------

    async def _process_memcell(self, memcell: MemCell) -> None:
        """Run parallel extractors on a MemCell, then cluster."""
        conv_text = self._format_messages(memcell.messages)
        timestamp_str = memcell.timestamp or datetime.utcnow().isoformat()

        # Parse timestamp for human-readable format
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt = datetime.utcnow()
        time_formatted = dt.strftime("%B %d, %Y (%A) at %I:%M %p UTC")

        # Run extractors in parallel (the EverMemOS secret sauce)
        tasks = [self._extract_episode(memcell, conv_text, time_formatted)]

        if self._enable_foresight:
            tasks.append(self._extract_foresights(memcell, conv_text, timestamp_str))
        if self._enable_event_log:
            tasks.append(self._extract_atomic_facts(memcell, conv_text, time_formatted))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 2: Cluster into MemScene
        if self._enable_clustering and self._episodes:
            latest_ep = self._episodes[-1]
            if latest_ep.memcell_id == memcell.event_id and latest_ep.embedding:
                await self._cluster_episode(latest_ep)

        # Rebuild BM25 index
        self._rebuild_bm25()

    async def _extract_episode(
        self, memcell: MemCell, conv_text: str, time_formatted: str
    ) -> None:
        """Extract episodic memory from MemCell."""
        # Use pseudo-JSON format for episode extraction (matches EverOS
        # get_conversation_json_text())
        episode_text = self._format_messages_json(memcell.messages)
        # Upstream EverCore passes ``custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS``
        # at format-time (episode_memory_extractor.py line 241+269). The prompt
        # template carries an explicit ``{custom_instructions}`` placeholder
        # we MUST fill — leaving it empty would inject the literal string
        # ``{custom_instructions}`` into the LLM prompt.
        prompt = EPISODE_GENERATION_PROMPT.format(
            conversation_start_time=time_formatted,
            conversation=episode_text,
            custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS,
        )

        for attempt in range(5):
            resp = await self._llm_generate(prompt)
            parsed = _parse_json(resp)
            if parsed and parsed.get("title") and parsed.get("content"):
                content = parsed["content"]
                embedding = await self._get_embedding(content)

                episode = EpisodeMemory(
                    title=parsed["title"],
                    content=content,
                    embedding=embedding,
                    timestamp=memcell.timestamp,
                    user_id=memcell.user_id,
                    memcell_id=memcell.event_id,
                )
                self._episodes.append(episode)
                logger.debug("EverMemOS: episode extracted — %s", parsed["title"])
                return

        # Fallback: use raw conversation as episode
        embedding = await self._get_embedding(conv_text[:1000])
        episode = EpisodeMemory(
            title="Conversation segment",
            content=conv_text[:2000],
            embedding=embedding,
            timestamp=memcell.timestamp,
            user_id=memcell.user_id,
            memcell_id=memcell.event_id,
        )
        self._episodes.append(episode)

    async def _extract_foresights(
        self, memcell: MemCell, conv_text: str, timestamp_str: str
    ) -> None:
        """Extract foresight predictions from MemCell.

        Matches EverOS ForesightExtractor.generate_foresights_for_conversation():
        - Up to 5 retries on parse failure
        - Date cleaning via _clean_date_string
        - If only one of end_time/duration_days is present, derive the other
        - Start-time fallback: MemCell timestamp formatted YYYY-MM-DD
        """
        # Derive YYYY-MM-DD fallback start from MemCell timestamp (EverOS parity).
        fallback_start = _format_timestamp_as_date(timestamp_str) or ""

        prompt = FORESIGHT_GENERATION_PROMPT.format(
            USER_ID=memcell.user_id,
            USER_NAME=memcell.user_id,
            CONVERSATION_TEXT=conv_text,
        )

        for attempt in range(5):
            resp = await self._llm_generate(prompt, temperature=0.3)
            parsed = _parse_json_array(resp)
            if not parsed:
                continue

            # Batch embed all foresight contents
            contents = [f.get("content", "") for f in parsed if f.get("content")]
            embeddings = await self._get_embeddings(contents) if contents else []

            emb_idx = 0
            for item in parsed:
                content = item.get("content", "")
                if not content:
                    continue

                start_time = _clean_date_string(item.get("start_time")) or fallback_start
                end_time = _clean_date_string(item.get("end_time"))
                duration_days = item.get("duration_days")

                if start_time:
                    if duration_days and not end_time:
                        end_time = _calculate_end_time_from_duration(
                            start_time, duration_days
                        )
                    elif end_time and not duration_days:
                        duration_days = _calculate_duration_days(start_time, end_time)

                foresight = Foresight(
                    content=content,
                    evidence=item.get("evidence", ""),
                    start_time=start_time or "",
                    end_time=end_time or "",
                    duration_days=duration_days,
                    embedding=embeddings[emb_idx] if emb_idx < len(embeddings) else [],
                    timestamp=timestamp_str,
                    user_id=memcell.user_id,
                    memcell_id=memcell.event_id,
                )
                self._foresights.append(foresight)
                emb_idx += 1

            logger.debug("EverMemOS: %d foresights extracted", len(parsed))
            return

    async def _extract_atomic_facts(
        self, memcell: MemCell, conv_text: str, time_formatted: str
    ) -> None:
        """Extract atomic facts from a MemCell."""
        # Format input text matching EverOS atomic_fact_extractor:
        # [timestamp] speaker: content\n  per message
        atomic_text = self._format_messages_for_atomic(memcell.messages)
        # Upstream EverCore uses ``str.replace`` (not ``str.format``) for
        # this prompt — atomic_fact_extractor.py:186-187 calls
        # ``self.atomic_fact_prompt.replace("{{INPUT_TEXT}}", input_text)``
        # then ``.replace("{{TIME}}", time_str)``. The placeholders are
        # literal ``{{TIME}}`` / ``{{INPUT_TEXT}}`` strings in the
        # template (escape-doubled braces), and ``str.replace`` substitutes
        # them. We mirror that exactly to avoid touching any of the
        # JSON-shaped single-brace examples in the prompt body.
        prompt = ATOMIC_FACT_PROMPT.replace(
            "{{INPUT_TEXT}}", atomic_text
        ).replace("{{TIME}}", time_formatted)

        for attempt in range(5):
            resp = await self._llm_generate(prompt)
            parsed = _parse_json(resp)
            if parsed and "atomic_facts" in parsed:
                af = parsed["atomic_facts"]
                facts = af.get("atomic_fact", [])
                if not isinstance(facts, list) or not facts:
                    continue

                # Batch embed all atomic facts
                embeddings = await self._get_embeddings(facts)

                entry = EventLogEntry(
                    time_str=af.get("time", time_formatted),
                    atomic_facts=facts,
                    fact_embeddings=embeddings,
                    timestamp=memcell.timestamp,
                    user_id=memcell.user_id,
                    memcell_id=memcell.event_id,
                )
                self._event_logs.append(entry)
                logger.debug(
                    "EverMemOS: %d atomic facts extracted", len(facts)
                )
                return

    # ------------------------------------------------------------------
    # Phase 2: Semantic Consolidation — Clustering
    # ------------------------------------------------------------------

    async def _cluster_episode(self, episode: EpisodeMemory) -> None:
        """Assign episode to a MemScene cluster using incremental centroid matching."""
        state = self._cluster_state
        vec = episode.embedding
        if not vec:
            return

        import numpy as np
        ep_vec = np.array(vec, dtype=float)
        ep_norm = np.linalg.norm(ep_vec)
        if ep_norm == 0:
            return

        # Try to parse timestamp for temporal gating
        ts_float = _parse_timestamp(episode.timestamp)

        best_sim = -1.0
        best_cid = None

        for cid, centroid_list in state.centroids.items():
            centroid = np.array(centroid_list, dtype=float)
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                continue

            # Temporal gating
            if ts_float is not None and cid in state.last_ts:
                if abs(ts_float - state.last_ts[cid]) > self._max_time_gap:
                    continue

            sim = float(np.dot(ep_vec, centroid) / (ep_norm * c_norm))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_sim >= self._similarity_threshold and best_cid is not None:
            # Add to existing cluster — update centroid incrementally
            old_centroid = np.array(state.centroids[best_cid], dtype=float)
            count = state.counts[best_cid]
            new_centroid = (old_centroid * count + ep_vec) / (count + 1)
            state.centroids[best_cid] = new_centroid.tolist()
            state.counts[best_cid] = count + 1
            if ts_float:
                state.last_ts[best_cid] = max(
                    state.last_ts.get(best_cid, ts_float), ts_float
                )
            state.memcell_to_cluster[episode.memcell_id] = best_cid
        else:
            # New cluster
            cid = f"scene_{state.next_idx:03d}"
            state.next_idx += 1
            state.centroids[cid] = ep_vec.tolist()
            state.counts[cid] = 1
            if ts_float:
                state.last_ts[cid] = ts_float
            state.memcell_to_cluster[episode.memcell_id] = cid

    # ------------------------------------------------------------------
    # Phase 3: Reconstructive Recollection — Retrieval
    # ------------------------------------------------------------------

    async def _retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Retrieve relevant memories. Returns (event_id, score) tuples."""
        method = self._retrieve_method

        if method == "keyword":
            return self._bm25_search(query, top_k)
        elif method == "vector":
            return await self._vector_search(query, top_k)
        elif method in ("hybrid", "rrf"):
            return await self._hybrid_search(query, top_k)
        elif method == "agentic":
            return await self._agentic_search(query, top_k)
        else:
            return await self._hybrid_search(query, top_k)

    # -- MemCell-centric helpers for retrieval --

    def _get_memcell_bm25_docs(self) -> list[tuple[str, str]]:
        """Build one BM25 document per MemCell (matching EverOS stage2).

        Each document = all atomic facts for that MemCell concatenated.
        Fallback to episode content if no atomic facts.

        Returns list of (event_id, bm25_text) tuples, one per MemCell.
        """
        # Index: which memcell_ids have event logs
        el_by_mc: dict[str, EventLogEntry] = {}
        for el in self._event_logs:
            el_by_mc[el.memcell_id] = el

        ep_by_mc: dict[str, EpisodeMemory] = {}
        for ep in self._episodes:
            ep_by_mc[ep.memcell_id] = ep

        docs: list[tuple[str, str]] = []
        for mc in self._memcells:
            el = el_by_mc.get(mc.event_id)
            if el and el.atomic_facts:
                # EverOS: all atomic facts concatenated with spaces
                text = " ".join(el.atomic_facts)
            else:
                # Fallback: episode content
                ep = ep_by_mc.get(mc.event_id)
                text = ep.content if ep else ""
            if text:
                docs.append((mc.event_id, text))
        return docs

    def _get_memcell_maxsim(self, query_emb: list[float]) -> dict[str, float]:
        """MaxSim score per MemCell: max cosine over all fact embeddings.

        Matches EverOS ``compute_maxsim_score()`` in stage3.
        """
        maxsim: dict[str, float] = {}
        for el in self._event_logs:
            best = 0.0
            for emb in el.fact_embeddings:
                if emb:
                    sim = _cosine_similarity(query_emb, emb)
                    if sim > best:
                        best = sim
            maxsim[el.memcell_id] = best

        # Also score MemCells that only have episodes (no event log)
        for ep in self._episodes:
            if ep.memcell_id not in maxsim and ep.embedding:
                maxsim[ep.memcell_id] = _cosine_similarity(query_emb, ep.embedding)

        return maxsim

    def _format_memcell_for_sufficiency(
        self, event_ids: list[str], max_docs: int = 10
    ) -> str:
        """Format MemCells for sufficiency check (matches EverOS format_documents_for_llm).

        Format:
            Document 1:
              Title: {episode.title}
              Content: {episode.content[:500]}
        """
        ep_by_mc: dict[str, EpisodeMemory] = {}
        for ep in self._episodes:
            ep_by_mc[ep.memcell_id] = ep

        lines = []
        for i, eid in enumerate(event_ids[:max_docs], start=1):
            ep = ep_by_mc.get(eid)
            if ep:
                content = ep.content[:500] + "..." if len(ep.content) > 500 else ep.content
                lines.append(
                    f"Document {i}:\n  Title: {ep.title}\n  Content: {content}\n"
                )
        return "\n".join(lines)

    def _format_context_for_answer(
        self, event_ids: list[str], top_k: int = 10
    ) -> str:
        """Build answer context from top-k MemCells (matches EverOS stage4).

        Format per MemCell: ``{title}: {episode}\\n---``
        Wrapped in: ``Episodes memories for conversation between ...``
        """
        ep_by_mc: dict[str, EpisodeMemory] = {}
        for ep in self._episodes:
            ep_by_mc[ep.memcell_id] = ep

        # Collect speaker names from memcells
        speakers = set()
        for mc in self._memcells:
            for msg in mc.messages:
                speakers.add(msg.get("speaker_name", ""))
        speaker_list = sorted(s for s in speakers if s)
        speaker_1 = speaker_list[0] if len(speaker_list) > 0 else "Speaker A"
        speaker_2 = speaker_list[1] if len(speaker_list) > 1 else "Speaker B"

        docs = []
        for eid in event_ids[:top_k]:
            ep = ep_by_mc.get(eid)
            if ep:
                docs.append(f"{ep.title}: {ep.content}\n---")

        speaker_memories = "\n\n".join(docs)
        return (
            f"Episodes memories for conversation between "
            f"{speaker_1} and {speaker_2}:\n\n    {speaker_memories}"
        )

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index at the MemCell level (matches EverOS stage2).

        Each MemCell = one BM25 document (all atomic facts concatenated).
        """
        mc_docs = self._get_memcell_bm25_docs()
        if not mc_docs:
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_event_ids = []
            return
        self._bm25_event_ids = [eid for eid, _ in mc_docs]
        self._bm25_docs = [text for _, text in mc_docs]
        self._bm25_index = _BM25Index(self._bm25_docs)

    def _bm25_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """BM25 keyword search at MemCell level.

        Returns (event_id, bm25_score) tuples.
        """
        if not self._bm25_index:
            return []

        scores = self._bm25_index.score(query)
        ranked = sorted(
            zip(self._bm25_event_ids, scores),
            key=lambda x: x[1], reverse=True,
        )
        return [(eid, s) for eid, s in ranked[:top_k] if s > 0]

    async def _vector_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Vector search at MemCell level using MaxSim.

        Returns (event_id, maxsim_score) tuples.
        """
        query_emb = await self._get_embedding(query)
        if not query_emb:
            return self._bm25_search(query, top_k)

        maxsim = self._get_memcell_maxsim(query_emb)
        ranked = sorted(maxsim.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    async def _hybrid_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Hybrid BM25 + Vector retrieval with RRF fusion at MemCell level.

        Returns (event_id, rrf_score) tuples.
        Matches EverOS hybrid with k=40 (config.hybrid_rrf_k).
        """
        bm25_results = self._bm25_search(query, top_k=50)
        vec_results = await self._vector_search(query, top_k=50)

        if not bm25_results and not vec_results:
            return []

        # RRF fusion (k=40, matches EverOS config.hybrid_rrf_k)
        k = 40
        rrf_scores: dict[str, float] = {}

        for rank, (eid, _) in enumerate(bm25_results, start=1):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank)
        for rank, (eid, _) in enumerate(vec_results, start=1):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    async def _agentic_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Agentic multi-round retrieval (matches EverOS stage3 agentic_retrieval).

        Round 1: hybrid search → top 20
        Sufficiency check on top 10 (using episode format)
        If insufficient: multi-query generation → round 2 → multi-RRF merge
        Returns (event_id, score) tuples.

        NOTE on reranker parity: EverOS agentic_retrieval performs two
        cross-encoder rerank passes (Top-20 → Top-10 before sufficiency check,
        and Top-40 → Top-20 after merging rounds). This embedded version has no
        reranker model wired up, so it follows the EverOS use_reranker=False
        branch — taking the first N by RRF score. Expect modest retrieval
        quality differences versus EverOS runs with use_reranker=True.
        """
        # Round 1: hybrid (top 20)
        round1 = await self._hybrid_search(query, top_k=20)

        if not round1:
            return []

        # Sufficiency check on top 10 (matches EverOS: use_episode=True, max_docs=10)
        top10_eids = [eid for eid, _ in round1[:10]]
        top_docs = self._format_memcell_for_sufficiency(top10_eids, max_docs=10)

        suf_prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query, retrieved_docs=top_docs
        )
        suf_resp = await self._llm_generate(suf_prompt, temperature=0.0)
        suf_parsed = _parse_json(suf_resp)

        is_sufficient = True
        missing_info: list[str] = []
        key_info: list[str] = []
        if suf_parsed:
            is_sufficient = suf_parsed.get("is_sufficient", True)
            missing_info = suf_parsed.get("missing_information", [])
            key_info = suf_parsed.get("key_information_found", [])

        if is_sufficient:
            return round1[:top_k]

        # Round 2: generate refined queries (matches EverOS multi-query flow)
        # Format docs for multi-query prompt (max_docs=5, matching EverOS)
        mq_docs = self._format_memcell_for_sufficiency(top10_eids, max_docs=5)
        mq_prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            original_query=query,
            key_info=", ".join(key_info) if key_info else "N/A",
            missing_info=", ".join(missing_info) if missing_info else "N/A",
            retrieved_docs=mq_docs,
        )
        mq_resp = await self._llm_generate(mq_prompt, temperature=0.4)
        mq_parsed = _parse_json(mq_resp)

        refined_queries: list[str] = []
        if mq_parsed and "queries" in mq_parsed:
            refined_queries = [str(q) for q in mq_parsed["queries"] if q][:3]
        if not refined_queries:
            refined_queries = [query]

        # Parallel retrieval for all refined queries (top 50 each)
        round2_tasks = [self._hybrid_search(q, top_k=50) for q in refined_queries]
        round2_results = await asyncio.gather(*round2_tasks, return_exceptions=True)

        # Multi-query RRF fusion (k=40, matching EverOS)
        k = 40
        all_result_sets: list[list[tuple[str, float]]] = []
        for r in round2_results:
            if isinstance(r, list):
                all_result_sets.append(r)

        mq_scores: dict[str, float] = {}
        for result_set in all_result_sets:
            for rank, (eid, _) in enumerate(result_set, start=1):
                mq_scores[eid] = mq_scores.get(eid, 0.0) + 1.0 / (k + rank)

        # Take top 40 from multi-RRF
        round2_ranked = sorted(mq_scores.items(), key=lambda x: x[1], reverse=True)[:40]

        # Merge round 1 (20) + unique from round 2 (up to 20)
        seen = {eid for eid, _ in round1}
        merged = list(round1)
        for eid, score in round2_ranked:
            if eid not in seen:
                merged.append((eid, score))
                seen.add(eid)
                if len(merged) >= 40:
                    break

        logger.debug(
            "EverMemOS agentic: round1=%d, refined_queries=%d, merged=%d",
            len(round1), len(refined_queries), len(merged),
        )
        return merged[:top_k]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_messages(
        messages: list[dict[str, Any]], include_timestamps: bool = True
    ) -> str:
        lines = []
        for msg in messages:
            speaker = msg.get("speaker_name", "Unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            if include_timestamps and ts:
                lines.append(f"[{ts}] {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_messages_json(messages: list[dict[str, Any]]) -> str:
        """Format messages as pseudo-JSON blocks for episode extraction.

        Matches EverOS ``get_conversation_json_text()`` output:
        each message becomes an indented JSON-like block with unquoted values.
        """
        lines = []
        for msg in messages:
            speaker = msg.get("speaker_name", "Unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            if ts:
                lines.append(
                    f"""
                {{
                    "timestamp": {ts},
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
            else:
                lines.append(
                    f"""
                {{
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
        return "\n".join(lines)

    @staticmethod
    def _format_messages_for_atomic(messages: list[dict[str, Any]]) -> str:
        """Format messages for atomic fact extraction.

        Matches EverOS ``extract_atomic_fact()`` input format:
        ``[timestamp] speaker: content\\n`` per message.
        """
        text = ""
        for msg in messages:
            speaker = msg.get("speaker_name", "Unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            if ts:
                text += f"[{ts}] {speaker}: {content}\n"
            else:
                text += f"{speaker}: {content}\n"
        return text

    # ------------------------------------------------------------------
    # is_indexed support
    # ------------------------------------------------------------------

    async def is_indexed(self) -> bool:
        """Return True when all MemCells have been processed."""
        return not self._processing


# ═══════════════════════════════════════════════════════════════════════════
# BM25 index (Okapi BM25 — same as rag_baseline)
# ═══════════════════════════════════════════════════════════════════════════

class _BM25Index:
    """Okapi BM25 implementation for keyword search."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._docs = [self._tokenize(d) for d in documents]
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        self._n = len(self._docs)
        self._df: Counter = Counter()
        for doc in self._docs:
            self._df.update(set(doc))

    def score(self, query: str) -> list[float]:
        query_terms = self._tokenize(query)
        scores = []
        for doc in self._docs:
            s = 0.0
            doc_len = len(doc)
            tf = Counter(doc)
            for term in query_terms:
                if term not in tf:
                    continue
                f = tf[term]
                df = self._df.get(term, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                numerator = f * (self._k1 + 1)
                denominator = f + self._k1 * (
                    1 - self._b + self._b * doc_len / self._avgdl
                )
                s += idf * numerator / denominator
            scores.append(s)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        try:
            from nltk.tokenize import word_tokenize
            from nltk.stem import PorterStemmer
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words("english"))
            stemmer = PorterStemmer()
            tokens = word_tokenize(text.lower())
            return [stemmer.stem(t) for t in tokens if t.isalpha() and len(t) >= 2 and t not in stop_words]
        except ImportError:
            return re.findall(r"\w+", text.lower())


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_timestamp(ts: str) -> float | None:
    """Parse a timestamp string to float (seconds since epoch).

    Handles ISO 8601, LoCoMo's "1:56 pm on 8 May, 2023", and other
    common date formats via dateutil fallback.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        pass
    # Fallback: python-dateutil or manual parsing
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(ts)
        return dt.timestamp()
    except Exception:
        pass
    return None


def _bow_vector(text: str, dim: int = 256) -> list[float]:
    """Bag-of-words hash vector as embedding fallback."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        idx = hash(word) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _parse_json(text: str) -> dict | None:
    """Extract and parse a JSON object from LLM output."""
    if not text:
        return None
    # Try ```json block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass
    # Try full/nested JSON first (greedy — handles nested objects like
    # {"atomic_facts": {"time": "...", "atomic_fact": [...]}} )
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback to non-nested JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _clean_date_string(date_str: Any) -> str | None:
    """Clean and validate a YYYY-MM-DD date string.

    Mirrors EverOS ForesightExtractor._clean_date_string. Strips non-digit
    non-hyphen characters, validates the YYYY-MM-DD shape, then verifies the
    calendar date is valid.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = re.sub(r"[^\d\-]", "", date_str)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        return None
    try:
        year, month, day = map(int, cleaned.split("-"))
        datetime(year, month, day)
        return cleaned
    except ValueError:
        return None


def _calculate_end_time_from_duration(
    start_time: str, duration_days: Any
) -> str | None:
    """Return YYYY-MM-DD end date = start_time + duration_days, or None."""
    try:
        if not start_time or duration_days is None:
            return None
        days = int(duration_days)
        return (
            datetime.strptime(start_time, "%Y-%m-%d") + timedelta(days=days)
        ).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _calculate_duration_days(start_time: str, end_time: str) -> int | None:
    """Return integer day span between two YYYY-MM-DD dates, or None."""
    try:
        if not start_time or not end_time:
            return None
        s = datetime.strptime(start_time, "%Y-%m-%d")
        e = datetime.strptime(end_time, "%Y-%m-%d")
        return (e - s).days
    except (ValueError, TypeError):
        return None


def _format_timestamp_as_date(ts: Any) -> str | None:
    """Best-effort convert an ISO/epoch timestamp to YYYY-MM-DD (UTC-agnostic)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d")
    if isinstance(ts, (int, float)):
        secs = ts / 1000 if ts > 1e12 else ts
        try:
            return datetime.utcfromtimestamp(secs).strftime("%Y-%m-%d")
        except (OSError, ValueError):
            return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            # Fall back to leading YYYY-MM-DD slice if present
            m = re.match(r"^(\d{4}-\d{2}-\d{2})", ts)
            return m.group(1) if m else None
    return None


def _parse_json_array(text: str) -> list | None:
    """Extract and parse a JSON array from LLM output."""
    if not text:
        return None
    # Try ```json block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                result = json.loads(text[start:end].strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
    # Try raw JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None
