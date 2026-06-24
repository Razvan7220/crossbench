"""Provider-specific prompt templates for conversational benchmarks.

Each memory provider returns context in a different format.  These templates
adapt the answer prompt and judge prompt to match how the provider structures
its search results, following MemoryBench's per-provider prompt customization.

Reference:
    https://github.com/supermemoryai/memorybench/tree/main/src/providers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Default prompts (used when no provider-specific prompt is configured)
# ---------------------------------------------------------------------------

DEFAULT_ANSWER_PROMPT = (
    "You are a helpful assistant with access to conversation history.\n"
    "Use the following retrieved memories to answer the question.\n"
    "If the memories don't contain enough information, say so honestly.\n\n"
    "Retrieved memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer concisely and directly."
)

DEFAULT_JUDGE_SYSTEM = (
    "You are a fair and objective judge. Your task is to evaluate whether "
    "a generated answer correctly addresses a given question, using a golden "
    "(reference) answer as the standard.\n\n"
    "Guidelines:\n"
    "- Be generous in your evaluation. If the generated answer captures the "
    "same meaning, topic, or key information as the golden answer, consider "
    "it CORRECT.\n"
    "- Minor differences in wording, format, or additional details should NOT "
    "be penalized.\n"
    "- Only mark as WRONG if the generated answer is clearly incorrect, "
    "contradicts the golden answer, or completely misses the point."
)

# ---------------------------------------------------------------------------
# Provider-specific answer prompts (adapted from MemoryBench)
# ---------------------------------------------------------------------------

MEM0_ANSWER_PROMPT = (
    "You are an intelligent memory assistant that uses memories from past "
    "conversations to answer questions accurately.\n\n"
    "Each memory has an associated timestamp indicating when it was stored.\n\n"
    "# INSTRUCTIONS:\n"
    "- Synthesize information across different memories to build a complete answer\n"
    "- Convert relative time references (e.g. 'last year', 'next month') to "
    "specific dates using the memory timestamps\n"
    "- When memories contradict each other, the most recent memory is the source "
    "of truth\n"
    "- Perform logical inference: when evidence strongly suggests a connection, "
    "state it\n"
    "- NEVER omit specific names, dates, numbers, or amounts\n"
    "- ALWAYS include exact details as they appear in the memories\n\n"
    "Memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible answer "
    "(usually 1-10 words). For yes/no questions answer `Yes` or `No`. For "
    "dates use the exact format from memories (e.g., `7 May 2023`). For lists "
    "use comma-separated terse items. Do NOT write full sentences, do NOT "
    "repeat the question, do NOT add explanations after the answer. If the "
    "memories don't contain the answer, write exactly `FINAL ANSWER: no "
    "information available`."
)

ZEP_ANSWER_PROMPT = (
    "You are a helpful assistant with access to conversation facts and "
    "entity information.\n\n"
    "FACTS:\n{facts}\n\n"
    "ENTITIES:\n{entities}\n\n"
    "Use these facts and entity summaries to answer the question.\n"
    "For temporal questions, pay attention to event_time fields.\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible answer "
    "(usually 1-10 words). For yes/no questions answer `Yes` or `No`. For "
    "dates use the exact format from memories (e.g., `7 May 2023`). For lists "
    "use comma-separated terse items. Do NOT write full sentences, do NOT "
    "repeat the question, do NOT add explanations after the answer. If the "
    "facts don't contain the answer, write exactly `FINAL ANSWER: no "
    "information available`."
)

EVERMEMOS_ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the SHORTEST POSSIBLE answer — usually 1-10 words. For yes/no questions answer "Yes" or "No". For dates use the exact format from memories (e.g., "7 May 2023"). For lists use comma-separated terse items. For "what kind of X" or "how" questions answer with the most direct phrase from memories. Do NOT write full sentences, do NOT add explanations, do NOT repeat the question. If the answer is not in the memories, reply exactly: "no information available".]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""

LETTA_ANSWER_PROMPT = (
    "You are a helpful assistant with a three-tier memory system:\n"
    "- Core memory: key facts always available\n"
    "- Archival memory: long-term stored passages\n"
    "- Recall memory: recent conversation history\n\n"
    "Retrieved context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer based on your memories."
)

# Mirrors MEM0_ANSWER_PROMPT's contract: synthesize across chunks,
# allow logical inference when evidence strongly suggests a connection,
# and emit a single-line ``FINAL ANSWER:`` so ``_extract_final_answer``
# can strip CoT before F1. Without the inference clause the model
# refuses every cat-3 (commonsense) question in lockstep ("the context
# does not contain the answer") and cat-3 accuracy collapses to 0%.
# Without the FINAL ANSWER line F1 token-overlap drowns in disclaimer
# text. Both fixes were paid for once already on the memory-system
# prompts; the same lift applies to RAG.
RAG_ANSWER_PROMPT = (
    "You are a helpful assistant answering questions about a long "
    "personal conversation, using only the retrieved context chunks "
    "below as evidence. Each chunk includes a relevance score.\n\n"
    "# INSTRUCTIONS:\n"
    "- Synthesize information across chunks to build a complete answer\n"
    "- Convert relative time references (e.g. 'last year', 'next month') "
    "to specific dates using the chunk timestamps when present\n"
    "- When chunks contradict each other, the most recent chunk is the "
    "source of truth\n"
    "- Perform logical inference: when evidence strongly suggests a "
    "connection (e.g. someone mentions wanting to be a counselor and "
    "previously received support → would they pursue counseling?), "
    "state it. Do NOT dismiss reasonable inferences as 'the context "
    "doesn't say'\n"
    "- NEVER omit specific names, dates, numbers, or amounts\n"
    "- ALWAYS include exact details as they appear in the chunks\n\n"
    "Context chunks:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single "
    "line starting with `FINAL ANSWER:` followed by the SHORTEST "
    "possible answer (usually 1-10 words). For yes/no questions answer "
    "`Yes` or `No`. For dates use the format from the chunks (e.g., "
    "`7 May 2023`). For lists use comma-separated terse items. Do NOT "
    "write full sentences, do NOT repeat the question, do NOT add "
    "explanations after the answer. If the chunks genuinely don't "
    "support any reasonable inference (e.g. a question about a person "
    "or topic never mentioned in the conversation), write exactly "
    "`FINAL ANSWER: no information available`."
)

MASTRA_ANSWER_PROMPT = (
    "You are a helpful assistant with access to a multi-layered memory system.\n"
    "Your memory consists of:\n"
    "- Observations: consolidated synthesized facts with date stamps\n"
    "- Working memory: a structured scratchpad of key facts about the user\n"
    "- Semantic recall + recent history: a single chronological message list\n"
    "  where each line is prefixed with its source-conversation timestamp\n\n"
    "Use all available memory layers to answer the question.\n"
    "For questions about user preferences or profile, prioritize working memory.\n"
    "For questions about specific past conversations, prioritize semantic recall.\n"
    "For questions about recent context, prioritize message history.\n"
    "For temporal questions, the bracketed timestamps next to each message and\n"
    "observation are from the conversation's own clock — ground dates there,\n"
    "NOT today's date.\n\n"
    "Retrieved memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible answer "
    "(usually 1-10 words). For yes/no questions answer `Yes` or `No`. For "
    "dates use the exact format from memories (e.g., `7 May 2023`). For lists "
    "use comma-separated terse items. Do NOT write full sentences, do NOT "
    "repeat the question, do NOT add explanations after the answer. If the "
    "memories don't contain the answer, write exactly `FINAL ANSWER: no "
    "information available`."
)

GRAPHITI_ANSWER_PROMPT = (
    "You are a helpful assistant with access to a temporally-aware knowledge graph.\n"
    "Your memory contains:\n"
    "- Facts: typed relationships between entities (e.g. Alice WORKS_AT Acme Corp)\n"
    "- Entity summaries: dense factual profiles of each known person/place/thing\n\n"
    "Memory is bi-temporal: outdated facts are invalidated when new information\n"
    "arrives, so only the most recent active facts are shown.\n"
    "Each fact includes source entities and a relevance score.\n\n"
    "For questions about current state, prefer higher-scored active facts.\n"
    "For questions about past events, reason over all provided facts.\n\n"
    "Retrieved memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible answer "
    "(usually 1-10 words). For yes/no questions answer `Yes` or `No`. For "
    "dates use the exact format from memories (e.g., `7 May 2023`). For lists "
    "use comma-separated terse items. Do NOT write full sentences, do NOT "
    "repeat the question, do NOT add explanations after the answer. If the "
    "memories don't contain the answer, write exactly `FINAL ANSWER: no "
    "information available`."
)

HINDSIGHT_ANSWER_PROMPT = (
    "You are a helpful assistant with access to a structured memory system.\n"
    "Your memory contains three types of facts:\n"
    "- World facts: information about the environment and other people\n"
    "- Experience facts: events and interactions you have participated in\n"
    "- Observations: consolidated knowledge synthesized from multiple facts\n\n"
    "Each memory includes a relevance score and may have an event date "
    "(in `date=YYYY-MM-DD` form) anchored to the source conversation's\n"
    "clock — NOT today's date.\n\n"
    "For temporal questions, ground dates in the `date=` field of the\n"
    "retrieved memories. NEVER substitute today's date for an event's\n"
    "actual date.\n"
    "For factual questions, prefer observations (consolidated knowledge) "
    "over raw facts when available.\n\n"
    "Retrieved memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible answer "
    "(usually 1-10 words). For yes/no questions answer `Yes` or `No`. For "
    "dates use the exact format from memories (e.g., `7 May 2023`). For lists "
    "use comma-separated terse items. Do NOT write full sentences, do NOT "
    "repeat the question, do NOT add explanations after the answer. If the "
    "memories don't contain the answer, write exactly `FINAL ANSWER: no "
    "information available`."
)

# Same shape as MEM0_ANSWER_PROMPT — the Mem0 v2 paper prompt is the
# best-tuned LoCoMo answer prompt available, and memos returns memory
# items in a comparable "list of opaque memory strings" shape. The
# middleware prefixes each retrieved memory with ``[<timestamp>]`` from
# the source conversation's clock when available (see deviations.md
# O27); the prompt below names that contract so the LLM grounds
# temporal questions there instead of today's wall-clock — the same
# trap H46 fixed for Hindsight and M38 fixed for Mastra.
MEMOS_ANSWER_PROMPT = (
    "You are an intelligent memory assistant that uses memories from past "
    "conversations to answer questions accurately.\n\n"
    "Each memory may be prefixed with a bracketed timestamp `[<time>]` "
    "from the source conversation's own clock.\n\n"
    "# INSTRUCTIONS:\n"
    "- Synthesize information across different memories to build a "
    "complete answer\n"
    "- Convert relative time references (e.g. 'last year', 'next month') "
    "to specific dates using the bracketed timestamps on the memories — "
    "NEVER substitute today's date for an event whose memory carries a "
    "timestamp\n"
    "- When memories contradict each other, the most recent memory is "
    "the source of truth\n"
    "- Perform logical inference: when evidence strongly suggests a "
    "connection, state it\n"
    "- NEVER omit specific names, dates, numbers, or amounts\n"
    "- ALWAYS include exact details as they appear in the memories\n\n"
    "Retrieved memories:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then output your final response on a single line "
    "starting with `FINAL ANSWER:` followed by the SHORTEST possible "
    "answer (usually 1-10 words). For yes/no questions answer `Yes` or "
    "`No`. For dates use the exact format from memories (e.g., "
    "`7 May 2023`). For lists use comma-separated terse items. Do NOT "
    "write full sentences, do NOT repeat the question, do NOT add "
    "explanations after the answer. If the memories don't contain the "
    "answer, write exactly `FINAL ANSWER: no information available`."
)

# ---------------------------------------------------------------------------
# Provider-specific judge prompts
# ---------------------------------------------------------------------------

ZEP_JUDGE_SYSTEM = (
    "You are a generous and fair judge evaluating answers about conversations.\n"
    "Be especially lenient when:\n"
    "- The answer captures the core meaning but uses different words\n"
    "- The answer provides additional context beyond the golden answer\n"
    "- Names or minor details differ but the substance is correct\n\n"
    "Only mark WRONG if the answer is clearly factually incorrect or "
    "completely misses the point."
)

# ---------------------------------------------------------------------------
# Registry: provider name -> prompt template
# ---------------------------------------------------------------------------

_ANSWER_PROMPTS: dict[str, str] = {
    "mem0v1_api": MEM0_ANSWER_PROMPT,
    "mem0v1_embedded": MEM0_ANSWER_PROMPT,
    "mem0v2_api": MEM0_ANSWER_PROMPT,
    "mem0v2_embedded": MEM0_ANSWER_PROMPT,
    "zep_api": ZEP_ANSWER_PROMPT,
    "zep_embedded": GRAPHITI_ANSWER_PROMPT,
    "evermemos_api": EVERMEMOS_ANSWER_PROMPT,
    "evermemos_embedded": EVERMEMOS_ANSWER_PROMPT,
    "graphiti_embedded": GRAPHITI_ANSWER_PROMPT,
    "letta_api": LETTA_ANSWER_PROMPT,
    "rag": RAG_ANSWER_PROMPT,
    "hindsight_api": HINDSIGHT_ANSWER_PROMPT,
    "hindsight_embedded": HINDSIGHT_ANSWER_PROMPT,
    "mastra_api": MASTRA_ANSWER_PROMPT,
    "mastra_embedded": MASTRA_ANSWER_PROMPT,
    "memos_embedded": MEMOS_ANSWER_PROMPT,
}

_JUDGE_PROMPTS: dict[str, str] = {
    "zep_api": ZEP_JUDGE_SYSTEM,
}


def get_answer_prompt(provider: str) -> str:
    """Get the answer prompt template for a memory provider."""
    return _ANSWER_PROMPTS.get(provider, DEFAULT_ANSWER_PROMPT)


def get_judge_system_prompt(provider: str) -> str:
    """Get the judge system prompt for a memory provider."""
    return _JUDGE_PROMPTS.get(provider, DEFAULT_JUDGE_SYSTEM)


def format_answer_prompt(
    provider: str,
    question: str,
    context: str,
    **extra: Any,
) -> str:
    """Build the answer prompt from the provider template + question + context."""
    template = get_answer_prompt(provider)
    return template.format(
        question=question,
        context=context,
        facts=extra.get("facts", context),
        entities=extra.get("entities", ""),
    )
