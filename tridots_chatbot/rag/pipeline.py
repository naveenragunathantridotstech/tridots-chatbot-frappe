from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

from tridots_chatbot.schemas.models import ChatMessage, ChatRequest, ChatResponse, ChatSource, LatencyBreakdown, RetrievedChunk, RetrievalQuery
from tridots_chatbot.rag.protocols import AnswerGenerator, QueryEmbedder, QueryRewriter, Reranker, Retriever


SYSTEM_PROMPT_TEMPLATE = """You are an expert assistant for Tridots Tech, a Frappe Gold Partner and software development company based in Chennai, India.

You help website visitors understand:
- Tridots Tech's services (ERPNext, custom development, cloud, AI integrations)
- ERPNext features and modules
- Frappe ecosystem (Frappe Framework, Frappe HR, Frappe CRM, etc.)
- Pricing guidance, implementation timelines, case studies

BASIC RULES:
1. Answer ONLY based on the provided context documents.
2. If unsure, say "I don't have that information — please contact us at contact@tridotstech.com"
3. Always be professional, concise, and helpful.
4. When mentioning services, link to the relevant page if available.
5. Never make up pricing, timelines, or technical specs not in the context.
6. Do NOT use any emojis or emoticons in your response. Use plain text only.

CONTEXT:
{retrieved_chunks}

RECENT CONVERSATION:
{last_3_turns}

RESPONSE FORMATTING & STYLING INSTRUCTIONS:
You MUST output your entire response as a valid, structured JSON array of UI blocks. Do NOT output any plain text, markdown blocks, or other content outside of the JSON array. Start with "[" and end with "]".

Every element in the array must be an object representing a block with one of the following schemas:

1. Text Block (for paragraph responses, lists, or standard text):
   {{
     "type": "text",
     "content": "Markdown formatted text goes here. Keep it concise."
   }}

2. Card Grid Block (for showcasing services, plans, or key details side-by-side):
   {{
     "type": "cards",
     "title": "Optional grid section title",
     "cards": [
       {{
         "title": "Card Title",
         "subtitle": "Optional subtitle",
         "content": "Card body description text.",
         "highlight": true
       }}
     ]
   }}

3. Table Block (for comparing plans, features, or metrics):
   {{
     "type": "table",
     "title": "Table Title",
     "headers": ["Header 1", "Header 2"],
     "rows": [
       ["Value 1", "Value 2"],
       ["Value 3", "Value 4"]
     ]
   }}

4. Chart Block (for showing statistics, distributions, growth trends, or graphical metrics):
   {{
     "type": "chart",
     "title": "Chart Title",
     "chart_type": "bar" | "line" | "pie",
     "x_key": "nameOfXAttribute",
     "y_key": "nameOfYAttribute",
     "series": [
       {{"nameOfXAttribute": "Label A", "nameOfYAttribute": 120}},
       {{"nameOfXAttribute": "Label B", "nameOfYAttribute": 80}}
     ]
   }}

5. Button Block (for a Call to Action):
   {{
     "type": "button",
     "label": "Button text",
     "url": "URL or email address",
     "highlight": true
   }}

Example Response 1 (simple text query):
[
  {{
    "type": "text",
    "content": "Tridots Tech is a software company specializing in ERPNext implementations."
  }}
]

Example Response 2 (query requiring structured sections and CTA):
[
  {{
    "type": "text",
    "content": "Here is information about our ERPNext customization services:"
  }},
  {{
    "type": "cards",
    "title": "Customization Modules",
    "cards": [
      {{
        "title": "Manufacturing Customization",
        "subtitle": "Production tracking",
        "content": "We customize work orders and BOM hierarchies for supply chain optimization."
      }},
      {{
        "title": "HR & Payroll Setup",
        "subtitle": "Compliance focus",
        "content": "We configure local tax rules, leave policies, and biometric sync."
      }}
    ]
  }},
  {{
    "type": "button",
    "label": "Book a Free Consultation",
    "url": "contact@tridotstech.com",
    "highlight": true
  }}
]

Example Response 3 (query asking for statistics or category distribution):
[
  {{
    "type": "text",
    "content": "Here is the category distribution of products in our systems:"
  }},
  {{
    "type": "chart",
    "title": "Product Category Distribution",
    "chart_type": "bar",
    "x_key": "category",
    "y_key": "count",
    "series": [
      {{"category": "Tops", "count": 115}},
      {{"category": "Bottoms", "count": 96}},
      {{"category": "Outerwear", "count": 88}}
    ]
  }}
]
"""

QUERY_REWRITE_PROMPT = """Given the conversation below, rewrite the LAST user message as a self-contained question. Use ONLY information present in the conversation. Do not add external assumptions. Return only the rewritten question, no explanation."""


FALLBACK_ANSWER = "I don't have that information — please contact us at contact@tridotstech.com"


FOLLOWUP_PROMPT = """Based on the conversation, the assistant's last answer, and the provided context documents, suggest 3 short follow-up questions the user might want to ask next.

Requirements:
- CRITICAL: Every suggested question MUST be directly answerable using the provided context documents.
- Do NOT suggest questions asking for details, metrics, examples, or stats not explicitly mentioned in the context.
- Each question must be a natural standalone question (no numbering, no quotes, no bullet points).
- Return exactly one question per line, no empty lines.
- Do NOT include any explanation or prefix, and do not use emojis. """


@dataclass(slots=True)
class RetrievalPreparation:
    query_text: str
    rewritten_query: str | None
    rewrite_skipped: bool
    trimmed_history: list[ChatMessage]
    raw_chunks: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]
    context_text: str
    latency: LatencyBreakdown


def trim_conversation_history(
    history: list[ChatMessage],
    *,
    max_pairs: int = 3,
) -> list[ChatMessage]:
    eligible = [
        (index, message)
        for index, message in enumerate(history)
        if message.role in {"user", "assistant"}
    ]
    user_indexes = [index for index, message in eligible if message.role == "user"]

    if len(user_indexes) <= max_pairs:
        return [message for _, message in eligible]

    start_index = user_indexes[-max_pairs]
    return [message for index, message in eligible if index >= start_index]


def format_history(history: list[ChatMessage]) -> str:
    if not history:
        return "No recent conversation."
    return "\n".join(f"{message.role.title()}: {message.content}" for message in history)


def approximate_tokens(text: str) -> int:
    return ceil(len(text) / 4) if text else 0


def chunk_to_context_block(chunk: RetrievedChunk) -> str:
    heading = f" | {chunk.section_heading}" if chunk.section_heading else ""
    return f"[Source: {chunk.title}{heading} | {chunk.url}]\n{chunk.content}\n---"


def dedupe_sources(
    chunks: list[RetrievedChunk], *, max_sources: int = 5, max_per_url: int = 2
) -> list[RetrievedChunk]:
    by_url: dict[str, list[RetrievedChunk]] = {}
    for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
        if chunk.url not in by_url:
            by_url[chunk.url] = []
        if len(by_url[chunk.url]) < max_per_url:
            by_url[chunk.url].append(chunk)
    flattened = [item for items in by_url.values() for item in items]
    return list(sorted(flattened, key=lambda item: item.score, reverse=True))[:max_sources]


def assemble_context(
    chunks: list[RetrievedChunk],
    *,
    token_budget: int = 2000,
) -> tuple[str, list[RetrievedChunk]]:
    ranked = sorted(chunks, key=lambda item: item.score, reverse=True)
    kept = ranked[:]

    while kept:
        context = "\n\n".join(chunk_to_context_block(chunk) for chunk in kept)
        if approximate_tokens(context) <= token_budget:
            return context, kept
        kept.pop()

    return "", []


def build_sources(chunks: list[RetrievedChunk]) -> list[ChatSource]:
    seen: set[str] = set()
    sources: list[ChatSource] = []
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        if chunk.url in seen:
            continue
        seen.add(chunk.url)
        excerpt = " ".join(chunk.content.split())
        sources.append(
            ChatSource(
                url=chunk.url,
                title=chunk.title,
                excerpt=excerpt[:280],
                page_type=chunk.page_type,
                score=round(chunk.score, 4),
            )
        )
    return sources


def build_system_prompt(context_text: str, history: list[ChatMessage]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_chunks=context_text,
        last_3_turns=format_history(history),
    )


class RAGService:
    def __init__(
        self,
        *,
        embedder: QueryEmbedder,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        retrieval_limit: int = 15,
        source_limit: int = 5,
        score_threshold: float = 0.25,
        context_token_budget: int = 2000,
        max_chunks_per_url: int = 3,
        fallback_answer: str = FALLBACK_ANSWER,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.rewriter = rewriter
        self.reranker = reranker
        self.retrieval_limit = retrieval_limit
        self.source_limit = source_limit
        self.score_threshold = score_threshold
        self.context_token_budget = context_token_budget
        self.max_chunks_per_url = max_chunks_per_url
        self.fallback_answer = fallback_answer

    async def prepare(self, request: ChatRequest) -> RetrievalPreparation:
        trimmed_history = trim_conversation_history(request.conversation_history)
        query_text = request.message.strip()
        rewritten_query: str | None = None
        latency = LatencyBreakdown()

        embedding_started = perf_counter()
        embedding = await self.embedder.embed(query_text)
        latency.embedding = elapsed_ms(embedding_started)

        retrieval_started = perf_counter()
        retrieval_query = RetrievalQuery(
            text=query_text,
            embedding=embedding,
            limit=self.retrieval_limit,
            score_threshold=self.score_threshold,
        )
        raw_chunks = await self.retriever.retrieve(retrieval_query)
        latency.retrieval = elapsed_ms(retrieval_started)
        latency.retrieval_db = latency.retrieval

        assembly_started = perf_counter()
        if self.reranker:
            reranked_chunks = await self.reranker.rerank(query_text, raw_chunks)
        else:
            reranked_chunks = raw_chunks
        deduped_chunks = dedupe_sources(
            reranked_chunks,
            max_sources=self.source_limit,
            max_per_url=self.max_chunks_per_url,
        )
        context_text, context_chunks = assemble_context(
            deduped_chunks,
            token_budget=self.context_token_budget,
        )
        latency.context_assembly = elapsed_ms(assembly_started)

        return RetrievalPreparation(
            query_text=query_text,
            rewritten_query=rewritten_query,
            rewrite_skipped=rewritten_query is None,
            trimmed_history=trimmed_history,
            raw_chunks=raw_chunks,
            context_chunks=context_chunks,
            context_text=context_text,
            latency=latency,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        prepared = await self.prepare(request)
        latency = prepared.latency

        if not prepared.context_chunks:
            latency.total = elapsed_ms(started)
            latency.total_end_to_end = latency.total
            return ChatResponse(
                answer=self.fallback_answer,
                sources=[],
                retrieval_count=0,
                latency_ms=latency,
            )

        llm_started = perf_counter()
        system_prompt = build_system_prompt(prepared.context_text, prepared.trimmed_history)
        answer = await self.answer_generator.generate(
            system_prompt=system_prompt,
            user_message=request.message,
            conversation_history=prepared.trimmed_history,
            sources=prepared.context_chunks,
        )
        latency.llm = elapsed_ms(llm_started)
        latency.llm_full_response = latency.llm
        latency.total = elapsed_ms(started)
        latency.total_end_to_end = latency.total

        final_answer = answer.strip() or self.fallback_answer
        return ChatResponse(
            answer=final_answer,
            sources=build_sources(prepared.context_chunks),
            retrieval_count=len(prepared.context_chunks),
            latency_ms=latency,
        )


def elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
