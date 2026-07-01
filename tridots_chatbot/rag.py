from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Protocol

from tridots_chatbot.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSource,
    LatencyBreakdown,
    RetrievedChunk,
    RetrievalQuery,
)

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
Read the instructions below carefully before generating your answer. They define how you must structure and format your response.

1. CONVERSATIONAL PARTITIONING:
   - When generating an interactive widget (described below), you MUST strictly partition your response:
     - Part 1: Brief conversational introduction (MAXIMUM 2 sentences).
     - Part 2: The `openui` component code block containing all structured details/comparisons/math.
     - Part 3: Brief conversational conclusion inviting the user to interact with the widget (MAXIMUM 1 sentence).
   - STRICTLY PROHIBITED: Do not write any Markdown tables, bulleted lists, numbered lists, or detailed comparisons in your conversational text. All structured information must reside EXCLUSIVELY inside the `openui` code block. This prevents duplication and data truncation.

2. RESPONSIVE DYNAMIC GENERATIVE UI:
   - If the user asks about processes, ROI, calculators, service details, pricing plans, comparisons, or module mappings, generate a dynamic interface block using the OpenUI Lang syntax.
   - To output the interface, output a codeblock of type `openui` containing line-oriented assignments. Every declaration must be on its own line: `varName = ComponentName(arguments)`.
   - You must end the codeblock with a `root = Stack(...)` or `root = Grid(...)` component.

Component Library Spec:
- Stack(children=[var1, var2], spacing="sm"|"md"|"lg", direction="vertical"|"horizontal")
- Grid(children=[var1, var2], columns=1|2|3)
- Card(title="...", subtitle="...", highlight=true|false, children=[...]) (Highlight=true applies brand blue gradient border)
- StatCard(value="...", label="...", description="...") (Renders large stat number/percentage)
- Timeline(steps=[step1, step2, ...]) (Renders vertical connector progress. Steps must be Step components)
- Step(title="...", desc="...") (timeline step)
- ComparisonTable(headers=["Column1", "Column2"], rows=[row1, row2, ...])
- Row(cells=["Cell1", "Cell2"]) (table row cells)
- Slider(label="...", min=10, max=1000, value=100, step=10) (Dynamic input slider, binds current value to its variable name)
- Formula(label="...", formula="var1 * var2", format="currency"|"percentage"|"number") (Dynamic output cell. Computes using safe math, currency formats to INR ₹)
- Accordion(title="...", children=[...]) (FAQ drop-down)
- Button(label="...", url="...", highlight=true|false) (Highlight=true applies green accent background)

Examples:

User Query: Estimate my potential cost savings.
Assistant Response:
Here is a dynamic ROI savings calculator to estimate your annual savings based on your team size and hours saved:

```openui
s1 = Slider(label="Total Employees", min=10, max=500, value=100, step=5)
s2 = Slider(label="Weekly Hours Saved per Employee", min=1, max=40, value=5, step=1)
savings = Formula(label="Estimated Annual Savings (₹)", formula="s1 * s2 * 500 * 52", format="currency")
btn = Button(label="Book a free consultation", url="contact@tridotstech.com", highlight=true)
root = Stack(children=[s1, s2, savings, btn], spacing="md")
```
Please adjust the sliders above to calculate your custom savings estimate.

User Query: Compare plan A and plan B.
Assistant Response:
Here is a side-by-side comparison of Plan A and Plan B:

```openui
row1 = Row(cells=["ERPNext Integration", "Included", "Add-on"])
row2 = Row(cells=["Custom Dashboard", "1 Included", "Unlimited"])
tbl = ComparisonTable(headers=["Feature", "Starter Plan", "Enterprise Plan"], rows=[row1, row2])
root = Card(title="Plan Comparison Matrix", children=[tbl])
```
Please review the comparison matrix above. Let me know if you would like details on standard features!
"""

QUERY_REWRITE_PROMPT = """Given the conversation below, rewrite the LAST user message as a self-contained question. Use ONLY information present in the conversation. Do not add external assumptions. Return only the rewritten question, no explanation."""


FALLBACK_ANSWER = "I don't have that information — please contact us at contact@tridotstech.com"


class QueryRewriter(Protocol):
    async def rewrite(
        self,
        *,
        prompt: str,
        conversation_history: list[ChatMessage],
        latest_message: str,
    ) -> str: ...


class QueryEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class Retriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


FOLLOWUP_PROMPT = """Based on the conversation, the assistant's last answer, and the provided context documents, suggest 3 short follow-up questions the user might want to ask next.

Requirements:
- CRITICAL: Every suggested question MUST be directly answerable using the provided context documents.
- Do NOT suggest questions asking for details, metrics, examples, or stats not explicitly mentioned in the context.
- Each question must be a natural standalone question (no numbering, no quotes, no bullet points).
- Return exactly one question per line, no empty lines.
- Do NOT include any explanation or prefix, and do not use emojis. """


class AnswerGenerator(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ChatMessage],
        sources: list[RetrievedChunk],
    ) -> str: ...

    async def generate_followups(
        self,
        *,
        conversation_history: list[ChatMessage],
        last_answer: str,
        context_text: str | None = None,
    ) -> list[str]: ...


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
    sources: list[ChatSource] = []
    for chunk in chunks:
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
