from __future__ import annotations

from typing import Protocol

from tridots_chatbot.schemas.models import ChatMessage, RetrievedChunk


class RetrievalQuery:
    text: str
    embedding: list[float]
    limit: int = 60
    score_threshold: float = 0.25


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
