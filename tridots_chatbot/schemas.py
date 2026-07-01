from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = None


class ChatSource(BaseModel):
    url: str
    title: str | None = None
    excerpt: str | None = None
    page_type: str | None = None
    score: float | None = None


class LatencyBreakdown(BaseModel):
    rewrite: int = 0
    query_rewrite: int = 0
    embedding: int = 0
    retrieval: int = 0
    retrieval_db: int = 0
    context_assembly: int = 0
    llm: int = 0
    llm_first_token: int | None = None
    llm_full_response: int | None = None
    total: int = 0
    total_end_to_end: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    retrieval_count: int = 0
    suggested_questions: list[str] = Field(default_factory=list)
    latency_ms: LatencyBreakdown


class RetrievedChunk(BaseModel):
    url: str
    title: str
    content: str
    page_type: str | None = None
    section_heading: str | None = None
    token_count: int = 0
    score: float

    model_config = ConfigDict(extra="ignore")


class RetrievalQuery(BaseModel):
    text: str
    embedding: list[float]
    limit: int = 60
    score_threshold: float = 0.25


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)
