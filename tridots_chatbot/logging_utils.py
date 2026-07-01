from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def preview_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported structured value: {type(value)!r}")


@dataclass(slots=True)
class InteractionLogBuilder:
    raw_query: str
    conversation_turns: int
    session_id: str | None = None
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=utc_now_iso)
    rewritten_query: str | None = None
    rewrite_skipped: bool = True
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    context_tokens: int = 0
    answer_parts: list[str] = field(default_factory=list)
    answer_tokens: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] | None = None

    def set_pipeline(
        self,
        *,
        rewritten_query: str | None,
        rewrite_skipped: bool,
        retrieved_chunks: list[dict[str, Any]],
        context_tokens: int,
    ) -> None:
        self.rewritten_query = rewritten_query
        self.rewrite_skipped = rewrite_skipped
        self.retrieved_chunks = [normalize_chunk_log(chunk) for chunk in retrieved_chunks]
        self.context_tokens = context_tokens

    def append_answer_text(self, text: str) -> None:
        if not text:
            return
        self.answer_parts.append(text)

    def finalize_output(
        self,
        *,
        answer: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        answer_tokens: int | None = None,
    ) -> None:
        final_answer = answer if answer is not None else "".join(self.answer_parts)
        self.answer_parts = [final_answer]
        self.sources = [normalize_source(source) for source in (sources or [])]
        self.answer_tokens = answer_tokens if answer_tokens is not None else estimate_token_count(final_answer)

    def set_latency(self, latency_ms: dict[str, Any]) -> None:
        self.latency_ms = dict(latency_ms)

    def set_error(self, message: str, *, error_type: str = "runtime_error") -> None:
        self.errors = {"type": error_type, "message": message}

    def build(self) -> dict[str, Any]:
        answer = self.answer_parts[0] if self.answer_parts else ""
        return {
            "id": self.interaction_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "input": {
                "raw_query": self.raw_query,
                "conversation_turns": self.conversation_turns,
            },
            "pipeline": {
                "rewritten_query": self.rewritten_query or self.raw_query,
                "rewrite_skipped": self.rewrite_skipped,
                "retrieved_chunks": self.retrieved_chunks,
                "chunks_retrieved": len(self.retrieved_chunks),
                "context_tokens": self.context_tokens,
            },
            "output": {
                "answer": answer,
                "answer_tokens": self.answer_tokens,
                "sources": self.sources,
            },
            "latency_ms": self.latency_ms,
            "errors": self.errors,
        }


def normalize_chunk_log(chunk: dict[str, Any]) -> dict[str, Any]:
    chunk = to_plain_dict(chunk)
    content = str(chunk.get("content") or chunk.get("text") or "")
    return {
        "chunk_id": chunk.get("chunk_id"),
        "url": chunk.get("url"),
        "page_type": chunk.get("page_type"),
        "score": chunk.get("score"),
        "token_count": chunk.get("token_count") or estimate_token_count(content),
        "text_preview": preview_text(content),
    }


def normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    source = to_plain_dict(source)
    return {
        "url": source.get("url"),
        "title": source.get("title"),
        "page_type": source.get("page_type"),
        **({"excerpt": source["excerpt"]} if source.get("excerpt") else {}),
    }


class JsonlInteractionLogger:
    def __init__(self, log_file: str | Path = "logs/chat_logs.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
