from __future__ import annotations

import json
import time
from typing import Any

from tridots_chatbot.utils.logging import InteractionLogBuilder
from tridots_chatbot.schemas.models import ChatMessage
from tridots_chatbot.rag.pipeline import build_sources


def merge_latency(
    retrieval_latency: dict[str, int | float],
    completion_latency: dict[str, int | float] | None = None,
    *,
    first_token_ms: int | None = None,
    total_ms: int | None = None,
) -> dict[str, int | float | None]:
    latency = dict(retrieval_latency)
    if completion_latency:
        latency.update(completion_latency)
    if first_token_ms is not None:
        latency["llm_first_token"] = first_token_ms
    if total_ms is not None:
        latency["total"] = total_ms
        latency.setdefault("total_end_to_end", total_ms)
    return latency


async def stream_chat_events(
    request: Any,
    backend: Any,
    logger: Any | None = None,
) -> AsyncIterator[str]:
    from collections.abc import AsyncIterator

    started = time.monotonic()
    session_id = getattr(request, "session_id", None)

    log_builder = InteractionLogBuilder(
        raw_query=request.message,
        conversation_turns=len(request.conversation_history),
        session_id=session_id,
    )

    yield _sse({"type": "phase", "phase": "retrieving"})

    try:
        retrieval = await backend.retrieve(request)
        log_builder.set_pipeline(
            rewritten_query=retrieval.rewritten_query,
            rewrite_skipped=retrieval.rewrite_skipped,
            retrieved_chunks=retrieval.chunks,
            context_tokens=retrieval.context_tokens,
        )

        first_token_ms: int | None = None
        full_answer_parts: list[str] = []
        async for token in backend.stream_complete(request, retrieval):
            if not token:
                continue
            if first_token_ms is None:
                first_token_ms = int((time.monotonic() - started) * 1000)
            full_answer_parts.append(token)
            yield _sse({"type": "token", "content": token})

        total_ms = int((time.monotonic() - started) * 1000)
        latency = merge_latency(
            retrieval.latency_ms,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
        )
        log_builder.finalize_output(sources=retrieval.sources)
        log_builder.set_latency(latency)

        yield _sse({"type": "sources", "sources": retrieval.sources})
        yield _sse({"type": "latency", "ms": latency})
        yield _sse({"type": "done"})

        if logger is not None:
            logger.log(log_builder.build())
    except Exception:
        import frappe
        frappe.log_error(frappe.get_traceback(), "tridots_chatbot.stream")
        yield _sse({"type": "error", "message": "Something went wrong. Please try again."})


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
