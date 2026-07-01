from __future__ import annotations

import json
import time
import uuid
from typing import Any

import frappe
from frappe import _

from tridots_chatbot.config import get_settings
from tridots_chatbot.schemas.models import ChatMessage, ChatRequest
from tridots_chatbot.rag.embedder import FastEmbedQueryEmbedder
from tridots_chatbot.rag.retriever import NumpyRetriever
from tridots_chatbot.rag.generator import GroqAnswerGenerator
from tridots_chatbot.rag.pipeline import RAGService, build_sources, FALLBACK_ANSWER
from tridots_chatbot.rag.streaming import merge_latency
from tridots_chatbot.utils.logging import JsonlInteractionLogger, InteractionLogBuilder, to_plain_dict


_BACKEND_CACHE: dict[str, Any] = {}


def _build_backend():
    if "backend" in _BACKEND_CACHE:
        return _BACKEND_CACHE["backend"]

    settings = get_settings()
    embedder = FastEmbedQueryEmbedder(settings.embedding_model)
    retriever = NumpyRetriever(settings.vectors_path)
    answer_gen = GroqAnswerGenerator(settings)

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        answer_generator=answer_gen,
        rewriter=None,
        reranker=None,
        retrieval_limit=settings.retrieval_limit,
        source_limit=settings.source_limit,
        score_threshold=settings.score_threshold,
        context_token_budget=settings.context_token_budget,
        max_chunks_per_url=settings.max_chunks_per_url,
    )
    _BACKEND_CACHE["backend"] = service
    _BACKEND_CACHE["logger"] = JsonlInteractionLogger(settings.log_file)
    return service


def _get_logger():
    if "logger" in _BACKEND_CACHE:
        return _BACKEND_CACHE["logger"]
    settings = get_settings()
    logger = JsonlInteractionLogger(settings.log_file)
    _BACKEND_CACHE["logger"] = logger
    return logger


@frappe.whitelist(allow_guest=True)
def chat():
    """POST /api/method/tridots_chatbot.api.chat

    Request body (JSON):
        message: str
        conversation_history: list[{"role": str, "content": str}] (optional)
        session_id: str (optional)
    """
    data = frappe.local.form_dict
    raw = frappe.request.data
    if raw:
        body = json.loads(raw)
    else:
        body = dict(data)

    message = body.get("message", "").strip()
    if not message:
        frappe.local.response["http_status_code"] = 400
        return {"error": "message is required"}

    history_raw = body.get("conversation_history", []) or []
    conversation_history = []
    for msg in history_raw:
        if isinstance(msg, dict):
            conversation_history.append(ChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")))

    session_id = body.get("session_id") or str(uuid.uuid4())

    request = ChatRequest(
        message=message,
        conversation_history=conversation_history,
        session_id=session_id,
    )

    started = time.monotonic()
    log_builder = InteractionLogBuilder(
        raw_query=message,
        conversation_turns=len(conversation_history),
        session_id=session_id,
    )

    try:
        service = _build_backend()
        response = frappe._dict({})

        async def _run():
            nonlocal response
            result = await service.chat(request)
            response = result

        import asyncio
        asyncio.run(_run())

        latency = merge_latency(
            {"embedding": response.latency_ms.embedding, "retrieval": response.latency_ms.retrieval,
             "context_assembly": response.latency_ms.context_assembly},
            {"llm": response.latency_ms.llm},
            total_ms=int((time.monotonic() - started) * 1000),
        )

        log_builder.set_pipeline(
            rewritten_query=message,
            rewrite_skipped=True,
            retrieved_chunks=[s.model_dump() for s in response.sources],
            context_tokens=sum(getattr(s, "token_count", 0) or 0 for s in response.sources),
        )
        log_builder.finalize_output(
            answer=response.answer,
            sources=[to_plain_dict(s) for s in response.sources],
        )
        log_builder.set_latency(latency)
        _get_logger().log(log_builder.build())

        suggested_questions: list[str] = []
        try:
            async def _gen_followups():
                return await service.answer_generator.generate_followups(
                    conversation_history=conversation_history,
                    last_answer=response.answer,
                    context_text=None,
                )
            suggested_questions = asyncio.run(_gen_followups())
        except Exception:
            frappe.log_error(frappe.get_traceback(), "tridots_chatbot.followups")

        return {
            "answer": response.answer,
            "sources": [s.model_dump() for s in response.sources],
            "retrieval_count": response.retrieval_count,
            "suggested_questions": suggested_questions,
            "session_id": session_id,
            "latency_ms": latency,
        }
    except Exception:
        frappe.log_error(frappe.get_traceback(), "tridots_chatbot.chat")
        frappe.local.response["http_status_code"] = 500
        return {"error": "Something went wrong. Please try again."}


def after_migrate():
    """Called after Frappe migration — a no-op for now."""
    pass
