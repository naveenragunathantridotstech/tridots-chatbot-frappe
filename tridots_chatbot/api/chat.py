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


def ensure_chat_session(session_id: str, title: str):
    """Creates a new Chat Session if it doesn't already exist."""
    if not frappe.db.exists("Chat Session", {"session_id": session_id}):
        doc = frappe.get_doc({
            "doctype": "Chat Session",
            "session_id": session_id,
            "title": title,
            "user": frappe.session.user if frappe.session.user != "Guest" else None
        })
        doc.insert(ignore_permissions=True)
        frappe.db.commit()


def save_chat_message(session_id: str, role: str, content: str, latency: str = None, sources: str = None, suggested_questions: str = None):
    """Saves a new Chat Message in the database."""
    session_name = frappe.db.get_value("Chat Session", {"session_id": session_id}, "name")
    if not session_name:
        ensure_chat_session(session_id, title=content[:50])
        session_name = frappe.db.get_value("Chat Session", {"session_id": session_id}, "name")
        
    doc = frappe.get_doc({
        "doctype": "Chat Message",
        "session": session_name,
        "role": role,
        "content": content,
        "latency": latency,
        "feedback": "None",
        "sources": sources,
        "suggested_questions": suggested_questions
    })
    doc.insert(ignore_permissions=True)
    frappe.db.commit()


@frappe.whitelist(allow_guest=True)
def get_sessions():
    """Lists all chat sessions for the active session user."""
    filters = {}
    if frappe.session.user and frappe.session.user != "Guest":
        filters["user"] = frappe.session.user
    
    sessions = frappe.get_all(
        "Chat Session",
        filters=filters,
        fields=["name", "session_id", "title", "modified"],
        order_by="modified desc"
    )
    return {"sessions": sessions}


@frappe.whitelist(allow_guest=True)
def get_session_messages(session_id):
    """Fetches past messages from the Chat Message DocType for a given session."""
    if not session_id:
        return {"messages": []}
    
    messages = frappe.get_all(
        "Chat Message",
        filters={"session": session_id},
        fields=["name", "role", "content", "latency", "feedback", "sources", "suggested_questions", "creation"],
        order_by="creation asc"
    )
    
    parsed_messages = []
    for msg in messages:
        sources_parsed = []
        if msg.get("sources"):
            try:
                sources_parsed = json.loads(msg["sources"])
            except Exception:
                pass
        
        suggestions_parsed = []
        if msg.get("suggested_questions"):
            try:
                suggestions_parsed = json.loads(msg["suggested_questions"])
            except Exception:
                pass
                
        parsed_messages.append({
            "id": msg["name"],
            "role": msg["role"].lower(),
            "content": msg["content"],
            "latency": json.loads(msg["latency"]) if msg.get("latency") else None,
            "feedback": msg["feedback"].lower() if msg.get("feedback") else "none",
            "sources": sources_parsed,
            "suggestedQuestions": suggestions_parsed,
            "timestamp": int(msg["creation"].timestamp() * 1000) if msg.get("creation") else None
        })
    return {"messages": parsed_messages}


@frappe.whitelist(allow_guest=True)
def update_message_feedback():
    """Updates the feedback rating of a Chat Message."""
    data = frappe.local.form_dict
    raw = frappe.request.data
    if raw:
        body = json.loads(raw)
    else:
        body = dict(data)
        
    message_id = body.get("message_id")
    rating = body.get("rating")
    
    if not message_id:
        frappe.local.response["http_status_code"] = 400
        return {"error": "message_id is required"}
        
    formatted_rating = "None"
    if rating:
        rating_lower = rating.lower()
        if rating_lower == "positive":
            formatted_rating = "Positive"
        elif rating_lower == "negative":
            formatted_rating = "Negative"
            
    try:
        doc = frappe.get_doc("Chat Message", message_id)
        doc.feedback = formatted_rating
        doc.save(ignore_permissions=True)
        frappe.db.commit()
        return {"status": "success"}
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "tridots_chatbot.update_message_feedback")
        frappe.local.response["http_status_code"] = 500
        return {"error": str(e)}


@frappe.whitelist(allow_guest=True)
def chat_stream():
    """HTTP streaming endpoint returning text/event-stream."""
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
    
    # Ensure session and user message are saved
    ensure_chat_session(session_id, title=message[:50])
    save_chat_message(session_id, "User", message)
    
    request = ChatRequest(
        message=message,
        conversation_history=conversation_history,
        session_id=session_id
    )
    
    def event_generator():
        started = time.monotonic()
        backend = _build_backend()
        
        # Stream phase retrieving
        yield f"data: {json.dumps({'type': 'phase', 'phase': 'retrieving'})}\n\n"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            retrieval_task = loop.run_until_complete(backend.retriever.retrieve(request))
            sources_list = [
                {"title": chunk.title, "url": chunk.url, "score": chunk.score} for chunk in retrieval_task.chunks[:5]
            ]
            # Stream sources
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_list})}\n\n"
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), "tridots_chatbot.retrieval_error")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Retrieval failed'})}\n\n"
            return
            
        # Stream phase streaming
        yield f"data: {json.dumps({'type': 'phase', 'phase': 'streaming'})}\n\n"
        
        full_answer = ""
        try:
            async def iterate_stream():
                nonlocal full_answer
                async for token in backend.answer_generator.stream_complete(request, retrieval_task):
                    full_answer += token
                    yield token
                    
            gen = iterate_stream()
            while True:
                try:
                    token = loop.run_until_complete(gen.__anext__())
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                except StopAsyncIteration:
                    break
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), "tridots_chatbot.generator_error")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Generation failed'})}\n\n"
            return
            
        latency_ms = merge_latency(
            {"embedding": retrieval_task.latency_ms.embedding, "retrieval": retrieval_task.latency_ms.retrieval,
             "context_assembly": retrieval_task.latency_ms.context_assembly},
            {"llm": 0},
            total_ms=int((time.monotonic() - started) * 1000),
        )
        yield f"data: {json.dumps({'type': 'latency', 'ms': latency_ms})}\n\n"
        
        suggested_questions = []
        try:
            async def _gen_followups():
                return await backend.answer_generator.generate_followups(
                    conversation_history=conversation_history,
                    last_answer=full_answer,
                    context_text=None,
                )
            suggested_questions = loop.run_until_complete(_gen_followups())
            yield f"data: {json.dumps({'type': 'suggestions', 'questions': suggested_questions})}\n\n"
        except Exception:
            frappe.log_error(frappe.get_traceback(), "tridots_chatbot.followup_error")
            
        # Save Assistant response with metadata
        save_chat_message(
            session_id, 
            "Assistant", 
            full_answer, 
            latency=json.dumps(latency_ms),
            sources=json.dumps(sources_list),
            suggested_questions=json.dumps(suggested_questions)
        )
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        loop.close()
        
    from werkzeug.wrappers import Response
    response = Response(event_generator(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    
    frappe.local.response_binary = True
    return response


@frappe.whitelist(allow_guest=True)
def chat():
    """Fallback standard POST chat endpoint for backward compatibility."""
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

    try:
        service = _build_backend()
        response = frappe._dict({})

        async def _run():
            nonlocal response
            result = await service.chat(request)
            response = result

        import asyncio
        asyncio.run(_run())

        return {
            "answer": response.answer,
            "sources": [s.model_dump() for s in response.sources],
            "retrieval_count": response.retrieval_count,
            "suggested_questions": [],
            "session_id": session_id,
        }
    except Exception:
        frappe.log_error(frappe.get_traceback(), "tridots_chatbot.chat_fallback")
        frappe.local.response["http_status_code"] = 500
        return {"error": "Something went wrong. Please try again."}


def after_migrate():
    pass
