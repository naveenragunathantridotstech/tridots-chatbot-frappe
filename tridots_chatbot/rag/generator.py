from __future__ import annotations

import asyncio
import json

import httpx

from tridots_chatbot.schemas.models import ChatMessage, RetrievedChunk
from tridots_chatbot.rag.protocols import AnswerGenerator
from tridots_chatbot.rag.pipeline import FOLLOWUP_PROMPT
from tridots_chatbot.config import get_settings


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


def _is_mistral_model(model: str) -> bool:
    return "mistral" in model or "ministral" in model


def _get_api_config(settings, model: str) -> tuple[str, dict[str, str]] | None:
    if _is_mistral_model(model):
        if not settings.mistral_api:
            return None
        return MISTRAL_API_URL, {
            "Authorization": f"Bearer {settings.mistral_api}",
            "Content-Type": "application/json",
        }
    else:
        if not settings.groq_api_key:
            return None
        return GROQ_API_URL, {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _build_messages(
    system_prompt: str,
    conversation_history: list[ChatMessage],
    user_message: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": m.role, "content": m.content} for m in conversation_history)
    messages.append({"role": "user", "content": user_message})
    return messages


class GroqAnswerGenerator(AnswerGenerator):
    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ChatMessage],
        sources: list[RetrievedChunk],
    ) -> str:
        models = [
            "mistral-small-latest",
            "ministral-3b-latest",
            "mistral-large-2512",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]
        last_exc = None

        for model in models:
            config = _get_api_config(self.settings, model)
            if not config:
                continue
            url, headers = config
            for attempt in range(3):
                try:
                    response = await self.client.post(
                        url,
                        headers=headers,
                        json={
                            "model": model,
                            "temperature": 0.2,
                            "max_tokens": 1536,
                            "messages": _build_messages(system_prompt, conversation_history, user_message),
                        },
                    )
                    response.raise_for_status()
                    return _extract_content(response.json())
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if exc.response.status_code == 429:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    break
                except httpx.RequestError as exc:
                    last_exc = exc
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
        if last_exc:
            raise last_exc
        raise RuntimeError("failed to generate response from any model")

    async def generate_followups(
        self,
        *,
        conversation_history: list[ChatMessage],
        last_answer: str,
        context_text: str | None = None,
    ) -> list[str]:
        models = [
            "ministral-3b-latest",
            "mistral-small-latest",
            "llama-3.1-8b-instant",
        ]
        formatted = "\n".join(f"{m.role.title()}: {m.content}" for m in conversation_history)
        context_header = f"Context Documents:\n{context_text}\n\n" if context_text else ""
        user_content = f"{context_header}Conversation:\n{formatted}\n\nLast answer:\n{last_answer}"
        last_exc = None

        for model in models:
            config = _get_api_config(self.settings, model)
            if not config:
                continue
            url, headers = config
            try:
                response = await self.client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "max_tokens": 256,
                        "messages": [
                            {"role": "system", "content": FOLLOWUP_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                    },
                )
                response.raise_for_status()
                text = _extract_content(response.json())
                questions = [q.strip().strip('"').strip("'") for q in text.strip().split("\n") if q.strip()]
                return [q for q in questions if q.endswith("?")][:3]
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exc = e
                continue

        if last_exc:
            import frappe
            frappe.log_error(f"Follow-up generation failed: {last_exc}", "tridots_chatbot.followups")
        return []

    async def stream_complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ChatMessage],
    ):
        models = [
            "mistral-small-latest",
            "ministral-3b-latest",
            "mistral-large-2512",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]
        for model in models:
            config = _get_api_config(self.settings, model)
            if not config:
                continue
            url, headers = config
            for attempt in range(3):
                try:
                    async with self.client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json={
                            "model": model,
                            "temperature": 0.2,
                            "max_tokens": 1536,
                            "stream": True,
                            "messages": _build_messages(system_prompt, conversation_history, user_message),
                        },
                    ) as response:
                        if response.status_code == 429:
                            raise httpx.HTTPStatusError("Rate limit", request=response.request, response=response)
                        if response.status_code != 200:
                            await response.aread()
                            raise httpx.HTTPStatusError(
                                f"HTTP status error {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            payload = line[6:].strip()
                            if payload == "[DONE]":
                                break
                            token = _extract_stream_token(payload)
                            if token:
                                yield token
                        return
                except (httpx.HTTPStatusError, httpx.RequestError):
                    continue


def _extract_content(payload: dict) -> str:
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def _extract_stream_token(payload: str) -> str:
    data = json.loads(payload)
    return data.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
