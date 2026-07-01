from __future__ import annotations

import asyncio
from typing import Any

from tridots_chatbot.schemas.models import RetrievedChunk
from tridots_chatbot.rag.protocols import Reranker


class FastEmbedReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._encoder: Any = None

    @property
    def encoder(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._encoder = TextCrossEncoder(self._model_name, threads=1)
        return self._encoder

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._rerank_sync, query, chunks)

    def _rerank_sync(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        texts = [chunk.content for chunk in chunks]
        scores = list(self.encoder.rerank(query, texts))

        rescored = []
        for chunk, score in zip(chunks, scores):
            rescored.append(chunk.model_copy(update={"score": float(score)}))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
