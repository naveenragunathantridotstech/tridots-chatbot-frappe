from __future__ import annotations

import asyncio
from typing import Any

from tridots_chatbot.rag import QueryEmbedder


class FastEmbedQueryEmbedder(QueryEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._encoder: Any = None

    @property
    def encoder(self):
        if self._encoder is None:
            from fastembed import TextEmbedding
            self._encoder = TextEmbedding(self._model_name, threads=1)
        return self._encoder

    async def embed(self, text: str) -> list[float]:
        embeddings = await asyncio.to_thread(self._embed_sync, text)
        return embeddings

    def _embed_sync(self, text: str) -> list[float]:
        embeddings = list(self.encoder.query_embed([text]))
        return [float(val) for val in embeddings[0]]
