from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tridots_chatbot.schemas import RetrievedChunk, RetrievalQuery
from tridots_chatbot.rag import Retriever


STOP_WORDS = {
    "what", "do", "u", "you", "have", "any", "offer", "offering", "does", "tridotstech",
    "tridots", "tech", "company", "is", "a", "the", "to", "for", "in", "of", "and", "or",
    "with", "on", "at", "by", "from", "about", "we",
}


class NumpyRetriever(Retriever):
    def __init__(self, vectors_path: str | Path) -> None:
        self.vectors_path = Path(vectors_path)
        self._loaded = False
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    def load(self) -> None:
        if self._loaded:
            return
        data = json.loads(self.vectors_path.read_text(encoding="utf-8"))
        self._chunks = data.get("chunks", data)  # support both {"chunks": [...]} and flat arrays
        if not self._chunks:
            self._embeddings = np.empty((0, 384), dtype=np.float64)
            self._norms = np.empty(0, dtype=np.float64)
            self._loaded = True
            return
        embeddings_list = [chunk["embedding"] for chunk in self._chunks]
        self._embeddings = np.array(embeddings_list, dtype=np.float64)
        self._norms = np.linalg.norm(self._embeddings, axis=1)
        self._loaded = True

    def _keywords_in_text(self, query: str, text: str) -> float:
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', query.lower())
        keywords = [w for w in words if w not in STOP_WORDS]
        if not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords) if keywords else 0.0

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._retrieve_sync, query)

    def _retrieve_sync(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.load()

        if not self._chunks:
            return []

        query_vec = np.array(query.embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        similarities = np.dot(self._embeddings, query_vec) / (self._norms * query_norm)

        top_indices = np.argsort(similarities)[::-1]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < query.score_threshold:
                continue

            chunk = self._chunks[idx]
            keyword_bonus = self._keywords_in_text(query.text, chunk.get("text", ""))
            final_score = score + 0.15 * keyword_bonus

            results.append(RetrievedChunk(
                url=chunk.get("url", ""),
                title=chunk.get("title", ""),
                content=chunk.get("text", ""),
                page_type=chunk.get("page_type"),
                section_heading=chunk.get("section_heading"),
                token_count=chunk.get("token_count", 0),
                score=final_score,
            ))

        results.sort(key=lambda c: c.score, reverse=True)
        return results[: query.limit]
