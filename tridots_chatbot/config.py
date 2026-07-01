from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    mistral_api: str = os.environ.get("MISTRAL_API_KEY", "")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    disable_reranker: bool = True
    retrieval_limit: int = 60
    source_limit: int = 5
    score_threshold: float = 0.25
    context_token_budget: int = 4000
    max_chunks_per_url: int = 5
    vectors_path: Path = Path(
        os.environ.get("VECTORS_PATH", "../vectors.json")
    )
    scraped_content_dir: Path = Path(
        os.environ.get("SCRAPED_CONTENT_DIR", "../scraped_content")
    )
    log_file: Path = Path(
        os.environ.get("CHAT_LOG_FILE", "../logs/chat_logs.jsonl")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
