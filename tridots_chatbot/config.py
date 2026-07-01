from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    @property
    def groq_api_key(self) -> str:
        import frappe
        if frappe.local and getattr(frappe.local, "site", None):
            return frappe.conf.get("groq_api_key") or os.environ.get("GROQ_API_KEY", "")
        return os.environ.get("GROQ_API_KEY", "")

    @property
    def mistral_api(self) -> str:
        import frappe
        if frappe.local and getattr(frappe.local, "site", None):
            return frappe.conf.get("mistral_api") or os.environ.get("MISTRAL_API_KEY", "")
        return os.environ.get("MISTRAL_API_KEY", "")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    disable_reranker: bool = True
    retrieval_limit: int = 60
    source_limit: int = 5
    score_threshold: float = 0.25
    context_token_budget: int = 4000
    max_chunks_per_url: int = 5
    @property
    def vectors_path(self) -> Path:
        import frappe
        if frappe.local and getattr(frappe.local, "site", None):
            return Path(frappe.get_site_path("private", "files", "vectors.json"))
        return Path(os.environ.get("VECTORS_PATH", "../vectors.json"))

    @property
    def scraped_content_dir(self) -> Path:
        import frappe
        if frappe.local and getattr(frappe.local, "site", None):
            return Path(frappe.get_site_path("private", "files", "scraped_content"))
        return Path(os.environ.get("SCRAPED_CONTENT_DIR", "../scraped_content"))

    @property
    def log_file(self) -> Path:
        import frappe
        if frappe.local and getattr(frappe.local, "site", None):
            return Path(frappe.get_site_path("private", "files", "chat_logs.jsonl"))
        return Path(os.environ.get("CHAT_LOG_FILE", "../logs/chat_logs.jsonl"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
