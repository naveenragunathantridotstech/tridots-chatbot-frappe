from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests

from tridots_chatbot.ingest.sitemap import SitemapEntry, filter_scrapable_entries, load_sitemap


DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; TridotsBot/1.0)"
DEFAULT_TIMEOUT = 15
RATE_LIMIT_SECONDS = 0.5
RETRY_WAIT_SECONDS = 10
RETRYABLE_STATUS_CODES = {429, 503}
REMOVABLE_TAGS = ("nav", "footer", "script", "style", "noscript", "header", "iframe")
MAIN_CONTENT_SELECTORS = (
    "main",
    '[role="main"]',
    "article",
    ".page-content",
    ".content",
    "#content",
)
BOILERPLATE_HINTS = (
    "menu",
    "navbar",
    "navigation",
    "breadcrumb",
    "footer",
    "header",
    "sidebar",
    "social",
)


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    file: str
    size_bytes: int
    last_modified: str


def slugify_url_to_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.replace("/", "_") or "home"
    return f"{slug}.md"


def render_frontmatter(url: str, title: str, last_modified: str, scraped_at: str, content: str) -> str:
    return (
        "---\n"
        f"url: {url}\n"
        f"title: {title}\n"
        f"last_modified: {last_modified}\n"
        f"scraped_at: {scraped_at}\n"
        "---\n\n"
        f"{content.rstrip()}\n"
    )


def clean_markdown(text: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = _drop_boilerplate_lines(cleaned)
    return cleaned.strip()


def extract_markdown(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(REMOVABLE_TAGS):
        tag.decompose()

    for node in soup.find_all(attrs={"id": True}):
        if node.attrs is None:
            continue
        node_id = " ".join(str(node.get("id", "")).lower().split())
        if any(hint in node_id for hint in BOILERPLATE_HINTS):
            node.decompose()

    for node in soup.find_all(attrs={"class": True}):
        if node.attrs is None:
            continue
        class_text = " ".join(" ".join(node.get("class", [])).lower().split())
        if any(hint in class_text for hint in BOILERPLATE_HINTS):
            node.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main = None
    for selector in MAIN_CONTENT_SELECTORS:
        main = soup.select_one(selector)
        if main:
            break
    main = main or soup.find("body")
    markdown = markdownify(str(main or ""), heading_style="ATX")
    return title, clean_markdown(markdown)


def _drop_boilerplate_lines(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    filtered: list[str] = []
    link_heavy_streak = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            link_heavy_streak = 0
            filtered.append("")
            continue

        if stripped.startswith("!["):
            continue

        links = re.findall(r"\[[^\]]+\]\([^)]+\)", stripped)
        text_without_links = re.sub(r"\[[^\]]+\]\([^)]+\)", "", stripped).strip()
        link_ratio = (len(" ".join(links)) / max(len(stripped), 1)) if links else 0.0
        is_link_heavy = len(links) >= 3 and (link_ratio > 0.6 or len(text_without_links) < 30)

        if is_link_heavy:
            link_heavy_streak += 1
            if link_heavy_streak >= 2:
                continue
        else:
            link_heavy_streak = 0

        filtered.append(line)

    return "\n".join(filtered)


def fetch_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> requests.Response:
    headers = {"User-Agent": user_agent}
    response = session.get(url, headers=headers, timeout=timeout)

    if response.status_code in RETRYABLE_STATUS_CODES:
        time.sleep(RETRY_WAIT_SECONDS)
        response = session.get(url, headers=headers, timeout=timeout)

    response.raise_for_status()
    return response


def scrape_page(
    session: requests.Session,
    entry: SitemapEntry,
    output_dir: Path,
    *,
    scraped_at: str,
) -> ScrapeResult:
    response = fetch_with_retry(session, entry.url)
    title, markdown = extract_markdown(response.text)
    if not markdown:
        raise ValueError("page content was empty after markdown conversion")

    filename = slugify_url_to_filename(entry.url)
    output_path = output_dir / filename
    output_path.write_text(
        render_frontmatter(entry.url, title, entry.last_modified, scraped_at, markdown),
        encoding="utf-8",
    )

    return ScrapeResult(
        url=entry.url,
        file=filename,
        size_bytes=output_path.stat().st_size,
        last_modified=entry.last_modified,
    )


def build_manifest(
    *,
    scraped_at: str,
    total_sitemap_urls: int,
    skipped_template_urls: int,
    scraped_files: list[ScrapeResult],
    failed_urls: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "scraped_at": scraped_at,
        "total_sitemap_urls": total_sitemap_urls,
        "skipped_template_urls": skipped_template_urls,
        "scraped_count": len(scraped_files),
        "failed_count": len(failed_urls),
        "failed_urls": failed_urls,
        "files": [asdict(item) for item in scraped_files],
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_scrape_pipeline(
    *,
    sitemap_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    session: requests.Session | None = None,
    sleep_seconds: float = RATE_LIMIT_SECONDS,
    incremental: bool = False,
) -> dict[str, Any]:
    entries = load_sitemap(sitemap_path)
    scrapable_entries, skipped_entries = filter_scrapable_entries(entries)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    owned_session = session is None
    session = session or requests.Session()
    scraped_files: list[ScrapeResult] = []
    failed_urls: list[dict[str, str]] = []

    # Load existing manifest for incremental check
    old_manifest_files = {}
    if incremental and Path(manifest_path).exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for file_entry in old_data.get("files", []):
                    old_manifest_files[file_entry["url"]] = file_entry
        except Exception:
            pass

    try:
        for index, entry in enumerate(scrapable_entries):
            # Check if we can skip (incremental run)
            if incremental and entry.url in old_manifest_files:
                old_entry = old_manifest_files[entry.url]
                old_file_path = output_dir / old_entry["file"]
                if old_entry.get("last_modified") == entry.last_modified and old_file_path.exists():
                    # Keep existing result without fetching
                    scraped_files.append(
                        ScrapeResult(
                            url=entry.url,
                            file=old_entry["file"],
                            size_bytes=old_entry["size_bytes"],
                            last_modified=entry.last_modified
                        )
                    )
                    continue

            try:
                scraped_files.append(
                    scrape_page(session, entry, output_dir, scraped_at=scraped_at)
                )
            except Exception as exc:  # noqa: BLE001
                failed_urls.append({"url": entry.url, "error": str(exc)})

            if index < len(scrapable_entries) - 1:
                time.sleep(sleep_seconds)
    finally:
        if owned_session:
            session.close()

    manifest = build_manifest(
        scraped_at=scraped_at,
        total_sitemap_urls=len(entries),
        skipped_template_urls=len(skipped_entries),
        scraped_files=scraped_files,
        failed_urls=failed_urls,
    )
    write_manifest(manifest, manifest_path)
    return manifest
