from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
BROKEN_TEMPLATE_MARKERS = ("%3Croute%3E", "/pages/page-")


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    last_modified: str = ""


def is_broken_template_url(url: str) -> bool:
    return any(marker in url for marker in BROKEN_TEMPLATE_MARKERS)


def parse_sitemap_xml(xml_text: str) -> list[SitemapEntry]:
    root = ET.fromstring(xml_text)
    entries: list[SitemapEntry] = []

    for url_node in root.findall("sm:url", SITEMAP_NS):
        loc_node = url_node.find("sm:loc", SITEMAP_NS)
        if loc_node is None or not loc_node.text:
            continue

        lastmod_node = url_node.find("sm:lastmod", SITEMAP_NS)
        entries.append(
            SitemapEntry(
                url=loc_node.text.strip(),
                last_modified=(lastmod_node.text or "").strip() if lastmod_node is not None else "",
            )
        )

    return entries


def load_sitemap(path_or_url: str | Path) -> list[SitemapEntry]:
    import requests
    if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        return parse_sitemap_xml(response.text)
    return parse_sitemap_xml(Path(path_or_url).read_text(encoding="utf-8"))


def filter_scrapable_entries(entries: Iterable[SitemapEntry]) -> tuple[list[SitemapEntry], list[SitemapEntry]]:
    scrapable: list[SitemapEntry] = []
    skipped: list[SitemapEntry] = []

    for entry in entries:
        (skipped if is_broken_template_url(entry.url) else scrapable).append(entry)

    return scrapable, skipped
