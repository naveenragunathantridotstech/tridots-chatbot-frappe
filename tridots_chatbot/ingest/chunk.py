from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import ceil
from pathlib import Path
import hashlib
import re
from typing import Any


FRONTMATTER_BOUNDARY = "---"
MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 750
TOKEN_CHAR_RATIO = 3.5
BLOCK_DUP_THRESHOLD = 3

CORPORATE_SLUGS = {
    "about",
    "about-us",
    "contact",
    "career",
    "tridots-service",
    "stories",
}


@dataclass(slots=True)
class ScrapedDocument:
    source_path: Path | None
    url: str
    title: str
    last_modified: date | None
    scraped_at: datetime | None
    slug: str
    page_type: str
    body: str


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    url: str
    title: str
    last_modified: date | None
    scraped_at: datetime | None
    page_type: str
    slug: str
    section_heading: str | None
    h1_heading: str | None
    h2_heading: str | None
    heading_prefix: str
    chunk_index: int
    chunk_total: int
    token_count: int
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_block_for_hash(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\]\([^)]*\)', ']', text)
    text = re.sub(r'!\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_duplicate_block_hashes(input_dir: Path, threshold: int = BLOCK_DUP_THRESHOLD) -> set[str]:
    hashes: dict[str, int] = {}
    for md_path in sorted(input_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
            _, body = parse_frontmatter(text)
        except (ValueError, OSError):
            continue
        seen_in_file: set[str] = set()
        for block in body.split("\n\n"):
            normalized = normalize_block_for_hash(block)
            if len(normalized) < 5:
                continue
            block_hash = hashlib.md5(normalized.encode()).hexdigest()
            seen_in_file.add(block_hash)
        for h in seen_in_file:
            hashes[h] = hashes.get(h, 0) + 1
    return {h for h, count in hashes.items() if count > threshold}


def _clean_heading(text: str) -> str:
    text = re.sub(r'\*\*', '', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'[\[\](){}]', '', text)
    return text.strip()


def build_heading_prefix(title: str, h1: str | None, h2: str | None, section_heading: str | None) -> str:
    parts: list[str] = [_clean_heading(title)]
    seen: set[str] = {title.lower()}
    for h in (h1, h2, section_heading):
        if h:
            cleaned = _clean_heading(h)
            if cleaned and cleaned.lower() not in seen:
                parts.append(cleaned)
                seen.add(cleaned.lower())
    return " > ".join(parts)


def parse_scraped_markdown_file(path: str | Path) -> ScrapedDocument:
    file_path = Path(path)
    return parse_scraped_markdown(file_path.read_text(encoding="utf-8"), source_path=file_path)


def parse_scraped_markdown(markdown_text: str, source_path: str | Path | None = None) -> ScrapedDocument:
    frontmatter, body = parse_frontmatter(markdown_text)
    url = str(frontmatter.get("url", "")).strip()
    if not url:
        raise ValueError("Frontmatter is missing required 'url'")

    title = str(frontmatter.get("title", "")).strip()
    slug = slug_from_url(url)
    return ScrapedDocument(
        source_path=Path(source_path) if source_path is not None else None,
        url=url,
        title=title,
        last_modified=_parse_date(frontmatter.get("last_modified")),
        scraped_at=_parse_datetime(frontmatter.get("scraped_at")),
        slug=slug,
        page_type=classify_page_type(url=url, slug=slug),
        body=normalize_markdown(strip_boilerplate(body)),
    )


def strip_boilerplate(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("![](") and "flagcdn.com" in stripped:
            continue
        if re.match(r"^\+\d+(\s+\d+)*$", stripped) and len(stripped) <= 6:
            continue
        if stripped.lower() in {
            "phone number is required*", "phone number is required\\*",
            "email is required*", "email is required\\*",
            "e-mail is required*", "e-mail is required\\*",
            "service is required*", "service is required\\*",
            "message is required*", "message is required\\*",
            "full name is required*", "full name is required\\*",
            "first name is required*", "first name is required\\*",
            "last name is required*", "last name is required\\*",
            "name is required*", "name is required\\*"
        }:
            continue
        cleaned_lines.append(line)
    markdown_text = "\n".join(cleaned_lines)

    blocks = markdown_text.split("\n\n")
    if not blocks:
        return ""

    start_block_idx = 0
    first_para_idx = 0
    for i, block in enumerate(blocks):
        text = block.strip()
        if not text:
            continue
        link_count = text.count("](") + text.count("](/")
        if len(text) > 60 and link_count == 0:
            first_para_idx = i
            break
        if text.startswith("# ") and "\n" not in text and link_count == 0:
            first_para_idx = i
            break

    start_block_idx = first_para_idx
    for j in range(first_para_idx - 1, max(-1, first_para_idx - 3), -1):
        prev_text = blocks[j].strip()
        if not prev_text:
            continue
        link_count = prev_text.count("](")
        if link_count == 0 and 0 < len(prev_text) < 60:
            start_block_idx = j
        else:
            break

    end_block_idx = len(blocks)
    footer_signatures = [
        "© 20", "Contact  us", "Contact us", "Let's create something new",
        "Submit Your Inquiry", "Privacy Policy", "Free Technical Consultation",
        "Commitment to Excellence"
    ]
    for i in range(len(blocks) - 1, start_block_idx, -1):
        text = blocks[i].strip()
        if not text:
            continue
        if any(sig.lower() in text.lower() for sig in footer_signatures):
            end_block_idx = i
            for j in range(i, max(start_block_idx, i - 10), -1):
                t = blocks[j].strip().lower()
                if "related services" in t or "contact us" in t or t.startswith("# **faq's**") or "let's create something new" in t:
                    end_block_idx = j
            break
        link_count = text.count("](")
        clean_text_len = len(re.sub(r'\[.*?\]\(.*?\)', '', text).strip())
        if link_count > 0 and clean_text_len < 50:
            end_block_idx = i
            continue
        break

    if start_block_idx >= end_block_idx:
        return markdown_text

    return "\n\n".join(blocks[start_block_idx:end_block_idx])


def parse_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    text = markdown_text.lstrip("\ufeff")
    if not text.startswith(FRONTMATTER_BOUNDARY):
        raise ValueError("Markdown file is missing expected frontmatter")

    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        raise ValueError("Frontmatter opening boundary is malformed")

    frontmatter: dict[str, str] = {}
    closing_index: int | None = None
    for index in range(1, len(lines)):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if stripped == FRONTMATTER_BOUNDARY:
            closing_index = index
            break
        if not stripped:
            continue
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        frontmatter[key.strip()] = value.strip()

    if closing_index is None:
        raise ValueError("Frontmatter closing boundary is missing")

    body = "\n".join(lines[closing_index + 1:]).strip()
    return frontmatter, body


def normalize_markdown(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def slug_from_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
        cleaned = cleaned.split("/", 1)[1] if "/" in cleaned else ""
    slug = cleaned.strip("/")
    if not slug:
        return "home"
    return slug.replace("/", "_")


def classify_page_type(url: str, slug: str | None = None) -> str:
    derived_slug = slug or slug_from_url(url)
    normalized_url = url.lower()
    normalized_slug = derived_slug.lower()

    if "/blog/" in normalized_url:
        return "blog"
    if normalized_slug.startswith("go1-"):
        return "go1_product"
    if normalized_slug.endswith("-erpnext-integration") or re.match(r"^[a-z0-9-]+-erpnext-", normalized_slug):
        return "integration"
    if normalized_slug.startswith("frappe-"):
        return "frappe_product"
    if normalized_slug.startswith("erpnext-services-") or normalized_slug.startswith("erpnext-implementation-"):
        return "geo"
    if normalized_slug in CORPORATE_SLUGS:
        return "corporate"
    if normalized_slug.startswith("erpnext-"):
        return "erpnext_feature"
    return "service"


def _filter_duplicate_blocks(body: str, duplicate_hashes: set[str] | None) -> str:
    filtered: list[str] = []
    seen_hashes: set[str] = set()
    for block in body.split("\n\n"):
        h = hashlib.md5(normalize_block_for_hash(block).encode()).hexdigest()
        if h in seen_hashes:
            continue
        if duplicate_hashes and h in duplicate_hashes:
            continue
        seen_hashes.add(h)
        filtered.append(block)
    return "\n\n".join(filtered)


MIN_MEANINGFUL_CHARS = 60
FLAG_DROP_THRESHOLD = 3


def _meaningful_text_length(text: str) -> int:
    text = re.sub(r'!\[.*?\]\([^)]*\)', '', text)
    text = re.sub(r'(?m)^-{3,}$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return len(text)


def _is_noise_chunk(content: str) -> bool:
    lines = content.splitlines()
    non_empty = [l.strip() for l in lines if l.strip()]

    flag_count = sum(1 for l in non_empty if 'flagcdn.com' in l)
    if flag_count > FLAG_DROP_THRESHOLD:
        return True

    hr_count = sum(1 for l in non_empty if re.match(r'^-{3,}$', l))
    img_count = sum(1 for l in non_empty if re.match(r'^!\[.*?\]\(.*\)$', l))
    real_lines = sum(1 for l in non_empty if len(l.split()) >= 2 and not re.match(r'^-{3,}$', l) and not re.match(r'^!\[.*?\]\(.*\)$', l))

    if img_count > 0 and real_lines == 0:
        return True
    if hr_count >= 3 and real_lines < 2:
        return True

    if _meaningful_text_length(content) < MIN_MEANINGFUL_CHARS:
        return True
    return False


def extract_page_introduction(body: str) -> str:
    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
    if not blocks:
        return ""

    intro_candidates = blocks[:8]

    h1_block = None
    first_para_block = None

    for block in intro_candidates:
        if block.startswith("# ") and "\n" not in block:
            h1_block = block.lstrip("# ").strip()
            h1_idx = blocks.index(block)
            for next_block in blocks[h1_idx + 1: h1_idx + 6]:
                if next_block.startswith("#") or next_block.startswith("!") or next_block.startswith("-") or next_block.startswith("*") or next_block.startswith("---") or next_block.startswith("["):
                    continue
                if len(next_block) < 25:
                    continue
                first_para_block = next_block
                break
            break

    if not h1_block:
        for block in intro_candidates:
            if block.startswith("#") or block.startswith("!") or block.startswith("-") or block.startswith("*") or block.startswith("---") or block.startswith("["):
                continue
            if len(block) >= 40:
                first_para_block = block
                break
        if not first_para_block:
            for block in intro_candidates:
                if block.startswith("#") or block.startswith("!") or block.startswith("-") or block.startswith("*") or block.startswith("---") or block.startswith("["):
                    continue
                if len(block) >= 15:
                    first_para_block = block
                    break

    parts = []
    if h1_block:
        parts.append(h1_block)
    if first_para_block:
        clean_para = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', first_para_block)
        clean_para = re.sub(r'\]\([^)]*\)', '', clean_para)
        clean_para = re.sub(r'\s+', ' ', clean_para).strip()
        parts.append(clean_para)

    intro = " : ".join(parts)
    if len(intro) > 350:
        intro = intro[:347] + "..."
    return intro


def chunk_document(
    document: ScrapedDocument,
    *,
    duplicate_hashes: set[str] | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[ChunkRecord]:
    if document.slug in CORPORATE_SLUGS:
        body = document.body
    else:
        body = _filter_duplicate_blocks(document.body, duplicate_hashes)
    raw_sections = split_markdown_sections(body)
    chunks: list[ChunkRecord] = []

    for h1, h2, section_heading, section_text in raw_sections:
        heading_prefix = build_heading_prefix(document.title, h1, h2, section_heading)
        if document.slug == "contact":
            heading_prefix = f"Contact Us > Office Locations, Addresses, and Phone Numbers > {heading_prefix}"

        for chunk_text in split_section_text(
            section_text,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        ):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.slug}__chunk_{0}",
                    url=document.url,
                    title=document.title,
                    last_modified=document.last_modified,
                    scraped_at=document.scraped_at,
                    page_type=document.page_type,
                    slug=document.slug,
                    section_heading=section_heading,
                    h1_heading=h1,
                    h2_heading=h2,
                    heading_prefix=heading_prefix,
                    chunk_index=0,
                    chunk_total=0,
                    token_count=approximate_token_count(chunk_text),
                    content=chunk_text,
                )
            )

    chunks = [c for c in chunks if not _is_noise_chunk(c.content)]
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{document.slug}__chunk_{i}"
        chunk.chunk_index = i
        chunk.chunk_total = total
    return chunks


def split_markdown_sections(markdown_text: str) -> list[tuple[str | None, str | None, str | None, str]]:
    if not markdown_text.strip():
        return []

    sections: list[tuple[str | None, str | None, str | None, str]] = []
    current_h1: str | None = None
    current_h2: str | None = None
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        h1_match = re.match(r"^#\s+(.*\S)\s*$", line)
        h2_match = re.match(r"^##\s+(.*\S)\s*$", line)
        h3_match = re.match(r"^###\s+(.*\S)\s*$", line)

        if h1_match or h2_match or h3_match:
            if current_lines:
                sections.append((current_h1, current_h2, current_heading, "\n".join(current_lines).strip()))

            if h1_match:
                current_h1 = h1_match.group(1).strip()
                current_h2 = None
                current_heading = current_h1
            elif h2_match:
                current_h2 = h2_match.group(1).strip()
                current_heading = current_h2
            elif h3_match:
                current_heading = h3_match.group(1).strip()

            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_h1, current_h2, current_heading, "\n".join(current_lines).strip()))

    return [(h1, h2, heading, text) for h1, h2, heading, text in sections if text.strip()]


def split_section_text(
    section_text: str,
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> list[str]:
    text = normalize_markdown(section_text)
    if len(text) < min_chunk_chars:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        sep_len = 2 if current else 0
        if current_len + sep_len + para_len > max_chunk_chars and current:
            chunk = "\n\n".join(current).strip()
            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += sep_len + para_len

    if current:
        chunk = "\n\n".join(current).strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

    return chunks or [text]


def approximate_token_count(text: str) -> int:
    return max(1, ceil(len(text) / TOKEN_CHAR_RATIO))


def _parse_date(value: Any) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    return date.fromisoformat(text)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
