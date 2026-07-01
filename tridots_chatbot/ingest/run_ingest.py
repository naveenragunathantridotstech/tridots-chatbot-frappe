from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv

from tridots_chatbot.ingest.chunk import (
    chunk_document,
    get_duplicate_block_hashes,
    parse_scraped_markdown_file,
    extract_page_introduction,
    approximate_token_count,
)
from tridots_chatbot.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest scraped markdown into vectors.json")
    parser.add_argument("--input-dir", default=None, help="Directory containing scraped markdown files")
    parser.add_argument("--output", default=None, help="Output path for vectors.json")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for number of markdown files")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    started = monotonic()

    input_dir = Path(args.input_dir) if args.input_dir else settings.scraped_content_dir
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", flush=True)
        return 1

    output_path = Path(args.output) if args.output else settings.vectors_path

    # Build embedding text for each chunk (same strategy as the original llamaindex_ingest)
    from fastembed import TextEmbedding
    embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    files = sorted(input_dir.glob("*.md"))
    if args.max_files > 0:
        files = files[: args.max_files]

    duplicate_hashes = get_duplicate_block_hashes(input_dir)

    chunks_data = []
    all_texts = []

    for md_path in files:
        document = parse_scraped_markdown_file(md_path)
        page_intro = extract_page_introduction(document.body)
        clean_category = document.slug.replace("-", " ").title()

        for chunk in chunk_document(document, duplicate_hashes=duplicate_hashes):
            heading_part = chunk.heading_prefix if chunk.heading_prefix else chunk.title
            embedding_text = f"{clean_category} | {heading_part} | {page_intro}\n\n{chunk.content}"
            all_texts.append(embedding_text)
            chunks_data.append({
                "text": embedding_text,
                "url": chunk.url,
                "title": chunk.title,
                "page_type": chunk.page_type,
                "section_heading": chunk.section_heading,
                "slug": chunk.slug,
                "heading_prefix": chunk.heading_prefix,
                "token_count": approximate_token_count(embedding_text),
                "chunk_id": chunk.chunk_id,
            })

    # Generate embeddings
    print(f"Generating embeddings for {len(all_texts)} chunks...", flush=True)
    embeddings = list(embed_model.embed(all_texts))

    # Attach embeddings to each chunk record
    for chunk_dict, emb in zip(chunks_data, embeddings):
        chunk_dict["embedding"] = [float(v) for v in emb]

    output = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "vector_dimensions": 384,
        "total_chunks": len(chunks_data),
        "chunks": chunks_data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    elapsed = monotonic() - started
    print(f"Done. {len(chunks_data)} chunks written to {output_path} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
