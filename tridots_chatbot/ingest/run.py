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
    parser.add_argument("--incremental", action="store_true", help="Only embed new or modified pages")
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

    # Load existing vectors if incremental
    old_chunks_by_url = {}
    if args.incremental and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for chunk in old_data.get("chunks", []):
                    old_chunks_by_url.setdefault(chunk["url"], []).append(chunk)
            print(f"Loaded {len(old_data.get('chunks', []))} existing chunks from {output_path}", flush=True)
        except Exception as e:
            print(f"Failed to load existing vectors: {e}. Proceeding with full ingestion.", flush=True)

    files = sorted(input_dir.glob("*.md"))
    if args.max_files > 0:
        files = files[: args.max_files]

    duplicate_hashes = get_duplicate_block_hashes(input_dir)

    chunks_data = []
    chunks_to_embed = []
    all_texts_to_embed = []

    reused_count = 0
    new_count = 0

    for md_path in files:
        document = parse_scraped_markdown_file(md_path)
        page_intro = extract_page_introduction(document.body)
        clean_category = document.slug.replace("-", " ").title()
        doc_last_mod = document.last_modified.isoformat() if document.last_modified else ""

        # Check if we can reuse old chunks
        if args.incremental and document.url in old_chunks_by_url:
            old_list = old_chunks_by_url[document.url]
            if old_list and old_list[0].get("last_modified") == doc_last_mod:
                # Reuse existing chunks
                chunks_data.extend(old_list)
                reused_count += len(old_list)
                continue

        # Otherwise, we chunk and embed
        for chunk in chunk_document(document, duplicate_hashes=duplicate_hashes):
            heading_part = chunk.heading_prefix if chunk.heading_prefix else chunk.title
            embedding_text = f"{clean_category} | {heading_part} | {page_intro}\n\n{chunk.content}"
            
            chunk_dict = {
                "text": embedding_text,
                "url": chunk.url,
                "title": chunk.title,
                "page_type": chunk.page_type,
                "section_heading": chunk.section_heading,
                "slug": chunk.slug,
                "heading_prefix": chunk.heading_prefix,
                "token_count": approximate_token_count(embedding_text),
                "chunk_id": chunk.chunk_id,
                "last_modified": doc_last_mod,  # Store this to enable incremental indexing later
            }
            chunks_data.append(chunk_dict)
            chunks_to_embed.append(chunk_dict)
            all_texts_to_embed.append(embedding_text)
            new_count += 1

    # Generate embeddings for new chunks if any
    if all_texts_to_embed:
        print(f"Generating embeddings for {len(all_texts_to_embed)} new/modified chunks...", flush=True)
        from fastembed import TextEmbedding
        embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        embeddings = list(embed_model.embed(all_texts_to_embed))

        # Attach embeddings to new chunks
        for chunk_dict, emb in zip(chunks_to_embed, embeddings):
            chunk_dict["embedding"] = [float(v) for v in emb]
    else:
        print("No new or modified chunks to embed.", flush=True)

    output = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "vector_dimensions": 384,
        "total_chunks": len(chunks_data),
        "chunks": chunks_data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    elapsed = monotonic() - started
    print(f"Done. Reused: {reused_count}, Newly Embedded: {new_count}. Total: {len(chunks_data)} chunks written to {output_path} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
