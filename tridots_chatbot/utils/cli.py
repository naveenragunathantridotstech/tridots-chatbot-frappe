from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tridots chatbot utility CLI (Frappe app).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest scraped markdown into vectors.json")
    subparsers.add_parser("verify", help="Verify vectors.json and print summary")

    return parser


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    command = args.command

    if command == "ingest":
        from tridots_chatbot.ingest.run import main as ingest_main
        sys.exit(ingest_main(unknown))
    elif command == "verify":
        _verify()
    else:
        parser.print_help()
        sys.exit(1)


def _verify() -> None:
    import json
    from pathlib import Path
    from tridots_chatbot.config import get_settings

    path = get_settings().vectors_path
    if not path.exists():
        print(f"vectors.json not found at {path}", flush=True)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    print(f"Total chunks: {len(chunks)}", flush=True)
    print(f"Embedding model: {data.get('embedding_model', 'N/A')}", flush=True)
    print(f"Vector dimensions: {data.get('vector_dimensions', 'N/A')}", flush=True)

    if chunks:
        first = chunks[0]
        emb = first.get("embedding", [])
        print(f"First chunk: {first.get('chunk_id', 'N/A')}", flush=True)
        print(f"  URL: {first.get('url', 'N/A')}", flush=True)
        print(f"  Title: {first.get('title', 'N/A')}", flush=True)
        print(f"  Embedding dims: {len(emb)}", flush=True)

    missing = [c for c in chunks if "embedding" not in c]
    if missing:
        print(f"WARNING: {len(missing)} chunks missing embeddings!", flush=True)
    else:
        print("All chunks have embeddings.", flush=True)

    urls = set(c.get("url", "") for c in chunks)
    print(f"Unique source URLs: {len(urls)}", flush=True)
