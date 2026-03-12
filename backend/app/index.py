

import argparse
import json
from pathlib import Path

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex


def load_docs(jsonl_path: Path) -> list:
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"[Index] Loaded {len(docs)} documents from {jsonl_path}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and vector indexes")
    parser.add_argument("--input", default="data/processed/docs.jsonl")
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"[Error] File not found: {jsonl_path}")
        print("Run ingest first: python -m app.ingest --download")
        return

    docs = load_docs(jsonl_path)

    print("\n[Index] Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(docs)
    bm25.save()

    print("\n[Index] Building vector index...")
    vec = VectorIndex()
    vec.build(docs)
    vec.save()

    print("\n[Index] All indexes built successfully!")


if __name__ == "__main__":
    main()