

import argparse
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Public-domain sample corpus (Project Gutenberg plain-text books)
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    ("pg1342", "Pride and Prejudice",        "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("pg11",   "Alice in Wonderland",         "https://www.gutenberg.org/files/11/11-0.txt"),
    ("pg1661", "Sherlock Holmes",             "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ("pg2701", "Moby Dick",                   "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("pg98",   "A Tale of Two Cities",        "https://www.gutenberg.org/files/98/98-0.txt"),
    ("pg1232", "The Prince - Machiavelli",    "https://www.gutenberg.org/files/1232/1232-0.txt"),
    ("pg84",   "Frankenstein",                "https://www.gutenberg.org/files/84/84-0.txt"),
    ("pg345",  "Dracula",                     "https://www.gutenberg.org/files/345/345-0.txt"),
    ("pg1080", "A Modest Proposal",           "https://www.gutenberg.org/files/1080/1080-0.txt"),
    ("pg76",   "Adventures of Huckleberry Finn", "https://www.gutenberg.org/files/76/76-0.txt"),
]


def clean_text(text: str) -> str:
    """Basic whitespace cleanup."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_chunks(text: str, doc_id: str, title: str,
                      source: str, chunk_size: int = 500) -> list[dict]:
    """Split long text into paragraph chunks of ~chunk_size words."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current, count = [], [], 0

    for para in paragraphs:
        words = para.split()
        if count + len(words) > chunk_size and current:
            chunk_text = ' '.join(' '.join(c) for c in current)
            chunks.append(chunk_text)
            current, count = [], 0
        current.append(words)
        count += len(words)

    if current:
        chunk_text = ' '.join(' '.join(c) for c in current)
        chunks.append(chunk_text)

    records = []
    for i, chunk in enumerate(chunks):
        cid = f"{doc_id}_chunk{i:04d}"
        records.append({
            "doc_id":     cid,
            "title":      f"{title} (part {i+1})",
            "text":       chunk,
            "source":     source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


def download_samples(raw_dir: Path) -> None:
    """Download Project Gutenberg books if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, title, url in SAMPLE_URLS:
        dest = raw_dir / f"{doc_id}.txt"
        if dest.exists():
            print(f"  [skip] {title} already downloaded")
            continue
        print(f"  [download] {title} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [ok] saved to {dest}")
        except Exception as exc:
            print(f"  [error] {title}: {exc}")


def ingest(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "docs.jsonl"

    all_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.md"))
    if not all_files:
        print("No .txt or .md files found in input directory.")
        return

    total = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for filepath in sorted(all_files):
            raw = filepath.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw)
            doc_id = filepath.stem
            title  = filepath.stem.replace("-", " ").replace("_", " ").title()
            records = split_into_chunks(cleaned, doc_id, title, str(filepath))
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"  [ingested] {filepath.name} -> {len(records)} chunks")

    print(f"\nDone. {total} total chunks written to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents to JSONL")
    parser.add_argument("--input",    default="data/raw",       help="Input folder")
    parser.add_argument("--out",      default="data/processed", help="Output folder")
    parser.add_argument("--download", action="store_true",       help="Download sample corpus first")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.out)

    if args.download:
        print("Downloading sample corpus...")
        download_samples(input_dir)

    print(f"Ingesting from {input_dir} ...")
    ingest(input_dir, output_dir)


if __name__ == "__main__":
    main()