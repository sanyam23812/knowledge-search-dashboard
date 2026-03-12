

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace/punctuation."""
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.docs: List[Dict[str, Any]] = []
        self.index_dir: Path = Path("data/index/bm25")

    def build(self, docs: List[Dict[str, Any]]) -> None:
        """Build BM25 index from list of doc dicts."""
        self.docs = docs
        corpus = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.bm25 = BM25Okapi(corpus)
        print(f"[BM25] Built index over {len(docs)} documents.")

    def save(self) -> None:
        """Save index and docs to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / "index.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.index_dir / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False)
        print(f"[BM25] Saved index to {self.index_dir}")

    def load(self) -> None:
        """Load index and docs from disk."""
        with open(self.index_dir / "index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.index_dir / "docs.json", "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        print(f"[BM25] Loaded index with {len(self.docs)} documents.")

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top_k results with BM25 scores."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call build() or load() first.")

        tokens = tokenize(query_text)
        scores = self.bm25.get_scores(tokens)

        # Pair each doc with its score and sort
        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in scored:
            doc = self.docs[idx].copy()
            doc["bm25_score"] = float(score)
            results.append(doc)

        return results