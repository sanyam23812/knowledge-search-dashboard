

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, fast, CPU-friendly


class VectorIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.docs: List[Dict[str, Any]] = []
        self.index_dir = Path("data/index/vector")
        self.dimension = None

    def _load_model(self):
        if self.model is None:
            print(f"[Vector] Loading model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

    def build(self, docs: List[Dict[str, Any]]) -> None:
        """Build FAISS index from docs."""
        self._load_model()
        self.docs = docs

        texts = [d["title"] + " " + d["text"] for d in docs]
        print(f"[Vector] Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True,
                                       convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine-like)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"[Vector] Built FAISS index with {self.index.ntotal} vectors.")

    def save(self) -> None:
        """Save FAISS index + metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

        with open(self.index_dir / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False)

        meta = {
            "model_name": self.model_name,
            "dimension":  self.dimension,
            "num_docs":   len(self.docs),
        }
        with open(self.index_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        print(f"[Vector] Saved index to {self.index_dir}")

    def load(self) -> None:
        """Load FAISS index + metadata from disk."""
        with open(self.index_dir / "meta.json") as f:
            meta = json.load(f)

        self.model_name = meta["model_name"]
        self.dimension  = meta["dimension"]

        self._load_model()

        # Startup validation — catch dimension mismatch early
        if self.dimension != meta["dimension"]:
            raise RuntimeError(
                f"Dimension mismatch! Index has {meta['dimension']} "
                f"but model produces {self.dimension}. Rebuild the index."
            )

        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))

        with open(self.index_dir / "docs.json", encoding="utf-8") as f:
            self.docs = json.load(f)

        print(f"[Vector] Loaded index: {len(self.docs)} docs, dim={self.dimension}")

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top_k results with vector scores."""
        if self.index is None:
            raise RuntimeError("Vector index not loaded. Call build() or load() first.")

        self._load_model()
        query_vec = self.model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.docs[idx].copy()
            doc["vector_score"] = float(score)
            results.append(doc)

        return results