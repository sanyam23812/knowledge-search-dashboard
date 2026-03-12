
import re
from typing import List, Dict, Any, Tuple
import numpy as np

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex


def minmax_norm(scores: List[float]) -> List[float]:
    """Normalise scores to [0, 1] using min-max."""
    arr = np.array(scores, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:          # all scores equal → avoid divide-by-zero
        return [0.0] * len(scores)
    return ((arr - mn) / (mx - mn)).tolist()


def zscore_norm(scores: List[float]) -> List[float]:
    """Normalise scores using z-score, then clip to [0, 1]."""
    arr = np.array(scores, dtype=float)
    std = arr.std()
    if std < 1e-9:               # avoid divide-by-zero
        return [0.0] * len(scores)
    normed = (arr - arr.mean()) / std
    # shift so minimum is 0, then scale to max 1
    normed = normed - normed.min()
    mx = normed.max()
    if mx < 1e-9:
        return [0.0] * len(scores)
    return (normed / mx).tolist()


def get_snippet(text: str, query: str, window: int = 40) -> str:
    """Return a short highlight snippet around the first query term match."""
    tokens = re.findall(r'\b\w+\b', query.lower())
    lower  = text.lower()
    best   = -1
    for tok in tokens:
        idx = lower.find(tok)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    if best == -1:
        return text[:200]
    start = max(0, best - window)
    end   = min(len(text), best + window + len(tokens[0]) if tokens else best + window)
    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
    return snippet


class HybridSearcher:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        norm_strategy: str = "minmax",   # "minmax" or "zscore"
    ):
        self.bm25   = bm25_index
        self.vector = vector_index
        self.norm_strategy = norm_strategy

    def _normalise(self, scores: List[float]) -> List[float]:
        if self.norm_strategy == "zscore":
            return zscore_norm(scores)
        return minmax_norm(scores)   # default

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search.
        hybrid_score = alpha * norm_bm25 + (1 - alpha) * norm_vector
        alpha=1.0  → pure BM25
        alpha=0.0  → pure vector
        alpha=0.5  → equal blend (default)
        """
        fetch_k = max(top_k * 3, 50)   # fetch more, then re-rank

        bm25_results   = self.bm25.query(query, top_k=fetch_k)
        vector_results = self.vector.query(query, top_k=fetch_k)

        # Build score maps keyed by doc_id
        bm25_map   = {r["doc_id"]: r["bm25_score"]   for r in bm25_results}
        vector_map = {r["doc_id"]: r["vector_score"]  for r in vector_results}
        doc_map    = {r["doc_id"]: r for r in bm25_results + vector_results}

        all_ids = list(set(bm25_map) | set(vector_map))

        raw_bm25   = [bm25_map.get(did, 0.0)   for did in all_ids]
        raw_vector = [vector_map.get(did, 0.0)  for did in all_ids]

        norm_bm25   = self._normalise(raw_bm25)
        norm_vector = self._normalise(raw_vector)

        results = []
        for i, did in enumerate(all_ids):
            doc = doc_map[did].copy()
            nb  = norm_bm25[i]
            nv  = norm_vector[i]
            doc["bm25_score"]    = raw_bm25[i]
            doc["vector_score"]  = raw_vector[i]
            doc["norm_bm25"]     = nb
            doc["norm_vector"]   = nv
            doc["hybrid_score"]  = alpha * nb + (1 - alpha) * nv
            doc["highlight"]     = get_snippet(doc.get("text", ""), query)
            results.append(doc)

        # Apply optional filters
        if filters:
            for key, val in filters.items():
                results = [r for r in results if r.get(key) == val]

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]