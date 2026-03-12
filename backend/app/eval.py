

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex
from app.search.hybrid import HybridSearcher


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def dcg(relevances: list, k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    relevances = [1 if did in relevant_ids else 0 for did in retrieved_ids[:k]]
    ideal      = sorted(relevances, reverse=True)
    actual_dcg = dcg(relevances, k)
    ideal_dcg  = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for did in retrieved_ids[:k] if did in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    for i, did in enumerate(retrieved_ids[:k]):
        if did in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def run_eval(
    queries_path: Path,
    qrels_path: Path,
    alpha: float = 0.5,
    norm: str = "minmax",
    k: int = 10,
):
    # Load queries
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # Load qrels
    with open(qrels_path, encoding="utf-8") as f:
        qrels = json.load(f)

    # Load indexes
    bm25 = BM25Index()
    bm25.load()
    vec = VectorIndex()
    vec.load()
    searcher = HybridSearcher(bm25, vec, norm_strategy=norm)

    ndcg_scores, recall_scores, mrr_scores = [], [], []

    for q in queries:
        qid      = q["query_id"]
        qtext    = q["query"]
        relevant = set(qrels.get(qid, []))

        results      = searcher.search(qtext, top_k=k, alpha=alpha)
        retrieved    = [r["doc_id"] for r in results]

        ndcg_scores.append(ndcg(retrieved, relevant, k))
        recall_scores.append(recall(retrieved, relevant, k))
        mrr_scores.append(mrr(retrieved, relevant, k))

    avg_ndcg   = sum(ndcg_scores)   / len(ndcg_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr    = sum(mrr_scores)    / len(mrr_scores)

    print(f"\n=== Evaluation Results (alpha={alpha}, norm={norm}) ===")
    print(f"  nDCG@{k}  : {avg_ndcg:.4f}")
    print(f"  Recall@{k}: {avg_recall:.4f}")
    print(f"  MRR@{k}   : {avg_mrr:.4f}")

    # Get git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    # Append to experiments CSV
    metrics_dir = Path("data/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "experiments.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "commit", "alpha", "norm",
            "ndcg_at_10", "recall_at_10", "mrr_at_10"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "commit":      commit,
            "alpha":       alpha,
            "norm":        norm,
            "ndcg_at_10":  round(avg_ndcg,   4),
            "recall_at_10": round(avg_recall, 4),
            "mrr_at_10":   round(avg_mrr,    4),
        })

    print(f"  Results appended to {csv_path}")
    return avg_ndcg, avg_recall, avg_mrr


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--qrels",   default="data/eval/qrels.json")
    parser.add_argument("--alpha",   type=float, default=0.5)
    parser.add_argument("--norm",    default="minmax", choices=["minmax", "zscore"])
    parser.add_argument("--k",       type=int, default=10)
    args = parser.parse_args()

    run_eval(
        Path(args.queries),
        Path(args.qrels),
        alpha = args.alpha,
        norm  = args.norm,
        k     = args.k,
    )


if __name__ == "__main__":
    main()