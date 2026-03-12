# Architecture Overview

## System Components
```
┌─────────────────────────────────────────────────────┐
│                     up.bat                          │
│         (orchestrates startup of all services)      │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐        ┌─────────────────┐
│  FastAPI      │        │   Streamlit     │
│  Backend      │        │   Dashboard     │
│  :8000        │        │   :8501         │
└──────┬────────┘        └────────┬────────┘
       │                          │
       │ imports directly         │
       ▼                          ▼
┌─────────────────────────────────────────┐
│           Search Layer                  │
│  ┌──────────┐  ┌──────────────────┐    │
│  │  BM25    │  │  Vector (FAISS)  │    │
│  │  Index   │  │  Index           │    │
│  └──────────┘  └──────────────────┘    │
│         └──────────┬─────────┘         │
│              HybridSearcher            │
│    hybrid = alpha*norm_bm25            │
│           + (1-alpha)*norm_vector      │
└──────────────────┬──────────────────── ┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│              Data Layer                 │
│  data/index/bm25/     — BM25 artifacts  │
│  data/index/vector/   — FAISS artifacts │
│  data/processed/      — JSONL corpus    │
│  data/eval/           — queries+qrels  │
│  data/metrics/        — experiments CSV│
│  data/search_logs.db  — SQLite logs    │
└─────────────────────────────────────────┘
```

## Request Flow

1. User types query in Streamlit UI
2. Streamlit calls `HybridSearcher.search(query, alpha, top_k)`
3. HybridSearcher fetches top-N from BM25 and Vector independently
4. Scores are normalised (min-max or z-score)
5. Hybrid score computed: `alpha * norm_bm25 + (1-alpha) * norm_vector`
6. Results ranked by hybrid score, highlight snippets generated
7. Request logged to SQLite with latency, query, result count
8. Results returned to UI with full score breakdown

## Database Schema
```sql
CREATE TABLE request_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,
    query        TEXT NOT NULL,
    latency_ms   REAL,
    top_k        INTEGER,
    alpha        REAL,
    result_count INTEGER,
    error        TEXT,
    created_at   TEXT NOT NULL
);
```

## Evaluation Pipeline
```
data/eval/queries.jsonl  +  data/eval/qrels.json
              │
              ▼
       app.eval (python -m app.eval)
              │
              ▼
   nDCG@10 / Recall@10 / MRR@10
              │
              ▼
   data/metrics/experiments.csv
              │
              ▼
   Streamlit Evaluation Page (trend chart)
```