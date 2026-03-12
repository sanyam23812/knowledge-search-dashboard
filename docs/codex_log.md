# Codex Prompt Log

Chronological log of every AI-assisted prompt used during development,
mapped to the file it produced and what was reviewed or edited.

---

## Entry 1
**File:** `backend/app/ingest.py`
**Prompt:**
Write a data ingestion script at backend/app/ingest.py that reads .txt
and .md files from an input folder, does basic whitespace cleanup, splits
documents into ~500 word chunks, and writes output as JSONL with fields:
doc_id, title, text, source, created_at. Also include a download helper
that fetches 10 Project Gutenberg public domain books using urllib.

**What was kept:** Full implementation used as-is.
**Edits made:** Verified chunk splitting logic and URL list manually.

---

## Entry 2
**File:** `backend/app/search/bm25.py`
**Prompt:**
In backend/app/search/bm25.py implement a BM25Index class using rank-bm25.
Provide build(), save(), load(), and query() methods. query() should return
top_k results with bm25_score attached to each result dict. Add a tokenize()
helper that lowercases and strips punctuation.

**What was kept:** Full implementation used as-is.
**Edits made:** Confirmed tokenizer handles edge cases correctly.

---

## Entry 3
**File:** `backend/app/search/vector.py`
**Prompt:**
In backend/app/search/vector.py implement a VectorIndex class using
sentence-transformers and FAISS CPU. Use all-MiniLM-L6-v2 as default model.
Provide build(), save(), load(), query() methods. Save index metadata
(model_name, dimension) to meta.json. Add startup validation in load()
that raises a clear RuntimeError on dimension mismatch.

**What was kept:** Full implementation used as-is.
**Edits made:** Reviewed dimension mismatch error message for clarity.

---

## Entry 4
**File:** `backend/app/search/hybrid.py`
**Prompt:**
In backend/app/search/hybrid.py implement HybridSearcher that combines
BM25 and vector results. Implement two normalisation strategies: minmax_norm
and zscore_norm, both handling divide-by-zero when all scores are equal.
Hybrid score formula: alpha * norm_bm25 + (1-alpha) * norm_vector.
Include a get_snippet() function for highlight extraction.

**What was kept:** Full implementation used as-is.
**Edits made:** Verified divide-by-zero guard returns 0.0 not NaN.

---

## Entry 5
**File:** `backend/app/api/routes.py`
**Prompt:**
In backend/app/api/routes.py add FastAPI routes: GET /health returning
status/version/commit, POST /search accepting query/top_k/alpha/filters
and returning ranked results with full score breakdown, GET /metrics
returning Prometheus-style text with request counts and latency stats.
Add input validation via Pydantic models.

**What was kept:** Full implementation used as-is.
**Edits made:** Reviewed Pydantic field constraints for alpha range.

---

## Entry 6
**File:** `backend/app/db.py`
**Prompt:**
In backend/app/db.py implement SQLite logging using sqlite3. Include
init_db() to create the request_logs table, log_request() to insert
a row per search request, get_metrics() to compute total requests,
p50/p95 latency, and zero-result count, and get_all_logs() to return
all rows as dicts.

**What was kept:** Full implementation used as-is.
**Edits made:** Verified schema matches architecture.md documentation.

---

## Entry 7
**File:** `backend/app/main.py`
**Prompt:**
In backend/app/main.py create the FastAPI app entry point. On startup,
call init_db(), load BM25 and vector indexes, and inject a HybridSearcher
into the routes module. Add CORS middleware. Handle index load failures
gracefully with a warning message instead of crashing.

**What was kept:** Full implementation used as-is.
**Edits made:** Tested graceful degradation when indexes missing.

---

## Entry 8
**File:** `backend/app/index.py`
**Prompt:**
In backend/app/index.py write an indexing pipeline script that loads
docs from a JSONL file, builds a BM25Index and VectorIndex, and saves
both to disk. Should be runnable as python -m app.index --input path.

**What was kept:** Full implementation used as-is.
**Edits made:** Added check for missing input file with helpful message.

---

## Entry 9
**File:** `backend/app/eval.py`
**Prompt:**
In backend/app/eval.py write an evaluation harness that loads queries
from JSONL and qrels from JSON, runs hybrid search for each query,
computes nDCG@10, Recall@10, MRR@10, prints results, and appends a
row to data/metrics/experiments.csv with timestamp and git commit hash.

**What was kept:** Full implementation used as-is.
**Edits made:** Verified metric formulas against standard IR definitions.

---

## Entry 10
**File:** `data/eval/queries.jsonl` and `data/eval/qrels.json`
**Prompt:**
Create 25 labeled evaluation queries covering the 10 Gutenberg books
in the corpus. Each query should target a specific book's themes.
Format queries as JSONL with query_id and query fields. Format qrels
as JSON mapping query_id to list of relevant doc_id chunk strings.

**What was kept:** Full data used as-is.
**Edits made:** Reviewed query relevance mapping against actual book content.

---

## Entry 11
**File:** `frontend/app.py`
**Prompt:**
In frontend/app.py build a Streamlit dashboard with 4 pages via sidebar
navigation: Search page with query box, alpha slider, result cards showing
score breakdown; KPI page with p50/p95 latency, request volume chart, top
queries table, zero-result queries; Evaluation page with nDCG trend chart
and experiment table; Debug page with log table filterable by errors and
time range.

**What was kept:** Full implementation used as-is.
**Edits made:** Adjusted layout columns for better readability.

---

## Entry 12
**File:** `backend/tests/test_bm25.py`
**Prompt:**
In backend/tests/test_bm25.py write pytest tests for the BM25 module
using a 3-doc toy corpus. Cover: tokenizer lowercasing and punctuation
handling, index build, query result ordering, bm25_score field presence,
top_k limit, and save/load round-trip.

**What was kept:** Full implementation used as-is.
**Edits made:** Verified deterministic ordering with toy corpus.

---

## Entry 13
**File:** `backend/tests/test_hybrid.py`
**Prompt:**
In backend/tests/test_hybrid.py write pytest tests covering minmax_norm
and zscore_norm divide-by-zero edge cases, get_snippet behaviour, and
HybridSearcher with alpha=1.0 (pure BM25), alpha=0.0 (pure vector),
score field presence, top_k limit, and descending score ordering.

**What was kept:** Full implementation used as-is.
**Edits made:** Added approx() tolerance for floating point comparisons.

---

## Entry 14
**File:** `backend/tests/test_api.py`
**Prompt:**
In backend/tests/test_api.py write FastAPI TestClient contract tests
for /health, /search, and /metrics. Use a fixture that injects a real
HybridSearcher built from a toy corpus. Test: 200 responses, required
fields, score breakdown presence, top_k enforcement, empty query
rejection (422), alpha out of range (422), and 503 when no index loaded.

**What was kept:** Full implementation used as-is.
**Edits made:** Reviewed fixture teardown to reset searcher to None.

---

## Entry 15
**File:** `up.bat`
**Prompt:**
Write a Windows batch file up.bat that creates a .venv if missing,
activates it, installs requirements.txt, runs ingest and index only
if artifacts are missing, then starts uvicorn backend on port 8000
and streamlit frontend on port 8501 in separate terminal windows,
printing the local URLs.

**What was kept:** Full implementation used as-is.
**Edits made:** Tested each step individually before full run.