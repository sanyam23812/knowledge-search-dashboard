# knowledge-search-dashboard
# Knowledge Search Dashboard

screen recording : https://drive.google.com/file/d/1nZMHEDJ2lnUz-wM58AyaStRShMAxzzlK/view?usp=sharing

A hybrid search engine with a KPI dashboard built as an internship assignment.
Combines BM25 lexical search + semantic vector search with a configurable
alpha blending parameter. Fully runs on CPU, no GPU or paid cloud needed.

---
## Prerequisites
- Python 3.11+
- Node is not required (Streamlit only)
- Run this once in PowerShell before anything else:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

Then commit:
```powershell
git add up.bat README.md
git commit -m "fix: update up.bat to handle protobuf fix and index path copying automatically"
git push
```

After this a reviewer can clone and run `up.bat` and it will work! 🚀

## Quick Start (Windows)
```bash
git clone https://github.com/sanyam23812/knowledge-search-dashboard.git
cd knowledge-search-dashboard
up.bat
```

That's it. `up.bat` will:
- Create a Python virtual environment
- Install all dependencies
- Download and ingest the sample corpus
- Build BM25 and vector indexes
- Start the backend API and dashboard UI

**URLs after startup:**
- Dashboard UI  → http://localhost:8501
- Backend API   → http://localhost:8000
- API Docs      → http://localhost:8000/docs

---

## How to Run Tests
```bash
call .venv\Scripts\activate.bat
cd backend
pytest tests/ -v
```

---

## How to Run Evaluation
```bash
call .venv\Scripts\activate.bat
cd backend

# Single run with default alpha=0.5
python -m app.eval --alpha 0.5

# Run 5 experiments varying alpha
python -m app.eval --alpha 0.2 --norm minmax
python -m app.eval --alpha 0.4 --norm minmax
python -m app.eval --alpha 0.6 --norm minmax
python -m app.eval --alpha 0.8 --norm minmax
python -m app.eval --alpha 0.5 --norm zscore
```

Results are saved to `data/metrics/experiments.csv` and visible
on the Evaluation page of the dashboard.

---

## How to Run Ingest + Index Manually
```bash
call .venv\Scripts\activate.bat
cd backend
python -m app.ingest --download --input data/raw --out data/processed
python -m app.index --input data/processed/docs.jsonl
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for full system diagram.

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Lexical Search | rank-bm25 |
| Semantic Search | sentence-transformers + FAISS CPU |
| Dashboard | Streamlit + Plotly |
| Logging | SQLite |
| Tests | pytest |

---

## Project Structure
```
backend/          FastAPI app, search logic, eval harness
frontend/         Streamlit dashboard
data/             Corpus, indexes, eval data, metrics
docs/             Architecture, decision log, codex log, break/fix log
up.bat            One-command startup script (Windows)
requirements.txt  Pinned Python dependencies
```

---

## Key Design Decisions

See [docs/decision_log.md](docs/decision_log.md) for full rationale.

- **Min-max normalisation** as default (z-score also available)
- **all-MiniLM-L6-v2** embedding model — best CPU speed/quality balance
- **FAISS IndexFlatIP** — exact search, fast enough for ~500 docs on CPU
- **Streamlit** over React for simpler Python-only stack

---

## Break/Fix Scenarios

See [docs/break_fix_log.md](docs/break_fix_log.md) for full details.

- **Scenario A** — Embedding model dimension mismatch on startup
- **Scenario B** — SQLite schema migration without column defaults
- **Scenario C** — Divide-by-zero in hybrid score normalisation
