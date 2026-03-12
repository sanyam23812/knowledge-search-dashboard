@echo off
echo ============================================
echo  Knowledge Search Dashboard — Startup
echo ============================================

:: Step 1 — Create virtual environment if missing
if not exist .venv (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/5] Virtual environment already exists.
)

:: Step 2 — Activate and install dependencies
echo [2/5] Installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

:: Step 3 — Download and ingest data if missing
if not exist data\processed\docs.jsonl (
    echo [3/5] Downloading and ingesting sample corpus...
    python -m app.ingest --download --input data/raw --out data/processed
) else (
    echo [3/5] Processed data already exists, skipping ingest.
)

:: Step 4 — Build indexes if missing
if not exist data\index\bm25\index.pkl (
    echo [4/5] Building BM25 and vector indexes...
    python -m app.index --input data/processed/docs.jsonl
) else (
    echo [4/5] Indexes already exist, skipping index build.
)

:: Step 5 — Start backend and frontend
echo [5/5] Starting backend and frontend...
echo.
echo Backend API  : http://localhost:8000
echo Frontend UI  : http://localhost:8501
echo API Docs     : http://localhost:8000/docs
echo.

:: Start FastAPI in background
start "Backend" cmd /k "call .venv\Scripts\activate.bat && cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start Streamlit in background
start "Frontend" cmd /k "call .venv\Scripts\activate.bat && cd frontend && streamlit run app.py --server.port 8501"

echo Both services started! Press any key to exit this window.
pause