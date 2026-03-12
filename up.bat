@echo off
echo ============================================
echo  Knowledge Search Dashboard — Startup
echo ============================================

:: Step 1 — Create virtual environment if missing
if not exist .venv (
    echo [1/6] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/6] Virtual environment already exists.
)

:: Step 2 — Activate and install dependencies
echo [2/6] Installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
pip install protobuf --upgrade --quiet

:: Step 3 — Download and ingest data if missing
if not exist data\processed\docs.jsonl (
    echo [3/6] Downloading and ingesting sample corpus...
    cd backend
    python -m app.ingest --download --input ../data/raw --out ../data/processed
    cd ..
) else (
    echo [3/6] Processed data already exists, skipping ingest.
)

:: Step 4 — Build indexes if missing
if not exist data\index\bm25\index.pkl (
    echo [4/6] Building BM25 and vector indexes...
    cd backend
    python -m app.index --input ../data/processed/docs.jsonl
    cd ..
) else (
    echo [4/6] Indexes already exist, skipping index build.
)

:: Step 5 — Copy indexes to root data folder if needed
if not exist data\index\bm25\index.pkl (
    echo [5/6] Copying indexes to correct location...
    xcopy /E /I /Y backend\data\index\bm25 data\index\bm25
    xcopy /E /I /Y backend\data\index\vector data\index\vector
) else (
    echo [5/6] Indexes already in correct location.
)

:: Step 6 — Start backend and frontend
echo [6/6] Starting backend and frontend...
echo.
echo Backend API  : http://localhost:8000
echo Frontend UI  : http://localhost:8501
echo API Docs     : http://localhost:8000/docs
echo.

start "Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
start "Frontend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && cd frontend && streamlit run app.py --server.port 8501"

echo Both services started!
echo.
echo NOTE: First run takes 5-10 minutes to download books and build indexes.
pause