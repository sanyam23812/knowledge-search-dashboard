

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, searcher as _searcher
from app.db import init_db
from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex
from app.search.hybrid import HybridSearcher
import app.api.routes as routes_module

app = FastAPI(title="Knowledge Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    init_db()
    try:
        bm25 = BM25Index()
        bm25.load()
        vec = VectorIndex()
        vec.load()
        routes_module.searcher = HybridSearcher(bm25, vec)
        print("[Startup] Search indexes loaded successfully.")
    except Exception as e:
        print(f"[Startup] Warning: could not load indexes: {e}")
        print("[Startup] Run ingest + index pipeline first.")