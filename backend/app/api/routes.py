

import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.search.hybrid import HybridSearcher
from app.db import log_request, get_metrics

router = APIRouter()
searcher: Optional[HybridSearcher] = None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    doc_id: str
    title: str
    highlight: str
    bm25_score: float
    vector_score: float
    hybrid_score: float
    norm_bm25: float
    norm_vector: float

class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]
    latency_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    import subprocess, os
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {"status": "ok", "version": "1.0.0", "commit": commit}


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search index not loaded yet.")

    request_id = str(uuid.uuid4())
    start      = time.time()
    error      = None
    results    = []

    try:
        results = searcher.search(
            query   = req.query,
            top_k   = req.top_k,
            alpha   = req.alpha,
            filters = req.filters,
        )
    except Exception as exc:
        error = str(exc)
        raise HTTPException(status_code=500, detail=error)
    finally:
        latency_ms = (time.time() - start) * 1000
        log_request({
            "request_id":   request_id,
            "query":        req.query,
            "latency_ms":   latency_ms,
            "top_k":        req.top_k,
            "alpha":        req.alpha,
            "result_count": len(results),
            "error":        error,
        })

    return SearchResponse(
        query      = req.query,
        total      = len(results),
        results    = results,
        latency_ms = latency_ms,
    )


@router.get("/metrics")
def metrics():
    data = get_metrics()
    lines = [
        "# HELP search_requests_total Total search requests",
        f"search_requests_total {data['total_requests']}",
        "# HELP search_latency_p50_ms Median latency in ms",
        f"search_latency_p50_ms {data['p50_ms']:.2f}",
        "# HELP search_latency_p95_ms P95 latency in ms",
        f"search_latency_p95_ms {data['p95_ms']:.2f}",
        "# HELP search_zero_results_total Requests with zero results",
        f"search_zero_results_total {data['zero_results']}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))