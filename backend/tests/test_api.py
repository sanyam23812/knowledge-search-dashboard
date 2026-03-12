

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
import app.api.routes as routes_module
from app.search.hybrid import HybridSearcher
from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex



# Setup mock searcher so tests don't need real indexes


SAMPLE_DOCS = [
    {"doc_id": "d1", "title": "Whale Hunting",
     "text": "The sailors hunted the great white whale.",
     "source": "test", "created_at": "2024-01-01"},
    {"doc_id": "d2", "title": "Alice Wonderland",
     "text": "Alice fell down the rabbit hole.",
     "source": "test", "created_at": "2024-01-01"},
]


@pytest.fixture(autouse=True)
def inject_searcher():
    """Build a real searcher with toy docs and inject it."""
    bm25 = BM25Index()
    bm25.build(SAMPLE_DOCS)
    vec = VectorIndex()
    vec.build(SAMPLE_DOCS)
    routes_module.searcher = HybridSearcher(bm25, vec)
    yield
    routes_module.searcher = None


client = TestClient(app)


# /health

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_has_version():
    response = client.get("/health")
    assert "version" in response.json()



# /search


def test_search_returns_200():
    response = client.post("/search", json={"query": "whale"})
    assert response.status_code == 200


def test_search_has_required_fields():
    response = client.post("/search", json={"query": "whale"})
    data = response.json()
    assert "results"    in data
    assert "query"      in data
    assert "total"      in data
    assert "latency_ms" in data


def test_search_result_has_score_breakdown():
    response = client.post("/search", json={"query": "whale"})
    results  = response.json()["results"]
    assert len(results) > 0
    r = results[0]
    assert "bm25_score"   in r
    assert "vector_score" in r
    assert "hybrid_score" in r
    assert "highlight"    in r


def test_search_top_k_respected():
    response = client.post("/search", json={"query": "the", "top_k": 1})
    assert len(response.json()["results"]) <= 1


def test_search_empty_query_rejected():
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


def test_search_alpha_out_of_range():
    response = client.post("/search", json={"query": "whale", "alpha": 1.5})
    assert response.status_code == 422


def test_search_no_index_returns_503():
    routes_module.searcher = None
    response = client.post("/search", json={"query": "whale"})
    assert response.status_code == 503


# /metrics


def test_metrics_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_contains_counters():
    response = client.get("/metrics")
    text = response.text
    assert "search_requests_total" in text
    assert "search_latency_p50_ms" in text