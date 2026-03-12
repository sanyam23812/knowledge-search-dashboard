"""
Unit tests for BM25 index.
"""

import pytest
from app.search.bm25 import BM25Index, tokenize


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = tokenize("Hello World!")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_lowercase():
    tokens = tokenize("Python IS Great")
    assert tokens == ["python", "is", "great"]


def test_tokenize_punctuation():
    tokens = tokenize("one, two. three!")
    assert tokens == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# BM25 index tests
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"doc_id": "d1", "title": "Whale Hunting",
     "text": "The sailors hunted the great white whale across the ocean.",
     "source": "test", "created_at": "2024-01-01"},
    {"doc_id": "d2", "title": "Alice Wonderland",
     "text": "Alice fell down the rabbit hole into a magical wonderland.",
     "source": "test", "created_at": "2024-01-01"},
    {"doc_id": "d3", "title": "Pride and Prejudice",
     "text": "Elizabeth Bennet met Mr Darcy at the ball and felt strong prejudice.",
     "source": "test", "created_at": "2024-01-01"},
]


@pytest.fixture
def bm25_index():
    idx = BM25Index()
    idx.build(SAMPLE_DOCS)
    return idx


def test_bm25_build(bm25_index):
    assert bm25_index.bm25 is not None
    assert len(bm25_index.docs) == 3


def test_bm25_query_returns_results(bm25_index):
    results = bm25_index.query("whale ocean", top_k=3)
    assert len(results) > 0


def test_bm25_query_correct_ordering(bm25_index):
    results = bm25_index.query("whale ocean", top_k=3)
    # d1 should rank first for whale/ocean query
    assert results[0]["doc_id"] == "d1"


def test_bm25_query_has_score(bm25_index):
    results = bm25_index.query("alice rabbit", top_k=3)
    for r in results:
        assert "bm25_score" in r
        assert isinstance(r["bm25_score"], float)


def test_bm25_top_k_respected(bm25_index):
    results = bm25_index.query("the", top_k=2)
    assert len(results) <= 2


def test_bm25_save_and_load(bm25_index, tmp_path):
    bm25_index.index_dir = tmp_path / "bm25"
    bm25_index.save()

    loaded = BM25Index()
    loaded.index_dir = tmp_path / "bm25"
    loaded.load()

    results = loaded.query("whale", top_k=1)
    assert results[0]["doc_id"] == "d1"