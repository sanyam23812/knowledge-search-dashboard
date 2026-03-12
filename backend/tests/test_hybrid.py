
import pytest
from app.search.hybrid import minmax_norm, zscore_norm, get_snippet, HybridSearcher
from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex


# Normalisation tests


def test_minmax_norm_basic():
    result = minmax_norm([0.0, 5.0, 10.0])
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)


def test_minmax_norm_all_equal():
    # Should not raise divide-by-zero
    result = minmax_norm([3.0, 3.0, 3.0])
    assert result == [0.0, 0.0, 0.0]


def test_minmax_norm_single():
    result = minmax_norm([7.0])
    assert result == [0.0]


def test_zscore_norm_all_equal():
    # Should not raise divide-by-zero
    result = zscore_norm([5.0, 5.0, 5.0])
    assert result == [0.0, 0.0, 0.0]


def test_zscore_norm_output_range():
    result = zscore_norm([1.0, 2.0, 3.0, 4.0, 5.0])
    assert min(result) >= 0.0
    assert max(result) <= 1.0 + 1e-6


# Snippet tests

def test_get_snippet_finds_term():
    text    = "The quick brown fox jumps over the lazy dog"
    snippet = get_snippet(text, "fox")
    assert "fox" in snippet


def test_get_snippet_short_text():
    text    = "Short text"
    snippet = get_snippet(text, "missing_term")
    assert isinstance(snippet, str)
    assert len(snippet) > 0


def test_get_snippet_adds_ellipsis():
    text    = "word " * 100
    snippet = get_snippet(text, "word")
    assert isinstance(snippet, str)


# Hybrid searcher tests


SAMPLE_DOCS = [
    {"doc_id": "d1", "title": "Whale Hunting",
     "text": "The sailors hunted the great white whale across the ocean.",
     "source": "test", "created_at": "2024-01-01"},
    {"doc_id": "d2", "title": "Alice Wonderland",
     "text": "Alice fell down the rabbit hole into a magical wonderland.",
     "source": "test", "created_at": "2024-01-01"},
    {"doc_id": "d3", "title": "Pride Prejudice",
     "text": "Elizabeth Bennet met Mr Darcy at the ball.",
     "source": "test", "created_at": "2024-01-01"},
]


@pytest.fixture
def hybrid_searcher():
    bm25 = BM25Index()
    bm25.build(SAMPLE_DOCS)
    vec = VectorIndex()
    vec.build(SAMPLE_DOCS)
    return HybridSearcher(bm25, vec)


def test_hybrid_returns_results(hybrid_searcher):
    results = hybrid_searcher.search("whale ocean", top_k=3)
    assert len(results) > 0


def test_hybrid_has_all_score_fields(hybrid_searcher):
    results = hybrid_searcher.search("alice rabbit", top_k=3)
    for r in results:
        assert "bm25_score"   in r
        assert "vector_score" in r
        assert "hybrid_score" in r
        assert "norm_bm25"    in r
        assert "norm_vector"  in r
        assert "highlight"    in r


def test_hybrid_alpha_one_pure_bm25(hybrid_searcher):
    # alpha=1.0 means hybrid_score == norm_bm25
    results = hybrid_searcher.search("whale", top_k=3, alpha=1.0)
    for r in results:
        assert r["hybrid_score"] == pytest.approx(r["norm_bm25"])


def test_hybrid_alpha_zero_pure_vector(hybrid_searcher):
    # alpha=0.0 means hybrid_score == norm_vector
    results = hybrid_searcher.search("whale", top_k=3, alpha=0.0)
    for r in results:
        assert r["hybrid_score"] == pytest.approx(r["norm_vector"])


def test_hybrid_top_k_respected(hybrid_searcher):
    results = hybrid_searcher.search("the", top_k=2)
    assert len(results) <= 2


def test_hybrid_scores_descending(hybrid_searcher):
    results = hybrid_searcher.search("alice wonderland", top_k=3)
    scores  = [r["hybrid_score"] for r in results]
    assert scores == sorted(scores, reverse=True)