"""
Tests for bll/faiss_search.py — FaissSearcher

All tests are pure unit tests — no embedding model required.
Random L2-normalized float32 vectors are used as stand-ins for real embeddings.

sklearn.preprocessing.normalize is used to produce valid L2-normalized vectors.
This is a test-only dependency; the production code uses sentence-transformers.

Run:
    pytest tests/test_faiss_search.py -v
"""

import numpy as np
import pytest
from sklearn.preprocessing import normalize

from bll.faiss_search import FaissSearcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_l2_normalized(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Return (n, d) float32 array of L2-normalized random vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, d)).astype("float32")
    return normalize(raw, axis=1).astype("float32")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_index_built_from_embeddings_b():
    """FaissSearcher constructs without error from valid L2-normalized embeddings."""
    embeddings_b = make_l2_normalized(20, 64)
    searcher = FaissSearcher(embeddings_b)
    assert searcher.n_total == 20


def test_empty_embeddings_b_raises():
    """FaissSearcher raises ValueError when Source B is empty."""
    empty = np.zeros((0, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one vector"):
        FaissSearcher(empty)


def test_non_2d_embeddings_b_raises():
    """FaissSearcher raises ValueError for non-2D input."""
    bad = np.zeros((64,), dtype=np.float32)
    with pytest.raises(ValueError, match="2-dimensional"):
        FaissSearcher(bad)


# ---------------------------------------------------------------------------
# search() — output shape
# ---------------------------------------------------------------------------

def test_search_returns_correct_shape():
    """search() returns (scores, indices) both of shape (n_a, top_k)."""
    embeddings_b = make_l2_normalized(20, 64)
    embeddings_a = make_l2_normalized(5, 64, seed=99)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=3)

    assert scores.shape == (5, 3)
    assert indices.shape == (5, 3)


def test_search_top_k_equals_one():
    """search() with top_k=1 returns shape (n, 1)."""
    embeddings_b = make_l2_normalized(10, 32)
    embeddings_a = make_l2_normalized(4, 32, seed=7)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=1)

    assert scores.shape == (4, 1)
    assert indices.shape == (4, 1)


def test_search_top_k_larger_than_b_clamped():
    """search() clamps top_k to len(B) when top_k > len(B) — no crash."""
    embeddings_b = make_l2_normalized(3, 32)   # only 3 vectors in B
    embeddings_a = make_l2_normalized(2, 32, seed=5)
    searcher = FaissSearcher(embeddings_b)

    # top_k=10 but only 3 vectors in index → clamped to 3
    scores, indices = searcher.search(embeddings_a, top_k=10)

    # Should return 3 columns, not 10
    assert scores.shape[1] == 3
    assert indices.shape[1] == 3


# ---------------------------------------------------------------------------
# search() — score range properties
# ---------------------------------------------------------------------------

def test_search_scores_are_finite():
    """All returned scores must be finite (no NaN or inf)."""
    embeddings_b = make_l2_normalized(15, 64)
    embeddings_a = make_l2_normalized(5, 64, seed=11)
    searcher = FaissSearcher(embeddings_b)

    scores, _ = searcher.search(embeddings_a, top_k=5)

    assert np.all(np.isfinite(scores)), f"Non-finite scores found: {scores}"


def test_search_top_result_is_highest_score():
    """The first column of scores (top-1) must be >= all subsequent columns per row."""
    embeddings_b = make_l2_normalized(20, 64)
    embeddings_a = make_l2_normalized(5, 64, seed=3)
    searcher = FaissSearcher(embeddings_b)

    scores, _ = searcher.search(embeddings_a, top_k=5)

    for i in range(scores.shape[0]):
        assert scores[i, 0] >= scores[i, -1], (
            f"Row {i}: top score {scores[i,0]:.4f} < last score {scores[i,-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# get_candidate() — score clipping and structure
# ---------------------------------------------------------------------------

def test_get_candidate_scores_clipped_to_unit_range():
    """All cosine scores returned by get_candidate() must be in [0.0, 1.0]."""
    embeddings_b = make_l2_normalized(20, 64)
    embeddings_a = make_l2_normalized(5, 64, seed=13)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=5)

    for i in range(embeddings_a.shape[0]):
        candidates = searcher.get_candidate(scores, indices, i)
        for b_idx, cosine_score in candidates:
            assert 0.0 <= cosine_score <= 1.0, (
                f"Entry {i}: score {cosine_score:.6f} out of [0.0, 1.0]"
            )


def test_get_candidate_returns_list_of_tuples():
    """get_candidate() returns a list of (int, float) tuples."""
    embeddings_b = make_l2_normalized(10, 32)
    embeddings_a = make_l2_normalized(2, 32, seed=17)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=3)
    candidates = searcher.get_candidate(scores, indices, i=0)

    assert isinstance(candidates, list)
    for item in candidates:
        assert isinstance(item, tuple) and len(item) == 2
        b_idx, score = item
        assert isinstance(b_idx, int)
        assert isinstance(score, float)


def test_get_candidate_indices_in_valid_range():
    """All b_index values must be valid Source B positions [0, n_total)."""
    embeddings_b = make_l2_normalized(20, 64)
    embeddings_a = make_l2_normalized(5, 64, seed=21)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=5)

    for i in range(embeddings_a.shape[0]):
        candidates = searcher.get_candidate(scores, indices, i)
        for b_idx, _ in candidates:
            assert 0 <= b_idx < 20, f"b_idx {b_idx} out of range [0, 20)"


def test_get_candidate_length_matches_top_k():
    """get_candidate() returns exactly top_k candidates when len(B) >= top_k."""
    embeddings_b = make_l2_normalized(20, 64)
    embeddings_a = make_l2_normalized(3, 64, seed=25)
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(embeddings_a, top_k=5)

    for i in range(embeddings_a.shape[0]):
        candidates = searcher.get_candidate(scores, indices, i)
        assert len(candidates) == 5, f"Entry {i}: expected 5 candidates, got {len(candidates)}"


def test_get_candidate_filters_sentinel_minus_one():
    """get_candidate() excludes FAISS sentinel (-1) indices cleanly."""
    # Use top_k larger than B so FAISS fills some slots with -1
    embeddings_b = make_l2_normalized(2, 32)   # only 2 vectors
    embeddings_a = make_l2_normalized(1, 32, seed=29)
    searcher = FaissSearcher(embeddings_b)

    # Clamped to 2 internally — but we verify no -1 indices leak through
    scores, indices = searcher.search(embeddings_a, top_k=10)
    candidates = searcher.get_candidate(scores, indices, i=0)

    b_indices = [b_idx for b_idx, _ in candidates]
    assert -1 not in b_indices, f"Sentinel -1 found in candidates: {b_indices}"
    assert len(candidates) <= 2


# ---------------------------------------------------------------------------
# Identical vector self-match
# ---------------------------------------------------------------------------

def test_identical_vector_gets_score_near_one():
    """An A vector identical to a B vector should get cosine score ≈ 1.0."""
    embeddings_b = make_l2_normalized(10, 64)
    # Query with the exact same vector as B[3]
    query = embeddings_b[3:4].copy()
    searcher = FaissSearcher(embeddings_b)

    scores, indices = searcher.search(query, top_k=1)
    candidates = searcher.get_candidate(scores, indices, i=0)

    assert len(candidates) == 1
    b_idx, score = candidates[0]
    assert b_idx == 3, f"Expected B[3] as top result, got B[{b_idx}]"
    assert score > 0.9999, f"Expected cosine ≈ 1.0 for identical vector, got {score:.6f}"
