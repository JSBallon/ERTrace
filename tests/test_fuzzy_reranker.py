"""
Tests for bll/fuzzy_reranker.py — FuzzyReranker

All tests are pure unit tests — no embedding model required, no FAISS.

Known score values are verified against manually computed results to confirm:
  - JaroWinkler.similarity() is used (not /100 — already [0.0, 1.0])
  - token_sort_ratio() is correctly divided by 100

Run:
    pytest tests/test_fuzzy_reranker.py -v
"""

import pytest
from bll.fuzzy_reranker import FuzzyReranker


@pytest.fixture
def reranker() -> FuzzyReranker:
    return FuzzyReranker()


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

def test_score_returns_tuple(reranker):
    """score() returns a tuple of exactly two elements."""
    result = reranker.score("deutsche bank", "deutsche bahn")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_score_returns_floats(reranker):
    """score() returns (float, float)."""
    jw, ts = reranker.score("deutsche bank", "deutsche bahn")
    assert isinstance(jw, float)
    assert isinstance(ts, float)


# ---------------------------------------------------------------------------
# Score range — both metrics must stay in [0.0, 1.0]
# ---------------------------------------------------------------------------

VARIED_PAIRS = [
    ("deutsche bank", "deutsche bahn"),
    ("bayerische landesbank", "bayernlb"),
    ("acme", "acme"),
    ("allianz", "alliance"),
    ("siemens", "siemens ag"),
    ("alpha beta gamma", "gamma beta alpha"),
    ("short", "a very much longer company name here"),
    ("abc", "xyz"),
    ("commerzbank", "commerzbank aktiengesellschaft"),
    ("x", "y"),
]

@pytest.mark.parametrize("a, b", VARIED_PAIRS)
def test_jaro_winkler_in_unit_range(reranker, a, b):
    """Jaro-Winkler score must be in [0.0, 1.0] for all inputs."""
    jw, _ = reranker.score(a, b)
    assert 0.0 <= jw <= 1.0, f"JW={jw:.6f} out of range for ({a!r}, {b!r})"


@pytest.mark.parametrize("a, b", VARIED_PAIRS)
def test_token_sort_ratio_in_unit_range(reranker, a, b):
    """Token Sort Ratio must be in [0.0, 1.0] for all inputs."""
    _, ts = reranker.score(a, b)
    assert 0.0 <= ts <= 1.0, f"TS={ts:.6f} out of range for ({a!r}, {b!r})"


# ---------------------------------------------------------------------------
# Identical inputs → both scores == 1.0
# ---------------------------------------------------------------------------

def test_score_identical_names(reranker):
    """Identical normalized names must yield jw=1.0 and ts=1.0."""
    jw, ts = reranker.score("deutsche bank", "deutsche bank")
    assert jw == 1.0, f"Expected JW=1.0 for identical names, got {jw}"
    assert ts == 1.0, f"Expected TS=1.0 for identical names, got {ts}"


def test_score_identical_single_token(reranker):
    """Single identical token: both scores == 1.0."""
    jw, ts = reranker.score("siemens", "siemens")
    assert jw == 1.0
    assert ts == 1.0


# ---------------------------------------------------------------------------
# Completely different → both scores low
# ---------------------------------------------------------------------------

def test_score_completely_different(reranker):
    """Completely different names should score low on both metrics."""
    jw, ts = reranker.score("alpha gmbh", "omega corp")
    assert jw < 0.7, f"Expected JW < 0.7 for different names, got {jw:.6f}"
    assert ts < 0.5, f"Expected TS < 0.5 for different names, got {ts:.6f}"


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_score_is_symmetric_jaro_winkler(reranker):
    """JW score must be identical regardless of argument order."""
    jw_ab, _ = reranker.score("deutsche bank", "deutsche bahn")
    jw_ba, _ = reranker.score("deutsche bahn", "deutsche bank")
    assert jw_ab == jw_ba, f"JW not symmetric: {jw_ab} vs {jw_ba}"


def test_score_is_symmetric_token_sort(reranker):
    """TS score must be identical regardless of argument order."""
    _, ts_ab = reranker.score("bayerische landesbank", "landesbank bayerische")
    _, ts_ba = reranker.score("landesbank bayerische", "bayerische landesbank")
    assert ts_ab == ts_ba, f"TS not symmetric: {ts_ab} vs {ts_ba}"


# ---------------------------------------------------------------------------
# Precision — rounded to 6 decimal places
# ---------------------------------------------------------------------------

def test_score_rounded_to_6_decimals(reranker):
    """Both scores must be rounded to at most 6 decimal places."""
    jw, ts = reranker.score("commerzbank", "dresdner bank")
    # Check by verifying re-rounding does not change the value
    assert round(jw, 6) == jw, f"JW not rounded to 6dp: {jw}"
    assert round(ts, 6) == ts, f"TS not rounded to 6dp: {ts}"


# ---------------------------------------------------------------------------
# Token Sort Ratio: order invariance
# ---------------------------------------------------------------------------

def test_token_sort_order_invariant(reranker):
    """Token Sort Ratio must be 1.0 when tokens are identical but reordered."""
    _, ts = reranker.score("bank deutsche", "deutsche bank")
    assert ts == 1.0, f"Expected TS=1.0 for reordered tokens, got {ts:.6f}"


def test_token_sort_three_tokens_reordered(reranker):
    """Three-token reordering must yield TS=1.0."""
    _, ts = reranker.score("gamma alpha beta", "alpha beta gamma")
    assert ts == 1.0, f"Expected TS=1.0 for reordered tokens, got {ts:.6f}"


# ---------------------------------------------------------------------------
# Jaro-Winkler: prefix sensitivity
# ---------------------------------------------------------------------------

def test_jaro_winkler_prefix_boost(reranker):
    """JW must score higher for strings sharing a common prefix vs. non-prefix match."""
    # "deut" prefix shared → JW prefix boost applies
    jw_prefix, _ = reranker.score("deutsche bank", "deutsche bahn")
    # No shared prefix
    jw_no_prefix, _ = reranker.score("alpha inc", "omega ltd")
    assert jw_prefix > jw_no_prefix, (
        f"Expected JW higher for prefix-sharing pair: {jw_prefix:.4f} vs {jw_no_prefix:.4f}"
    )


# ---------------------------------------------------------------------------
# Known pair: Deutsche Bank vs Deutsche Bahn (governance validation pair)
# ---------------------------------------------------------------------------

def test_deutsche_bank_vs_bahn_known_scores(reranker):
    """
    Deutsche Bank vs Deutsche Bahn: verified known scores.

    This pair is the canonical false-positive risk case. Both fuzzy metrics
    score it highly because the names differ by only one letter.
    The composite scorer (with embedding cosine) must prevent AUTO_MATCH.

    Verified values (rapidfuzz 3.14.3):
      JW = 0.969231
      TS = 0.923077
    """
    jw, ts = reranker.score("deutsche bank", "deutsche bahn")
    assert jw > 0.95, f"Expected JW > 0.95 for Deutsche Bank/Bahn, got {jw:.6f}"
    assert ts > 0.85, f"Expected TS > 0.85 for Deutsche Bank/Bahn, got {ts:.6f}"


# ---------------------------------------------------------------------------
# API correctness guard: JW must NOT be in millirange (catches /100 bug)
# ---------------------------------------------------------------------------

def test_jaro_winkler_not_divided_by_100(reranker):
    """
    Guard against the /100 scaling bug from the old fuzz.jaro_winkler_similarity API.

    JaroWinkler.similarity() returns [0.0, 1.0] directly.
    If someone mistakenly adds /100, identical names would return 0.01 instead of 1.0.
    """
    jw, _ = reranker.score("allianz se", "allianz se")
    assert jw > 0.5, (
        f"JW={jw:.6f} looks like a /100 bug — identical names should give JW=1.0"
    )


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------

def test_score_batch_returns_correct_length(reranker):
    """score_batch() returns a list of the same length as the input."""
    pairs = [
        ("deutsche bank", "deutsche bahn"),
        ("bayerische landesbank", "bayernlb"),
        ("acme gmbh", "acme ltd"),
        ("siemens", "siemens ag"),
        ("allianz", "allianz se"),
    ]
    results = reranker.score_batch(pairs)
    assert len(results) == 5


def test_score_batch_matches_individual_calls(reranker):
    """score_batch() must produce identical results to individual score() calls."""
    pairs = [
        ("deutsche bank", "deutsche bahn"),
        ("bayerische landesbank", "bayernlb"),
        ("acme gmbh", "acme ltd"),
    ]
    batch_results = reranker.score_batch(pairs)
    individual_results = [reranker.score(a, b) for a, b in pairs]
    assert batch_results == individual_results


def test_score_batch_empty_input(reranker):
    """score_batch() with empty input returns empty list."""
    assert reranker.score_batch([]) == []


def test_score_batch_single_pair(reranker):
    """score_batch() with a single pair returns a list with one tuple."""
    results = reranker.score_batch([("allianz", "allianz se")])
    assert len(results) == 1
    jw, ts = results[0]
    assert 0.0 <= jw <= 1.0
    assert 0.0 <= ts <= 1.0
