    """score() result must be rounded to at most 6 decimal places."""
    result = scorer.score(0.333333, 0.666666, 0.111111, 0.888888)
    assert round(result, 6) == result, f"Result not rounded to 6dp: {result}"


# ---------------------------------------------------------------------------
# verify() — consistency guard
# ---------------------------------------------------------------------------

def test_verify_passes_for_own_output(scorer):
    """verify() must return True when computed = score(...)."""
    inputs = (0.90, 0.80, 0.85, 1.0)
    computed = scorer.score(*inputs)
    assert scorer.verify(computed, *inputs) is True


def test_verify_passes_all_zeros(scorer):
    """verify() returns True for all-zero inputs."""
    computed = scorer.score(0.0, 0.0, 0.0, 0.0)
    assert scorer.verify(computed, 0.0, 0.0, 0.0, 0.0) is True


def test_verify_passes_all_ones(scorer):
    """verify() returns True for all-one inputs."""
    computed = scorer.score(1.0, 1.0, 1.0, 1.0)
    assert scorer.verify(computed, 1.0, 1.0, 1.0, 1.0) is True


def test_verify_fails_on_large_deviation(scorer):
    """verify() returns False when stored value is far from recomputed value."""
    # actual score(0.0, 0.0, 0.0, 0.0) = 0.0, but stored as 0.99
    assert scorer.verify(0.99, 0.0, 0.0, 0.0, 0.0) is False


def test_verify_fails_on_moderate_deviation(scorer):
    """verify() returns False for deviation of ~0.5."""
    computed = scorer.score(1.0, 1.0, 1.0, 1.0)   # = 1.0
    # Pass zero inputs — recomputed = 0.0, deviation = 1.0
    assert scorer.verify(computed, 0.0, 0.0, 0.0, 0.0) is False


def test_verify_passes_within_tolerance_boundary(scorer):
    """
    verify() returns True when |deviation| < 0.001.

    Note: float addition makes exactly 0.001 unreachable cleanly
    (0.88 + 0.001 = 0.8810000...0009 which is > 0.001 by float epsilon).
    We use 0.0009 which is cleanly within tolerance.
    """
    inputs = (0.90, 0.80, 0.85, 1.0)
    computed = scorer.score(*inputs)
    # 0.0009 is cleanly within the 0.001 tolerance
    assert scorer.verify(computed + 0.0009, *inputs) is True
    assert scorer.verify(computed - 0.0009, *inputs) is True


def test_verify_fails_just_above_tolerance(scorer):
    """verify() returns False when |deviation| = 0.0011 (just above 0.001)."""
    inputs = (0.90, 0.80, 0.85, 1.0)
    computed = scorer.score(*inputs)
    assert scorer.verify(computed + 0.0011, *inputs) is False
    assert scorer.verify(computed - 0.0011, *inputs) is False


def test_verify_returns_bool(scorer):
    """verify() must return a Python bool."""
    inputs = (0.5, 0.5, 0.5, 0.5)
    result = scorer.verify(scorer.score(*inputs), *inputs)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# WeightsConfig validation — enforced by Pydantic, not CompositeScorer itself
# ---------------------------------------------------------------------------

def test_weights_sum_constraint_rejects_invalid():
    """WeightsConfig raises ValidationError when weights do not sum to 1.0 ± 0.001."""
    with pytest.raises(ValidationError):
        WeightsConfig(
            w_embedding=0.50,
            w_jaro_winkler=0.20,
            w_token_sort=0.20,
            w_legal_form=0.20,   # sum = 1.10 → invalid
        )


def test_weights_sum_constraint_accepts_within_tolerance():
    """
    WeightsConfig accepts weights whose sum is within ±0.001 of 1.0.

    Note: float addition is imprecise — 0.501+0.2+0.2+0.1 = 1.0010000000000001
    which exceeds the tolerance. We use weights that sum to exactly 1.0 in
    float arithmetic to confirm valid configs are accepted without error.
    """
    # These sum to 0.9999999999999999 in float64 — well within tolerance
    config = WeightsConfig(
        w_embedding=0.5005,
        w_jaro_winkler=0.1995,
        w_token_sort=0.2000,
        w_legal_form=0.1000,
    )
    assert config is not None