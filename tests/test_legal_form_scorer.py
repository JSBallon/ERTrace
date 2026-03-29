"""
Tests for bll/legal_form_scorer.py — LegalFormScorer

All tests are pure unit tests — no embedding model required, no FAISS.

Key behaviour verified:
  - Return types: (float, str)
  - Score range: [0.0, 1.0]
  - Relation literals: one of {"identical", "related", "conflict", "unknown"}
  - Default config scores match spec: identical=1.0, related=0.5, conflict=0.0, unknown=0.5
  - Configurable scores are respected
  - GmbH↔AG → "conflict" (not "related") — DAL strict-AND, type mismatch
  - GmbH↔Ltd. → "conflict"
  - GmbH↔GmbH → "identical", score=1.0
  - No legal form → "unknown", score=0.5

Run:
    pytest tests/test_legal_form_scorer.py -v
"""

import pytest
from bll.legal_form_scorer import LegalFormScorer
from bll.schemas import LegalFormConfig

VALID_RELATIONS = {"identical", "related", "conflict", "unknown"}


@pytest.fixture
def default_scorer() -> LegalFormScorer:
    """Scorer with default LegalFormConfig (identical=1.0, related=0.5, conflict=0.0, unknown=0.5)."""
    return LegalFormScorer(LegalFormConfig())


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

def test_score_returns_tuple(default_scorer):
    """score() returns a tuple."""
    result = default_scorer.score("Alpha GmbH", "Beta GmbH")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_score_returns_float_and_str(default_scorer):
    """score() returns (float, str)."""
    score, relation = default_scorer.score("Alpha GmbH", "Beta GmbH")
    assert isinstance(score, float)
    assert isinstance(relation, str)


def test_relation_is_valid_literal(default_scorer):
    """relation must be one of the four valid literals."""
    pairs = [
        ("Alpha GmbH", "Beta GmbH"),
        ("Alpha GmbH", "Beta AG"),
        ("Alpha GmbH", "Beta Ltd."),
        ("Alpha", "Beta"),
        ("Alpha GmbH", "Beta"),
    ]
    for name_a, name_b in pairs:
        _, relation = default_scorer.score(name_a, name_b)
        assert relation in VALID_RELATIONS, (
            f"Unexpected relation {relation!r} for ({name_a!r}, {name_b!r})"
        )


def test_score_in_unit_range(default_scorer):
    """Score must be in [0.0, 1.0] for all inputs."""
    pairs = [
        ("Alpha GmbH", "Beta GmbH"),
        ("Alpha GmbH", "Beta AG"),
        ("Alpha GmbH", "Beta Ltd."),
        ("Alpha", "Beta"),
        ("Alpha GmbH", "Beta"),
    ]
    for name_a, name_b in pairs:
        score, _ = default_scorer.score(name_a, name_b)
        assert 0.0 <= score <= 1.0, (
            f"Score {score} out of [0.0, 1.0] for ({name_a!r}, {name_b!r})"
        )


# ---------------------------------------------------------------------------
# identical relation
# ---------------------------------------------------------------------------

def test_identical_gmbh_gmbh(default_scorer):
    """Same GmbH term on both sides → identical, score=1.0."""
    score, relation = default_scorer.score("Alpha GmbH", "Beta GmbH")
    assert relation == "identical", f"Expected 'identical', got {relation!r}"
    assert score == 1.0, f"Expected score=1.0, got {score}"


def test_identical_ag_ag(default_scorer):
    """Same AG term on both sides → identical, score=1.0."""
    score, relation = default_scorer.score("Alpha AG", "Beta AG")
    assert relation == "identical"
    assert score == 1.0


def test_identical_llc_llc(default_scorer):
    """Same LLC term on both sides → identical, score=1.0."""
    score, relation = default_scorer.score("Alpha LLC", "Beta LLC")
    assert relation == "identical"
    assert score == 1.0


# ---------------------------------------------------------------------------
# conflict relation
# ---------------------------------------------------------------------------

def test_conflict_gmbh_ltd(default_scorer):
    """GmbH (DE) vs Ltd. (UK) → conflict, score=0.0 (type mismatch + no country overlap)."""
    score, relation = default_scorer.score("ACME GmbH", "ACME Ltd.")
    assert relation == "conflict", f"Expected 'conflict', got {relation!r}"
    assert score == 0.0, f"Expected score=0.0, got {score}"


def test_conflict_gmbh_ag(default_scorer):
    """
    GmbH vs AG → conflict, score=0.0.

    This is the critical ADR-M2-004 case: the system patterns doc shows GmbH↔AG as
    'related', but the DAL strict-AND logic classifies it as 'conflict' because
    GmbH type='Limited' and AG type='Corporation' — type mismatch overrides country overlap.
    """
    score, relation = default_scorer.score("ACME GmbH", "ACME AG")
    assert relation == "conflict", (
        f"Expected 'conflict' for GmbH↔AG (strict-AND, type mismatch), got {relation!r}"
    )
    assert score == 0.0


def test_conflict_gmbh_llc(default_scorer):
    """GmbH (Limited/DE) vs LLC (Limited Liability Company/US) → conflict, score=0.0."""
    score, relation = default_scorer.score("Alpha GmbH", "Alpha LLC")
    assert relation == "conflict"
    assert score == 0.0


# ---------------------------------------------------------------------------
# unknown relation
# ---------------------------------------------------------------------------

def test_unknown_no_legal_form_on_a(default_scorer):
    """No legal form on A → unknown, score=0.5."""
    score, relation = default_scorer.score("ACME", "Beta GmbH")
    assert relation == "unknown", f"Expected 'unknown', got {relation!r}"
    assert score == 0.5


def test_unknown_no_legal_form_on_b(default_scorer):
    """No legal form on B → unknown, score=0.5."""
    score, relation = default_scorer.score("Alpha GmbH", "ACME")
    assert relation == "unknown"
    assert score == 0.5


def test_unknown_both_no_legal_form(default_scorer):
    """No legal form on either side → unknown, score=0.5."""
    score, relation = default_scorer.score("Alpha", "Beta")
    assert relation == "unknown"
    assert score == 0.5


# ---------------------------------------------------------------------------
# Configurable scores are respected
# ---------------------------------------------------------------------------

def test_custom_config_scores():
    """Custom LegalFormConfig values must be reflected in score output."""
    config = LegalFormConfig(
        identical_score=0.9,
        related_score=0.4,
        conflict_score=0.1,
        unknown_score=0.3,
    )
    scorer = LegalFormScorer(config)

    score_id, _ = scorer.score("Alpha GmbH", "Beta GmbH")   # identical
    assert score_id == 0.9

    score_un, _ = scorer.score("Alpha", "Beta GmbH")        # unknown
    assert score_un == 0.3

    score_cf, _ = scorer.score("Alpha GmbH", "Beta Ltd.")   # conflict
    assert score_cf == 0.1


def test_default_config_identical_score():
    """Default LegalFormConfig identical_score must be 1.0."""
    scorer = LegalFormScorer(LegalFormConfig())
    score, _ = scorer.score("Commerzbank AG", "Deutsche Bank AG")
    # Both AG → identical; default identical_score = 1.0
    assert score == 1.0


def test_default_config_unknown_score():
    """Default LegalFormConfig unknown_score must be 0.5."""
    scorer = LegalFormScorer(LegalFormConfig())
    score, relation = scorer.score("Holding Company", "Another Holding")
    assert relation == "unknown"
    assert score == 0.5


# ---------------------------------------------------------------------------
# Governance validation pair: ACME GmbH ↔ ACME Ltd.
# (M2 exit criteria: legal_form_relation='conflict', legal_form_score=0.0)
# ---------------------------------------------------------------------------

def test_acme_gmbh_vs_acme_ltd_exit_criteria(default_scorer):
    """
    M2 exit criteria validation: ACME GmbH ↔ ACME Ltd.
    Must yield legal_form_relation='conflict' and legal_form_score=0.0.
    """
    score, relation = default_scorer.score("ACME GmbH", "ACME Ltd.")
    assert relation == "conflict", (
        f"M2 exit criteria FAILED: expected relation='conflict', got {relation!r}"
    )
    assert score == 0.0, (
        f"M2 exit criteria FAILED: expected score=0.0, got {score}"
    )
