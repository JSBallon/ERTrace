"""
Tests for bll/router.py — Router

Covers:
  - route(): threshold routing at all zones, exact boundary values
  - compute_priority(): all 12 matrix cells, default fallback
  - FR-LF-05 guardrail: normal path (no event), triggered path (monkeypatched matrix)
  - apply(): immutability, correct field updates
  - compute_review_rate(): empty list, all-REVIEW, mixed
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from bll.router import Router
from bll.schemas import (
    MatchCandidate, MatchResult, ScoreVector,
    ThresholdConfig, WeightsConfig, LegalFormConfig,
)
from governance.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def threshold_config():
    return ThresholdConfig(
        auto_match_threshold=0.92,
        review_lower_threshold=0.70,
    )


@pytest.fixture
def mock_logger():
    """AuditLogger with log_guardrail replaced by a MagicMock for call inspection."""
    logger = MagicMock(spec=AuditLogger)
    return logger


@pytest.fixture
def router(threshold_config, mock_logger):
    return Router(
        config=threshold_config,
        run_id="test_run_router_001",
        audit_logger=mock_logger,
    )


def _make_candidate(
    composite_score: float,
    legal_form_relation: str = "identical",
    routing_zone: str = "REVIEW",
    review_priority: int = 0,
    rank: int = 0,
) -> MatchCandidate:
    """Helper: build a minimal MatchCandidate with given score fields."""
    return MatchCandidate(
        source_b_id="core_001",
        source_b_name="Test Company AG",
        source_b_name_normalized="test company",
        source_b_legal_form="ag",
        score=ScoreVector(
            embedding_cosine_score=composite_score,  # use composite as proxy for simplicity
            jaro_winkler_score=composite_score,
            token_sort_ratio=composite_score,
            legal_form_score=0.5,
            legal_form_relation=legal_form_relation,
            composite_score=composite_score,
        ),
        routing_zone=routing_zone,
        review_priority=review_priority,
        rank=rank,
    )


def _make_match_result(routing_zone: str, run_id: str = "run_001") -> MatchResult:
    """Helper: build a minimal MatchResult with given routing_zone."""
    return MatchResult(
        source_a_id="crm_001",
        source_a_name="Test GmbH",
        source_a_name_normalized="test",
        embedding_cosine_score=0.80,
        jaro_winkler_score=0.80,
        token_sort_ratio=0.80,
        legal_form_score=0.5,
        legal_form_relation="related",
        composite_score=0.80,
        routing_zone=routing_zone,
        review_priority=2,
        run_id=run_id,
        trace_id="trace_001",
        timestamp="2026-03-30T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# route() — threshold routing
# ---------------------------------------------------------------------------

class TestRoute:

    def test_above_auto_match_threshold(self, router):
        """Score clearly above auto_match_threshold → AUTO_MATCH."""
        assert router.route(0.95) == "AUTO_MATCH"

    def test_at_auto_match_threshold(self, router):
        """Exact boundary: score == auto_match_threshold → AUTO_MATCH (>=)."""
        assert router.route(0.92) == "AUTO_MATCH"

    def test_just_below_auto_match_threshold(self, router):
        """Score just below auto_match_threshold → REVIEW (if >= review_lower)."""
        assert router.route(0.919) == "REVIEW"

    def test_in_review_zone(self, router):
        """Score between thresholds → REVIEW."""
        assert router.route(0.80) == "REVIEW"

    def test_at_review_lower_threshold(self, router):
        """Exact boundary: score == review_lower_threshold → REVIEW (>=)."""
        assert router.route(0.70) == "REVIEW"

    def test_just_below_review_lower_threshold(self, router):
        """Score just below review_lower_threshold → NO_MATCH."""
        assert router.route(0.699) == "NO_MATCH"

    def test_zero_score(self, router):
        """Score 0.0 → NO_MATCH."""
        assert router.route(0.0) == "NO_MATCH"

    def test_perfect_score(self, router):
        """Score 1.0 → AUTO_MATCH."""
        assert router.route(1.0) == "AUTO_MATCH"

    def test_return_type_is_str(self, router):
        """Return value is a string (Literal)."""
        result = router.route(0.85)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# compute_priority() — all 12 matrix cells
# ---------------------------------------------------------------------------

class TestComputePriority:

    # Full matrix — 12 cells
    @pytest.mark.parametrize("zone,relation,expected", [
        ("AUTO_MATCH", "identical", 0),
        ("AUTO_MATCH", "related",   2),
        ("AUTO_MATCH", "conflict",  1),
        ("AUTO_MATCH", "unknown",   2),
        ("REVIEW",     "identical", 3),
        ("REVIEW",     "related",   2),
        ("REVIEW",     "conflict",  1),
        ("REVIEW",     "unknown",   2),
        ("NO_MATCH",   "identical", 0),
        ("NO_MATCH",   "related",   0),
        ("NO_MATCH",   "conflict",  0),
        ("NO_MATCH",   "unknown",   0),
    ])
    def test_matrix_cell(self, router, zone, relation, expected):
        """All 12 explicit matrix cells return the correct priority."""
        assert router.compute_priority(zone, relation) == expected

    def test_default_fallback_unknown_zone(self, router):
        """Unknown zone/relation combination falls back to 0."""
        assert router.compute_priority("UNKNOWN_ZONE", "identical") == 0

    def test_default_fallback_unknown_relation(self, router):
        """Unknown legal form relation falls back to 0."""
        assert router.compute_priority("REVIEW", "UNKNOWN_RELATION") == 0

    def test_return_type_is_int(self, router):
        """Return value is always an int."""
        assert isinstance(router.compute_priority("REVIEW", "related"), int)


# ---------------------------------------------------------------------------
# FR-LF-05 guardrail
# ---------------------------------------------------------------------------

class TestFrLf05:

    def test_auto_match_conflict_yields_priority_1(self, router):
        """
        AUTO_MATCH + conflict → review_priority == 1.
        This is the normal path: matrix already returns 1, guardrail does not fire.
        """
        priority = router.compute_priority("AUTO_MATCH", "conflict")
        assert priority == 1

    def test_fr_lf_05_does_not_log_in_normal_case(self, router, mock_logger):
        """
        Normal operation: matrix returns 1 for AUTO_MATCH + conflict.
        The FR-LF-05 if-block condition (priority != 1) is False → no log event.
        """
        router.compute_priority("AUTO_MATCH", "conflict")
        mock_logger.log_guardrail.assert_not_called()

    def test_fr_lf_05_fires_when_matrix_is_wrong(self, router, mock_logger, monkeypatch):
        """
        Defensive path: monkeypatch _PRIORITY_MATRIX so (AUTO_MATCH, conflict) returns 0.
        The FR-LF-05 if-block detects priority != 1 → logs guardrail event, corrects to 1.
        """
        # Patch the class-level constant for this test only
        patched_matrix = dict(Router._PRIORITY_MATRIX)
        patched_matrix[("AUTO_MATCH", "conflict")] = 0  # simulate matrix regression
        monkeypatch.setattr(Router, "_PRIORITY_MATRIX", patched_matrix)

        priority = router.compute_priority("AUTO_MATCH", "conflict")

        # Priority must be corrected to 1 by the guardrail
        assert priority == 1

        # Guardrail log event must have been written
        mock_logger.log_guardrail.assert_called_once()
        call_kwargs = mock_logger.log_guardrail.call_args
        assert call_kwargs.kwargs["guardrail_name"] == "priority_override_FR_LF_05"
        assert call_kwargs.kwargs["triggered"] is True
        assert call_kwargs.kwargs["context"]["computed_priority"] == 0
        assert call_kwargs.kwargs["context"]["routing_zone"] == "AUTO_MATCH"
        assert call_kwargs.kwargs["context"]["legal_form_relation"] == "conflict"

    def test_fr_lf_05_context_contains_run_id(self, router, mock_logger, monkeypatch):
        """Guardrail context dict contains the run_id for audit correlation."""
        patched_matrix = dict(Router._PRIORITY_MATRIX)
        patched_matrix[("AUTO_MATCH", "conflict")] = 0
        monkeypatch.setattr(Router, "_PRIORITY_MATRIX", patched_matrix)

        router.compute_priority("AUTO_MATCH", "conflict")

        context = mock_logger.log_guardrail.call_args.kwargs["context"]
        assert context["run_id"] == "test_run_router_001"

    def test_review_conflict_is_not_subject_to_fr_lf_05(self, router, mock_logger):
        """REVIEW + conflict → priority 1, but no FR-LF-05 log (only AUTO_MATCH triggers it)."""
        priority = router.compute_priority("REVIEW", "conflict")
        assert priority == 1
        mock_logger.log_guardrail.assert_not_called()


# ---------------------------------------------------------------------------
# apply() — immutability and field updates
# ---------------------------------------------------------------------------

class TestApply:

    def test_apply_returns_new_candidate_not_same_object(self, router):
        """apply() returns a new MatchCandidate — does not mutate the input."""
        candidate = _make_candidate(composite_score=0.95, legal_form_relation="identical")
        result = router.apply(candidate)
        assert result is not candidate

    def test_apply_sets_routing_zone(self, router):
        """apply() sets routing_zone correctly based on composite_score."""
        candidate = _make_candidate(composite_score=0.95, legal_form_relation="identical")
        result = router.apply(candidate)
        assert result.routing_zone == "AUTO_MATCH"

    def test_apply_sets_review_priority(self, router):
        """apply() sets review_priority from the 2D matrix."""
        candidate = _make_candidate(composite_score=0.95, legal_form_relation="identical")
        result = router.apply(candidate)
        assert result.review_priority == 0  # AUTO_MATCH + identical = 0

    def test_apply_auto_match_conflict_priority_1(self, router):
        """apply() on AUTO_MATCH + conflict candidate → review_priority == 1."""
        candidate = _make_candidate(composite_score=0.95, legal_form_relation="conflict")
        result = router.apply(candidate)
        assert result.routing_zone == "AUTO_MATCH"
        assert result.review_priority == 1

    def test_apply_review_identical_priority_3(self, router):
        """apply() on REVIEW + identical → review_priority == 3."""
        candidate = _make_candidate(composite_score=0.80, legal_form_relation="identical")
        result = router.apply(candidate)
        assert result.routing_zone == "REVIEW"
        assert result.review_priority == 3

    def test_apply_no_match_priority_0(self, router):
        """apply() on NO_MATCH zone → review_priority == 0."""
        candidate = _make_candidate(composite_score=0.50, legal_form_relation="conflict")
        result = router.apply(candidate)
        assert result.routing_zone == "NO_MATCH"
        assert result.review_priority == 0

    def test_apply_preserves_score_fields(self, router):
        """apply() does not alter any score fields on the returned candidate."""
        candidate = _make_candidate(composite_score=0.80, legal_form_relation="related")
        result = router.apply(candidate)
        assert result.score.composite_score == pytest.approx(0.80)
        assert result.score.legal_form_relation == "related"

    def test_apply_preserves_source_b_fields(self, router):
        """apply() preserves source B identification fields unchanged."""
        candidate = _make_candidate(composite_score=0.80)
        result = router.apply(candidate)
        assert result.source_b_id == candidate.source_b_id
        assert result.source_b_name == candidate.source_b_name
        assert result.rank == candidate.rank

    def test_apply_preserves_rank(self, router):
        """apply() does not change the rank field."""
        candidate = _make_candidate(composite_score=0.80, rank=2)
        result = router.apply(candidate)
        assert result.rank == 2

    def test_apply_input_unchanged(self, router):
        """Original candidate fields are unchanged after apply() call."""
        candidate = _make_candidate(
            composite_score=0.95,
            legal_form_relation="conflict",
            routing_zone="REVIEW",   # deliberately wrong initial zone
            review_priority=0,
        )
        _ = router.apply(candidate)
        # Input must be unchanged (Pydantic model_copy is immutable)
        assert candidate.routing_zone == "REVIEW"
        assert candidate.review_priority == 0


# ---------------------------------------------------------------------------
# compute_review_rate()
# ---------------------------------------------------------------------------

class TestComputeReviewRate:

    def test_empty_list_returns_zero(self, router):
        """Empty results list → 0.0."""
        assert router.compute_review_rate([]) == pytest.approx(0.0)

    def test_all_review(self, router):
        """All entries REVIEW → 1.0."""
        results = [_make_match_result("REVIEW") for _ in range(5)]
        assert router.compute_review_rate(results) == pytest.approx(1.0)

    def test_no_review(self, router):
        """No REVIEW entries → 0.0."""
        results = [_make_match_result("AUTO_MATCH") for _ in range(3)]
        results += [_make_match_result("NO_MATCH") for _ in range(2)]
        assert router.compute_review_rate(results) == pytest.approx(0.0)

    def test_mixed_rate(self, router):
        """3 REVIEW out of 10 total → 0.3."""
        results = (
            [_make_match_result("REVIEW")     for _ in range(3)] +
            [_make_match_result("AUTO_MATCH") for _ in range(5)] +
            [_make_match_result("NO_MATCH")   for _ in range(2)]
        )
        assert router.compute_review_rate(results) == pytest.approx(0.3)

    def test_single_review(self, router):
        """1 REVIEW out of 1 → 1.0."""
        assert router.compute_review_rate([_make_match_result("REVIEW")]) == pytest.approx(1.0)

    def test_single_no_match(self, router):
        """1 NO_MATCH out of 1 → 0.0."""
        assert router.compute_review_rate([_make_match_result("NO_MATCH")]) == pytest.approx(0.0)

    def test_return_type_is_float(self, router):
        """Return type is always float."""
        rate = router.compute_review_rate([_make_match_result("REVIEW")])
        assert isinstance(rate, float)
