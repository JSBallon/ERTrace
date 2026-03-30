"""
BLL — Router

Threshold routing and 2D review priority assignment for the TGFR pipeline.
Replaces the M2 inline routing placeholder in pipeline.py.

Responsibilities:
  1. route()           — map a composite score to AUTO_MATCH / REVIEW / NO_MATCH
                         using the configured ThresholdConfig.
  2. compute_priority()— map (routing_zone, legal_form_relation) to a review priority
                         integer (0–3) via the 2D _PRIORITY_MATRIX.
                         FR-LF-05 standalone if-block enforces priority=1 for
                         AUTO_MATCH + conflict as a redundant defensive control.
  3. apply()           — convenience wrapper: route + compute_priority on a single
                         MatchCandidate; returns an immutably updated copy.
  4. compute_review_rate() — fraction of MatchResult entries with routing_zone=REVIEW;
                             used by Streamlit and CLI for the REVIEW rate warning.

Design decisions (ADR-M3-002):
  - _PRIORITY_MATRIX is a class-level constant so tests can monkeypatch individual cells
    without restructuring the Router or wrapping the method.
  - apply() uses model_copy(update={...}) — immutable, no side effects on input candidate.
  - FR-LF-05 guardrail fires only when the matrix is incorrect (priority != 1 for
    AUTO_MATCH + conflict). In normal operation the matrix already returns 1 for this cell,
    so the event does not appear in the JSONL — by design (defensive redundancy).
  - log_guardrail() is called with triggered= (not trigger=) to match the actual
    AuditLogger signature. run_id is passed inside the context dict, not as a kwarg.

No Streamlit imports. No filesystem access. No external API calls.
"""

from typing import Literal

from bll.schemas import MatchCandidate, MatchResult, ThresholdConfig
from governance.audit_logger import AuditLogger


class Router:
    """
    Threshold router and 2D review priority calculator.

    Instantiated once per pipeline run with the active ThresholdConfig and
    AuditLogger. All methods are stateless with respect to per-entry data —
    the only instance state is the shared config, run_id, and logger.

    Usage (pipeline.py Task 3):
        router = Router(config.threshold_config, config.run_id, audit_logger)
        routed_candidate = router.apply(candidate)
        review_rate = router.compute_review_rate(results)
    """

    # -----------------------------------------------------------------------
    # 2D Review Priority Matrix (ADR-M3-002)
    # -----------------------------------------------------------------------
    # Class-level constant — domain knowledge, not run-time config.
    # Exposed at class level to allow targeted monkeypatching in tests
    # (verifying the FR-LF-05 defensive guardrail fires on matrix error).
    #
    # Priority semantics:
    #   0 = no review needed
    #   1 = mandatory review (highest urgency)
    #   2 = standard review
    #   3 = low-urgency review
    #
    # All 12 cells are explicit. Default fallback in compute_priority() is 0.
    # -----------------------------------------------------------------------
    _PRIORITY_MATRIX: dict[tuple[str, str], int] = {
        # AUTO_MATCH zone
        ("AUTO_MATCH", "identical"): 0,  # High score + same legal form → auto-approve
        ("AUTO_MATCH", "related"):   2,  # High score + related form → standard review
        ("AUTO_MATCH", "conflict"):  1,  # High score + conflicting form → MANDATORY (FR-LF-05)
        ("AUTO_MATCH", "unknown"):   2,  # High score + unknown form → standard review
        # REVIEW zone
        ("REVIEW",     "identical"): 3,  # Mid score + same form → low-urgency review
        ("REVIEW",     "related"):   2,  # Mid score + related form → standard review
        ("REVIEW",     "conflict"):  1,  # Mid score + conflicting form → mandatory review
        ("REVIEW",     "unknown"):   2,  # Mid score + unknown form → standard review
        # NO_MATCH zone — no review needed regardless of legal form
        ("NO_MATCH",   "identical"): 0,
        ("NO_MATCH",   "related"):   0,
        ("NO_MATCH",   "conflict"):  0,
        ("NO_MATCH",   "unknown"):   0,
    }

    def __init__(
        self,
        config: ThresholdConfig,
        run_id: str,
        audit_logger: AuditLogger,
    ) -> None:
        """
        Initialise the Router for a single pipeline run.

        Args:
            config:       ThresholdConfig — auto_match_threshold and review_lower_threshold.
            run_id:       Run identifier — included in FR-LF-05 guardrail context dict.
            audit_logger: AuditLogger for this run — used by the FR-LF-05 guardrail only.
        """
        self.config = config
        self.run_id = run_id
        self.logger = audit_logger

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def route(
        self,
        composite_score: float,
    ) -> Literal["AUTO_MATCH", "REVIEW", "NO_MATCH"]:
        """
        Map a composite score to a routing zone.

        Uses strict >= comparisons against both thresholds.
        Boundary values belong to the higher zone
        (score == auto_match_threshold → AUTO_MATCH).

        Args:
            composite_score: Weighted composite score in [0.0, 1.0].

        Returns:
            "AUTO_MATCH" if score >= auto_match_threshold,
            "REVIEW"     if score >= review_lower_threshold,
            "NO_MATCH"   otherwise.
        """
        if composite_score >= self.config.auto_match_threshold:
            return "AUTO_MATCH"
        elif composite_score >= self.config.review_lower_threshold:
            return "REVIEW"
        else:
            return "NO_MATCH"

    def compute_priority(
        self,
        routing_zone: str,
        legal_form_relation: str,
    ) -> int:
        """
        Map (routing_zone, legal_form_relation) to a review priority integer.

        Performs a lookup against _PRIORITY_MATRIX (12 explicit cells).
        Unknown combinations fall back to 0 (no review).

        FR-LF-05 governance guardrail (post-matrix, standalone if-block):
        If routing_zone == AUTO_MATCH and legal_form_relation == conflict,
        priority MUST be 1. The matrix already encodes this — the if-block is a
        redundant defensive control that fires only if the matrix is incorrect.
        On trigger: logs a priority_override_FR_LF_05 guardrail event and sets
        priority = 1. This ensures the governance requirement can never be silently
        bypassed by a matrix modification.

        Args:
            routing_zone:        "AUTO_MATCH", "REVIEW", or "NO_MATCH".
            legal_form_relation: "identical", "related", "conflict", or "unknown".

        Returns:
            Review priority integer in [0, 3].
        """
        priority = self._PRIORITY_MATRIX.get((routing_zone, legal_form_relation), 0)

        # -----------------------------------------------------------------------
        # FR-LF-05 — governance-critical guardrail. MUST remain a standalone
        # if-block. NEVER fold into the matrix. Visibly auditable as a separate
        # defensive control. See ADR-M3-002 section 1 for full rationale.
        # -----------------------------------------------------------------------
        if routing_zone == "AUTO_MATCH" and legal_form_relation == "conflict":
            if priority != 1:
                self.logger.log_guardrail(
                    guardrail_name="priority_override_FR_LF_05",
                    triggered=True,
                    action="review_priority overridden to 1",
                    context={
                        "computed_priority": priority,
                        "routing_zone": routing_zone,
                        "legal_form_relation": legal_form_relation,
                        "run_id": self.run_id,
                    },
                )
                priority = 1

        return priority

    def apply(self, candidate: MatchCandidate) -> MatchCandidate:
        """
        Route a single MatchCandidate and assign its review priority.

        Calls route() and compute_priority() on the candidate's score fields,
        then returns an immutably updated copy with routing_zone and
        review_priority set. The input candidate is never mutated.

        Used by pipeline.py (Task 3) to apply routing to all Top-K candidates.

        Args:
            candidate: MatchCandidate with a fully populated ScoreVector.

        Returns:
            New MatchCandidate with routing_zone and review_priority set.
        """
        zone     = self.route(candidate.score.composite_score)
        priority = self.compute_priority(zone, candidate.score.legal_form_relation)
        return candidate.model_copy(update={"routing_zone": zone, "review_priority": priority})

    def compute_review_rate(self, results: list[MatchResult]) -> float:
        """
        Calculate the fraction of results with routing_zone == REVIEW.

        Used by Streamlit and CLI to decide whether to display the REVIEW rate
        warning when the rate exceeds the configured monitoring threshold.

        Args:
            results: List of MatchResult from a completed pipeline run.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 for an empty list.
        """
        if not results:
            return 0.0
        review_count = sum(1 for r in results if r.routing_zone == "REVIEW")
        return review_count / len(results)
