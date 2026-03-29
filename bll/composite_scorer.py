"""
BLL — CompositeScorer

Aggregates the four TGFR score components into a single weighted composite score
that drives routing zone assignment (AUTO_MATCH / REVIEW / NO_MATCH).

Design decisions (see ADR-M2-005):
  - Linear weighted sum — interpretable, calibratable, audit-explainable
  - np.clip(result, 0.0, 1.0): mandatory defense-in-depth because:
      (a) WeightsConfig allows sum tolerance ±0.001 → sum=1.001 × inputs=1.0 → 1.001
      (b) Float32 FAISS inputs can be slightly above 1.0 before Task 2 clipping
      Without clip: MatchResult.composite_score (le=1.0 Pydantic) would raise ValidationError
  - round(..., 6): precision consistency across all score vector components
  - verify() calls score() directly — same rounding path, zero spurious guardrail failures

No Streamlit imports. No DAL access. No filesystem access. No external API calls.
"""

import numpy as np
from bll.schemas import WeightsConfig


class CompositeScorer:
    """
    Weighted linear composite scorer for the TGFR pipeline.

    Stateless after construction — score() and verify() take explicit inputs.

    Usage:
        config = WeightsConfig(
            w_embedding=0.50, w_jaro_winkler=0.20,
            w_token_sort=0.20, w_legal_form=0.10
        )
        scorer = CompositeScorer(config)
        s = scorer.score(
            embedding_score=0.90,
            jw_score=0.80,
            ts_score=0.85,
            lf_score=1.0,
        )
        # s = 0.5*0.90 + 0.2*0.80 + 0.2*0.85 + 0.1*1.0 = 0.88

        consistent = scorer.verify(s, 0.90, 0.80, 0.85, 1.0)
        # consistent = True
    """

    def __init__(self, config: WeightsConfig) -> None:
        """
        Initialise with weight configuration.

        Args:
            config: WeightsConfig with w_embedding, w_jaro_winkler, w_token_sort,
                    w_legal_form. Pydantic validator already enforces sum = 1.0 ± 0.001.
        """
        self._config = config

    def score(
        self,
        embedding_score: float,
        jw_score: float,
        ts_score: float,
        lf_score: float,
    ) -> float:
        """
        Compute the weighted composite score.

        Formula:
            result = w_embedding * embedding_score
                   + w_jaro_winkler * jw_score
                   + w_token_sort   * ts_score
                   + w_legal_form   * lf_score

        Args:
            embedding_score: Cosine similarity from FAISS Stage 1, [0.0, 1.0].
            jw_score:        Jaro-Winkler similarity from FuzzyReranker, [0.0, 1.0].
            ts_score:        Token Sort Ratio from FuzzyReranker, [0.0, 1.0].
            lf_score:        Legal form score from LegalFormScorer, [0.0, 1.0].

        Returns:
            float in [0.0, 1.0], rounded to 6 decimal places.

        Notes:
            - np.clip is applied before rounding as defense-in-depth against float32
              precision edge cases and WeightsConfig sum tolerance (±0.001).
            - All inputs are expected to be in [0.0, 1.0]; this is not re-validated
              here — Pydantic MatchResult validation catches out-of-range values upstream.
        """
        result = (
            self._config.w_embedding    * embedding_score
            + self._config.w_jaro_winkler * jw_score
            + self._config.w_token_sort   * ts_score
            + self._config.w_legal_form   * lf_score
        )
        return round(float(np.clip(result, 0.0, 1.0)), 6)

    def verify(
        self,
        computed: float,
        embedding_score: float,
        jw_score: float,
        ts_score: float,
        lf_score: float,
    ) -> bool:
        """
        Consistency check — recompute and verify deviation ≤ 0.001.

        Used as the composite_inconsistency guardrail in the pipeline.
        Returns False if the stored composite score deviates from the recomputed value
        by more than 0.001, which triggers a guardrail_event in the audit log.

        Delegates recompute to score() to ensure identical precision path.
        Tolerance of 0.001 matches WeightsConfig's sum-to-1 tolerance — prevents
        spurious failures from floating-point rounding in the 6th decimal place.

        Args:
            computed:        The composite score stored in MatchResult.
            embedding_score: Same embedding_score used to produce computed.
            jw_score:        Same jw_score used to produce computed.
            ts_score:        Same ts_score used to produce computed.
            lf_score:        Same lf_score used to produce computed.

        Returns:
            True  if |computed − recomputed| ≤ 0.001 (consistent).
            False if deviation exceeds tolerance (triggers composite_inconsistency event).
        """
        recomputed = self.score(embedding_score, jw_score, ts_score, lf_score)
        return abs(computed - recomputed) <= 0.001