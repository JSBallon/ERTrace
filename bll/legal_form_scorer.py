"""
BLL — LegalFormScorer

Converts a pair of company names into a legal form score and relation label.
Both values become named dimensions of the MatchResult score vector.

Design decisions (see ADR-M2-004):
  - Accepts two raw name strings (not pre-extracted types or relation strings)
    → delegates fully to LegalFormExtractor.extract_and_classify() in the DAL
  - LegalFormExtractor is stateless; one instance created at construction, reused
  - Score levels are configurable via LegalFormConfig (defaults from v1.0-default.yaml)
  - Cross-layer call (BLL → DAL) is the one permitted exception documented in
    Active Context M2 section 8.5 and ADR-M2-004

GmbH↔AG resolves to "conflict" (not "related") per DAL strict-AND logic:
  GmbH type="Limited", AG type="Corporation" → type mismatch → conflict.
  See ADR-M2-004 section 6 for full explanation.

No Streamlit imports. No filesystem access. No external API calls.
The only DAL access is extract_and_classify() — no other DAL function may be called.
"""

from dal.legal_form_extractor import LegalFormExtractor
from bll.schemas import LegalFormConfig


class LegalFormScorer:
    """
    Legal form scorer for the TGFR pipeline.

    Maps the legal form relation between two company names to a configurable
    numerical score and returns the relation label for the score vector.

    Usage:
        config = LegalFormConfig()   # uses defaults: identical=1.0, related=0.5,
                                     #                conflict=0.0, unknown=0.5
        scorer = LegalFormScorer(config)
        score, relation = scorer.score("ACME GmbH", "ACME GmbH & Co. KG")
        # score=0.0, relation="conflict"  (GmbH vs GmbH & Co. KG = different terms)

        score, relation = scorer.score("Alpha GmbH", "Beta GmbH")
        # score=1.0, relation="identical"  (same term "gmbh")
    """

    def __init__(self, config: LegalFormConfig) -> None:
        """
        Initialise with configurable score levels.

        Args:
            config: LegalFormConfig with score levels for each relation class.
                    Defaults: identical=1.0, related=0.5, conflict=0.0, unknown=0.5.
        """
        self._config = config
        # One LegalFormExtractor instance — stateless, safe to reuse across all calls
        self._extractor = LegalFormExtractor()

    def score(self, name_a: str, name_b: str) -> tuple[float, str]:
        """
        Compute the legal form score and relation for a pair of company names.

        Delegates to LegalFormExtractor.extract_and_classify() for full extraction
        and classification in one call. This is the only permitted BLL→DAL call
        (see ADR-M2-004 and Active Context M2 section 8.5).

        Args:
            name_a: Raw company name from Source A (pre-normalization or normalized).
                    Legal form extraction works on the raw name.
            name_b: Raw company name from Source B (candidate).

        Returns:
            Tuple of (legal_form_score, legal_form_relation):
              legal_form_score    : float in [0.0, 1.0] from LegalFormConfig.
              legal_form_relation : one of "identical", "related", "conflict", "unknown".

        Score semantics (default config):
            "identical" → 1.0  : same legal form term (GmbH↔GmbH, AG↔AG)
            "related"   → 0.5  : same type + overlapping countries (LLC↔LLC variants)
            "conflict"  → 0.0  : type mismatch or no country overlap (GmbH↔Ltd., GmbH↔AG)
            "unknown"   → 0.5  : one or both names have no recognisable legal form
        """
        # extract_and_classify returns a 7-tuple; the last element is the relation
        *_, relation = self._extractor.extract_and_classify(name_a, name_b)

        score_map = {
            "identical": self._config.identical_score,
            "related":   self._config.related_score,
            "conflict":  self._config.conflict_score,
            "unknown":   self._config.unknown_score,
        }
        return score_map[relation], relation
