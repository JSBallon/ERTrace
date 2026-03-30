"""
BLL — FuzzyReranker

Stage 2 of the TGFR pipeline (Fuzzy-Reconsider): syntactic verification of
Top-K semantic candidates retrieved by FAISS in Stage 1.

Computes two complementary fuzzy metrics per candidate pair:
  - Jaro-Winkler similarity : character-level, prefix-sensitive
  - Token Sort Ratio         : token-level, word-order-invariant

Design decisions (see ADR-M2-003):
  - JaroWinkler.similarity() from rapidfuzz.distance — NOT fuzz.jaro_winkler_similarity()
    which does not exist in rapidfuzz >= 3.x (moved + rescaled to [0.0, 1.0])
  - token_sort_ratio() from rapidfuzz.fuzz — returns [0, 100], divided by 100.0
  - Both scores rounded to 6 decimal places for consistent precision in score vector
  - score_batch() is a plain loop — rapidfuzz calls are already fast C++ implementations

Inputs must be normalized names (legal form stripped, lowercase, whitespace cleaned)
as produced by dal/normalizer.py. Raw company names must NOT be passed directly.

No Streamlit imports. No DAL access. No filesystem access. No external API calls.
"""

from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz


class FuzzyReranker:
    """
    Syntactic fuzzy scorer for candidate pairs in the TGFR pipeline.

    Stateless — no configuration, no model to load.
    Instantiate once and reuse across all A entries.

    Usage:
        reranker = FuzzyReranker()
        jw, ts = reranker.score("deutsche bank", "deutsche bahn")
        # jw ≈ 0.969, ts ≈ 0.923

    Both scores are in [0.0, 1.0] and can be fed directly into CompositeScorer.
    """

    def score(
        self,
        name_a_normalized: str,
        name_b_normalized: str,
    ) -> tuple[float, float]:
        """
        Compute Jaro-Winkler and Token Sort Ratio for a single normalized name pair.

        Both inputs must be normalized (legal form stripped, lowercase, whitespace cleaned).
        Normalization is the caller's responsibility — this method applies no preprocessing.

        Args:
            name_a_normalized: Normalized company name from Source A.
            name_b_normalized: Normalized company name from Source B (candidate).

        Returns:
            Tuple of (jaro_winkler_score, token_sort_ratio), both in [0.0, 1.0],
            rounded to 6 decimal places.

        Notes:
            - JaroWinkler.similarity() returns [0.0, 1.0] directly (rapidfuzz >= 3.x).
              Do NOT divide by 100 — see ADR-M2-003.
            - fuzz.token_sort_ratio() returns [0, 100] — divided by 100.0 here.
            - Both metrics are symmetric: score(a, b) == score(b, a).
            - Empty string pairs: JW('', '') = 1.0, TS('', '') = 1.0 (rapidfuzz default).
              The normalizer upstream should prevent empty strings from reaching here.
        """
        # Jaro-Winkler: [0.0, 1.0] — prefix-sensitive character similarity
        # rapidfuzz >= 3.x: JaroWinkler.similarity, NOT fuzz.jaro_winkler_similarity
        jw = JaroWinkler.similarity(name_a_normalized, name_b_normalized)

        # Token Sort Ratio: [0, 100] → [0.0, 1.0] — order-invariant token similarity
        ts = fuzz.token_sort_ratio(name_a_normalized, name_b_normalized) / 100.0

        return round(float(jw), 6), round(float(ts), 6)

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """
        Compute fuzzy scores for a list of normalized name pairs.

        Applies score() to each pair in order. Results are in the same order as input.

        Args:
            pairs: List of (name_a_normalized, name_b_normalized) tuples.

        Returns:
            List of (jaro_winkler_score, token_sort_ratio) tuples, same length as input.
            Empty list if pairs is empty.
        """
        return [self.score(a, b) for a, b in pairs]
