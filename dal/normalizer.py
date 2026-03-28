"""
DAL — CompanyNameNormalizer

Five-step deterministic normalization pipeline (operation order is fixed — see ADR-005):

  Step 1: Unicode NFC normalization       (encoding consistency)
  Step 2: Whitespace cleanup              (tabs, newlines, multiple spaces)
  Step 3: Legal form stripping            (cleanco primary + regex fallback)
  Step 4: Lowercase                       (AFTER stripping — cleanco is case-sensitive)
  Step 5: Special character cleanup       (retain alphanumeric + spaces only)

IMPORTANT: This module only STRIPS legal forms. It does NOT extract them.
Legal form extraction must run first via dal/legal_form_extractor.py (Task 8)
before this normalizer is called. See ADR-005.

No Streamlit imports. No BLL imports. No external API calls.
"""

import re
import unicodedata

from cleanco import basename as cleanco_basename


# ---------------------------------------------------------------------------
# Regex fallback patterns for legal forms cleanco may miss
# Order matters — longer/more specific patterns first
# ---------------------------------------------------------------------------

_REGEX_FALLBACK_PATTERNS = [
    r"\bGmbH\s*&\s*Co\.?\s*KG\b",
    r"\bUG\s*\(haftungsbeschränkt\)\b",
    r"\bUG\s*\(haftungsbeschraenkt\)\b",
    r"\bAktiengesellschaft\b",
    r"\bGesellschaft\s+mit\s+beschränkter\s+Haftung\b",
    r"\bGesellschaft\s+mbH\b",
    r"\bG\.m\.b\.H\.\b",
    r"\bKommanditgesellschaft\b",
    r"\bNaamloze\s+Vennootschap\b",
    r"\bSociété\s+Anonyme\b",
    r"\bSociete\s+Anonyme\b",
    r"\bCorporation\b",
    r"\bIncorporated\b",
    r"\bLimited\b",
    r"\bGmbH\b",
    r"\bAG\b",
    r"\bKG\b",
    r"\bUG\b",
    r"\bSE\b",
    r"\bLtd\.\b",
    r"\bLtd\b",
    r"\bInc\.\b",
    r"\bInc\b",
    r"\bCorp\.\b",
    r"\bCorp\b",
    r"\bLLC\b",
    r"\bL\.L\.C\.\b",
    r"\bS\.A\.\b",
    r"\bSA\b",
    r"\bN\.V\.\b",
    r"\bNV\b",
    r"\bPlc\b",
    r"\bPLC\b",
]

# Pre-compile for performance
_COMPILED_FALLBACKS = [re.compile(p, re.IGNORECASE) for p in _REGEX_FALLBACK_PATTERNS]


class CompanyNameNormalizer:
    """
    Deterministic company name normalizer for the TGFR pipeline.

    Operation order is fixed per ADR-005. Do not reorder steps.
    """

    def normalize(self, name: str) -> str:
        """
        Apply the full five-step normalization pipeline to a single name.

        Args:
            name: Raw company name string.

        Returns:
            Normalized base name (lowercase, no legal form, no special chars).
        """
        if not name or not name.strip():
            return ""

        # Step 1: Unicode NFC normalization
        name = unicodedata.normalize("NFC", name)

        # Step 2: Whitespace cleanup
        name = " ".join(name.split())

        # Step 3a: Legal form stripping via cleanco (case-sensitive — runs BEFORE lowercase)
        try:
            stripped = cleanco_basename(name)
            # cleanco returns None or empty string when it finds nothing to strip
            if stripped and stripped.strip():
                name = stripped.strip()
        except Exception:
            pass  # cleanco failure is non-fatal — continue with original name

        # Step 3b: Regex fallback for forms cleanco may have missed
        name = self._regex_strip_fallback(name)

        # Step 4: Lowercase (AFTER stripping — cleanco is case-sensitive)
        name = name.lower()

        # Step 5: Special character cleanup — retain alphanumeric + spaces
        # \w matches [a-zA-Z0-9_] + Unicode word chars (correct for German umlauts)
        name = re.sub(r"[^\w\s]", " ", name)
        name = " ".join(name.split())

        return name.strip()

    def normalize_batch(self, names: list[str]) -> list[str]:
        """
        Apply normalize() to each name in a list.

        Args:
            names: List of raw company name strings.

        Returns:
            List of normalized strings, same length and order as input.
        """
        return [self.normalize(name) for name in names]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _regex_strip_fallback(self, name: str) -> str:
        """
        Strip legal form tokens that cleanco did not catch,
        using the pre-compiled regex fallback list.

        Strips trailing and inline legal form tokens.
        Returns the cleaned name, or the original if nothing matched.
        """
        result = name
        for pattern in _COMPILED_FALLBACKS:
            result = pattern.sub("", result)

        # Clean up any leftover punctuation artifacts (e.g. trailing "&", "-")
        result = re.sub(r"[\s&\-,\.]+$", "", result)
        result = " ".join(result.split())

        # If stripping removed everything, return the original
        return result.strip() if result.strip() else name
