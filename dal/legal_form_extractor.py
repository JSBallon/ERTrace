"""
DAL — LegalFormExtractor

Rule-based extraction of legal form strings from raw company names,
plus classification of the relation between two legal forms.

Extraction uses an ordered regex scan — compound forms (e.g. "GmbH & Co. KG")
are matched BEFORE their component forms ("GmbH") to prevent misclassification.
See ADR-006 for design rationale.

Pipeline position: runs BEFORE normalization (normalizer strips the legal form
after this module has already captured it as a structured field).

Public API:
  extractor = LegalFormExtractor()
  legal_form_str, class_id = extractor.extract("Bayerische Landesbank GmbH & Co. KG")
  # → ("GmbH & Co. KG", "DE_GMBH_COKG")

  relation = extractor.classify_relation("DE_GMBH", "DE_AG")
  # → "related"

No Streamlit imports. No BLL imports. No external API calls.
"""

import re
from typing import Literal


# ---------------------------------------------------------------------------
# Legal form pattern list — ORDER IS CRITICAL (see ADR-006)
# Compound / longer forms MUST precede their component forms.
# First match wins.
# ---------------------------------------------------------------------------

_RAW_PATTERNS: list[tuple[str, str]] = [
    # --- DE: compound forms first ---
    (r"\bGmbH\s*&\s*Co\.?\s*KG\b",                             "DE_GMBH_COKG"),
    (r"\bUG\s*\(haftungsbeschr[äa]nkt\)\b",                    "DE_UG"),
    # --- DE: long-form names before abbreviations ---
    (r"\bGesellschaft\s+mit\s+beschr[äa]nkter\s+Haftung\b",    "DE_GMBH"),
    (r"\bGesellschaft\s+mbH\b",                                 "DE_GMBH"),
    (r"\bG\.m\.b\.H\.\b",                                       "DE_GMBH"),
    (r"\bAktiengesellschaft\b",                                 "DE_AG"),
    (r"\bKommanditgesellschaft\b",                              "DE_KG"),
    # --- DE: abbreviations ---
    (r"\bGmbH\b",                                               "DE_GMBH"),
    (r"\bAG\b",                                                 "DE_AG"),
    (r"\bKG\b",                                                 "DE_KG"),
    (r"\bUG\b",                                                 "DE_UG"),
    (r"\bSE\b",                                                 "DE_SE"),
    (r"\beG\b",                                                 "DE_EG"),
    # --- UK/US: long forms before abbreviations ---
    (r"\bLimited\b",                                            "UK_LTD"),
    (r"\bCorporation\b",                                        "US_CORP"),
    (r"\bIncorporated\b",                                       "US_CORP"),
    (r"\bL\.L\.C\.\b",                                         "US_CORP"),
    (r"\bLLC\b",                                               "US_CORP"),
    (r"\bInc\.\b",                                             "US_CORP"),
    (r"\bInc\b",                                               "US_CORP"),
    (r"\bCorp\.\b",                                            "US_CORP"),
    (r"\bCorp\b",                                              "US_CORP"),
    (r"\bLtd\.\b",                                             "UK_LTD"),
    (r"\bLtd\b",                                               "UK_LTD"),
    (r"\bPlc\b",                                               "UK_PLC"),
    (r"\bPLC\b",                                               "UK_PLC"),
    # --- EU: long forms before abbreviations ---
    (r"\bSoci[ée]t[ée]\s+Anonyme\b",                           "FR_SA"),
    (r"\bNaamloze\s+Vennootschap\b",                           "NL_NV"),
    (r"\bBesloten\s+Vennootschap\b",                           "NL_BV"),
    (r"\bSoci[ée]t[ée]\s+[àa]\s+Responsabilit[ée]\s+Limit[ée]e\b", "FR_SARL"),
    # --- EU: abbreviations (placed AFTER longer EU forms) ---
    (r"\bS\.A\.R\.L\.\b",                                      "FR_SARL"),
    (r"\bSARL\b",                                              "FR_SARL"),
    (r"\bS\.A\.\b",                                            "FR_SA"),
    # SA must come after S.A. and SARL to avoid partial matches
    (r"\bSA\b",                                                "FR_SA"),
    (r"\bN\.V\.\b",                                            "NL_NV"),
    (r"\bNV\b",                                                "NL_NV"),
    (r"\bB\.V\.\b",                                            "NL_BV"),
    (r"\bBV\b",                                                "NL_BV"),
    (r"\bS\.p\.A\.\b",                                         "IT_SPA"),
    (r"\bSpA\b",                                               "IT_SPA"),
    (r"\bS\.L\.\b",                                            "ES_SL"),
]

# Pre-compile all patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(raw, re.IGNORECASE), class_id)
    for raw, class_id in _RAW_PATTERNS
]

# ---------------------------------------------------------------------------
# Related groups — same group = "related", different groups = "conflict"
# ---------------------------------------------------------------------------

_RELATED_GROUPS: list[frozenset[str]] = [
    frozenset({"DE_GMBH", "DE_AG", "DE_KG", "DE_GMBH_COKG", "DE_UG", "DE_SE", "DE_EG"}),
    frozenset({"UK_LTD", "UK_PLC", "US_CORP"}),
    frozenset({"FR_SA", "FR_SARL", "NL_NV", "NL_BV", "IT_SPA", "ES_SL"}),
]

# Build a fast class_id → group_index lookup
_CLASS_TO_GROUP: dict[str, int] = {}
for _idx, _group in enumerate(_RELATED_GROUPS):
    for _cls in _group:
        _CLASS_TO_GROUP[_cls] = _idx


# ---------------------------------------------------------------------------
# LegalFormExtractor
# ---------------------------------------------------------------------------

class LegalFormExtractor:
    """
    Rule-based legal form extractor and relation classifier.

    Extraction: ordered regex scan, compound/specific forms first (ADR-006).
    Classification: lookup table — identical / related / conflict / unknown.
    """

    def extract(self, name: str) -> tuple[str | None, str | None]:
        """
        Extract the legal form string and its class ID from a company name.

        Args:
            name: Raw company name string.

        Returns:
            Tuple of (legal_form_string, class_id).
            Both are None if no legal form is recognized.

        Examples:
            "Bayerische Landesbank GmbH & Co. KG" → ("GmbH & Co. KG", "DE_GMBH_COKG")
            "Deutsche Bank AG"                    → ("AG", "DE_AG")
            "Bridgewater Associates"              → (None, None)
        """
        if not name or not name.strip():
            return None, None

        for pattern, class_id in _COMPILED_PATTERNS:
            match = pattern.search(name)
            if match:
                return match.group(0).strip(), class_id

        return None, None

    def classify_relation(
        self,
        class_a: str | None,
        class_b: str | None,
    ) -> Literal["identical", "related", "conflict", "unknown"]:
        """
        Classify the relation between two legal form class IDs.

        Rules:
          - Either class is None / unrecognized → "unknown"
          - Same class ID                       → "identical"
          - Same related group                  → "related"
          - Different groups                    → "conflict"

        Args:
            class_a: Legal form class ID from source A (e.g. "DE_GMBH").
            class_b: Legal form class ID from source B (e.g. "DE_AG").

        Returns:
            One of: "identical", "related", "conflict", "unknown".

        Examples:
            ("DE_GMBH", "DE_GMBH") → "identical"
            ("DE_GMBH", "DE_AG")   → "related"
            ("DE_GMBH", "UK_LTD")  → "conflict"
            (None,      "DE_AG")   → "unknown"
        """
        # Unknown if either side is missing or unrecognized
        if not class_a or not class_b:
            return "unknown"
        if class_a not in _CLASS_TO_GROUP or class_b not in _CLASS_TO_GROUP:
            return "unknown"

        # Identical
        if class_a == class_b:
            return "identical"

        # Related (same group)
        if _CLASS_TO_GROUP[class_a] == _CLASS_TO_GROUP[class_b]:
            return "related"

        # Conflict (different groups)
        return "conflict"

    def extract_and_classify(
        self, name_a: str, name_b: str
    ) -> tuple[str | None, str | None, str | None, str | None, Literal["identical", "related", "conflict", "unknown"]]:
        """
        Convenience method: extract from both names and classify the relation.

        Args:
            name_a: Raw company name from source A.
            name_b: Raw company name from source B.

        Returns:
            Tuple of:
              (lf_string_a, class_id_a, lf_string_b, class_id_b, relation)
        """
        lf_a, cls_a = self.extract(name_a)
        lf_b, cls_b = self.extract(name_b)
        relation = self.classify_relation(cls_a, cls_b)
        return lf_a, cls_a, lf_b, cls_b, relation
