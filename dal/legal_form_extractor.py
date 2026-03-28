"""
DAL — LegalFormExtractor

Rule-based extraction of legal form strings from raw company names,
plus classification of the relation between two legal forms.

Uses BOTH cleanco sources as complementary dimensions (see ADR-006):
  - typesources()    → matched_term + legal_type (e.g. "Limited", "Corporation")
  - countrysources() → jurisdiction list (e.g. ["Germany", "Switzerland"])

Relation classification uses strict AND — both dimensions must agree:
  - identical : same matched term
  - related   : same legal_type AND overlapping country sets
  - conflict  : any mismatch on either dimension
  - unknown   : term not recognized on one or both sides

Rationale for strict AND:
  GmbH ↔ AG  : Limited vs Corporation (type differs) → conflict
               despite sharing Germany (country overlaps)
  GmbH ↔ Ltd.: Limited vs Limited (type same) → NOT related
               because DE vs UK (no country overlap) → conflict

Accepted gaps (Option A, agreed 2026-03-27):
  "UG (haftungsbeschränkt)", "Aktiengesellschaft" not in cleanco
  → (None, None, []) → "unknown" → legal_form_score = 0.5

See ADR-006 for full design rationale and revision history.

Pipeline position: runs BEFORE normalization. The normalizer strips the legal
form after this module has already captured it as a structured field.

No Streamlit imports. No BLL imports. No external API calls.
"""

import re
from typing import Literal

from cleanco.classify import typesources, countrysources


# ---------------------------------------------------------------------------
# Module-level: load cleanco sources once at import time
#
# typesources()    → list of (legal_type, term) sorted by term length desc
# countrysources() → list of (country, term)  sorted by term length desc
# ---------------------------------------------------------------------------

_TSRC: list[tuple[str, str]] = typesources()
_CSRC: list[tuple[str, str]] = countrysources()

# Pre-build term → sorted list[country] for O(1) country lookup
# Sorted at build time once — deterministic order, plain list, JSON-serializable
_TERM_TO_COUNTRIES: dict[str, list[str]] = {}
for _country, _term in _CSRC:
    if _term not in _TERM_TO_COUNTRIES:
        _TERM_TO_COUNTRIES[_term] = []
    if _country not in _TERM_TO_COUNTRIES[_term]:
        _TERM_TO_COUNTRIES[_term].append(_country)
for _term in _TERM_TO_COUNTRIES:
    _TERM_TO_COUNTRIES[_term].sort()

# Pre-build term → legal_type from typesources (first/longest match wins)
_TERM_TO_TYPE: dict[str, str] = {}
for _legal_type, _term in _TSRC:
    if _term not in _TERM_TO_TYPE:
        _TERM_TO_TYPE[_term] = _legal_type


# ---------------------------------------------------------------------------
# LegalFormExtractor
# ---------------------------------------------------------------------------

class LegalFormExtractor:
    """
    Combined typesource + countrysource legal form extractor and classifier.

    Extraction : scans typesources() (longest-first) for matched_term and
                 legal_type; looks up countries from countrysources() index.
    Classification : strict AND — same legal_type AND overlapping countries
                     required for "related". Any mismatch → "conflict".
    """

    def extract(self, name: str) -> tuple[str | None, str | None, list[str]]:
        """
        Extract the legal form term, its type, and its jurisdiction list.

        Scans typesources() first (pre-sorted longest-first by cleanco).
        Falls back to countrysources() for terms present there but not in
        typesources() (rare — covers edge cases like some jurisdiction-only terms).

        Args:
            name: Raw company name string.

        Returns:
            Tuple of (matched_term, legal_type, countries).
            - matched_term : lowercase term string, e.g. "gmbh", "gmbh & co. kg"
            - legal_type   : cleanco type label, e.g. "Limited", "Corporation"
                             None if term found only in countrysources
            - countries    : sorted list of country names, e.g. ["Austria", "Germany"]
                             [] if term not in countrysources

            All three are (None, None, []) if no legal form recognized.

        Examples:
            "Bayerische Landesbank GmbH & Co. KG"
                → ("gmbh & co. kg", "Limited Partnership", ["Germany"])
            "Deutsche Bank AG"
                → ("ag", "Corporation", ["Austria", "Germany"])
            "ACME Ltd."
                → ("ltd.", "Limited", ["India", "Nigeria", ..., "United Kingdom", ...])
            "No legal form here"
                → (None, None, [])
        """
        if not name or not name.strip():
            return None, None, []

        name_lower = name.lower()

        # Primary scan: typesources (longest-first, covers most legal forms)
        for legal_type, term in _TSRC:
            pattern = r"(?<![a-z])" + re.escape(term) + r"(?![a-z])"
            if re.search(pattern, name_lower):
                countries = list(_TERM_TO_COUNTRIES.get(term, []))
                return term, legal_type, countries

        # Fallback scan: countrysources (catches terms not in typesources)
        for _country, term in _CSRC:
            pattern = r"(?<![a-z])" + re.escape(term) + r"(?![a-z])"
            if re.search(pattern, name_lower):
                countries = list(_TERM_TO_COUNTRIES.get(term, []))
                return term, None, countries

        return None, None, []

    def classify_relation(
        self,
        type_a: str | None,
        countries_a: list[str],
        type_b: str | None,
        countries_b: list[str],
    ) -> Literal["identical", "related", "conflict", "unknown"]:
        """
        Classify the relation between two legal forms using strict AND logic.

        Both legal_type AND country overlap must agree simultaneously.
        A mismatch on either dimension → "conflict".

        Rules (applied in order):
          1. type_a is None OR type_b is None → "unknown"
          2. type_a == type_b AND countries_a ∩ countries_b non-empty → "related"
          3. everything else → "conflict"

        Note: "identical" is determined at the term level in extract_and_classify(),
        not here — this method receives types and countries only.

        Args:
            type_a     : legal_type from extract() for source A.
            countries_a: sorted list of countries from extract() for source A.
            type_b     : legal_type from extract() for source B.
            countries_b: sorted list of countries from extract() for source B.

        Returns:
            One of: "related", "conflict", "unknown".

        Examples:
            ("Limited", ["Germany"], "Limited", ["Germany"])
                → "related"   (same type, shared Germany)
            ("Limited", ["Germany"], "Corporation", ["Germany"])
                → "conflict"  (type differs despite shared country)
            ("Limited", ["Germany"], "Limited", ["United Kingdom"])
                → "conflict"  (type same but no shared country)
            (None, [], "Corporation", ["Germany"])
                → "unknown"
        """
        if type_a is None or type_b is None:
            return "unknown"

        if type_a == type_b and bool(set(countries_a) & set(countries_b)):
            return "related"

        return "conflict"

    def extract_and_classify(
        self,
        name_a: str,
        name_b: str,
    ) -> tuple[
        str | None,
        str | None,
        list[str],
        str | None,
        str | None,
        list[str],
        Literal["identical", "related", "conflict", "unknown"],
    ]:
        """
        Convenience method: extract from both names and classify the relation.

        Handles "identical" detection at the term level before delegating
        to classify_relation() for the type+country AND logic.

        Args:
            name_a: Raw company name from source A.
            name_b: Raw company name from source B.

        Returns:
            Tuple of:
              (term_a, type_a, countries_a, term_b, type_b, countries_b, relation)
        """
        term_a, type_a, countries_a = self.extract(name_a)
        term_b, type_b, countries_b = self.extract(name_b)

        # identical: exact same term (before type/country comparison)
        if term_a is not None and term_a == term_b:
            return term_a, type_a, countries_a, term_b, type_b, countries_b, "identical"

        relation = self.classify_relation(type_a, countries_a, type_b, countries_b)
        return term_a, type_a, countries_a, term_b, type_b, countries_b, relation
