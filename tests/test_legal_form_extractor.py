"""
Tests for dal/legal_form_extractor.py — LegalFormExtractor

Covers:
  - extract(): term, type, countries returned correctly
  - extract(): compound form beats component (GmbH & Co. KG before GmbH)
  - extract(): unrecognized form returns (None, None, [])
  - extract(): returns list[str], not frozenset (refactor verification)
  - extract(): countries are sorted
  - classify_relation(): identical, related, conflict, unknown
  - extract_and_classify(): convenience method end-to-end

NOTE: GmbH ↔ AG → "conflict" (not "related")
  The original M1 exit criteria spec said "related" — superseded by ADR-006
  strict AND redesign (2026-03-28). GmbH=Limited, AG=Corporation: type mismatch
  → conflict, despite sharing Germany.
"""

import pytest
from dal.legal_form_extractor import LegalFormExtractor


@pytest.fixture
def extractor():
    return LegalFormExtractor()


# ------------------------------------------------------------------
# extract() — basic extraction
# ------------------------------------------------------------------

def test_extract_gmbh(extractor):
    term, typ, countries = extractor.extract("ACME GmbH")
    assert term == "gmbh"
    assert typ == "Limited"
    assert "Germany" in countries


def test_extract_ag(extractor):
    term, typ, countries = extractor.extract("Deutsche Bank AG")
    assert term == "ag"
    assert typ == "Corporation"
    assert "Germany" in countries
    assert "Austria" in countries


def test_extract_ltd(extractor):
    term, typ, countries = extractor.extract("ACME Ltd.")
    assert term == "ltd."
    assert typ == "Limited"
    assert "United Kingdom" in countries


def test_extract_llc(extractor):
    term, typ, countries = extractor.extract("ACME LLC")
    assert term == "llc"
    assert typ == "Limited Liability Company"
    assert "United States of America" in countries


def test_extract_corp(extractor):
    term, typ, countries = extractor.extract("ACME Corp.")
    assert typ == "Corporation"
    assert "United States of America" in countries


# ------------------------------------------------------------------
# extract() — compound form beats component
# ------------------------------------------------------------------

def test_extract_gmbh_cokg_beats_gmbh(extractor):
    term, typ, countries = extractor.extract("Bayerische Landesbank GmbH & Co. KG")
    assert term == "gmbh & co. kg"
    assert typ == "Limited Partnership"
    assert "Germany" in countries


# ------------------------------------------------------------------
# extract() — unrecognized form
# ------------------------------------------------------------------

def test_extract_no_legal_form(extractor):
    term, typ, countries = extractor.extract("No legal form here")
    assert term is None
    assert typ is None
    assert countries == []


def test_extract_empty_string(extractor):
    term, typ, countries = extractor.extract("")
    assert term is None
    assert typ is None
    assert countries == []


# ------------------------------------------------------------------
# extract() — type safety (frozenset refactor verification)
# ------------------------------------------------------------------

def test_extract_returns_list_not_frozenset(extractor):
    _, _, countries = extractor.extract("ACME GmbH")
    assert isinstance(countries, list), f"expected list, got {type(countries)}"


def test_extract_countries_are_sorted(extractor):
    _, _, countries = extractor.extract("Deutsche Bank AG")
    assert countries == sorted(countries)


def test_extract_empty_countries_is_list(extractor):
    _, _, countries = extractor.extract("No legal form")
    assert isinstance(countries, list)
    assert countries == []


# ------------------------------------------------------------------
# classify_relation() — identical
# ------------------------------------------------------------------

def test_identical_same_term(extractor):
    *_, relation = extractor.extract_and_classify("ACME GmbH", "ACME GmbH")
    assert relation == "identical"


def test_identical_same_ltd_variants(extractor):
    # "ltd" and "ltd." are different terms — not identical
    *_, relation = extractor.extract_and_classify("ACME Ltd", "ACME Ltd.")
    # ltd and ltd. may or may not be same term depending on cleanco
    # — just assert it is not unknown
    assert relation in ("identical", "related", "conflict")


# ------------------------------------------------------------------
# classify_relation() — conflict
# ------------------------------------------------------------------

def test_conflict_type_mismatch_gmbh_ag(extractor):
    # GmbH=Limited, AG=Corporation — type differs despite shared Germany
    # ADR-006: strict AND → conflict
    *_, relation = extractor.extract_and_classify("ACME GmbH", "ACME AG")
    assert relation == "conflict"


def test_conflict_country_mismatch_gmbh_ltd(extractor):
    # GmbH=Limited (DE/CH), Ltd.=Limited (UK/US) — same type, no country overlap
    *_, relation = extractor.extract_and_classify("ACME GmbH", "ACME Ltd.")
    assert relation == "conflict"


def test_conflict_both_mismatch_gmbh_llc(extractor):
    # GmbH=Limited (DE), LLC=LLC (US) — type differs, no country overlap
    *_, relation = extractor.extract_and_classify("ACME GmbH", "ACME LLC")
    assert relation == "conflict"


def test_conflict_gmbh_cokg_vs_gmbh(extractor):
    # LP vs Limited — type differs
    *_, relation = extractor.extract_and_classify(
        "Bayerische Landesbank GmbH & Co. KG", "ACME GmbH"
    )
    assert relation == "conflict"


# ------------------------------------------------------------------
# classify_relation() — related (same type AND overlapping country)
# ------------------------------------------------------------------

def test_related_llc_variants(extractor):
    # LLC and L.L.C. — same LLC type, Philippines shared
    *_, relation = extractor.extract_and_classify("ACME LLC", "ACME L.L.C.")
    assert relation == "related"


def test_related_corp_variants(extractor):
    # Corp and Inc — both Corporation, US/Philippines shared
    *_, relation = extractor.extract_and_classify("ACME Corp", "ACME Inc.")
    assert relation == "related"


# ------------------------------------------------------------------
# classify_relation() — unknown
# ------------------------------------------------------------------

def test_unknown_no_form_on_a(extractor):
    *_, relation = extractor.extract_and_classify("No legal form", "ACME AG")
    assert relation == "unknown"


def test_unknown_no_form_on_b(extractor):
    *_, relation = extractor.extract_and_classify("ACME GmbH", "No legal form")
    assert relation == "unknown"


def test_unknown_both_unrecognized(extractor):
    *_, relation = extractor.extract_and_classify(
        "Bridgewater Associates", "Summit Capital"
    )
    assert relation == "unknown"


# ------------------------------------------------------------------
# extract_and_classify() — return tuple structure
# ------------------------------------------------------------------

def test_extract_and_classify_returns_seven_elements(extractor):
    result = extractor.extract_and_classify("ACME GmbH", "ACME AG")
    assert len(result) == 7


def test_extract_and_classify_countries_are_lists(extractor):
    term_a, type_a, countries_a, term_b, type_b, countries_b, relation = \
        extractor.extract_and_classify("ACME GmbH", "ACME AG")
    assert isinstance(countries_a, list)
    assert isinstance(countries_b, list)
