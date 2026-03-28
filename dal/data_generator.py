"""
DAL — FakerDataGenerator

Generates synthetic company name datasets for pipeline testing.
Uses a curated base pool of realistic German/international banking names
combined with four controlled mutation strategies:

  - Typo simulation       (~10% of entries)
  - Noise addition        (~20% of entries)
  - Legal form variant    (~10% of matching pairs)
  - Abbreviation variant  (curated lookup table, used in matching pairs)

See ADR-004 for design rationale.

No Streamlit imports. No BLL imports. No external API calls.
"""

import random
import string
import uuid
from faker import Faker


# ---------------------------------------------------------------------------
# Curated base name pool — realistic German/international banking sector names
# ---------------------------------------------------------------------------

BASE_NAMES_DE = [
    "Bayerische Landesbank",
    "Norddeutsche Landesbank",
    "Hamburger Sparkasse",
    "Deutsche Hypothekenbank",
    "Commerzbank",
    "Deutsche Bank",
    "DZ Bank",
    "Landesbank Baden-Württemberg",
    "Helaba Landesbank Hessen-Thüringen",
    "Stadtsparkasse München",
    "Kreissparkasse Köln",
    "Volksbank Raiffeisenbank Bayern Mitte",
    "Mittelbrandenburgische Sparkasse",
    "Sparkasse KölnBonn",
    "Bremer Landesbank",
    "HSH Nordbank",
    "Investitionsbank Berlin",
    "Thüringer Aufbaubank",
    "Saarländische Investitionskreditbank",
    "NRW Bank",
    "Euler Hermes",
    "Allianz Lebensversicherungs",
    "Münchener Hypothekenbank",
    "Wüstenrot Bausparkasse",
    "Deutsche Pfandbriefbank",
    "Aareal Bank",
    "Corealcredit Bank",
    "IKB Deutsche Industriebank",
    "Deka Bank Deutsche Girozentrale",
    "Westdeutsche ImmobilienBank",
]

BASE_NAMES_EN = [
    "First National Bank",
    "Global Trust Finance",
    "Northern Capital Partners",
    "Atlantic Investment Holdings",
    "Pacific Merchant Bank",
    "Commonwealth Financial Services",
    "Sterling Asset Management",
    "Continental Credit Union",
    "Meridian Banking Group",
    "Horizon Financial Corporation",
    "Summit Private Equity",
    "Apex Wealth Management",
    "Pinnacle Investment Trust",
    "Cascade Capital Partners",
    "Bridgewater Asset Management",
]

# ---------------------------------------------------------------------------
# Abbreviation lookup table (curated) — used in generate_matching_pair
# Produces "embedding finds it, fuzzy struggles" scenarios
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    "Bayerische Landesbank": "BayernLB",
    "Norddeutsche Landesbank": "NORD/LB",
    "Landesbank Baden-Württemberg": "LBBW",
    "Helaba Landesbank Hessen-Thüringen": "Helaba",
    "DZ Bank": "DZ",
    "Deka Bank Deutsche Girozentrale": "DekaBank",
    "Deutsche Pfandbriefbank": "pbb",
    "IKB Deutsche Industriebank": "IKB",
    "NRW Bank": "NRW.BANK",
    "Commerzbank": "Coba",
    "Deutsche Bank": "DB",
    "Hamburger Sparkasse": "Haspa",
    "Bridgewater Asset Management": "Bridgewater",
    "Commonwealth Financial Services": "CFS",
}

# ---------------------------------------------------------------------------
# Legal form pools
# ---------------------------------------------------------------------------

LEGAL_FORMS_DE = ["GmbH", "AG", "KG", "GmbH & Co. KG", "UG (haftungsbeschränkt)"]
LEGAL_FORMS_RELATED = ["GmbH", "AG", "KG"]        # same related group — legal_form_relation=related
LEGAL_FORMS_CONFLICT = ["Ltd.", "Corp.", "LLC", "S.A.", "N.V."]  # conflict group

# Noise additions
NOISE_SUFFIXES = ["Group", "Holding", "International", "Europe", "Deutschland", "Solutions"]
NOISE_PREFIXES = ["New", "United", "European", "Global"]


# ---------------------------------------------------------------------------
# FakerDataGenerator
# ---------------------------------------------------------------------------

class FakerDataGenerator:
    """
    Generates synthetic company name datasets for TGFR pipeline testing.

    Args:
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._faker_de = Faker("de_DE")
        self._faker_en = Faker("en_US")
        if seed is not None:
            Faker.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_company_list(self, n: int, language: str = "de") -> list[dict]:
        """
        Generate n synthetic company records.

        ~10% of entries include a simulated typo.
        ~20% of entries include a noise addition (suffix/prefix).
        Legal forms are randomly assigned from the DE or EN pool.

        Args:
            n: Number of records to generate.
            language: "de" (German base pool) or "en" (English base pool).

        Returns:
            List of dicts with keys: source_id, source_name.
        """
        base_pool = BASE_NAMES_DE if language == "de" else BASE_NAMES_EN
        records = []

        for i in range(n):
            base = self._rng.choice(base_pool)
            name = self._apply_legal_form(base, language)
            name = self._maybe_add_noise(name, rate=0.20)
            name = self._maybe_add_typo(name, rate=0.10)
            records.append({
                "source_id": f"{'crm' if language == 'de' else 'src'}_{i+1:04d}",
                "source_name": name,
            })

        return records

    def generate_matching_pair(
        self, base_name: str | None = None
    ) -> tuple[dict, dict]:
        """
        Generate a pair of records that should match (same underlying entity,
        realistic variance).

        Variant mutations (applied in order, independently):
          - If base_name is in ABBREVIATIONS (~always): use abbreviation as variant
          - ~10%: swap legal form (related or conflict)
          - ~20%: add noise suffix/prefix
          - ~10%: add typo

        Args:
            base_name: Optional specific base name. If None, randomly chosen.

        Returns:
            Tuple of (source_a_record, source_b_record).
        """
        if base_name is None:
            base_name = self._rng.choice(BASE_NAMES_DE + BASE_NAMES_EN)

        # Source A: base name with a DE legal form
        name_a = self._apply_legal_form(base_name, "de")

        # Source B: start from abbreviation if available, else base name
        if base_name in ABBREVIATIONS and self._rng.random() < 0.5:
            name_b = ABBREVIATIONS[base_name]
        else:
            name_b = base_name

        # ~10%: legal form change on B
        if self._rng.random() < 0.10:
            if self._rng.random() < 0.5:
                # related form change
                lf = self._rng.choice(LEGAL_FORMS_RELATED)
            else:
                # conflict form change (exercises priority-1 review cases)
                lf = self._rng.choice(LEGAL_FORMS_CONFLICT)
            name_b = f"{name_b} {lf}"
        else:
            name_b = self._apply_legal_form(name_b, "de")

        # ~20%: noise addition on B
        name_b = self._maybe_add_noise(name_b, rate=0.20)

        # ~10%: typo on B
        name_b = self._maybe_add_typo(name_b, rate=0.10)

        pair_id = uuid.uuid4().hex[:8]
        return (
            {"source_id": f"a_{pair_id}", "source_name": name_a},
            {"source_id": f"b_{pair_id}", "source_name": name_b},
        )

    def generate_non_matching_pair(self) -> tuple[dict, dict]:
        """
        Generate a pair of records that should clearly NOT match.

        Picks two names from different ends of the base pool to maximize
        semantic distance.

        Returns:
            Tuple of (source_a_record, source_b_record).
        """
        pool = BASE_NAMES_DE + BASE_NAMES_EN
        name_a = self._rng.choice(pool[:len(pool) // 2])
        name_b = self._rng.choice(pool[len(pool) // 2:])

        # Ensure they are actually different
        while name_a == name_b:
            name_b = self._rng.choice(pool[len(pool) // 2:])

        name_a = self._apply_legal_form(name_a, "de")
        name_b = self._apply_legal_form(name_b, "en")

        pair_id = uuid.uuid4().hex[:8]
        return (
            {"source_id": f"a_{pair_id}", "source_name": name_a},
            {"source_id": f"b_{pair_id}", "source_name": name_b},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_legal_form(self, name: str, language: str) -> str:
        """Append a randomly chosen legal form appropriate for the language."""
        if language == "de":
            lf = self._rng.choice(LEGAL_FORMS_DE)
        else:
            lf = self._rng.choice(LEGAL_FORMS_CONFLICT)  # English forms
        return f"{name} {lf}"

    def _maybe_add_noise(self, name: str, rate: float) -> str:
        """Add a noise suffix or prefix at the given probability rate."""
        if self._rng.random() < rate:
            if self._rng.random() < 0.5:
                noise = self._rng.choice(NOISE_SUFFIXES)
                return f"{name} {noise}"
            else:
                noise = self._rng.choice(NOISE_PREFIXES)
                return f"{noise} {name}"
        return name

    def _maybe_add_typo(self, name: str, rate: float) -> str:
        """Introduce a single character-level typo at the given probability rate."""
        if self._rng.random() >= rate or len(name) < 4:
            return name

        typo_type = self._rng.choice(["swap", "delete", "insert", "substitute"])
        chars = list(name)
        # Pick a position in the middle of the string (avoid first/last char)
        pos = self._rng.randint(1, len(chars) - 2)

        if typo_type == "swap" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo_type == "delete":
            chars.pop(pos)
        elif typo_type == "insert":
            chars.insert(pos, self._rng.choice(string.ascii_lowercase))
        elif typo_type == "substitute":
            chars[pos] = self._rng.choice(string.ascii_lowercase)

        return "".join(chars)
