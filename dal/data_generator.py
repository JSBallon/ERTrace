"""
DAL — FakerDataGenerator

Generates synthetic company name datasets for pipeline testing and Streamlit demos.

Two-phase generation strategy (generate_paired_datasets):

  PHASE 1 — Pool building (no distortion):
    Seed X → shared pool (k entries, both A and B)
    Seed Y → A-exclusive pool (M-k entries), rng_y continues into Phase 2
    Seed Z → B-exclusive pool (N-k entries), rng_z continues into Phase 3
    Per entry: optional SUFFIX (noise_rate), optional PREFIX (noise_rate),
               always a LEGAL_FORM — produces (built_name, raw_base) tuples

  PHASE 2 — List A end-to-end (rng_y continues from pool building):
    Combine shared + A-exclusive → shuffle (rng_y) → distort each entry (rng_y)
    Distortion: abbreviation → NOISE/NOISE1 compound → NOISE2 → typo

  PHASE 3 — List B end-to-end (rng_z continues from pool building):
    Combine shared + B-exclusive → shuffle (rng_z) → distort each entry (rng_z)
    Same distortion pipeline, fully independent from Phase 2

generate_company_list() uses the same _build_base_entry + _distort helpers
for single-list generation (used by run_manual_test.py and CLI).

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
# Abbreviation lookup table — produces "embedding finds it, fuzzy struggles"
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    "Bayerische Landesbank":           "BayernLB",
    "Norddeutsche Landesbank":         "NORD/LB",
    "Landesbank Baden-Württemberg":    "LBBW",
    "Helaba Landesbank Hessen-Thüringen": "Helaba",
    "DZ Bank":                         "DZ",
    "Deka Bank Deutsche Girozentrale": "DekaBank",
    "Deutsche Pfandbriefbank":         "pbb",
    "IKB Deutsche Industriebank":      "IKB",
    "NRW Bank":                        "NRW.BANK",
    "Commerzbank":                     "Coba",
    "Deutsche Bank":                   "DB",
    "Hamburger Sparkasse":             "Haspa",
    "Bridgewater Asset Management":    "Bridgewater",
    "Commonwealth Financial Services": "CFS",
}

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

# Legal forms — unified pool (no DE/EN split)
LEGAL_FORMS = [
    "GmbH", "AG", "KG", "GmbH & Co. KG", "UG (haftungsbeschränkt)",
    "Ltd.", "Corp.", "LLC", "S.A.", "N.V.",
]

# Base-list embellishments (applied during pool building, Phase 1)
SUFFIXES = ["Group", "Holding", "International", "Europe", "Deutschland", "Solutions"]
PREFIXES = ["New", "United", "European", "Global"]

# Distortion noise (applied during Phase 2 / Phase 3)
NOISE  = ["Regional", "Local", "Marketing", "Sales", "Customer",
          "Research", "Service", "Support", "Logistics"]
NOISE1 = ["Office", "Subsidiary", "Representation", "Division", "Department",
          "Entity", "Site", "Center", "Hub", "Unit"]
NOISE2 = ["Partner", "Distributor", "Reseller", "Vendor",
          "Supplier", "Contractor", "Franchise", "Agent"]


# ---------------------------------------------------------------------------
# Module-level pure helpers (no self._rng dependency)
# ---------------------------------------------------------------------------

def _build_base_entry(
    raw_base: str,
    rng: random.Random,
    noise_rate: float,
) -> tuple[str, str]:
    """
    Build one pool entry from a raw base name — Phase 1, no distortion.

    Applies SUFFIX and PREFIX independently at noise_rate, then always
    appends a LEGAL_FORM. Returns (built_name, raw_base) — raw_base is
    preserved so the distortion phase can look it up in ABBREVIATIONS.

    Args:
        raw_base:   Clean base name from BASE_NAMES_DE / BASE_NAMES_EN.
        rng:        RNG instance for this pool (X, Y, or Z).
        noise_rate: Independent probability for suffix and prefix roll.

    Returns:
        (built_name, raw_base) tuple.
    """
    name = raw_base

    # 1. SUFFIX — independent roll
    if rng.random() < noise_rate:
        name = f"{name} {rng.choice(SUFFIXES)}"

    # 2. PREFIX — independent roll
    if rng.random() < noise_rate:
        name = f"{rng.choice(PREFIXES)} {name}"

    # 3. LEGAL FORM — always
    name = f"{name} {rng.choice(LEGAL_FORMS)}"

    return name, raw_base


def _insert_at_random_blank(name: str, word: str, rng: random.Random) -> str:
    """
    Insert word at a uniformly random position in name.

    Positions include: before the first word, between any two adjacent words,
    and after the last word — i.e. len(parts) + 1 candidates for N words.

    Args:
        name: Current name string.
        word: Word (or compound) to insert.
        rng:  RNG instance for position selection.

    Returns:
        New name string with word inserted.
    """
    parts = name.split(" ")
    pos   = rng.randint(0, len(parts))    # 0 = before first, len = after last
    parts.insert(pos, word)
    return " ".join(parts)


def _distort(
    built: str,
    raw_base: str,
    rng: random.Random,
    typo_rate: float,
) -> str:
    """
    Apply all distortion strategies to a built name — Phase 2 / Phase 3.

    Order (must not be changed — abbreviation must precede noise insertion):
      1. Abbreviation  — ~50%, replaces raw_base portion with its abbreviation
      2a. NOISE compound — two independent 30% rolls for NOISE and NOISE1,
                           insert compound at random position (incl. start/end)
      2b. NOISE2         — independent 30% roll, insert at random position
      3. Typo            — typo_rate, single character mutation

    Args:
        built:     Base-list-built name (with suffix, prefix, legal form).
        raw_base:  Original raw pool name for abbreviation lookup.
        rng:       RNG instance for this list (Y or Z).
        typo_rate: Probability of a single-character typo.

    Returns:
        Distorted name string.
    """
    name = built

    # -----------------------------------------------------------------------
    # Step 1 — Abbreviation (~50%, if raw_base is in lookup table)
    # Applied first so noise inserts into the abbreviated form.
    # -----------------------------------------------------------------------
    if raw_base in ABBREVIATIONS and rng.random() < 0.50:
        name = name.replace(raw_base, ABBREVIATIONS[raw_base], 1)

    # -----------------------------------------------------------------------
    # Step 2a — NOISE compound (two independent 30% rolls)
    # Four outcomes: NOISE+NOISE1, NOISE only, NOISE1 only, neither
    # -----------------------------------------------------------------------
    use_noise  = rng.random() < 0.30
    use_noise1 = rng.random() < 0.30

    if use_noise and use_noise1:
        compound = f"{rng.choice(NOISE)} {rng.choice(NOISE1)}"
    elif use_noise:
        compound = rng.choice(NOISE)
    elif use_noise1:
        compound = rng.choice(NOISE1)
    else:
        compound = None

    if compound is not None:
        name = _insert_at_random_blank(name, compound, rng)

    # -----------------------------------------------------------------------
    # Step 2b — NOISE2 (independent 30% roll)
    # -----------------------------------------------------------------------
    if rng.random() < 0.30:
        name = _insert_at_random_blank(name, rng.choice(NOISE2), rng)

    # -----------------------------------------------------------------------
    # Step 3 — Typo (single character mutation at typo_rate)
    # -----------------------------------------------------------------------
    if rng.random() < typo_rate and len(name) >= 4:
        typo_type = rng.choice(["swap", "delete", "insert", "substitute"])
        chars = list(name)
        pos   = rng.randint(1, len(chars) - 2)

        if typo_type == "swap" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo_type == "delete":
            chars.pop(pos)
        elif typo_type == "insert":
            chars.insert(pos, rng.choice(string.ascii_lowercase))
        elif typo_type == "substitute":
            chars[pos] = rng.choice(string.ascii_lowercase)

        name = "".join(chars)

    return name


# ---------------------------------------------------------------------------
# FakerDataGenerator
# ---------------------------------------------------------------------------

class FakerDataGenerator:
    """
    Generates synthetic company name datasets for TGFR pipeline testing.

    For single-list generation: generate_company_list()
    For controlled-overlap paired generation: generate_paired_datasets()

    Args:
        seed: Optional random seed for self._rng (used by generate_company_list
              and the generate_*_pair helpers). generate_paired_datasets uses
              explicit seed_x/y/z parameters and ignores self._rng entirely.
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

    def generate_company_list(
        self,
        n: int,
        language: str = "de",
        id_prefix: str = "src",
        noise_rate: float = 0.30,
        typo_rate: float  = 0.30,
    ) -> list[dict]:
        """
        Generate n synthetic company records using self._rng.

        Each entry goes through Phase 1 (build) and Phase 2 (distort) in
        a single pass — no shuffle, since list order is not significant here.

        Args:
            n:          Number of records to generate.
            language:   "de" (German base pool) or "en" (English base pool).
            id_prefix:  Prefix for source_id values.
            noise_rate: Independent probability for SUFFIX and PREFIX in pool
                        building, and for NOISE/NOISE1/NOISE2 in distortion.
            typo_rate:  Probability of a single-character typo.

        Returns:
            List of dicts with keys: source_id, source_name.
        """
        base_pool = BASE_NAMES_DE if language == "de" else BASE_NAMES_EN
        records   = []

        for i in range(n):
            raw_base          = self._rng.choice(base_pool)
            built, raw_base   = _build_base_entry(raw_base, self._rng, noise_rate)
            distorted         = _distort(built, raw_base, self._rng, typo_rate)
            records.append({
                "source_id":   f"{id_prefix}_{i+1:04d}",
                "source_name": distorted,
            })

        return records

    def generate_paired_datasets(
        self,
        n_a: int,
        n_b: int,
        overlap_pct: float,
        noise_rate:  float = 0.30,
        typo_rate:   float = 0.30,
        seed_x: int = 42,
        seed_y: int = 137,
        seed_z: int = 271,
        id_prefix_a: str = "src-a",
        id_prefix_b: str = "src-b",
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate two paired lists with controlled overlap.

        Three-phase strategy:

        PHASE 1 — Pool building (no distortion):
          rng_x → shared pool (k entries)           ← same raw_base in both A and B
          rng_y → A-exclusive pool (n_a - k entries) ← rng_y continues into Phase 2
          rng_z → B-exclusive pool (n_b - k entries) ← rng_z continues into Phase 3

          Per entry: _build_base_entry applies SUFFIX, PREFIX (noise_rate each),
          then always appends LEGAL_FORM. Returns (built_name, raw_base).

        PHASE 2 — List A end-to-end (rng_y continues from Phase 1):
          raw_a = shared + excl_a → rng_y.shuffle → _distort each with rng_y

        PHASE 3 — List B end-to-end (rng_z continues from Phase 1):
          raw_b = shared + excl_b → rng_z.shuffle → _distort each with rng_z

        The shared entries receive independent distortion in A and B — the same
        underlying entity (same raw_base) looks different in each list, which
        is the correct scenario for exercising the TGFR pipeline.

        Args:
            n_a:         Number of Source A entries (M).
            n_b:         Number of Source B entries (N).
            overlap_pct: Fraction of Source A that is shared with B (0.0–1.0).
                         k = floor(n_a * overlap_pct), capped at n_b.
            noise_rate:  Base-list SUFFIX/PREFIX rate + distortion NOISE/NOISE1/NOISE2 rate.
            typo_rate:   Distortion typo probability.
            seed_x:      Seed for shared pool construction.
            seed_y:      Seed for A-exclusive pool + List A shuffle + distortion.
            seed_z:      Seed for B-exclusive pool + List B shuffle + distortion.
            id_prefix_a: source_id prefix for Source A.
            id_prefix_b: source_id prefix for Source B.

        Returns:
            (records_a, records_b) — lists of {source_id, source_name} dicts.
        """
        # k — shared entry count, capped at n_b
        k = min(int(n_a * overlap_pct), n_b)

        # -----------------------------------------------------------------------
        # PHASE 1 — Pool building, three independent RNGs
        # -----------------------------------------------------------------------
        rng_x = random.Random(seed_x)
        rng_y = random.Random(seed_y)
        rng_z = random.Random(seed_z)

        shared_pool = [
            _build_base_entry(rng_x.choice(BASE_NAMES_DE), rng_x, noise_rate)
            for _ in range(k)
        ]
        excl_a_pool = [
            _build_base_entry(rng_y.choice(BASE_NAMES_DE), rng_y, noise_rate)
            for _ in range(n_a - k)
        ]
        excl_b_pool = [
            _build_base_entry(rng_z.choice(BASE_NAMES_DE), rng_z, noise_rate)
            for _ in range(n_b - k)
        ]

        # -----------------------------------------------------------------------
        # PHASE 2 — List A (rng_y continues from Phase 1)
        # -----------------------------------------------------------------------
        raw_a: list[tuple[str, str]] = shared_pool + excl_a_pool   # M entries
        rng_y.shuffle(raw_a)

        records_a = []
        for i, (built, raw_base) in enumerate(raw_a):
            distorted = _distort(built, raw_base, rng_y, typo_rate)
            records_a.append({
                "source_id":   f"{id_prefix_a}_{i+1:04d}",
                "source_name": distorted,
            })

        # -----------------------------------------------------------------------
        # PHASE 3 — List B (rng_z continues from Phase 1)
        # -----------------------------------------------------------------------
        raw_b: list[tuple[str, str]] = shared_pool + excl_b_pool   # N entries
        rng_z.shuffle(raw_b)

        records_b = []
        for i, (built, raw_base) in enumerate(raw_b):
            distorted = _distort(built, raw_base, rng_z, typo_rate)
            records_b.append({
                "source_id":   f"{id_prefix_b}_{i+1:04d}",
                "source_name": distorted,
            })

        return records_a, records_b

    def generate_matching_pair(
        self, base_name: str | None = None
    ) -> tuple[dict, dict]:
        """
        Generate a pair of records that should match (same underlying entity).

        Variant mutations applied to Source B:
          - Abbreviation (~50% if available)
          - Legal form variant (~20%)
          - Noise addition (~20%)
          - Typo (~10%)
        """
        if base_name is None:
            base_name = self._rng.choice(BASE_NAMES_DE + BASE_NAMES_EN)

        name_a = f"{base_name} {self._rng.choice(LEGAL_FORMS)}"

        if base_name in ABBREVIATIONS and self._rng.random() < 0.5:
            name_b = ABBREVIATIONS[base_name]
        else:
            name_b = base_name

        if self._rng.random() < 0.20:
            name_b = f"{name_b} {self._rng.choice(LEGAL_FORMS)}"
        else:
            name_b = f"{name_b} {self._rng.choice(LEGAL_FORMS)}"

        name_b = self._maybe_add_noise(name_b, rate=0.20)
        name_b = self._maybe_add_typo(name_b, rate=0.10)

        pair_id = uuid.uuid4().hex[:8]
        return (
            {"source_id": f"a_{pair_id}", "source_name": name_a},
            {"source_id": f"b_{pair_id}", "source_name": name_b},
        )

    def generate_non_matching_pair(self) -> tuple[dict, dict]:
        """
        Generate a pair of records that should clearly NOT match.

        Picks two names from different ends of the base pool to maximise
        semantic distance.
        """
        pool   = BASE_NAMES_DE + BASE_NAMES_EN
        name_a = self._rng.choice(pool[:len(pool) // 2])
        name_b = self._rng.choice(pool[len(pool) // 2:])

        while name_a == name_b:
            name_b = self._rng.choice(pool[len(pool) // 2:])

        name_a = f"{name_a} {self._rng.choice(LEGAL_FORMS)}"
        name_b = f"{name_b} {self._rng.choice(LEGAL_FORMS)}"

        pair_id = uuid.uuid4().hex[:8]
        return (
            {"source_id": f"a_{pair_id}", "source_name": name_a},
            {"source_id": f"b_{pair_id}", "source_name": name_b},
        )

    # ------------------------------------------------------------------
    # Private helpers (use self._rng — for generate_company_list path)
    # ------------------------------------------------------------------

    def _apply_legal_form(self, name: str, language: str = "de") -> str:
        """Append a randomly chosen legal form. language param kept for API compat."""
        return f"{name} {self._rng.choice(LEGAL_FORMS)}"

    def _maybe_add_noise(
        self,
        name: str,
        rate: float,
        rng: random.Random | None = None,
    ) -> str:
        """Add a noise suffix or prefix at the given probability rate."""
        _rng = rng if rng is not None else self._rng
        if _rng.random() < rate:
            if _rng.random() < 0.5:
                return f"{name} {_rng.choice(SUFFIXES)}"
            else:
                return f"{_rng.choice(PREFIXES)} {name}"
        return name

    def _maybe_add_typo(
        self,
        name: str,
        rate: float,
        rng: random.Random | None = None,
    ) -> str:
        """Introduce a single character-level typo at the given probability rate."""
        _rng = rng if rng is not None else self._rng
        if _rng.random() >= rate or len(name) < 4:
            return name

        typo_type = _rng.choice(["swap", "delete", "insert", "substitute"])
        chars = list(name)
        pos   = _rng.randint(1, len(chars) - 2)

        if typo_type == "swap" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo_type == "delete":
            chars.pop(pos)
        elif typo_type == "insert":
            chars.insert(pos, _rng.choice(string.ascii_lowercase))
        elif typo_type == "substitute":
            chars[pos] = _rng.choice(string.ascii_lowercase)

        return "".join(chars)
