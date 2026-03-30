"""
BLL Schemas — Pydantic v2 data models for the TGFR pipeline.

All score fields are constrained to [0.0, 1.0].
WeightsConfig enforces sum = 1.0 (± 0.001 floating-point tolerance).
ThresholdConfig enforces review_lower_threshold < auto_match_threshold.

M3 additions (ADR-M3-001):
  - ScoreVector: 6 score fields as a sub-model — stored at compute time, never recomputed.
  - MatchCandidate: one Top-K candidate from Source B — uniform type for rerank list and
    best match. Routing fields (routing_zone, review_priority) belong here; set by Router.
  - MatchResult: extended with rerank_candidates: list[MatchCandidate]. All flat fields
    remain unchanged for audit log compatibility.
  - RunSummary: extended with total_rerank_candidates: int for run_end JSONL event.

No Streamlit imports. No filesystem access. No external API calls.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal


class CompanyRecord(BaseModel):
    """Normalized company entry produced by the DAL."""
    source_id: str
    source_name: str
    name_normalized: str
    legal_form: str | None = None
    legal_form_class: Literal["identical", "related", "conflict", "unknown"] | None = None


# ---------------------------------------------------------------------------
# M3: ScoreVector + MatchCandidate (ADR-M3-001)
# ---------------------------------------------------------------------------

class ScoreVector(BaseModel):
    """Score components for one candidate pair — stored at compute time, never recomputed.

    Used as MatchCandidate.score. All six components are required and constrained [0.0, 1.0]
    (or a valid Literal for legal_form_relation).

    Design rule: OutputWriter reads these fields to build the nested JSON output.
    It must never recompute or derive scores — only read what is stored here.
    """
    embedding_cosine_score: float = Field(ge=0.0, le=1.0)
    jaro_winkler_score: float = Field(ge=0.0, le=1.0)
    token_sort_ratio: float = Field(ge=0.0, le=1.0)
    legal_form_score: float = Field(ge=0.0, le=1.0)
    legal_form_relation: Literal["identical", "related", "conflict", "unknown"]
    composite_score: float = Field(ge=0.0, le=1.0)


class MatchCandidate(BaseModel):
    """One Top-K candidate from Source B.

    Uniform type used for both the selected best match and all rerank list entries.
    Routing fields (routing_zone, review_priority) belong here — set once by Router,
    not by pipeline scoring logic.

    rank is set once after the Top-K list is sorted by composite_score descending.
    rank=0 is the best (selected) candidate; rank=1,2,... are the remaining candidates.
    """
    source_b_id: str | None = None
    source_b_name: str | None = None
    source_b_name_normalized: str | None = None
    source_b_legal_form: str | None = None
    score: ScoreVector
    routing_zone: Literal["AUTO_MATCH", "REVIEW", "NO_MATCH"] = "REVIEW"
    review_priority: int = Field(ge=0, le=3, default=0)
    rank: int = Field(ge=0, default=0)


class WeightsConfig(BaseModel):
    """Configurable weights for the composite score. Must sum to 1.0."""
    w_embedding: float = Field(ge=0.0, le=1.0)
    w_jaro_winkler: float = Field(ge=0.0, le=1.0)
    w_token_sort: float = Field(ge=0.0, le=1.0)
    w_legal_form: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def weights_must_sum_to_one(self) -> 'WeightsConfig':
        total = self.w_embedding + self.w_jaro_winkler + self.w_token_sort + self.w_legal_form
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got: {total:.4f} "
                f"(w_embedding={self.w_embedding}, w_jaro_winkler={self.w_jaro_winkler}, "
                f"w_token_sort={self.w_token_sort}, w_legal_form={self.w_legal_form})"
            )
        return self


class ThresholdConfig(BaseModel):
    """Configurable routing thresholds. review_lower_threshold must be < auto_match_threshold."""
    auto_match_threshold: float = Field(ge=0.0, le=1.0, default=0.92)
    review_lower_threshold: float = Field(ge=0.0, le=1.0, default=0.70)

    @model_validator(mode='after')
    def review_must_be_below_auto_match(self) -> 'ThresholdConfig':
        if self.review_lower_threshold >= self.auto_match_threshold:
            raise ValueError(
                f"review_lower_threshold ({self.review_lower_threshold}) must be "
                f"< auto_match_threshold ({self.auto_match_threshold})"
            )
        return self


class LegalFormConfig(BaseModel):
    """Configurable score levels for each legal form relation class."""
    identical_score: float = Field(ge=0.0, le=1.0, default=1.0)
    related_score: float = Field(ge=0.0, le=1.0, default=0.5)
    conflict_score: float = Field(ge=0.0, le=1.0, default=0.0)
    unknown_score: float = Field(ge=0.0, le=1.0, default=0.5)


class RunConfig(BaseModel):
    """Full run configuration — captured verbatim in the run_start audit event."""
    run_id: str
    embedding_model: str
    faiss_top_k: int = Field(ge=1, le=50)
    threshold_config: ThresholdConfig
    weights_config: WeightsConfig
    legal_form_config: LegalFormConfig
    threshold_config_version: str
    weights_config_version: str
    legal_form_config_version: str
    timestamp: str


class MatchResult(BaseModel):
    """Complete matching result — flat internal type for pipeline and audit log.

    All flat score fields remain unchanged from M2 for audit log compatibility.
    OutputWriter translates this to the nested entry/match/rerank JSON structure
    at serialization time — no nesting happens inside this model.

    rerank_candidates carries the full Top-K list (populated in M3 pipeline).
    OutputWriter serializes it as the rerank[] array in the output JSON.
    AuditLogger writes only rerank_count (integer) — not the full list.

    See ADR-M3-001 for the rationale behind the flat+rerank_candidates dual structure.
    """
    # Source A fields
    source_a_id: str
    source_a_name: str
    source_a_name_normalized: str
    source_a_legal_form: str | None = None

    # Source B fields — None for NO_MATCH entries
    source_b_id: str | None = None
    source_b_name: str | None = None
    source_b_name_normalized: str | None = None
    source_b_legal_form: str | None = None

    # Score vector — flat fields, all constrained [0.0, 1.0]
    # Mirrors the best MatchCandidate's ScoreVector for audit log compatibility.
    embedding_cosine_score: float = Field(ge=0.0, le=1.0)
    jaro_winkler_score: float = Field(ge=0.0, le=1.0)
    token_sort_ratio: float = Field(ge=0.0, le=1.0)
    legal_form_score: float = Field(ge=0.0, le=1.0)
    legal_form_relation: Literal["identical", "related", "conflict", "unknown"]

    # Composite + routing
    composite_score: float = Field(ge=0.0, le=1.0)
    routing_zone: Literal["AUTO_MATCH", "REVIEW", "NO_MATCH"]
    review_priority: int = Field(ge=0, le=3)

    # Full Top-K rerank list (M3) — serialized as rerank[] in output JSON.
    # Empty list for NO_MATCH entries. Default keeps M2 compatibility.
    rerank_candidates: list[MatchCandidate] = Field(default_factory=list)

    # Traceability
    run_id: str
    trace_id: str
    timestamp: str


class RunSummary(BaseModel):
    """Run summary — captured in the run_end audit event and displayed in Streamlit."""
    run_id: str
    timestamp_start: str
    timestamp_end: str
    total_entries_a: int
    count_auto_match: int
    count_review: int
    count_no_match: int
    count_error: int
    auto_match_quote: float
    review_quote: float
    no_match_quote: float
    review_quote_warning: bool
    output_file_path: str
    review_file_path: str
    audit_log_path: str
    # M3: total rerank candidates across the run — written to run_end JSONL event.
    # Enables completeness verification: total_rerank_candidates == total_entries_a * top_k
    # for all non-NO_MATCH entries.
    total_rerank_candidates: int = 0
