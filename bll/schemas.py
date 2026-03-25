"""
BLL Schemas — Pydantic v2 data models for the TGFR pipeline.

All score fields are constrained to [0.0, 1.0].
WeightsConfig enforces sum = 1.0 (± 0.001 floating-point tolerance).
ThresholdConfig enforces review_lower_threshold < auto_match_threshold.

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
    input_file_a: str
    input_file_b: str
    timestamp: str


class MatchResult(BaseModel):
    """Complete matching result with full score vector — primary audit artifact."""
    # Source A fields
    source_a_id: str
    source_a_name: str
    source_a_name_normalized: str
    source_a_legal_form: str | None = None

    # Source B fields (None for NO_MATCH entries)
    source_b_id: str | None = None
    source_b_name: str | None = None
    source_b_name_normalized: str | None = None
    source_b_legal_form: str | None = None

    # Score vector — all components constrained [0.0, 1.0]
    embedding_cosine_score: float = Field(ge=0.0, le=1.0)
    jaro_winkler_score: float = Field(ge=0.0, le=1.0)
    token_sort_ratio: float = Field(ge=0.0, le=1.0)
    legal_form_score: float = Field(ge=0.0, le=1.0)
    legal_form_relation: Literal["identical", "related", "conflict", "unknown"]

    # Composite + routing
    composite_score: float = Field(ge=0.0, le=1.0)
    routing_zone: Literal["AUTO_MATCH", "REVIEW", "NO_MATCH"]
    review_priority: int = Field(ge=0, le=3)

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
