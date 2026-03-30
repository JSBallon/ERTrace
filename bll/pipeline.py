"""
BLL — TGFRPipeline

TGFR (Transformer-Gather, Fuzzy-Reconsider) pipeline orchestrator.
Wires all five BLL components (Tasks 1–5) into a complete entity resolution run.

Processing sequence (see ADR-M2-006):
  Stage 1 — Semantic Retrieval:
    Batch-embed A and B → build FAISS index on B → batch Top-K search
  Stage 2 — Per-entry Syntactic Verification and Scoring:
    For each A entry: fuzzy score + legal form score + composite score
    Select best candidate by composite_score
  Stage 3 — Routing (M2 placeholder, inline threshold comparison):
    AUTO_MATCH | REVIEW | NO_MATCH
    review_priority = 0 (M2 sentinel — M3 ThresholdRouter overwrites with 1/2/3)

Governance controls exercised in this module (see ADR-M2-006):
  - run_start event logged at entry
  - match_result / no_match event per A entry
  - composite_inconsistency guardrail per candidate
  - validation_error + fallback MatchResult on any per-entry exception
  - run_end event with RunSummary at exit

Design constraints:
  - No Streamlit imports
  - No YAML / filesystem access (config loaded externally via config/config_loader.py)
  - No direct DAL access except CompanyRecord (schema) — normalisation done by caller
  - TGFRPipeline is framework-agnostic; run_entity_resolution() is the top-level entry point
"""

import uuid
import traceback
from collections.abc import Callable
from datetime import datetime, timezone

from bll.schemas import (
    CompanyRecord,
    LegalFormConfig,
    MatchResult,
    RunConfig,
    RunSummary,
    WeightsConfig,
)
from bll.embedder import SentenceTransformerEmbedder
from bll.faiss_search import FaissSearcher
from bll.fuzzy_reranker import FuzzyReranker
from bll.legal_form_scorer import LegalFormScorer
from bll.composite_scorer import CompositeScorer
from governance.audit_logger import AuditLogger


class TGFRPipeline:
    """
    Framework-agnostic TGFR pipeline orchestrator.

    Receives a fully-constructed RunConfig and AuditLogger — no filesystem access,
    no YAML parsing. All governance events are emitted through AuditLogger.

    Usage:
        pipeline = TGFRPipeline(config, audit_logger)
        results = pipeline.run(records_a, records_b, progress_callback=cb)
    """

    def __init__(self, config: RunConfig, audit_logger: AuditLogger) -> None:
        """
        Initialise all BLL components from RunConfig.

        Args:
            config:       Fully validated RunConfig (from config_loader.py).
            audit_logger: Initialised AuditLogger for this run.
        """
        self.config = config
        self.logger = audit_logger

        # Instantiate all scoring components from config
        self.embedder   = SentenceTransformerEmbedder(config.embedding_model)
        self.fuzzy      = FuzzyReranker()
        self.lf_scorer  = LegalFormScorer(config.legal_form_config)
        self.composite  = CompositeScorer(config.weights_config)

    def run(
        self,
        records_a: list[CompanyRecord],
        records_b: list[CompanyRecord],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[MatchResult]:
        """
        Execute the full TGFR matching pipeline.

        Produces exactly one MatchResult per Source A entry (100% left-join coverage).
        Errors on individual entries are caught, logged, and produce a fallback
        NO_MATCH MatchResult — the pipeline never aborts mid-run.

        Args:
            records_a:         Normalised CompanyRecord list for Source A (CRM).
            records_b:         Normalised CompanyRecord list for Source B (Core Banking).
            progress_callback: Optional callback(completed: int, total: int) for UI progress.

        Returns:
            List of MatchResult, same length as records_a.
        """
        n = len(records_a)
        results: list[MatchResult] = []
        top_k = self.config.faiss_top_k

        # -----------------------------------------------------------------------
        # Stage 1 — Semantic Retrieval
        # -----------------------------------------------------------------------

        # Embed all names in a single batch call each (ADR-M2-001)
        names_a = [r.name_normalized for r in records_a]
        names_b = [r.name_normalized for r in records_b]

        embeddings_a = self.embedder.embed_batch(names_a)
        embeddings_b = self.embedder.embed_batch(names_b)

        # Build FAISS index on B; batch-search all A entries (ADR-M2-002)
        searcher = FaissSearcher(embeddings_b)
        scores_mat, indices_mat = searcher.search(embeddings_a, top_k=top_k)

        # -----------------------------------------------------------------------
        # Stage 2 + 3 — Per-entry scoring, routing, audit
        # -----------------------------------------------------------------------

        auto_thresh   = self.config.threshold_config.auto_match_threshold
        review_thresh = self.config.threshold_config.review_lower_threshold
        ts_now        = datetime.now(timezone.utc).isoformat()

        for i, rec_a in enumerate(records_a):
            trace_id = str(uuid.uuid4())
            try:
                result = self._score_entry(
                    i, rec_a, records_b,
                    searcher, scores_mat, indices_mat,
                    auto_thresh, review_thresh,
                    trace_id, ts_now,
                )
            except Exception as exc:
                # Entry-level error isolation — log and produce fallback NO_MATCH
                self.logger.log_validation_error(
                    error_type="pipeline_entry_error",
                    context={
                        "source_a_id":   rec_a.source_id,
                        "source_a_name": rec_a.source_name,
                        "trace_id":      trace_id,
                        "error":         str(exc),
                        "traceback":     traceback.format_exc(),
                    },
                )
                result = self._no_match_result(
                    rec_a, trace_id, ts_now, best_score=0.0, log=False
                )

            results.append(result)

            if progress_callback is not None:
                progress_callback(i + 1, n)

        return results

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _score_entry(
        self,
        i: int,
        rec_a: CompanyRecord,
        records_b: list[CompanyRecord],
        searcher: FaissSearcher,
        scores_mat,
        indices_mat,
        auto_thresh: float,
        review_thresh: float,
        trace_id: str,
        ts: str,
    ) -> MatchResult:
        """Score a single A entry and return its MatchResult."""

        candidates = searcher.get_candidate(scores_mat, indices_mat, i)

        # No FAISS candidates → immediate NO_MATCH
        if not candidates:
            return self._no_match_result(rec_a, trace_id, ts, best_score=0.0)

        # --- Score every candidate ---
        scored: list[tuple[float, int, float, float, float, float, str]] = []
        # (composite_score, b_idx, cosine, jw, ts_ratio, lf_score, lf_relation)

        for b_idx, cosine_score in candidates:
            rec_b = records_b[b_idx]

            jw, ts_ratio = self.fuzzy.score(
                rec_a.name_normalized, rec_b.name_normalized
            )
            lf_score, lf_relation = self.lf_scorer.score(
                rec_a.source_name, rec_b.source_name
            )
            cs = self.composite.score(cosine_score, jw, ts_ratio, lf_score)

            # Composite consistency guardrail (ADR-M2-005)
            if not self.composite.verify(cs, cosine_score, jw, ts_ratio, lf_score):
                self.logger.log_guardrail(
                    guardrail_name="composite_inconsistency",
                    triggered=True,
                    action="entry flagged; best score retained",
                    context={
                        "source_a_id":   rec_a.source_id,
                        "b_idx":         b_idx,
                        "computed":      cs,
                        "trace_id":      trace_id,
                    },
                )

            scored.append((cs, b_idx, cosine_score, jw, ts_ratio, lf_score, lf_relation))

        # --- Select best candidate ---
        best = max(scored, key=lambda x: x[0])
        best_cs, best_b_idx, best_cosine, best_jw, best_ts, best_lf, best_lf_rel = best
        best_rec_b = records_b[best_b_idx]

        # --- M2 placeholder routing (replaced by ThresholdRouter in M3) ---
        if best_cs >= auto_thresh:
            routing_zone = "AUTO_MATCH"
        elif best_cs >= review_thresh:
            routing_zone = "REVIEW"
        else:
            routing_zone = "NO_MATCH"

        # review_priority=0 is the M2 sentinel (ge=0 passes Pydantic; M3 router
        # will overwrite with 1/2/3 — see ADR-M2-006)
        review_priority = 0

        if routing_zone == "NO_MATCH":
            return self._no_match_result(rec_a, trace_id, ts, best_score=best_cs)

        # --- Construct MatchResult (Pydantic validates all score ranges) ---
        match_result = MatchResult(
            source_a_id=rec_a.source_id,
            source_a_name=rec_a.source_name,
            source_a_name_normalized=rec_a.name_normalized,
            source_a_legal_form=rec_a.legal_form,
            source_b_id=best_rec_b.source_id,
            source_b_name=best_rec_b.source_name,
            source_b_name_normalized=best_rec_b.name_normalized,
            source_b_legal_form=best_rec_b.legal_form,
            embedding_cosine_score=best_cosine,
            jaro_winkler_score=best_jw,
            token_sort_ratio=best_ts,
            legal_form_score=best_lf,
            legal_form_relation=best_lf_rel,
            composite_score=best_cs,
            routing_zone=routing_zone,
            review_priority=review_priority,
            run_id=self.config.run_id,
            trace_id=trace_id,
            timestamp=ts,
        )

        self.logger.log_match_result(match_result)
        return match_result

    def _no_match_result(
        self,
        rec_a: CompanyRecord,
        trace_id: str,
        ts: str,
        best_score: float,
        log: bool = True,
    ) -> MatchResult:
        """
        Construct a NO_MATCH MatchResult for a Source A entry.

        All Source B fields are None. All scores are 0.0.
        legal_form_relation defaults to "unknown" (no information).
        """
        if log:
            self.logger.log_no_match(
                source_a_id=rec_a.source_id,
                best_candidate_score=best_score,
                trace_id=trace_id,
            )

        return MatchResult(
            source_a_id=rec_a.source_id,
            source_a_name=rec_a.source_name,
            source_a_name_normalized=rec_a.name_normalized,
            source_a_legal_form=rec_a.legal_form,
            source_b_id=None,
            source_b_name=None,
            source_b_name_normalized=None,
            source_b_legal_form=None,
            embedding_cosine_score=0.0,
            jaro_winkler_score=0.0,
            token_sort_ratio=0.0,
            legal_form_score=0.0,
            legal_form_relation="unknown",
            composite_score=0.0,
            routing_zone="NO_MATCH",
            review_priority=0,
            run_id=self.config.run_id,
            trace_id=trace_id,
            timestamp=ts,
        )


# ---------------------------------------------------------------------------
# Top-level entry point — callable without Streamlit (CLI, Python API)
# ---------------------------------------------------------------------------

def run_entity_resolution(
    source_a_path: str,
    source_b_path: str,
    config_path: str = "config/config.yaml",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[MatchResult], RunSummary]:
    """
    Top-level entity resolution entry point.

    Handles config loading, DAL normalisation, pipeline execution, output writing,
    and audit logging. Framework-agnostic — callable from CLI, tests, or Streamlit.

    Args:
        source_a_path:     Path to Source A CSV or JSON file.
        source_b_path:     Path to Source B CSV or JSON file.
        config_path:       Path to config/config.yaml (active version pointer).
        progress_callback: Optional callback(completed: int, total: int).

    Returns:
        Tuple of (list[MatchResult], RunSummary).
    """
    # Deferred imports to keep pipeline.py free of filesystem/DAL concerns
    # when used in unit-test contexts that instantiate TGFRPipeline directly.
    from config.config_loader import load_run_config
    from dal.input_loader import InputLoader
    from dal.normalizer import CompanyNameNormalizer
    from dal.legal_form_extractor import LegalFormExtractor
    from dal.output_writer import OutputWriter

    # -----------------------------------------------------------------------
    # Config — algorithm parameters only, no data paths (see ADR-M2-006)
    # -----------------------------------------------------------------------
    config = load_run_config(config_path)

    # -----------------------------------------------------------------------
    # Load raw records — InputLoader validates paths before they are recorded
    # -----------------------------------------------------------------------
    loader = InputLoader()
    raw_a = loader.load(source_a_path)
    raw_b = loader.load(source_b_path)

    # -----------------------------------------------------------------------
    # Normalise → CompanyRecord
    # -----------------------------------------------------------------------
    normalizer = CompanyNameNormalizer()
    extractor  = LegalFormExtractor()

    def to_company_record(raw: dict) -> CompanyRecord:
        raw_name      = str(raw["source_name"])
        term, _, _    = extractor.extract(raw_name)
        normalized    = normalizer.normalize(raw_name)
        return CompanyRecord(
            source_id=str(raw["source_id"]),
            source_name=raw_name,
            name_normalized=normalized,
            legal_form=term,
        )

    records_a = [to_company_record(r) for r in raw_a]
    records_b = [to_company_record(r) for r in raw_b]

    # -----------------------------------------------------------------------
    # Pipeline execution
    # Paths passed to log_run_start only after InputLoader has validated them —
    # the audit record captures confirmed, real file paths (see ADR-M2-006)
    # -----------------------------------------------------------------------
    ts_start    = datetime.now(timezone.utc).isoformat()
    audit_logger = AuditLogger(run_id=config.run_id)
    audit_logger.log_run_start(
        config,
        input_file_a=source_a_path,
        input_file_b=source_b_path,
    )

    pipeline = TGFRPipeline(config, audit_logger)
    results  = pipeline.run(records_a, records_b, progress_callback)

    # -----------------------------------------------------------------------
    # Output files
    # -----------------------------------------------------------------------
    ts_end  = datetime.now(timezone.utc).isoformat()
    writer  = OutputWriter()
    out_path    = writer.write_output_json(results, config.run_id, ts_end)
    review_path = writer.write_review_json(results, config.run_id, ts_end)

    # -----------------------------------------------------------------------
    # RunSummary + run_end audit event
    # -----------------------------------------------------------------------
    n_total      = len(results)
    n_auto       = sum(1 for r in results if r.routing_zone == "AUTO_MATCH")
    n_review     = sum(1 for r in results if r.routing_zone == "REVIEW")
    n_no_match   = sum(1 for r in results if r.routing_zone == "NO_MATCH")
    n_error      = 0  # error entries land in NO_MATCH — tracked via validation_error events

    review_warn_thresh = 0.30  # default monitoring threshold

    summary = RunSummary(
        run_id=config.run_id,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        total_entries_a=n_total,
        count_auto_match=n_auto,
        count_review=n_review,
        count_no_match=n_no_match,
        count_error=n_error,
        auto_match_quote=round(n_auto / n_total, 4) if n_total else 0.0,
        review_quote=round(n_review / n_total, 4) if n_total else 0.0,
        no_match_quote=round(n_no_match / n_total, 4) if n_total else 0.0,
        review_quote_warning=(n_review / n_total > review_warn_thresh) if n_total else False,
        output_file_path=out_path,
        review_file_path=review_path,
        audit_log_path=str(audit_logger.path),
    )

    audit_logger.log_run_end(summary)

    return results, summary
