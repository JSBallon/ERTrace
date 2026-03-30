"""
BLL — ERTracePipeline

Entity Resolution engine with full score and audit Trace.

Implements the TGFR algorithm (Transformer-Gather, Fuzzy-Reconsider):
  Stage 1 — Semantic Retrieval:
    Batch-embed A and B → build FAISS index on B → batch Top-K search
  Stage 2 — Per-entry Scoring:
    For each A entry and each Top-K candidate:
      fuzzy score + legal form score + composite score → ScoreVector → MatchCandidate
  Stage 3 — Rank + Route (ADR-M3-003):
    Sort candidates by composite_score descending → assign rank (0 = best)
    Apply Router to ALL candidates → routing_zone + review_priority per candidate
    Select best_candidate = candidates[0]
  Stage 4 — MatchResult construction:
    Flat fields read from best_candidate.score.* (never recomputed — ADR-M3-001)
    rerank_candidates = full ranked+routed list

Governance controls (ADR-M2-006, ADR-M3-002, ADR-M3-003):
  - match_result / no_match event per A entry
  - composite_inconsistency guardrail per candidate (ADR-M2-005)
  - FR-LF-05 guardrail via Router.apply() (ADR-M3-002)
  - validation_error + fallback MatchResult on any per-entry exception

Design constraints (ADR-M3-003b):
  - No Streamlit imports
  - No config/YAML/filesystem access — RunConfig loaded externally by app_service.py
  - No direct DAL access except CompanyRecord (schema) — normalisation done by caller
  - No OutputWriter — output writing handled by app_service.py
  - Fully framework-agnostic; usable directly in tests without file I/O

See bll/app_service.py for the cross-layer entry point that wires this engine
to config loading, DAL normalisation, OutputWriter, and audit logging.
"""

import uuid
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal, cast

from bll.schemas import (
    CompanyRecord,
    MatchCandidate,
    MatchResult,
    RunConfig,
    ScoreVector,
)
from bll.embedder import SentenceTransformerEmbedder
from bll.faiss_search import FaissSearcher
from bll.fuzzy_reranker import FuzzyReranker
from bll.legal_form_scorer import LegalFormScorer
from bll.composite_scorer import CompositeScorer
from bll.router import Router
from governance.audit_logger import AuditLogger


class ERTracePipeline:
    """
    Entity Resolution engine — pure BLL, no filesystem or framework dependencies.

    Receives a fully-constructed RunConfig and AuditLogger; executes the full
    TGFR matching sequence; returns one MatchResult per Source A entry.

    Usage:
        engine = ERTracePipeline(config, audit_logger)
        results = engine.run(records_a, records_b, progress_callback=cb)
    """

    def __init__(self, config: RunConfig, audit_logger: AuditLogger) -> None:
        """
        Initialise all BLL scoring and routing components from RunConfig.

        Args:
            config:       Fully validated RunConfig (from config_loader.py).
            audit_logger: Initialised AuditLogger for this run.
        """
        self.config = config
        self.logger = audit_logger

        self.embedder   = SentenceTransformerEmbedder(config.embedding_model)
        self.fuzzy      = FuzzyReranker()
        self.lf_scorer  = LegalFormScorer(config.legal_form_config)
        self.composite  = CompositeScorer(config.weights_config)
        self.router     = Router(
            config=config.threshold_config,
            run_id=config.run_id,
            audit_logger=audit_logger,
        )

    def run(
        self,
        records_a: list[CompanyRecord],
        records_b: list[CompanyRecord],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[MatchResult]:
        """
        Execute the full TGFR matching pipeline.

        Produces exactly one MatchResult per Source A entry (100% left-join coverage).
        Per-entry exceptions are caught, logged, and produce a fallback NO_MATCH —
        the engine never aborts mid-run.

        Args:
            records_a:         Normalised CompanyRecord list for Source A (CRM).
            records_b:         Normalised CompanyRecord list for Source B (Core Banking).
            progress_callback: Optional callback(completed: int, total: int) for UI progress.

        Returns:
            List of MatchResult, same length as records_a.
        """
        n      = len(records_a)
        top_k  = self.config.faiss_top_k
        results: list[MatchResult] = []

        # -----------------------------------------------------------------------
        # Stage 1 — Semantic Retrieval
        # Batch-embed all names; single model call each (ADR-M2-001).
        # -----------------------------------------------------------------------
        names_a = [r.name_normalized for r in records_a]
        names_b = [r.name_normalized for r in records_b]

        embeddings_a = self.embedder.embed_batch(names_a)
        embeddings_b = self.embedder.embed_batch(names_b)

        # Build FAISS index on B; batch-search all A entries (ADR-M2-002)
        searcher = FaissSearcher(embeddings_b)
        scores_mat, indices_mat = searcher.search(embeddings_a, top_k=top_k)

        # -----------------------------------------------------------------------
        # Stages 2–4 — Per-entry scoring, rank, routing, MatchResult
        # -----------------------------------------------------------------------
        ts_now = datetime.now(timezone.utc).isoformat()

        for i, rec_a in enumerate(records_a):
            trace_id = str(uuid.uuid4())
            try:
                result = self._score_entry(
                    i, rec_a, records_b,
                    searcher, scores_mat, indices_mat,
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
        trace_id: str,
        ts: str,
    ) -> MatchResult:
        """
        Score all Top-K candidates for one A entry, route them, return MatchResult.

        Steps (ADR-M3-003):
          1. Retrieve Top-K (b_idx, cosine) from FAISS.
          2. Per candidate: four scores → ScoreVector → MatchCandidate (store-at-compute-time).
          3. Sort descending by composite_score; assign rank (0 = best).
          4. Router.apply() on all candidates — routing_zone + review_priority per candidate.
          5. If best.routing_zone == NO_MATCH → _no_match_result().
          6. Construct MatchResult: flat fields from best.score.*, rerank_candidates=all.
        """
        raw_candidates = searcher.get_candidate(scores_mat, indices_mat, i)

        if not raw_candidates:
            return self._no_match_result(rec_a, trace_id, ts, best_score=0.0)

        # -----------------------------------------------------------------------
        # Step 2 — Score every candidate; build MatchCandidate + ScoreVector
        # -----------------------------------------------------------------------
        candidates: list[MatchCandidate] = []

        for b_idx, cosine_score in raw_candidates:
            rec_b = records_b[b_idx]

            jw, ts_ratio = self.fuzzy.score(
                rec_a.name_normalized, rec_b.name_normalized
            )
            lf_score, lf_relation_str = self.lf_scorer.score(
                rec_a.source_name, rec_b.source_name
            )
            lf_relation = cast(
                Literal["identical", "related", "conflict", "unknown"], lf_relation_str
            )
            cs = self.composite.score(cosine_score, jw, ts_ratio, lf_score)

            # Composite consistency guardrail (ADR-M2-005)
            if not self.composite.verify(cs, cosine_score, jw, ts_ratio, lf_score):
                self.logger.log_guardrail(
                    guardrail_name="composite_inconsistency",
                    triggered=True,
                    action="entry flagged; best score retained",
                    context={
                        "source_a_id": rec_a.source_id,
                        "b_idx":       b_idx,
                        "computed":    cs,
                        "trace_id":    trace_id,
                    },
                )

            candidates.append(MatchCandidate(
                source_b_id=rec_b.source_id,
                source_b_name=rec_b.source_name,
                source_b_name_normalized=rec_b.name_normalized,
                source_b_legal_form=rec_b.legal_form,
                score=ScoreVector(
                    embedding_cosine_score=cosine_score,
                    jaro_winkler_score=jw,
                    token_sort_ratio=ts_ratio,
                    legal_form_score=lf_score,
                    legal_form_relation=lf_relation,
                    composite_score=cs,
                ),
            ))

        # -----------------------------------------------------------------------
        # Step 3 — Sort descending; assign rank (ADR-M3-003)
        # -----------------------------------------------------------------------
        candidates.sort(key=lambda c: c.score.composite_score, reverse=True)
        candidates = [
            c.model_copy(update={"rank": idx}) for idx, c in enumerate(candidates)
        ]

        # -----------------------------------------------------------------------
        # Step 4 — Route all candidates (ADR-M3-002, ADR-M3-003)
        # FR-LF-05 guardrail fires inside Router.apply() when applicable.
        # -----------------------------------------------------------------------
        candidates = [self.router.apply(c) for c in candidates]

        # -----------------------------------------------------------------------
        # Step 5 — NO_MATCH check on best candidate
        # -----------------------------------------------------------------------
        best = candidates[0]

        if best.routing_zone == "NO_MATCH":
            return self._no_match_result(
                rec_a, trace_id, ts, best_score=best.score.composite_score
            )

        # -----------------------------------------------------------------------
        # Step 6 — Construct MatchResult
        # Flat fields from best.score.* — never recomputed (ADR-M3-001).
        # -----------------------------------------------------------------------
        match_result = MatchResult(
            source_a_id=rec_a.source_id,
            source_a_name=rec_a.source_name,
            source_a_name_normalized=rec_a.name_normalized,
            source_a_legal_form=rec_a.legal_form,
            source_b_id=best.source_b_id,
            source_b_name=best.source_b_name,
            source_b_name_normalized=best.source_b_name_normalized,
            source_b_legal_form=best.source_b_legal_form,
            embedding_cosine_score=best.score.embedding_cosine_score,
            jaro_winkler_score=best.score.jaro_winkler_score,
            token_sort_ratio=best.score.token_sort_ratio,
            legal_form_score=best.score.legal_form_score,
            legal_form_relation=best.score.legal_form_relation,
            composite_score=best.score.composite_score,
            routing_zone=best.routing_zone,
            review_priority=best.review_priority,
            rerank_candidates=candidates,
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
        rerank_candidates is explicitly empty — no candidates were above threshold.
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
            rerank_candidates=[],
            run_id=self.config.run_id,
            trace_id=trace_id,
            timestamp=ts,
        )
