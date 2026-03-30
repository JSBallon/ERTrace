"""
Governance — AuditLogger

Append-only JSONL audit logger for the TGFR pipeline.
All events are written to logs/audit/audit_<YYYYMMDD-HHmm>_<run_id>.jsonl.

Filename includes a YYYYMMDD-HHmm datetime prefix (ADR-M3-004) for chronological
sortability in file explorers. The prefix is derived from datetime.now() at
construction time — AuditLogger is always instantiated at run-start in app_service.py.

Every event carries:
  - event_type  : identifies the event category
  - run_id      : correlates all events in a single pipeline run
  - timestamp   : UTC ISO 8601 write time (injected by _write())

Per-write open/close (mode='a') guarantees crash safety — events written
before a crash are never lost.

Event types:
  run_start          : full RunConfig at start of run
  match_result       : explicit score vector fields + rerank_count (not full list)
  no_match           : NO_MATCH entry with best candidate score for diagnosis
  guardrail_triggered: guardrail name, trigger condition, action taken
  validation_error   : error type and affected entry context
  run_end            : RunSummary with counts, rates, output file paths

Design note (ADR-M3-004):
  log_match_result emits an explicit field set — NOT result.model_dump().
  After M3, model_dump() would include rerank_candidates (full list), making
  the JSONL verbose. Only rerank_count (integer) is written to JSONL.
  The full rerank list is in the output JSON (dal/output_writer.py).

No Streamlit imports. No BLL logic. No external API calls.
"""

import jsonlines
from datetime import datetime, timezone
from pathlib import Path

from bll.schemas import MatchResult, RunConfig, RunSummary


class AuditLogger:
    """
    Append-only JSONL audit logger.

    Initialized with run_id — log path is fixed at construction.
    All log methods delegate to _write() which opens/closes the file
    on every call (crash-safe, never overwrites).
    """

    def __init__(self, run_id: str, audit_dir: str = "logs/audit"):
        self.run_id = run_id
        # Datetime prefix for chronological sortability (ADR-M3-004).
        # datetime.now() at construction is always the run-start time —
        # AuditLogger is instantiated at run-start in app_service.py.
        _prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        self.path = Path(audit_dir) / f"audit_{_prefix}_{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal write primitive
    # ------------------------------------------------------------------

    def _write(self, event: dict) -> None:
        """
        Append a single event to the JSONL log.

        Injects run_id and timestamp. Opens and closes the file on every
        call — crash-safe, never overwrites existing entries.
        """
        event["run_id"] = self.run_id
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with jsonlines.open(self.path, mode="a") as writer:
            writer.write(event)

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        config: RunConfig,
        input_file_a: str = "",
        input_file_b: str = "",
    ) -> None:
        """
        Log the start of a pipeline run with the full configuration.

        Satisfies EU AI Act Art. 11 (technical documentation) and
        MaRisk AT 7.2 (run configuration traceability).

        Input file paths are passed separately from RunConfig because they are
        run-time data pointers, not algorithm configuration parameters.
        They are validated by InputLoader before being recorded here — ensuring
        the audit record only captures paths confirmed to exist.

        Args:
            config:       Full RunConfig — all algorithm parameters captured verbatim.
            input_file_a: Validated path to Source A input file (CRM).
            input_file_b: Validated path to Source B input file (Core Banking).
        """
        self._write({
            "event_type": "run_start",
            **config.model_dump(),
            "input_file_a": input_file_a,
            "input_file_b": input_file_b,
        })

    def log_match_result(self, result: MatchResult) -> None:
        """
        Log a single matching decision with its complete flat score vector.

        Emits an explicit field set — NOT result.model_dump() — to exclude
        rerank_candidates (full list) and instead emit rerank_count (integer).
        The full rerank list is written to the output JSON by OutputWriter.
        This keeps the JSONL audit log compact and scannable (ADR-M3-004).

        rerank_count enables run-level completeness verification:
          sum(rerank_count across all match_result events) == RunSummary.total_rerank_candidates

        Satisfies EU AI Act Art. 12 (record-keeping), FR-SCR-02 (all individual
        scores captured per match decision), and MaRisk AT 4.3.4 (data quality
        decision traceability).

        Args:
            result: MatchResult with fully populated score vector and rerank list.
        """
        self._write({
            "event_type":               "match_result",
            "trace_id":                 result.trace_id,
            "source_a_id":              result.source_a_id,
            "source_a_name":            result.source_a_name,
            "source_a_name_normalized": result.source_a_name_normalized,
            "source_a_legal_form":      result.source_a_legal_form,
            "source_b_id":              result.source_b_id,
            "source_b_name":            result.source_b_name,
            "source_b_name_normalized": result.source_b_name_normalized,
            "source_b_legal_form":      result.source_b_legal_form,
            "embedding_cosine_score":   result.embedding_cosine_score,
            "jaro_winkler_score":       result.jaro_winkler_score,
            "token_sort_ratio":         result.token_sort_ratio,
            "legal_form_score":         result.legal_form_score,
            "legal_form_relation":      result.legal_form_relation,
            "composite_score":          result.composite_score,
            "routing_zone":             result.routing_zone,
            "review_priority":          result.review_priority,
            "rerank_count":             len(result.rerank_candidates),
        })

    def log_no_match(
        self,
        source_a_id: str,
        best_candidate_score: float,
        trace_id: str,
    ) -> None:
        """
        Log a NO_MATCH decision with the best candidate score for diagnosis.

        The best_candidate_score allows post-hoc analysis of why an entry
        did not reach the REVIEW threshold.

        Args:
            source_a_id          : Source A record identifier.
            best_candidate_score : Highest composite score found among Top-K candidates.
            trace_id             : Trace identifier for this entry.
        """
        self._write({
            "event_type": "no_match",
            "source_a_id": source_a_id,
            "routing_zone": "NO_MATCH",
            "best_candidate_score": best_candidate_score,
            "trace_id": trace_id,
        })

    def log_guardrail(
        self,
        guardrail_name: str,
        triggered: bool,
        action: str,
        context: dict,
    ) -> None:
        """
        Log a guardrail evaluation.

        Written for both triggered and non-triggered evaluations when
        relevant (e.g. priority override check, REVIEW rate warning).

        Satisfies EU AI Act Art. 9 (risk management) and
        MaRisk AT 4.3.2 (control activity documentation).

        Args:
            guardrail_name: Name of the guardrail (e.g. "priority_override").
            triggered     : True if the guardrail condition was met.
            action        : Action taken (e.g. "review_priority set to 1").
            context       : Dict with relevant entry or run context.
        """
        self._write({
            "event_type": "guardrail_triggered",
            "guardrail_name": guardrail_name,
            "triggered": triggered,
            "action": action,
            "context": context,
        })

    def log_validation_error(
        self,
        error_type: str,
        context: dict,
    ) -> None:
        """
        Log a validation error for an individual entry.

        Written when a score component falls outside [0.0, 1.0] or
        another Pydantic validation constraint is violated.

        Satisfies EU AI Act Art. 9 (risk management) and
        MaRisk AT 8.2 (error handling documentation).

        Args:
            error_type: Type of error (e.g. "score_out_of_range",
                        "composite_inconsistency").
            context   : Dict with the affected entry and error details.
        """
        self._write({
            "event_type": "validation_error",
            "error_type": error_type,
            "context": context,
        })

    def log_run_end(self, summary: RunSummary) -> None:
        """
        Log the end of a pipeline run with counts, rates, and output paths.

        Satisfies EU AI Act Art. 12 (record-keeping) and
        MaRisk AT 7.2 (run summary documentation).

        Args:
            summary: RunSummary — total entries, routing counts, rates,
                     review_quote_warning flag, and output file paths.
        """
        self._write({
            "event_type": "run_end",
            **summary.model_dump(),
        })
