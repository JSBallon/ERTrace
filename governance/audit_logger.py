"""
Governance — AuditLogger

Append-only JSONL audit logger for the TGFR pipeline.
All events are written to logs/audit/audit_<run_id>.jsonl.

Every event carries:
  - event_type  : identifies the event category
  - run_id      : correlates all events in a single pipeline run
  - timestamp   : UTC ISO 8601 write time

Per-write open/close (mode='a') guarantees crash safety — events written
before a crash are never lost. See ADR-009 for design rationale.

frozenset fields are serialized via make_serializable() from dal/utils.py.
See ADR-010 for the shared utility rationale.

Event types:
  run_start          : full RunConfig at start of run
  match_result       : MatchResult with complete score vector
  no_match           : NO_MATCH entry with best candidate score for diagnosis
  guardrail_triggered: guardrail name, trigger condition, action taken
  validation_error   : error type and affected entry context
  run_end            : RunSummary with counts, rates, output file paths

No Streamlit imports. No BLL logic. No external API calls.
"""

import jsonlines
from datetime import datetime, timezone
from pathlib import Path

from bll.schemas import MatchResult, RunConfig, RunSummary
from dal.utils import make_serializable


class AuditLogger:
    """
    Append-only JSONL audit logger.

    Initialized with run_id — log path is fixed at construction.
    All log methods delegate to _write() which opens/closes the file
    on every call (crash-safe, never overwrites).
    """

    def __init__(self, run_id: str, audit_dir: str = "logs/audit"):
        self.run_id = run_id
        self.path = Path(audit_dir) / f"audit_{run_id}.jsonl"
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
            writer.write(make_serializable(event))

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_run_start(self, config: RunConfig) -> None:
        """
        Log the start of a pipeline run with the full configuration.

        Satisfies EU AI Act Art. 11 (technical documentation) and
        MaRisk AT 7.2 (run configuration traceability).

        Args:
            config: Full RunConfig — all parameters captured verbatim.
        """
        self._write({
            "event_type": "run_start",
            **config.model_dump(),
        })

    def log_match_result(self, result: MatchResult) -> None:
        """
        Log a single matching decision with its complete score vector.

        Satisfies EU AI Act Art. 12 (record-keeping) and
        MaRisk AT 4.3.4 (data quality decision traceability).

        Args:
            result: MatchResult — all score components, routing zone,
                    review priority, source IDs, and trace ID.
        """
        self._write({
            "event_type": "match_result",
            **result.model_dump(),
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
