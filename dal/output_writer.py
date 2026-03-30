"""
DAL — OutputWriter

Serializes pipeline results to disk (ADR-M3-004):
  - output_<YYYYMMDD-HHmm>_<run_id>.json  : all MatchResult entries (including NO_MATCH)
                                             nested entry / match / rerank structure
  - review_<YYYYMMDD-HHmm>_<run_id>.json  : entries with review_priority > 0,
                                             sorted ascending by review_priority (P1 first)

Nested output format per entry:
  entry  — Source A identification fields
  match  — Source B best-match: id, name, normalized name, legal form, score sub-object,
            routing_zone, review_priority, rank (always 0 for the selected best match)
  rerank — Full Top-K candidate list (MatchCandidate.model_dump() per element)

JSONL audit logging is NOT the responsibility of OutputWriter.
Use governance/audit_logger.py (AuditLogger) exclusively for audit events.

Review filter: review_priority > 0 (not routing_zone == REVIEW).
This ensures AUTO_MATCH + legal form conflict entries (P1) appear in the review file
per FR-LF-05, even though their routing zone is AUTO_MATCH.

Design rules (ADR-M3-004):
  - _to_nested() is a read-model projection of the flat MatchResult (CQRS pattern).
    It reads stored field values and never recomputes or derives scores.
  - _to_nested() is called only from OutputWriter. Never from BLL or ERTracePipeline.
  - "rank": 0 in the match section is a documented pipeline invariant, not a magic number.

No Streamlit imports. No BLL logic. No external API calls.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from bll.schemas import MatchResult


class OutputWriter:
    """
    Writes pipeline output files to the outputs/ directory.

    All write methods return the absolute file path of the written file as a string,
    so the caller can capture paths for the RunSummary and run_end audit event.

    Filenames include a YYYYMMDD-HHmm datetime prefix for chronological sortability
    in file explorers without requiring a database or index (ADR-M3-004).
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _datetime_prefix(ts: str) -> str:
        """
        Derive a YYYYMMDD-HHmm prefix string from an ISO 8601 timestamp.

        HHmm resolution is sufficient — no two runs in the same minute will
        produce a filename collision because run_id (UUID) is always appended.

        Args:
            ts: ISO 8601 timestamp string (e.g. "2026-03-30T14:23:45+00:00").

        Returns:
            String of format "YYYYMMDD-HHmm" (e.g. "20260330-1423").
        """
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y%m%d-%H%M")

    def _to_nested(self, result: MatchResult) -> dict:
        """
        Translate a flat MatchResult to the three-section nested output dict.

        This is the read-model projection of MatchResult (CQRS pattern, ADR-M3-004):
          - MatchResult (flat) is the write model — audit-log compatible, append-only.
          - _to_nested() is the read model — human-readable, DQM-navigable.

        Sections:
          entry  — Source A: who was being matched
          match  — Best candidate: identity + score sub-object + routing + rank
          rerank — Full Top-K evaluated candidates (MatchCandidate.model_dump())

        Design rules:
          - All values READ from stored fields — never recomputed here.
          - "rank": 0 is a documented pipeline invariant (best candidate is always
            rank=0 by ERTracePipeline design). It is not a magic number.
          - MatchCandidate.model_dump() produces a "score" sub-object (ScoreVector
            nested) — consistent with the match.score structure.
          - NO_MATCH entries: source_b_* = None, all scores = 0.0, rerank = [].
            No special-casing needed — structurally valid.

        Args:
            result: A MatchResult from the pipeline (any routing zone).

        Returns:
            Dict with keys: entry, match, rerank, trace_id, timestamp.
        """
        return {
            "entry": {
                "source_a_id":              result.source_a_id,
                "source_a_name":            result.source_a_name,
                "source_a_name_normalized": result.source_a_name_normalized,
                "source_a_legal_form":      result.source_a_legal_form,
            },
            "match": {
                "source_b_id":              result.source_b_id,
                "source_b_name":            result.source_b_name,
                "source_b_name_normalized": result.source_b_name_normalized,
                "source_b_legal_form":      result.source_b_legal_form,
                "score": {
                    "embedding_cosine_score": result.embedding_cosine_score,
                    "jaro_winkler_score":     result.jaro_winkler_score,
                    "token_sort_ratio":       result.token_sort_ratio,
                    "legal_form_score":       result.legal_form_score,
                    "legal_form_relation":    result.legal_form_relation,
                    "composite_score":        result.composite_score,
                },
                "routing_zone":    result.routing_zone,
                "review_priority": result.review_priority,
                "rank": 0,  # pipeline invariant: selected best candidate is always rank=0
            },
            "rerank":    [c.model_dump() for c in result.rerank_candidates],
            "trace_id":  result.trace_id,
            "timestamp": result.timestamp,
        }

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_output_json(
        self,
        results: list[MatchResult],
        run_id: str,
        timestamp: str | None = None,
    ) -> str:
        """
        Write all MatchResult entries to output_<YYYYMMDD-HHmm>_<run_id>.json.

        Includes ALL entries: AUTO_MATCH, REVIEW, NO_MATCH, and error entries.
        Each entry is serialized using the nested entry/match/rerank structure.

        Args:
            results  : List of MatchResult objects from the pipeline.
            run_id   : Run identifier — appended to the filename for correlation.
            timestamp: Optional ISO timestamp string. Defaults to now() if not provided.

        Returns:
            Absolute path of the written file as a string.
        """
        ts   = timestamp or datetime.now(timezone.utc).isoformat()
        path = self.output_dir / f"output_{self._datetime_prefix(ts)}_{run_id}.json"

        payload = {
            "run_id":        run_id,
            "generated_at":  ts,
            "total_entries": len(results),
            "results":       [self._to_nested(r) for r in results],
        }

        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    def write_review_json(
        self,
        results: list[MatchResult],
        run_id: str,
        timestamp: str | None = None,
    ) -> str:
        """
        Write prioritized review entries to review_<YYYYMMDD-HHmm>_<run_id>.json.

        Filter: review_priority > 0 (not routing_zone == REVIEW).
        This captures AUTO_MATCH + legal form conflict entries (P1) per FR-LF-05,
        which must appear in the review file even though their zone is AUTO_MATCH.

        Sort: ascending by review_priority — P1 (value=1, mandatory) appears first,
        P3 (value=3, low-urgency) appears last. DQM expert works top-to-bottom.

        Args:
            results  : List of MatchResult objects from the pipeline.
            run_id   : Run identifier — appended to the filename for correlation.
            timestamp: Optional ISO timestamp string. Defaults to now() if not provided.

        Returns:
            Absolute path of the written file as a string.
        """
        ts   = timestamp or datetime.now(timezone.utc).isoformat()
        path = self.output_dir / f"review_{self._datetime_prefix(ts)}_{run_id}.json"

        review_entries = [r for r in results if r.review_priority > 0]
        review_entries.sort(key=lambda r: r.review_priority)  # P1 first (ascending)

        payload = {
            "run_id":               run_id,
            "generated_at":         ts,
            "total_review_entries": len(review_entries),
            "sorted_by":            "review_priority ascending (1=mandatory, 3=low-urgency)",
            "entries":              [self._to_nested(r) for r in review_entries],
        }

        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)
