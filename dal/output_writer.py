"""
DAL — OutputWriter

Serializes pipeline results to disk:
  - output_<run_id>.json   : all MatchResult entries (including NO_MATCH)
  - review_<run_id>.json   : REVIEW + AUTO_MATCH entries with review_priority > 0,
                             sorted ascending by review_priority (P1 first)

JSONL audit logging is NOT the responsibility of OutputWriter.
Use governance/audit_logger.py (AuditLogger) exclusively for audit events.

All fields in MatchResult are JSON-serializable via Pydantic model_dump() directly —
no pre-processor needed (frozenset removed from domain model, see ADR-006 refactor).

See ADR-008 for design rationale.

No Streamlit imports. No BLL imports. No external API calls.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from bll.schemas import MatchResult


class OutputWriter:
    """
    Writes pipeline output files to the outputs/ directory.

    All methods return the absolute file path of the written file as a string,
    so the caller can capture paths for the run_end audit event.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main output
    # ------------------------------------------------------------------

    def write_output_json(
        self,
        results: list[MatchResult],
        run_id: str,
        timestamp: str | None = None,
    ) -> str:
        """
        Write all MatchResult entries to output_<run_id>.json.

        Includes ALL entries: AUTO_MATCH, REVIEW, NO_MATCH, and error entries.

        Args:
            results  : List of MatchResult objects from the pipeline.
            run_id   : Run identifier — used in the filename.
            timestamp: Optional ISO timestamp string for the metadata header.

        Returns:
            Absolute path of the written file as a string.
        """
        path = self.output_dir / f"output_{run_id}.json"
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        payload = {
            "run_id": run_id,
            "generated_at": ts,
            "total_entries": len(results),
            "results": [r.model_dump() for r in results],
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
        Write prioritized review entries to review_<run_id>.json.

        Includes only entries where review_priority > 0 (i.e. REVIEW zone entries
        and AUTO_MATCH entries with legal form conflict). Sorted ascending by
        review_priority so P1 (value=1, mandatory) appears first.

        Args:
            results  : List of MatchResult objects from the pipeline.
            run_id   : Run identifier — used in the filename.
            timestamp: Optional ISO timestamp string for the metadata header.

        Returns:
            Absolute path of the written file as a string.
        """
        path = self.output_dir / f"review_{run_id}.json"
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        review_entries = [r for r in results if r.review_priority > 0]
        review_entries.sort(key=lambda r: r.review_priority)  # P1 first

        payload = {
            "run_id": run_id,
            "generated_at": ts,
            "total_review_entries": len(review_entries),
            "sorted_by": "review_priority ascending (1=mandatory, 3=low-urgency)",
            "entries": [r.model_dump() for r in review_entries],
        }

        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


