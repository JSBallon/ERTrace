"""
BLL — Application Service

Cross-layer entry point that wires ERTracePipeline to the outside world:
config loading, DAL normalisation, OutputWriter, and AuditLogger construction.

Separation of concerns (ADR-M3-003b):
  - ERTracePipeline  (bll/ertrace_pipeline.py): pure BLL engine — zero filesystem access
  - app_service.py   (this file)              : application service — bridges all layers

This module is the primary public API for:
  - CLI:       python -m gui.cli --source-a ... --source-b ...
  - Streamlit: from bll.app_service import run_entity_resolution
  - Tests:     from bll.app_service import run_entity_resolution
  - Manual:    run_manual_test.py

All DAL imports are deferred (inside the function body) so that importing this module
in test contexts that do not touch the filesystem does not trigger config/DAL loading.

No Streamlit imports. Filesystem access via config/ and dal/ layers only.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from bll.schemas import CompanyRecord, MatchResult, RunSummary
from bll.ertrace_pipeline import ERTracePipeline
from governance.audit_logger import AuditLogger


def run_entity_resolution(
    source_a_path: str,
    source_b_path: str,
    config_path: str = "config/config.yaml",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[MatchResult], RunSummary]:
    """
    Top-level entity resolution entry point.

    Handles config loading, DAL normalisation, engine execution, output writing,
    and audit logging. Framework-agnostic — callable from CLI, tests, or Streamlit.

    Args:
        source_a_path:     Path to Source A CSV or JSON file.
        source_b_path:     Path to Source B CSV or JSON file.
        config_path:       Path to config/config.yaml (active version pointer).
        progress_callback: Optional callback(completed: int, total: int).

    Returns:
        Tuple of (list[MatchResult], RunSummary).
    """
    # Deferred imports — keeps this module importable in unit-test contexts
    # that instantiate ERTracePipeline directly without any filesystem access.
    from config.config_loader import load_run_config
    from dal.input_loader import InputLoader
    from dal.normalizer import CompanyNameNormalizer
    from dal.legal_form_extractor import LegalFormExtractor
    from dal.output_writer import OutputWriter

    # -----------------------------------------------------------------------
    # Config — algorithm parameters only, no data paths (ADR-M2-006)
    # -----------------------------------------------------------------------
    config = load_run_config(config_path)

    # -----------------------------------------------------------------------
    # Load raw records — InputLoader validates paths before they are recorded
    # -----------------------------------------------------------------------
    loader = InputLoader()
    raw_a  = loader.load(source_a_path)
    raw_b  = loader.load(source_b_path)

    # -----------------------------------------------------------------------
    # Normalise → CompanyRecord
    # -----------------------------------------------------------------------
    normalizer = CompanyNameNormalizer()
    extractor  = LegalFormExtractor()

    def to_company_record(raw: dict) -> CompanyRecord:
        raw_name   = str(raw["source_name"])
        term, _, _ = extractor.extract(raw_name)
        normalized = normalizer.normalize(raw_name)
        return CompanyRecord(
            source_id=str(raw["source_id"]),
            source_name=raw_name,
            name_normalized=normalized,
            legal_form=term,
        )

    records_a = [to_company_record(r) for r in raw_a]
    records_b = [to_company_record(r) for r in raw_b]

    # -----------------------------------------------------------------------
    # Audit + engine
    # Paths passed to log_run_start only after InputLoader has validated them —
    # the audit record only captures confirmed, real file paths (ADR-M2-006).
    # -----------------------------------------------------------------------
    ts_start     = datetime.now(timezone.utc).isoformat()
    audit_logger = AuditLogger(run_id=config.run_id)
    audit_logger.log_run_start(
        config,
        input_file_a=source_a_path,
        input_file_b=source_b_path,
    )

    engine  = ERTracePipeline(config, audit_logger)
    results = engine.run(records_a, records_b, progress_callback)

    # -----------------------------------------------------------------------
    # Output files
    # -----------------------------------------------------------------------
    ts_end      = datetime.now(timezone.utc).isoformat()
    writer      = OutputWriter()
    out_path    = writer.write_output_json(results, config.run_id, ts_end)
    review_path = writer.write_review_json(results, config.run_id, ts_end)

    # -----------------------------------------------------------------------
    # RunSummary + run_end audit event
    # -----------------------------------------------------------------------
    n_total    = len(results)
    n_auto     = sum(1 for r in results if r.routing_zone == "AUTO_MATCH")
    n_review   = sum(1 for r in results if r.routing_zone == "REVIEW")
    n_no_match = sum(1 for r in results if r.routing_zone == "NO_MATCH")
    n_error    = 0       # error entries land in NO_MATCH — tracked via validation_error events
    n_rerank   = sum(len(r.rerank_candidates) for r in results)

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
        total_rerank_candidates=n_rerank,
    )

    audit_logger.log_run_end(summary)

    return results, summary
