"""
Tests for governance/audit_logger.py — AuditLogger

Covers:
  - Log file created on first write
  - All 6 event types write correct structure
  - run_id and timestamp injected into every event
  - Append-only behavior — no overwriting
  - Each JSONL line is valid JSON
  - Full run sequence: all 6 events in correct order
"""

import json
import pytest
from pathlib import Path
from governance.audit_logger import AuditLogger
from bll.schemas import (
    MatchCandidate, MatchResult, RunConfig, RunSummary, ScoreVector,
    WeightsConfig, ThresholdConfig, LegalFormConfig,
)


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture
def run_id():
    return "test_run_001"


@pytest.fixture
def logger(run_id, tmp_path):
    return AuditLogger(run_id=run_id, audit_dir=str(tmp_path))


@pytest.fixture
def log_path(logger):
    # Derive from the actual logger path — works with any filename format,
    # including the YYYYMMDD-HHmm datetime prefix introduced in ADR-M3-004.
    return logger.path


@pytest.fixture
def run_config(run_id):
    return RunConfig(
        run_id=run_id,
        embedding_model="test-model",
        faiss_top_k=5,
        threshold_config=ThresholdConfig(
            auto_match_threshold=0.92,
            review_lower_threshold=0.70,
        ),
        weights_config=WeightsConfig(
            w_embedding=0.50,
            w_jaro_winkler=0.20,
            w_token_sort=0.20,
            w_legal_form=0.10,
        ),
        legal_form_config=LegalFormConfig(),
        threshold_config_version="v1.0-default",
        weights_config_version="v1.0-default",
        legal_form_config_version="v1.0-default",
        timestamp="2026-03-28T00:00:00Z",
    )


@pytest.fixture
def match_result(run_id):
    return MatchResult(
        source_a_id="crm_001",
        source_a_name="ACME GmbH",
        source_a_name_normalized="acme",
        source_a_legal_form="GmbH",
        source_b_id="cb_001",
        source_b_name="ACME AG",
        source_b_name_normalized="acme",
        source_b_legal_form="AG",
        embedding_cosine_score=0.91,
        jaro_winkler_score=0.85,
        token_sort_ratio=0.90,
        legal_form_score=0.0,
        legal_form_relation="conflict",
        composite_score=0.80,
        routing_zone="REVIEW",
        review_priority=1,
        run_id=run_id,
        trace_id="trace_001",
        timestamp="2026-03-28T00:00:00Z",
    )


@pytest.fixture
def run_summary(run_id):
    return RunSummary(
        run_id=run_id,
        timestamp_start="2026-03-28T00:00:00Z",
        timestamp_end="2026-03-28T00:01:00Z",
        total_entries_a=10,
        count_auto_match=5,
        count_review=3,
        count_no_match=2,
        count_error=0,
        auto_match_quote=0.5,
        review_quote=0.3,
        no_match_quote=0.2,
        review_quote_warning=False,
        output_file_path="outputs/output_test_run_001.json",
        review_file_path="outputs/review_test_run_001.json",
        audit_log_path="logs/audit/audit_test_run_001.jsonl",
    )


def _read_events(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------
# File creation
# ------------------------------------------------------------------

def test_log_file_created(logger, log_path, run_config):
    logger.log_run_start(run_config)
    assert log_path.exists()


# ------------------------------------------------------------------
# run_start event
# ------------------------------------------------------------------

def test_run_start_event_type(logger, log_path, run_config):
    logger.log_run_start(run_config)
    events = _read_events(log_path)
    assert events[0]["event_type"] == "run_start"


def test_run_start_contains_run_id(logger, log_path, run_config, run_id):
    logger.log_run_start(run_config)
    events = _read_events(log_path)
    assert events[0]["run_id"] == run_id


def test_run_start_contains_timestamp(logger, log_path, run_config):
    logger.log_run_start(run_config)
    events = _read_events(log_path)
    assert "timestamp" in events[0]
    assert events[0]["timestamp"]  # non-empty


def test_run_start_contains_embedding_model(logger, log_path, run_config):
    logger.log_run_start(run_config)
    events = _read_events(log_path)
    assert events[0]["embedding_model"] == "test-model"


def test_run_start_contains_config_versions(logger, log_path, run_config):
    logger.log_run_start(run_config)
    events = _read_events(log_path)
    assert events[0]["threshold_config_version"] == "v1.0-default"
    assert events[0]["weights_config_version"] == "v1.0-default"
    assert events[0]["legal_form_config_version"] == "v1.0-default"


# ------------------------------------------------------------------
# match_result event
# ------------------------------------------------------------------

def test_match_result_event_type(logger, log_path, match_result):
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    assert events[0]["event_type"] == "match_result"


def test_match_result_contains_score_vector(logger, log_path, match_result):
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    e = events[0]
    assert "embedding_cosine_score" in e
    assert "jaro_winkler_score" in e
    assert "token_sort_ratio" in e
    assert "legal_form_score" in e
    assert "composite_score" in e


def test_match_result_contains_routing(logger, log_path, match_result):
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    assert events[0]["routing_zone"] == "REVIEW"
    assert events[0]["review_priority"] == 1


def test_match_result_contains_trace_id(logger, log_path, match_result):
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    assert events[0]["trace_id"] == "trace_001"


def test_match_result_contains_rerank_count(logger, log_path, match_result):
    """rerank_count must be present in the match_result event as an integer (ADR-M3-005)."""
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    e = events[0]
    assert "rerank_count" in e
    assert isinstance(e["rerank_count"], int)
    # match_result fixture has no rerank_candidates (default empty list) → count is 0
    assert e["rerank_count"] == 0


def test_match_result_rerank_count_reflects_candidates(logger, log_path, run_id):
    """rerank_count must equal len(result.rerank_candidates) — here 3 candidates."""
    def _sv() -> ScoreVector:
        return ScoreVector(
            embedding_cosine_score=0.90,
            jaro_winkler_score=0.85,
            token_sort_ratio=0.88,
            legal_form_score=1.0,
            legal_form_relation="identical",
            composite_score=0.88,
        )

    candidates = [
        MatchCandidate(source_b_id=f"b{i}", score=_sv(), rank=i)
        for i in range(3)
    ]

    result_with_rerank = MatchResult(
        source_a_id="crm_rc_001",
        source_a_name="Test GmbH",
        source_a_name_normalized="test",
        embedding_cosine_score=0.90,
        jaro_winkler_score=0.85,
        token_sort_ratio=0.88,
        legal_form_score=1.0,
        legal_form_relation="identical",
        composite_score=0.88,
        routing_zone="AUTO_MATCH",
        review_priority=0,
        rerank_candidates=candidates,
        run_id=run_id,
        trace_id="trace_rc_001",
        timestamp="2026-03-30T00:00:00Z",
    )

    logger.log_match_result(result_with_rerank)
    events = _read_events(log_path)
    assert events[0]["rerank_count"] == 3


def test_match_result_does_not_contain_full_rerank_list(logger, log_path, match_result):
    """Full rerank_candidates list must NOT appear in the JSONL event — only rerank_count.

    This is the governance contract from ADR-M3-005: the JSONL audit log stays compact.
    The full list lives in the output JSON (OutputWriter, ADR-M3-004).
    """
    logger.log_match_result(match_result)
    events = _read_events(log_path)
    e = events[0]
    assert "rerank_candidates" not in e


# ------------------------------------------------------------------
# no_match event
# ------------------------------------------------------------------

def test_no_match_event_type(logger, log_path):
    logger.log_no_match("crm_002", 0.45, "trace_002")
    events = _read_events(log_path)
    assert events[0]["event_type"] == "no_match"


def test_no_match_contains_fields(logger, log_path):
    logger.log_no_match("crm_002", 0.45, "trace_002")
    events = _read_events(log_path)
    e = events[0]
    assert e["source_a_id"] == "crm_002"
    assert e["best_candidate_score"] == pytest.approx(0.45)
    assert e["trace_id"] == "trace_002"
    assert e["routing_zone"] == "NO_MATCH"


# ------------------------------------------------------------------
# guardrail_triggered event
# ------------------------------------------------------------------

def test_guardrail_event_type(logger, log_path):
    logger.log_guardrail("priority_override", True, "review_priority set to 1", {"source_a_id": "crm_001"})
    events = _read_events(log_path)
    assert events[0]["event_type"] == "guardrail_triggered"


def test_guardrail_contains_fields(logger, log_path):
    logger.log_guardrail("priority_override", True, "review_priority set to 1", {"source_a_id": "crm_001"})
    events = _read_events(log_path)
    e = events[0]
    assert e["guardrail_name"] == "priority_override"
    assert e["triggered"] is True
    assert e["action"] == "review_priority set to 1"
    assert e["context"]["source_a_id"] == "crm_001"


# ------------------------------------------------------------------
# validation_error event
# ------------------------------------------------------------------

def test_validation_error_event_type(logger, log_path):
    logger.log_validation_error("score_out_of_range", {"value": 1.5})
    events = _read_events(log_path)
    assert events[0]["event_type"] == "validation_error"


def test_validation_error_contains_fields(logger, log_path):
    logger.log_validation_error("score_out_of_range", {"value": 1.5})
    events = _read_events(log_path)
    e = events[0]
    assert e["error_type"] == "score_out_of_range"
    assert e["context"]["value"] == pytest.approx(1.5)


# ------------------------------------------------------------------
# run_end event
# ------------------------------------------------------------------

def test_run_end_event_type(logger, log_path, run_summary):
    logger.log_run_end(run_summary)
    events = _read_events(log_path)
    assert events[0]["event_type"] == "run_end"


def test_run_end_contains_counts(logger, log_path, run_summary):
    logger.log_run_end(run_summary)
    events = _read_events(log_path)
    e = events[0]
    assert e["total_entries_a"] == 10
    assert e["count_auto_match"] == 5
    assert e["count_review"] == 3
    assert e["count_no_match"] == 2
    assert e["count_error"] == 0


def test_run_end_contains_output_paths(logger, log_path, run_summary):
    logger.log_run_end(run_summary)
    events = _read_events(log_path)
    e = events[0]
    assert "output_file_path" in e
    assert "review_file_path" in e
    assert "audit_log_path" in e


def test_run_end_contains_total_rerank_candidates(logger, log_path, run_summary):
    """total_rerank_candidates from RunSummary must appear in the run_end event (ADR-M3-005).

    Flows through automatically via RunSummary.model_dump() — no explicit logger code needed.
    The run_summary fixture uses the default value (0); the test verifies presence and type.
    """
    logger.log_run_end(run_summary)
    events = _read_events(log_path)
    e = events[0]
    assert "total_rerank_candidates" in e
    assert isinstance(e["total_rerank_candidates"], int)


# ------------------------------------------------------------------
# Append-only behavior
# ------------------------------------------------------------------

def test_append_only_multiple_writes(logger, log_path, run_config, match_result, run_summary):
    logger.log_run_start(run_config)
    logger.log_match_result(match_result)
    logger.log_run_end(run_summary)
    events = _read_events(log_path)
    assert len(events) == 3


def test_append_only_no_overwrite(run_id, tmp_path, run_config, run_summary):
    # Write run_start with first logger instance
    logger1 = AuditLogger(run_id=run_id, audit_dir=str(tmp_path))
    logger1.log_run_start(run_config)

    # Write run_end with second logger instance on same file
    logger2 = AuditLogger(run_id=run_id, audit_dir=str(tmp_path))
    logger2.log_run_end(run_summary)

    # Use logger1.path — works with any filename format including datetime prefix.
    # logger1 and logger2 are instantiated within the same second so their
    # strftime("%Y%m%d-%H%M") prefix is identical — they write to the same file.
    log_path = logger1.path
    events = _read_events(log_path)
    assert len(events) == 2
    assert events[0]["event_type"] == "run_start"
    assert events[1]["event_type"] == "run_end"


def test_append_adds_to_existing(logger, log_path, run_config):
    # Write 3 times, verify count grows
    logger.log_run_start(run_config)
    assert len(_read_events(log_path)) == 1
    logger.log_run_start(run_config)
    assert len(_read_events(log_path)) == 2
    logger.log_run_start(run_config)
    assert len(_read_events(log_path)) == 3


# ------------------------------------------------------------------
# JSON validity
# ------------------------------------------------------------------

def test_each_line_is_valid_json(logger, log_path, run_config, match_result, run_summary):
    logger.log_run_start(run_config)
    logger.log_match_result(match_result)
    logger.log_no_match("crm_002", 0.45, "trace_002")
    logger.log_run_end(run_summary)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        parsed = json.loads(line)  # raises if invalid
        assert isinstance(parsed, dict), f"line {i} is not a JSON object"


# ------------------------------------------------------------------
# Full run sequence — all 6 event types in correct order
# ------------------------------------------------------------------

def test_full_run_sequence(logger, log_path, run_config, match_result, run_summary):
    logger.log_run_start(run_config)
    logger.log_match_result(match_result)
    logger.log_no_match("crm_002", 0.45, "trace_002")
    logger.log_guardrail("priority_override", True, "review_priority set to 1", {})
    logger.log_validation_error("score_out_of_range", {"value": 1.5})
    logger.log_run_end(run_summary)

    events = _read_events(log_path)
    assert len(events) == 6
    types = [e["event_type"] for e in events]
    assert types == [
        "run_start",
        "match_result",
        "no_match",
        "guardrail_triggered",
        "validation_error",
        "run_end",
    ]


def test_all_events_have_run_id(logger, log_path, run_config, match_result, run_summary, run_id):
    logger.log_run_start(run_config)
    logger.log_match_result(match_result)
    logger.log_no_match("crm_002", 0.45, "trace_002")
    logger.log_guardrail("priority_override", True, "action", {})
    logger.log_validation_error("type", {})
    logger.log_run_end(run_summary)

    events = _read_events(log_path)
    assert all(e["run_id"] == run_id for e in events)


def test_all_events_have_timestamp(logger, log_path, run_config, match_result, run_summary):
    logger.log_run_start(run_config)
    logger.log_match_result(match_result)
    logger.log_no_match("crm_002", 0.45, "trace_002")
    logger.log_guardrail("g", True, "a", {})
    logger.log_validation_error("e", {})
    logger.log_run_end(run_summary)

    events = _read_events(log_path)
    assert all("timestamp" in e and e["timestamp"] for e in events)
