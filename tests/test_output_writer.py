"""
Tests for dal/output_writer.py — OutputWriter (ADR-M3-004)

Covers:
  - _datetime_prefix(): format correctness, timezone handling
  - _to_nested(): entry/match/rerank structure, all fields, NO_MATCH handling
  - write_output_json(): file creation, filename pattern, payload structure
  - write_review_json(): filter (review_priority > 0), sort order, filename pattern,
                         AUTO_MATCH+P1 inclusion, NO_MATCH exclusion
"""

import json
import pytest
from pathlib import Path

from dal.output_writer import OutputWriter
from bll.schemas import MatchCandidate, MatchResult, ScoreVector


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RUN_ID   = "test_run_output_001"
TS_ISO   = "2026-03-30T14:23:45+00:00"
TS_PREFIX = "20260330-1423"


def _make_score_vector(composite: float = 0.85, relation: str = "identical") -> ScoreVector:
    return ScoreVector(
        embedding_cosine_score=0.90,
        jaro_winkler_score=0.88,
        token_sort_ratio=0.85,
        legal_form_score=1.0 if relation == "identical" else 0.0,
        legal_form_relation=relation,
        composite_score=composite,
    )


def _make_candidate(rank: int = 0, composite: float = 0.85,
                    relation: str = "identical", zone: str = "AUTO_MATCH",
                    priority: int = 0) -> MatchCandidate:
    return MatchCandidate(
        source_b_id=f"core_{rank:03d}",
        source_b_name=f"Candidate {rank}",
        source_b_name_normalized=f"candidate {rank}",
        source_b_legal_form="gmbh",
        score=_make_score_vector(composite, relation),
        routing_zone=zone,
        review_priority=priority,
        rank=rank,
    )


def _make_result(
    source_a_id: str = "crm_001",
    routing_zone: str = "AUTO_MATCH",
    review_priority: int = 0,
    legal_form_relation: str = "identical",
    rerank_candidates: list | None = None,
) -> MatchResult:
    """Build a minimal MatchResult with optional rerank candidates."""
    candidates = rerank_candidates if rerank_candidates is not None else [
        _make_candidate(rank=0, composite=0.95, relation=legal_form_relation,
                        zone=routing_zone, priority=review_priority),
        _make_candidate(rank=1, composite=0.75, relation="unknown",
                        zone="REVIEW", priority=2),
    ]
    return MatchResult(
        source_a_id=source_a_id,
        source_a_name="ACME GmbH",
        source_a_name_normalized="acme",
        source_a_legal_form="gmbh",
        source_b_id=candidates[0].source_b_id if candidates else None,
        source_b_name=candidates[0].source_b_name if candidates else None,
        source_b_name_normalized=candidates[0].source_b_name_normalized if candidates else None,
        source_b_legal_form=candidates[0].source_b_legal_form if candidates else None,
        embedding_cosine_score=0.90,
        jaro_winkler_score=0.88,
        token_sort_ratio=0.85,
        legal_form_score=1.0 if legal_form_relation == "identical" else 0.0,
        legal_form_relation=legal_form_relation,
        composite_score=0.95,
        routing_zone=routing_zone,
        review_priority=review_priority,
        rerank_candidates=candidates,
        run_id="test_run_001",
        trace_id="trace_001",
        timestamp=TS_ISO,
    )


def _make_no_match_result(source_a_id: str = "crm_nm_001") -> MatchResult:
    """Build a NO_MATCH MatchResult with empty rerank_candidates."""
    return MatchResult(
        source_a_id=source_a_id,
        source_a_name="Unknown Corp",
        source_a_name_normalized="unknown corp",
        source_a_legal_form=None,
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
        run_id="test_run_001",
        trace_id="trace_nm_001",
        timestamp=TS_ISO,
    )


@pytest.fixture
def writer(tmp_path):
    return OutputWriter(output_dir=str(tmp_path))


@pytest.fixture
def auto_match_result():
    return _make_result(routing_zone="AUTO_MATCH", review_priority=0,
                        legal_form_relation="identical")


@pytest.fixture
def review_result():
    return _make_result(routing_zone="REVIEW", review_priority=3,
                        legal_form_relation="identical")


@pytest.fixture
def auto_match_p1_result():
    """AUTO_MATCH + legal form conflict → review_priority=1 (FR-LF-05)."""
    return _make_result(routing_zone="AUTO_MATCH", review_priority=1,
                        legal_form_relation="conflict")


@pytest.fixture
def no_match_result():
    return _make_no_match_result()


# ---------------------------------------------------------------------------
# _datetime_prefix()
# ---------------------------------------------------------------------------

class TestDatetimePrefix:

    def test_format_utc(self):
        assert OutputWriter._datetime_prefix("2026-03-30T14:23:45+00:00") == "20260330-1423"

    def test_format_z_suffix(self):
        assert OutputWriter._datetime_prefix("2026-03-30T14:23:45Z") == "20260330-1423"

    def test_midnight(self):
        assert OutputWriter._datetime_prefix("2026-01-01T00:00:00+00:00") == "20260101-0000"

    def test_end_of_day(self):
        assert OutputWriter._datetime_prefix("2026-12-31T23:59:00+00:00") == "20261231-2359"

    def test_returns_string(self):
        assert isinstance(OutputWriter._datetime_prefix(TS_ISO), str)

    def test_length_is_13(self):
        # "YYYYMMDD-HHmm" = 4+2+2+1+2+2 = 13 chars
        assert len(OutputWriter._datetime_prefix(TS_ISO)) == 13


# ---------------------------------------------------------------------------
# _to_nested()
# ---------------------------------------------------------------------------

class TestToNested:

    def test_top_level_keys(self, writer, auto_match_result):
        nested = writer._to_nested(auto_match_result)
        assert set(nested.keys()) == {"entry", "match", "rerank", "trace_id", "timestamp"}

    def test_entry_section_fields(self, writer, auto_match_result):
        entry = writer._to_nested(auto_match_result)["entry"]
        assert entry["source_a_id"] == "crm_001"
        assert entry["source_a_name"] == "ACME GmbH"
        assert entry["source_a_name_normalized"] == "acme"
        assert entry["source_a_legal_form"] == "gmbh"

    def test_match_section_identity_fields(self, writer, auto_match_result):
        match = writer._to_nested(auto_match_result)["match"]
        assert match["source_b_id"] is not None
        assert match["source_b_name"] is not None
        assert match["source_b_name_normalized"] is not None

    def test_match_section_score_sub_object(self, writer, auto_match_result):
        score = writer._to_nested(auto_match_result)["match"]["score"]
        assert "embedding_cosine_score" in score
        assert "jaro_winkler_score" in score
        assert "token_sort_ratio" in score
        assert "legal_form_score" in score
        assert "legal_form_relation" in score
        assert "composite_score" in score

    def test_match_section_routing_fields(self, writer, auto_match_result):
        match = writer._to_nested(auto_match_result)["match"]
        assert match["routing_zone"] == "AUTO_MATCH"
        assert match["review_priority"] == 0

    def test_match_section_rank_is_zero(self, writer, auto_match_result):
        """rank in match section is always 0 — pipeline invariant."""
        assert writer._to_nested(auto_match_result)["match"]["rank"] == 0

    def test_rerank_is_list(self, writer, auto_match_result):
        rerank = writer._to_nested(auto_match_result)["rerank"]
        assert isinstance(rerank, list)

    def test_rerank_length_matches_candidates(self, writer, auto_match_result):
        rerank = writer._to_nested(auto_match_result)["rerank"]
        assert len(rerank) == len(auto_match_result.rerank_candidates)

    def test_rerank_entry_has_score_sub_object(self, writer, auto_match_result):
        """Each rerank candidate has a nested 'score' sub-object (MatchCandidate.model_dump())."""
        rerank = writer._to_nested(auto_match_result)["rerank"]
        assert len(rerank) > 0
        assert "score" in rerank[0]
        assert "composite_score" in rerank[0]["score"]

    def test_rerank_entry_has_rank_field(self, writer, auto_match_result):
        rerank = writer._to_nested(auto_match_result)["rerank"]
        assert "rank" in rerank[0]

    def test_rerank_entry_has_routing_zone(self, writer, auto_match_result):
        rerank = writer._to_nested(auto_match_result)["rerank"]
        assert "routing_zone" in rerank[0]

    def test_trace_id_and_timestamp_present(self, writer, auto_match_result):
        nested = writer._to_nested(auto_match_result)
        assert nested["trace_id"] == "trace_001"
        assert nested["timestamp"] == TS_ISO

    def test_no_match_source_b_fields_are_none(self, writer, no_match_result):
        match = writer._to_nested(no_match_result)["match"]
        assert match["source_b_id"] is None
        assert match["source_b_name"] is None

    def test_no_match_rerank_is_empty_list(self, writer, no_match_result):
        assert writer._to_nested(no_match_result)["rerank"] == []

    def test_no_match_scores_are_zero(self, writer, no_match_result):
        score = writer._to_nested(no_match_result)["match"]["score"]
        assert score["composite_score"] == 0.0
        assert score["embedding_cosine_score"] == 0.0


# ---------------------------------------------------------------------------
# write_output_json()
# ---------------------------------------------------------------------------

class TestWriteOutputJson:

    def test_creates_file(self, writer, auto_match_result, tmp_path):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        assert Path(path).exists()

    def test_filename_pattern(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        filename = Path(path).name
        assert filename == f"output_{TS_PREFIX}_{RUN_ID}.json"

    def test_returns_absolute_path(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        assert Path(path).is_absolute()

    def test_top_level_keys(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert set(payload.keys()) == {"run_id", "generated_at", "total_entries", "results"}

    def test_run_id_in_payload(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["run_id"] == RUN_ID

    def test_total_entries_count(self, writer, auto_match_result, no_match_result):
        results = [auto_match_result, no_match_result]
        path = writer.write_output_json(results, RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_entries"] == 2

    def test_results_use_nested_structure(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = payload["results"][0]
        assert "entry" in result
        assert "match" in result
        assert "rerank" in result

    def test_results_not_flat_model_dump(self, writer, auto_match_result):
        """Results must NOT be flat model_dump — flat keys like 'source_a_id' at top level are wrong."""
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = payload["results"][0]
        # Top-level flat keys from model_dump must not be present
        assert "source_a_id" not in result
        assert "embedding_cosine_score" not in result
        assert "rerank_candidates" not in result

    def test_no_match_included_in_output(self, writer, auto_match_result, no_match_result):
        """ALL entries including NO_MATCH appear in output JSON."""
        path = writer.write_output_json([auto_match_result, no_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = [r["match"]["routing_zone"] for r in payload["results"]]
        assert "NO_MATCH" in zones

    def test_empty_results_list(self, writer, tmp_path):
        path = writer.write_output_json([], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_entries"] == 0
        assert payload["results"] == []

    def test_file_is_valid_json(self, writer, auto_match_result):
        path = writer.write_output_json([auto_match_result], RUN_ID, TS_ISO)
        content = Path(path).read_text(encoding="utf-8")
        parsed = json.loads(content)   # raises if invalid
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# write_review_json()
# ---------------------------------------------------------------------------

class TestWriteReviewJson:

    def test_creates_file(self, writer, review_result):
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        assert Path(path).exists()

    def test_filename_pattern(self, writer, review_result):
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        filename = Path(path).name
        assert filename == f"review_{TS_PREFIX}_{RUN_ID}.json"

    def test_top_level_keys(self, writer, review_result):
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "run_id" in payload
        assert "total_review_entries" in payload
        assert "sorted_by" in payload
        assert "entries" in payload

    def test_excludes_priority_zero(self, writer, auto_match_result):
        """AUTO_MATCH with review_priority=0 must NOT appear in review file."""
        path = writer.write_review_json([auto_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == 0
        assert payload["entries"] == []

    def test_includes_review_zone_entries(self, writer, review_result):
        """REVIEW zone entries with priority > 0 appear in review file."""
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == 1

    def test_includes_auto_match_p1_conflict(self, writer, auto_match_p1_result):
        """AUTO_MATCH + legal form conflict (P1, FR-LF-05) must appear in review file."""
        path = writer.write_review_json([auto_match_p1_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == 1
        entry = payload["entries"][0]
        assert entry["match"]["routing_zone"] == "AUTO_MATCH"
        assert entry["match"]["review_priority"] == 1

    def test_excludes_no_match(self, writer, no_match_result):
        """NO_MATCH entries never appear in the review file."""
        path = writer.write_review_json([no_match_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == 0

    def test_sorted_ascending_by_priority(self, writer):
        """Entries sorted ascending: P1 (mandatory) first, P3 (low-urgency) last."""
        p1 = _make_result(source_a_id="a1", routing_zone="REVIEW", review_priority=1,
                          legal_form_relation="conflict")
        p2 = _make_result(source_a_id="a2", routing_zone="REVIEW", review_priority=2,
                          legal_form_relation="related")
        p3 = _make_result(source_a_id="a3", routing_zone="REVIEW", review_priority=3,
                          legal_form_relation="identical")
        # Feed in reverse order — must come out sorted
        path = writer.write_review_json([p3, p1, p2], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        priorities = [e["match"]["review_priority"] for e in payload["entries"]]
        assert priorities == sorted(priorities)
        assert priorities[0] == 1  # P1 first

    def test_entries_use_nested_structure(self, writer, review_result):
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = payload["entries"][0]
        assert "entry" in entry
        assert "match" in entry
        assert "rerank" in entry

    def test_total_count_matches_filtered_entries(self, writer, auto_match_result,
                                                   review_result, no_match_result,
                                                   auto_match_p1_result):
        """total_review_entries must equal len(entries) in the payload."""
        results = [auto_match_result, review_result, no_match_result, auto_match_p1_result]
        path = writer.write_review_json(results, RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == len(payload["entries"])

    def test_empty_input_produces_empty_review(self, writer):
        path = writer.write_review_json([], RUN_ID, TS_ISO)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["total_review_entries"] == 0
        assert payload["entries"] == []

    def test_file_is_valid_json(self, writer, review_result):
        path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        content = Path(path).read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_output_and_review_have_different_filenames(self, writer, review_result):
        """Output and review files for the same run must have distinct names."""
        out_path    = writer.write_output_json([review_result], RUN_ID, TS_ISO)
        review_path = writer.write_review_json([review_result], RUN_ID, TS_ISO)
        assert Path(out_path).name != Path(review_path).name
        assert "output_" in Path(out_path).name
        assert "review_" in Path(review_path).name
