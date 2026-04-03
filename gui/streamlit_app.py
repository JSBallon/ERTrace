"""
GUI — Streamlit Pipeline Controller

Full pipeline controller for PoC2 Entity Resolution.

Layering rules (ADR-M3-006):
  - No dal.* imports — file I/O via tempfile only; all normalisation happens inside
    run_entity_resolution()
  - No BLL scoring logic — only schema types (RunConfig, RunSummary, MatchResult)
  - import streamlit only in this file — never in bll/, dal/, or governance/
  - All pipeline interaction via run_entity_resolution() or save_ui_config()

Flow:
  1. _load_defaults()        → RunConfig from YAML (slider default values, cached)
  2. Sidebar widgets         → user may adjust values
  3. File upload / Faker     → temp CSV paths stored in session_state
  4. ▶ Start Run             → save_ui_config() → (RunConfig, was_adjusted, changed_fields)
                             → run_entity_resolution(run_config=..., config_adjustments=...)
                             → results + summary stored in session_state
  5. Results display         → metrics, REVIEW rate warning, review table, full table, downloads

Config versioning (ADR-M3-006):
  If slider values differ from YAML defaults, save_ui_config() writes a new
  config/versions/v_ui_<YYYYMMDD-HHmm>.yaml and updates config/config.yaml.
  The config_adjustments diff is embedded in the run_start JSONL event.
"""

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from bll.app_service import run_entity_resolution
from bll.schemas import MatchResult, RunSummary
from config.config_loader import load_run_config, save_ui_config
from dal.data_generator import FakerDataGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Vsevolod/company-names-similarity-sentence-transformer"
ALT_MODELS = [
    "Vsevolod/company-names-similarity-sentence-transformer",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "deutsche-telekom/gbert-large-paraphrase-cosine",
    "all-MiniLM-L6-v2",
]
WEIGHT_TOLERANCE = 0.001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_defaults():
    """Load YAML defaults once per session (cached). Used to pre-fill sliders."""
    try:
        return load_run_config("config/config.yaml")
    except Exception:
        # Fallback to hardcoded defaults if YAML missing (e.g. fresh checkout)
        from bll.schemas import (
            LegalFormConfig, ThresholdConfig, WeightsConfig, RunConfig,
        )
        import uuid
        from datetime import datetime, timezone
        return RunConfig(
            run_id=str(uuid.uuid4()),
            embedding_model=DEFAULT_MODEL,
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _write_temp_csv(records: list[dict], prefix: str) -> str:
    """Write a list of {source_id, source_name} dicts to a temp CSV. Returns path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=prefix, delete=False, encoding="utf-8", newline=""
    )
    writer = csv.DictWriter(tmp, fieldnames=["source_id", "source_name"])
    writer.writeheader()
    writer.writerows(records)
    tmp.close()
    return tmp.name


def _cleanup_temp(path: str | None) -> None:
    """Silently remove a temp file if it exists."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _save_upload_to_temp(uploaded_file, prefix: str) -> str:
    """Save a Streamlit UploadedFile to a named temp file. Returns path."""
    suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".json"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=prefix, delete=False
    )
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _build_review_df(results: list[MatchResult]) -> pd.DataFrame:
    """Build a DataFrame of review-priority entries, sorted P1 first."""
    rows = []
    for r in results:
        if r.review_priority > 0:
            rows.append({
                "Priority":          r.review_priority,
                "Zone":              r.routing_zone,
                "Source A ID":       r.source_a_id,
                "Source A Name":     r.source_a_name,
                "Source B ID":       r.source_b_id or "—",
                "Source B Name":     r.source_b_name or "—",
                "Legal Form Rel.":   r.legal_form_relation,
                "Composite Score":   round(r.composite_score, 4),
                "LF Score":          round(r.legal_form_score, 4),
                "Embed Score":       round(r.embedding_cosine_score, 4),
                "Rerank Count":      len(r.rerank_candidates),
                "Trace ID":          r.trace_id,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Priority").reset_index(drop=True)
    return df


def _build_results_df(
    results: list[MatchResult],
    zone_filter: list[str],
) -> pd.DataFrame:
    """Build a DataFrame of all results, filtered by routing zone."""
    rows = []
    for r in results:
        if r.routing_zone not in zone_filter:
            continue
        rows.append({
            "Zone":              r.routing_zone,
            "Priority":          r.review_priority,
            "Source A ID":       r.source_a_id,
            "Source A Name":     r.source_a_name,
            "Source B ID":       r.source_b_id or "—",
            "Source B Name":     r.source_b_name or "—",
            "Composite Score":   round(r.composite_score, 4),
            "Legal Form Rel.":   r.legal_form_relation,
            "LF Score":          round(r.legal_form_score, 4),
            "Embed Score":       round(r.embedding_cosine_score, 4),
            "JW Score":          round(r.jaro_winkler_score, 4),
            "TS Ratio":          round(r.token_sort_ratio, 4),
            "Rerank Count":      len(r.rerank_candidates),
            "Trace ID":          r.trace_id,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PoC2 — Entity Resolution",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session state initialisation
    for key, default in [
        ("results", None),
        ("summary", None),
        ("temp_a_path", None),
        ("temp_b_path", None),
        ("faker_generated", False),
        ("last_run_adjusted", False),
        ("last_run_version", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    defaults = _load_defaults()

    # -----------------------------------------------------------------------
    # Sidebar — Configuration
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        # Embedding model
        model_idx = ALT_MODELS.index(defaults.embedding_model) if defaults.embedding_model in ALT_MODELS else 0
        embedding_model = st.selectbox(
            "Embedding Model",
            options=ALT_MODELS,
            index=model_idx,
            help="Sentence Transformer model — must be cached locally.",
        )

        faiss_top_k = st.slider(
            "FAISS Top-K", min_value=1, max_value=20,
            value=defaults.faiss_top_k,
            help="Number of nearest-neighbour candidates retrieved per Source A entry.",
        )

        st.subheader("Thresholds")
        auto_match_threshold = st.slider(
            "Auto-Match Threshold", min_value=0.50, max_value=1.00,
            value=defaults.threshold_config.auto_match_threshold,
            step=0.01,
            help="Composite score >= this → AUTO_MATCH.",
        )
        review_lower_threshold = st.slider(
            "Review Lower Threshold", min_value=0.00, max_value=0.99,
            value=defaults.threshold_config.review_lower_threshold,
            step=0.01,
            help="Composite score >= this (and < auto-match) → REVIEW.",
        )
        threshold_valid = review_lower_threshold < auto_match_threshold
        if not threshold_valid:
            st.error(
                f"Review threshold ({review_lower_threshold:.2f}) must be "
                f"< Auto-match threshold ({auto_match_threshold:.2f})."
            )

        st.subheader("Composite Weights")
        w_embedding    = st.slider("w_embedding",    0.00, 1.00, defaults.weights_config.w_embedding,    0.05)
        w_jaro_winkler = st.slider("w_jaro_winkler", 0.00, 1.00, defaults.weights_config.w_jaro_winkler, 0.05)
        w_token_sort   = st.slider("w_token_sort",   0.00, 1.00, defaults.weights_config.w_token_sort,   0.05)
        w_legal_form   = st.slider("w_legal_form",   0.00, 1.00, defaults.weights_config.w_legal_form,   0.05)

        total_w = w_embedding + w_jaro_winkler + w_token_sort + w_legal_form
        weight_ok = abs(total_w - 1.0) <= WEIGHT_TOLERANCE
        st.metric("Weight Sum", f"{total_w:.3f}", delta=None)
        if not weight_ok:
            st.warning(f"Weights sum to {total_w:.3f} — must equal 1.000")

        st.subheader("Monitoring")
        review_warn_threshold = st.slider(
            "REVIEW Rate Warning", 0.00, 1.00, 0.30, 0.05,
            help="Display warning when REVIEW rate exceeds this fraction.",
        )

    # -----------------------------------------------------------------------
    # Main area
    # -----------------------------------------------------------------------
    st.title("PoC2 — Entity Resolution Pipeline")
    st.caption("TGFR: Transformer-Gather, Fuzzy-Reconsider · Locally executable · Fully auditable")

    # --- Run Metadata Panel ---
    with st.expander("Run Parameters", expanded=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("**Model & FAISS**")
            st.caption(f"Model: `{embedding_model}`")
            st.caption(f"Top-K: `{faiss_top_k}`")
        with col_m2:
            st.markdown("**Thresholds**")
            st.caption(f"Auto-match: `{auto_match_threshold:.2f}`")
            st.caption(f"Review lower: `{review_lower_threshold:.2f}`")
        with col_m3:
            st.markdown("**Weights**")
            st.caption(
                f"Emb: `{w_embedding:.2f}` · JW: `{w_jaro_winkler:.2f}` · "
                f"TS: `{w_token_sort:.2f}` · LF: `{w_legal_form:.2f}`"
            )
        if st.session_state.last_run_version:
            st.info(
                f"Last run config version: `{st.session_state.last_run_version}` "
                + ("(UI-adjusted)" if st.session_state.last_run_adjusted else "(YAML default)")
            )

    st.divider()

    # --- Input section ---
    st.subheader("Input Files")
    col_a, col_b = st.columns(2)
    with col_a:
        source_a_file = st.file_uploader(
            "Source A (CRM)", type=["csv", "json"],
            help="CSV or JSON with columns: source_id, source_name",
        )
    with col_b:
        source_b_file = st.file_uploader(
            "Source B (Core Banking)", type=["csv", "json"],
            help="CSV or JSON with columns: source_id, source_name",
        )

    col_n, col_fbtn, _ = st.columns([1, 2, 4])
    with col_n:
        n_faker = st.number_input(
            "Faker N (each)", min_value=10, max_value=500, value=50, step=10,
        )
    with col_fbtn:
        st.markdown(" ")   # vertical alignment spacer
        faker_button = st.button("Generate Faker Data", use_container_width=True)

    if faker_button:
        _cleanup_temp(st.session_state.temp_a_path)
        _cleanup_temp(st.session_state.temp_b_path)

        gen = FakerDataGenerator(seed=42)
        records_a = gen.generate_company_list(n_faker, "de")
        records_b = gen.generate_company_list(n_faker, "de")

        st.session_state.temp_a_path = _write_temp_csv(records_a, prefix="faker_a_")
        st.session_state.temp_b_path = _write_temp_csv(records_b, prefix="faker_b_")
        st.session_state.faker_generated = True
        st.success(
            f"Generated {n_faker} Source A + {n_faker} Source B Faker entries. "
            "Ready to run."
        )

    # Resolve active file paths
    active_a_path: str | None = None
    active_b_path: str | None = None

    if source_a_file is not None:
        _cleanup_temp(st.session_state.temp_a_path)
        st.session_state.temp_a_path = _save_upload_to_temp(source_a_file, prefix="upload_a_")
        st.session_state.faker_generated = False
    if source_b_file is not None:
        _cleanup_temp(st.session_state.temp_b_path)
        st.session_state.temp_b_path = _save_upload_to_temp(source_b_file, prefix="upload_b_")
        st.session_state.faker_generated = False

    active_a_path = st.session_state.temp_a_path
    active_b_path = st.session_state.temp_b_path

    if active_a_path:
        st.caption(
            f"Source A: `{Path(active_a_path).name}`"
            + (" (Faker)" if st.session_state.faker_generated and source_a_file is None else " (uploaded)")
        )
    if active_b_path:
        st.caption(
            f"Source B: `{Path(active_b_path).name}`"
            + (" (Faker)" if st.session_state.faker_generated and source_b_file is None else " (uploaded)")
        )

    st.divider()

    # --- Pipeline Control ---
    files_ready = active_a_path is not None and active_b_path is not None
    run_disabled = not weight_ok or not threshold_valid or not files_ready

    if not files_ready:
        st.info("Upload Source A and Source B files, or use Generate Faker Data to continue.")

    progress_bar = st.progress(0)
    status_text  = st.empty()

    run_button = st.button(
        "▶ Start Run",
        disabled=run_disabled,
        type="primary",
        use_container_width=False,
    )

    if run_button and active_a_path and active_b_path:
        progress_bar.progress(0)
        status_text.text("Resolving configuration...")

        try:
            run_config, was_adjusted, changed_fields = save_ui_config(
                embedding_model=embedding_model,
                faiss_top_k=faiss_top_k,
                auto_match_threshold=auto_match_threshold,
                review_lower_threshold=review_lower_threshold,
                w_embedding=w_embedding,
                w_jaro_winkler=w_jaro_winkler,
                w_token_sort=w_token_sort,
                w_legal_form=w_legal_form,
            )

            if was_adjusted:
                st.info(
                    f"Configuration adjusted — new version saved: "
                    f"`{run_config.threshold_config_version}`. "
                    f"Changed: {list(changed_fields.keys())}"
                )

            def _progress(completed: int, total: int) -> None:
                pct = int(completed / total * 100)
                progress_bar.progress(pct)
                status_text.text(f"Processing entry {completed} / {total}...")

            status_text.text("Running pipeline...")

            results, summary = run_entity_resolution(
                source_a_path=active_a_path,
                source_b_path=active_b_path,
                run_config=run_config,
                config_adjustments=changed_fields if was_adjusted else None,
                progress_callback=_progress,
            )

            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.last_run_adjusted = was_adjusted
            st.session_state.last_run_version  = run_config.threshold_config_version

            progress_bar.progress(100)
            status_text.text("Run complete.")

        except Exception as exc:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"Pipeline error: {exc}")

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------
    summary: RunSummary | None = st.session_state.summary
    results: list[MatchResult] | None = st.session_state.results

    if summary is not None and results is not None:
        st.divider()
        st.subheader("Run Summary")

        col_am, col_rv, col_nm, col_err = st.columns(4)
        col_am.metric(
            "AUTO_MATCH",
            summary.count_auto_match,
            f"{summary.auto_match_quote:.1%}",
        )
        col_rv.metric(
            "REVIEW",
            summary.count_review,
            f"{summary.review_quote:.1%}",
        )
        col_nm.metric(
            "NO_MATCH",
            summary.count_no_match,
            f"{summary.no_match_quote:.1%}",
        )
        col_err.metric("ERRORS", summary.count_error)

        # REVIEW rate warning (FR-THR-04)
        if summary.review_quote > review_warn_threshold:
            st.warning(
                f"REVIEW rate {summary.review_quote:.1%} exceeds warning threshold "
                f"({review_warn_threshold:.1%}). Consider recalibrating thresholds."
            )

        st.caption(
            f"Run ID: `{summary.run_id}` · "
            f"Total rerank candidates: {summary.total_rerank_candidates}"
        )

        # --- Prioritized review list ---
        st.subheader("Prioritized Review List")
        st.caption("Entries with review_priority > 0, sorted P1 (mandatory) → P3 (low-urgency).")

        review_df = _build_review_df(results)
        if review_df.empty:
            st.info("No review entries in this run.")
        else:
            # Colour-hint: P1 rows are most critical
            st.dataframe(review_df, use_container_width=True, hide_index=True)

        # --- Full results table ---
        st.subheader("Full Results")
        zone_filter = st.multiselect(
            "Filter by Routing Zone",
            options=["AUTO_MATCH", "REVIEW", "NO_MATCH"],
            default=["AUTO_MATCH", "REVIEW", "NO_MATCH"],
        )
        full_df = _build_results_df(results, zone_filter)
        if full_df.empty:
            st.info("No results match the selected filter.")
        else:
            st.dataframe(full_df, use_container_width=True, hide_index=True)

        # --- Download buttons ---
        st.subheader("Download Outputs")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        output_path = Path(summary.output_file_path)
        review_path = Path(summary.review_file_path)
        audit_path  = Path(summary.audit_log_path)

        with col_dl1:
            if output_path.exists():
                st.download_button(
                    "⬇ Output JSON",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.caption("Output JSON not found.")

        with col_dl2:
            if review_path.exists():
                st.download_button(
                    "⬇ Review JSON (prioritized)",
                    data=review_path.read_bytes(),
                    file_name=review_path.name,
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.caption("Review JSON not found.")

        with col_dl3:
            if audit_path.exists():
                st.download_button(
                    "⬇ Audit JSONL",
                    data=audit_path.read_bytes(),
                    file_name=audit_path.name,
                    mime="application/x-ndjson",
                    use_container_width=True,
                )
            else:
                st.caption("Audit JSONL not found.")


if __name__ == "__main__":
    main()
