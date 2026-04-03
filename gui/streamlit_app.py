"""
GUI — Streamlit Pipeline Controller

Full pipeline controller for PoC2 Entity Resolution.

Layering rules (ADR-M3-006):
  - No dal.* imports — file I/O via tempfile only
  - No BLL scoring logic — only schema types (RunConfig, RunSummary, MatchResult)
  - import streamlit only in this file
  - All pipeline interaction via run_entity_resolution()

Config versioning (ADR-M3-006 — simplified):
  No YAML files are written for UI-adjusted runs. config.yaml is never modified
  by the Streamlit app. Instead, version strings signal the origin:

    "streamlit_custom"  — parameter group was changed via sliders
    "<yaml_version>"    — parameter group matches the active YAML exactly

  The config_adjustments diff is embedded in the run_start JSONL event so that
  exact before/after values are auditable without reading any file.

  Detective control: any downstream consumer can detect non-standard runs with
  a simple grep for "streamlit_custom" in the audit JSONL.
"""

import csv
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path("inputs")

import pandas as pd
import streamlit as st

from bll.app_service import run_entity_resolution
from bll.schemas import (
    LegalFormConfig,
    MatchResult,
    RunConfig,
    RunSummary,
    ThresholdConfig,
    WeightsConfig,
)
from config.config_loader import load_run_config
from dal.data_generator import FakerDataGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALT_MODELS = [
    "Vsevolod/company-names-similarity-sentence-transformer",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "deutsche-telekom/gbert-large-paraphrase-cosine",
    "all-MiniLM-L6-v2",
]
WEIGHT_TOLERANCE = 0.001
VERSION_CUSTOM   = "streamlit_custom"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_defaults() -> RunConfig:
    """Load YAML defaults once per session. Used to pre-fill sliders."""
    try:
        return load_run_config("config/config.yaml")
    except Exception:
        return RunConfig(
            run_id=str(uuid.uuid4()),
            embedding_model=ALT_MODELS[0],
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


def _build_run_config(
    defaults: RunConfig,
    embedding_model: str,
    faiss_top_k: int,
    auto_match_threshold: float,
    review_lower_threshold: float,
    w_embedding: float,
    w_jaro_winkler: float,
    w_token_sort: float,
    w_legal_form: float,
) -> tuple[RunConfig, dict]:
    """
    Build a RunConfig from UI slider values in memory — no file writes.

    Version string logic (ADR-M3-006):
      - If threshold values differ from YAML: threshold_config_version = "streamlit_custom"
      - If weight values differ from YAML:    weights_config_version   = "streamlit_custom"
      - If model/faiss differ from YAML:      both version strings     = "streamlit_custom"
      - Otherwise: retain the original YAML version strings unchanged

    Returns (RunConfig, changed_fields).
    changed_fields: {field: {"from": old, "to": new}} — empty if nothing changed.
    """
    d = defaults

    def _fc(a: float, b: float) -> bool:
        return abs(a - b) > 1e-6

    threshold_changed = (
        _fc(d.threshold_config.auto_match_threshold,   auto_match_threshold) or
        _fc(d.threshold_config.review_lower_threshold, review_lower_threshold)
    )
    weights_changed = (
        _fc(d.weights_config.w_embedding,    w_embedding)    or
        _fc(d.weights_config.w_jaro_winkler, w_jaro_winkler) or
        _fc(d.weights_config.w_token_sort,   w_token_sort)   or
        _fc(d.weights_config.w_legal_form,   w_legal_form)
    )
    model_changed = (
        embedding_model != d.embedding_model or faiss_top_k != d.faiss_top_k
    )

    # Model change marks all parameter groups as custom
    thr_version = VERSION_CUSTOM if (threshold_changed or model_changed) else d.threshold_config_version
    wts_version = VERSION_CUSTOM if (weights_changed   or model_changed) else d.weights_config_version
    lf_version  = d.legal_form_config_version  # not exposed in UI — always from YAML

    # Build diff
    changed_fields: dict = {}
    for field, old_val, new_val in [
        ("embedding_model",        d.embedding_model,                              embedding_model),
        ("faiss_top_k",            d.faiss_top_k,                                  faiss_top_k),
        ("auto_match_threshold",   d.threshold_config.auto_match_threshold,        auto_match_threshold),
        ("review_lower_threshold", d.threshold_config.review_lower_threshold,      review_lower_threshold),
        ("w_embedding",            d.weights_config.w_embedding,                   w_embedding),
        ("w_jaro_winkler",         d.weights_config.w_jaro_winkler,                w_jaro_winkler),
        ("w_token_sort",           d.weights_config.w_token_sort,                  w_token_sort),
        ("w_legal_form",           d.weights_config.w_legal_form,                  w_legal_form),
    ]:
        if isinstance(old_val, float):
            if _fc(old_val, new_val):
                changed_fields[field] = {"from": old_val, "to": new_val}
        elif old_val != new_val:
            changed_fields[field] = {"from": old_val, "to": new_val}

    run_config = RunConfig(
        run_id=str(uuid.uuid4()),
        embedding_model=embedding_model,
        faiss_top_k=faiss_top_k,
        threshold_config=ThresholdConfig(
            auto_match_threshold=auto_match_threshold,
            review_lower_threshold=review_lower_threshold,
        ),
        weights_config=WeightsConfig(
            w_embedding=w_embedding,
            w_jaro_winkler=w_jaro_winkler,
            w_token_sort=w_token_sort,
            w_legal_form=w_legal_form,
        ),
        legal_form_config=LegalFormConfig(
            identical_score=d.legal_form_config.identical_score,
            related_score=d.legal_form_config.related_score,
            conflict_score=d.legal_form_config.conflict_score,
            unknown_score=d.legal_form_config.unknown_score,
        ),
        threshold_config_version=thr_version,
        weights_config_version=wts_version,
        legal_form_config_version=lf_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return run_config, changed_fields


def _write_input_csv(records: list[dict], source: str) -> str:
    """
    Write Faker-generated records to inputs/<source>_<YYYYMMDD-HHmm>.csv.

    Files are kept permanently — inputs/ mirrors outputs/ and logs/audit/ as a
    first-class artifact folder. The path is recorded in the run_start audit event.

    Args:
        records: List of {source_id, source_name} dicts.
        source:  "source_a" or "source_b" — used as the filename prefix.

    Returns:
        Absolute path of the written file as a string.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    path = INPUT_DIR / f"{source}_{ts}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "source_name"])
        writer.writeheader()
        writer.writerows(records)
    return str(path)


def _save_upload_to_temp(uploaded_file, prefix: str) -> str:
    """Save a Streamlit UploadedFile to a named temp file. Returns path."""
    suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".json"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _cleanup_temp(path: str | None) -> None:
    """Silently remove a temp file if it exists."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _build_review_df(results: list[MatchResult]) -> pd.DataFrame:
    """DataFrame of review-priority entries, sorted P1 first."""
    rows = []
    for r in results:
        if r.review_priority > 0:
            rows.append({
                "Priority":        r.review_priority,
                "Zone":            r.routing_zone,
                "Source A ID":     r.source_a_id,
                "Source A Name":   r.source_a_name,
                "Source B ID":     r.source_b_id or "—",
                "Source B Name":   r.source_b_name or "—",
                "LF Relation":     r.legal_form_relation,
                "Composite Score": round(r.composite_score, 4),
                "LF Score":        round(r.legal_form_score, 4),
                "Embed Score":     round(r.embedding_cosine_score, 4),
                "Rerank Count":    len(r.rerank_candidates),
                "Trace ID":        r.trace_id,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Priority").reset_index(drop=True)
    return df


def _build_results_df(results: list[MatchResult], zone_filter: list[str]) -> pd.DataFrame:
    """DataFrame of all results filtered by routing zone."""
    rows = []
    for r in results:
        if r.routing_zone not in zone_filter:
            continue
        rows.append({
            "Zone":            r.routing_zone,
            "Priority":        r.review_priority,
            "Source A ID":     r.source_a_id,
            "Source A Name":   r.source_a_name,
            "Source B ID":     r.source_b_id or "—",
            "Source B Name":   r.source_b_name or "—",
            "Composite Score": round(r.composite_score, 4),
            "LF Relation":     r.legal_form_relation,
            "LF Score":        round(r.legal_form_score, 4),
            "Embed Score":     round(r.embedding_cosine_score, 4),
            "JW Score":        round(r.jaro_winkler_score, 4),
            "TS Ratio":        round(r.token_sort_ratio, 4),
            "Rerank Count":    len(r.rerank_candidates),
            "Trace ID":        r.trace_id,
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

    for key, default in [
        ("results",              None),
        ("summary",              None),
        ("temp_a_path",          None),
        ("temp_b_path",          None),
        ("faker_a_generated",    False),
        ("faker_b_generated",    False),
        ("last_run_config",      None),
        ("last_changed_fields",  {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    defaults = _load_defaults()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        model_idx = ALT_MODELS.index(defaults.embedding_model) if defaults.embedding_model in ALT_MODELS else 0
        embedding_model = st.selectbox(
            "Embedding Model", options=ALT_MODELS, index=model_idx,
            help="Sentence Transformer model — must be cached locally.",
        )
        faiss_top_k = st.slider(
            "FAISS Top-K", min_value=1, max_value=20, value=defaults.faiss_top_k,
        )

        st.subheader("Thresholds")
        auto_match_threshold = st.slider(
            "Auto-Match Threshold", 0.50, 1.00,
            value=defaults.threshold_config.auto_match_threshold, step=0.01,
        )
        review_lower_threshold = st.slider(
            "Review Lower Threshold", 0.00, 0.99,
            value=defaults.threshold_config.review_lower_threshold, step=0.01,
        )
        threshold_valid = review_lower_threshold < auto_match_threshold
        if not threshold_valid:
            st.error(
                f"Review lower ({review_lower_threshold:.2f}) must be "
                f"< Auto-match ({auto_match_threshold:.2f})."
            )

        st.subheader("Composite Weights")
        w_embedding    = st.slider("w_embedding",    0.00, 1.00, defaults.weights_config.w_embedding,    0.05)
        w_jaro_winkler = st.slider("w_jaro_winkler", 0.00, 1.00, defaults.weights_config.w_jaro_winkler, 0.05)
        w_token_sort   = st.slider("w_token_sort",   0.00, 1.00, defaults.weights_config.w_token_sort,   0.05)
        w_legal_form   = st.slider("w_legal_form",   0.00, 1.00, defaults.weights_config.w_legal_form,   0.05)

        total_w   = w_embedding + w_jaro_winkler + w_token_sort + w_legal_form
        weight_ok = abs(total_w - 1.0) <= WEIGHT_TOLERANCE
        st.metric("Weight Sum", f"{total_w:.3f}")
        if not weight_ok:
            st.warning(f"Weights sum to {total_w:.3f} — must equal 1.000")

        st.subheader("Monitoring")
        review_warn_threshold = st.slider("REVIEW Rate Warning", 0.00, 1.00, 0.30, 0.05)

    # -----------------------------------------------------------------------
    # Main area
    # -----------------------------------------------------------------------
    st.title("PoC2 — Entity Resolution Pipeline")
    st.caption("TGFR: Transformer-Gather, Fuzzy-Reconsider · Locally executable · Fully auditable")

    with st.expander("Run Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Model & FAISS**")
            st.caption(f"Model: `{embedding_model}`")
            st.caption(f"Top-K: `{faiss_top_k}`")
        with c2:
            st.markdown("**Thresholds**")
            st.caption(f"Auto-match: `{auto_match_threshold:.2f}`")
            st.caption(f"Review lower: `{review_lower_threshold:.2f}`")
        with c3:
            st.markdown("**Weights**")
            st.caption(
                f"Emb: `{w_embedding:.2f}` · JW: `{w_jaro_winkler:.2f}` · "
                f"TS: `{w_token_sort:.2f}` · LF: `{w_legal_form:.2f}`"
            )
        if st.session_state.last_run_config is not None:
            rc: RunConfig = st.session_state.last_run_config
            is_custom = VERSION_CUSTOM in (rc.threshold_config_version, rc.weights_config_version)
            st.info(
                f"Last run · thr: `{rc.threshold_config_version}` · "
                f"wts: `{rc.weights_config_version}`"
                + (" · **streamlit_custom settings active**" if is_custom else "")
            )
            if st.session_state.last_changed_fields:
                with st.expander("Changed vs. YAML defaults"):
                    for f, diff in st.session_state.last_changed_fields.items():
                        st.caption(f"`{f}`: {diff['from']} → {diff['to']}")

    st.divider()

    # --- Input ---
    st.subheader("Input Files")
    col_a, col_b = st.columns(2)

    with col_a:
        source_a_file = st.file_uploader("Source A", type=["csv", "json"])
        n_faker_a = st.number_input(
            "Faker entries", min_value=10, max_value=500, value=50, step=10,
            key="n_faker_a",
        )
        faker_a_button = st.button("Generate Faker A", key="btn_faker_a", use_container_width=True)

        if faker_a_button:
            gen = FakerDataGenerator(seed=42)
            path_a = _write_input_csv(
                gen.generate_company_list(n_faker_a, "de", id_prefix="src-a"),
                "source_a",
            )
            st.session_state.temp_a_path      = path_a
            st.session_state.faker_a_generated = True
            st.success(f"Generated {n_faker_a} Source A entries → `{Path(path_a).name}`")

        if source_a_file is not None:
            _cleanup_temp(st.session_state.temp_a_path)
            st.session_state.temp_a_path       = _save_upload_to_temp(source_a_file, "upload_a_")
            st.session_state.faker_a_generated = False

        if st.session_state.temp_a_path:
            tag = " (Faker)" if st.session_state.faker_a_generated else " (uploaded)"
            st.caption(f"`{Path(st.session_state.temp_a_path).name}`{tag}")

    with col_b:
        source_b_file = st.file_uploader("Source B", type=["csv", "json"])
        n_faker_b = st.number_input(
            "Faker entries", min_value=10, max_value=500, value=50, step=10,
            key="n_faker_b",
        )
        faker_b_button = st.button("Generate Faker B", key="btn_faker_b", use_container_width=True)

        if faker_b_button:
            gen = FakerDataGenerator(seed=42)
            path_b = _write_input_csv(
                gen.generate_company_list(n_faker_b, "de", id_prefix="src-b"),
                "source_b",
            )
            st.session_state.temp_b_path      = path_b
            st.session_state.faker_b_generated = True
            st.success(f"Generated {n_faker_b} Source B entries → `{Path(path_b).name}`")

        if source_b_file is not None:
            _cleanup_temp(st.session_state.temp_b_path)
            st.session_state.temp_b_path       = _save_upload_to_temp(source_b_file, "upload_b_")
            st.session_state.faker_b_generated = False

        if st.session_state.temp_b_path:
            tag = " (Faker)" if st.session_state.faker_b_generated else " (uploaded)"
            st.caption(f"`{Path(st.session_state.temp_b_path).name}`{tag}")

    active_a = st.session_state.temp_a_path
    active_b = st.session_state.temp_b_path

    st.divider()

    # --- Run control ---
    files_ready  = active_a is not None and active_b is not None
    run_disabled = not weight_ok or not threshold_valid or not files_ready

    if not files_ready:
        st.info("Upload Source A and Source B, or use Generate Faker Data.")

    progress_bar = st.progress(0)
    status_text  = st.empty()
    run_button   = st.button("▶ Start Run", disabled=run_disabled, type="primary")

    if run_button and active_a and active_b:
        progress_bar.progress(0)
        status_text.text("Building configuration...")
        try:
            run_config, changed_fields = _build_run_config(
                defaults=defaults,
                embedding_model=embedding_model,
                faiss_top_k=faiss_top_k,
                auto_match_threshold=auto_match_threshold,
                review_lower_threshold=review_lower_threshold,
                w_embedding=w_embedding,
                w_jaro_winkler=w_jaro_winkler,
                w_token_sort=w_token_sort,
                w_legal_form=w_legal_form,
            )

            is_custom = VERSION_CUSTOM in (run_config.threshold_config_version, run_config.weights_config_version)
            if is_custom:
                st.warning(
                    "Custom settings active — tagged `streamlit_custom` in audit log. "
                    "Changed: " + str(list(changed_fields.keys()))
                )

            def _cb(completed: int, total: int) -> None:
                progress_bar.progress(int(completed / total * 100))
                status_text.text(f"Processing {completed} / {total}...")

            status_text.text("Running pipeline...")
            results, summary = run_entity_resolution(
                source_a_path=active_a,
                source_b_path=active_b,
                run_config=run_config,
                config_adjustments=changed_fields if changed_fields else None,
                progress_callback=_cb,
            )

            st.session_state.results             = results
            st.session_state.summary             = summary
            st.session_state.last_run_config     = run_config
            st.session_state.last_changed_fields = changed_fields
            progress_bar.progress(100)
            status_text.text("Run complete.")

        except Exception as exc:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"Pipeline error: {exc}")

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    summary: RunSummary | None = st.session_state.summary
    results: list[MatchResult] | None = st.session_state.results

    if summary is not None and results is not None:
        st.divider()
        st.subheader("Run Summary")

        c_am, c_rv, c_nm, c_err = st.columns(4)
        c_am.metric("AUTO_MATCH", summary.count_auto_match, f"{summary.auto_match_quote:.1%}")
        c_rv.metric("REVIEW",     summary.count_review,     f"{summary.review_quote:.1%}")
        c_nm.metric("NO_MATCH",   summary.count_no_match,   f"{summary.no_match_quote:.1%}")
        c_err.metric("ERRORS",    summary.count_error)

        if summary.review_quote > review_warn_threshold:
            st.warning(
                f"REVIEW rate {summary.review_quote:.1%} exceeds warning threshold "
                f"({review_warn_threshold:.1%}). Consider recalibrating thresholds."
            )

        st.caption(f"Run ID: `{summary.run_id}` · Total rerank candidates: {summary.total_rerank_candidates}")

        st.subheader("Prioritized Review List")
        st.caption("Entries with review_priority > 0, sorted P1 (mandatory) → P3 (low-urgency).")
        review_df = _build_review_df(results)
        if review_df.empty:
            st.info("No review entries in this run.")
        else:
            st.dataframe(review_df, use_container_width=True, hide_index=True)

        st.subheader("Full Results")
        zone_filter = st.multiselect(
            "Filter by Routing Zone",
            ["AUTO_MATCH", "REVIEW", "NO_MATCH"],
            default=["AUTO_MATCH", "REVIEW", "NO_MATCH"],
        )
        full_df = _build_results_df(results, zone_filter)
        if full_df.empty:
            st.info("No results match the selected filter.")
        else:
            st.dataframe(full_df, use_container_width=True, hide_index=True)

        st.subheader("Download Outputs")
        cd1, cd2, cd3 = st.columns(3)
        for col, path_str, label, mime in [
            (cd1, summary.output_file_path, "⬇ Output JSON",  "application/json"),
            (cd2, summary.review_file_path, "⬇ Review JSON",  "application/json"),
            (cd3, summary.audit_log_path,   "⬇ Audit JSONL",  "application/x-ndjson"),
        ]:
            p = Path(path_str)
            with col:
                if p.exists():
                    st.download_button(label, data=p.read_bytes(), file_name=p.name, mime=mime, use_container_width=True)
                else:
                    st.caption(f"{p.name} not found.")


if __name__ == "__main__":
    main()
