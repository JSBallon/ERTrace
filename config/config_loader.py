"""
Config — config_loader

Loads the active versioned YAML configuration and constructs a fully validated RunConfig.

Two-file resolution:
  1. config/config.yaml          → reads active_version + config_path
  2. config/versions/<ver>.yaml  → reads all parameters (embedding, faiss, thresholds,
                                     weights, legal_form, monitoring)

Constructs RunConfig (Pydantic) — the only input ERTracePipeline needs.
All YAML parsing is isolated here. ERTracePipeline never touches the filesystem.

M3 addition (ADR-M3-006):
  save_ui_config() — compares Streamlit slider values to the active YAML config.
  If any value differs, writes a new config/versions/v_ui_<YYYYMMDD-HHmm>.yaml and
  updates config/config.yaml to point to the new version.
  Returns (RunConfig, was_adjusted: bool, changed_fields: dict).

No Streamlit imports. No BLL logic. No external API calls.
"""

import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path

from bll.schemas import (
    RunConfig,
    ThresholdConfig,
    WeightsConfig,
    LegalFormConfig,
)


def load_run_config(
    config_path: str = "config/config.yaml",
) -> RunConfig:
    """
    Load the active versioned YAML config and construct a RunConfig.

    Owns only algorithm configuration (model, FAISS, thresholds, weights, legal form).
    Input file paths are NOT part of RunConfig — they are run-time data pointers
    that belong on the audit event directly (see ADR-M2-006 refactor).

    Args:
        config_path: Path to config/config.yaml (contains active_version pointer).

    Returns:
        Fully validated RunConfig ready for ERTracePipeline.

    Raises:
        FileNotFoundError: If config.yaml or the versioned YAML cannot be found.
        KeyError: If required fields are missing from the versioned YAML.
        pydantic.ValidationError: If config values fail schema constraints.
    """
    root = Path(config_path)
    if not root.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(root, "r", encoding="utf-8") as f:
        top = yaml.safe_load(f)

    versioned_path = Path(top["config_path"])
    if not versioned_path.exists():
        raise FileNotFoundError(
            f"Versioned config not found: {versioned_path} "
            f"(referenced from {config_path})"
        )

    with open(versioned_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    meta = cfg["metadata"]
    emb  = cfg["embedding"]
    fais = cfg["faiss"]
    thr  = cfg["thresholds"]
    wts  = cfg["weights"]
    lf   = cfg["legal_form"]

    return RunConfig(
        run_id=str(uuid.uuid4()),
        embedding_model=emb["model"],
        faiss_top_k=fais["top_k"],
        threshold_config=ThresholdConfig(
            auto_match_threshold=thr["auto_match_threshold"],
            review_lower_threshold=thr["review_lower_threshold"],
        ),
        weights_config=WeightsConfig(
            w_embedding=wts["w_embedding"],
            w_jaro_winkler=wts["w_jaro_winkler"],
            w_token_sort=wts["w_token_sort"],
            w_legal_form=wts["w_legal_form"],
        ),
        legal_form_config=LegalFormConfig(
            identical_score=lf["identical_score"],
            related_score=lf["related_score"],
            conflict_score=lf["conflict_score"],
            unknown_score=lf["unknown_score"],
        ),
        threshold_config_version=meta["threshold_config_version"],
        weights_config_version=meta["weights_config_version"],
        legal_form_config_version=meta["legal_form_config_version"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_ui_config(
    embedding_model: str,
    faiss_top_k: int,
    auto_match_threshold: float,
    review_lower_threshold: float,
    w_embedding: float,
    w_jaro_winkler: float,
    w_token_sort: float,
    w_legal_form: float,
    config_path: str = "config/config.yaml",
) -> tuple["RunConfig", bool, dict]:
    """
    Compare Streamlit UI slider values to the current active YAML config.

    If any value differs from the active config:
      1. Write a new config/versions/v_ui_<YYYYMMDD-HHmm>.yaml.
      2. Update config/config.yaml → active_version + config_path.
    If values are identical, no files are written.

    Args:
        embedding_model:        Embedding model identifier string.
        faiss_top_k:            FAISS top-K candidate count.
        auto_match_threshold:   Auto-match routing threshold.
        review_lower_threshold: Review lower routing threshold.
        w_embedding:            Weight for embedding cosine score.
        w_jaro_winkler:         Weight for Jaro-Winkler score.
        w_token_sort:           Weight for token sort ratio.
        w_legal_form:           Weight for legal form score.
        config_path:            Path to config/config.yaml.

    Returns:
        Tuple of:
          - RunConfig: validated config with new version strings if adjusted.
          - bool: True if any value was different from active config.
          - dict: changed_fields — {field: {"from": old, "to": new}} per changed param.
                  Empty dict if nothing changed.
    """
    # Load current active config for comparison
    current = load_run_config(config_path)

    # Build comparison map: field_name → (current_value, ui_value)
    comparisons: dict[str, tuple] = {
        "embedding_model":        (current.embedding_model,                             embedding_model),
        "faiss_top_k":            (current.faiss_top_k,                                 faiss_top_k),
        "auto_match_threshold":   (current.threshold_config.auto_match_threshold,        auto_match_threshold),
        "review_lower_threshold": (current.threshold_config.review_lower_threshold,      review_lower_threshold),
        "w_embedding":            (current.weights_config.w_embedding,                   w_embedding),
        "w_jaro_winkler":         (current.weights_config.w_jaro_winkler,               w_jaro_winkler),
        "w_token_sort":           (current.weights_config.w_token_sort,                  w_token_sort),
        "w_legal_form":           (current.weights_config.w_legal_form,                  w_legal_form),
    }

    changed_fields: dict = {}
    for field, (old_val, new_val) in comparisons.items():
        # Use tolerance for float comparisons
        if isinstance(old_val, float):
            if abs(old_val - new_val) > 1e-6:
                changed_fields[field] = {"from": old_val, "to": new_val}
        else:
            if old_val != new_val:
                changed_fields[field] = {"from": old_val, "to": new_val}

    if not changed_fields:
        # No changes — return existing config unchanged
        return current, False, {}

    # --- Values differ: write a new versioned YAML ---
    version_name = f"v_ui_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    versions_dir = Path(config_path).parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_path = versions_dir / f"{version_name}.yaml"

    new_yaml_content = {
        "metadata": {
            "threshold_config_version": version_name,
            "weights_config_version":   version_name,
            "legal_form_config_version": version_name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "rationale": "UI-adjusted configuration (Streamlit session)",
        },
        "embedding": {
            "model": embedding_model,
            "alternatives": [
                "paraphrase-multilingual-MiniLM-L12-v2",
                "deutsche-telekom/gbert-large-paraphrase-cosine",
                "all-MiniLM-L6-v2",
            ],
        },
        "faiss": {
            "top_k": faiss_top_k,
        },
        "thresholds": {
            "auto_match_threshold":   auto_match_threshold,
            "review_lower_threshold": review_lower_threshold,
        },
        "weights": {
            "w_embedding":    w_embedding,
            "w_jaro_winkler": w_jaro_winkler,
            "w_token_sort":   w_token_sort,
            "w_legal_form":   w_legal_form,
        },
        "legal_form": {
            "identical_score": current.legal_form_config.identical_score,
            "related_score":   current.legal_form_config.related_score,
            "conflict_score":  current.legal_form_config.conflict_score,
            "unknown_score":   current.legal_form_config.unknown_score,
        },
        "monitoring": {
            "review_quote_warning_threshold": 0.30,
        },
    }

    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(new_yaml_content, f, allow_unicode=True, sort_keys=False)

    # Update config.yaml to point to the new version
    root = Path(config_path)
    with open(root, "w", encoding="utf-8") as f:
        yaml.dump(
            {"active_version": version_name, "config_path": str(new_yaml_path)},
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    # Build and return the new RunConfig
    new_config = RunConfig(
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
            identical_score=current.legal_form_config.identical_score,
            related_score=current.legal_form_config.related_score,
            conflict_score=current.legal_form_config.conflict_score,
            unknown_score=current.legal_form_config.unknown_score,
        ),
        threshold_config_version=version_name,
        weights_config_version=version_name,
        legal_form_config_version=version_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return new_config, True, changed_fields
