"""
Config — config_loader

Loads the active versioned YAML configuration and constructs a fully validated RunConfig.

Two-file resolution (see ADR-M2-006):
  1. config/config.yaml          → reads active_version + config_path
  2. config/versions/<ver>.yaml  → reads all parameters (embedding, faiss, thresholds,
                                    weights, legal_form, monitoring)

Constructs RunConfig (Pydantic) — the only input ERTracePipeline needs.
All YAML parsing is isolated here. ERTracePipeline never touches the filesystem.

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
