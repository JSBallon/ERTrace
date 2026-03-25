# PoC_2 — Entity Resolution
### Auditable Company Name Matching for Regulated Financial Institutions

> **Status:** Work in Progress — Banking GenAI PoC Series
> **License:** Apache 2.0
> **Stack:** Python 3.12 · sentence-transformers · FAISS · rapidfuzz · Pydantic v2 · Streamlit

---

## What This Is

A locally executable, LLM-free entity resolution pipeline that matches company names across two banking systems (CRM ↔ Core Banking) using a hybrid TGFR approach (Transformer-Gather, Fuzzy-Reconsider). Every matching decision is fully auditable via a structured score vector and a JSONL audit trail.

**The core design principle:** Governance is not an afterthought — it is the architecture.

---

## The Problem

Manual master data reconciliation between CRM and core banking systems is slow, error-prone, and produces no auditable decision trail. Classical string-distance metrics fail at semantic equivalence ("BayernLB" ↔ "Bayerische Landesbank") and produce false positives at lexical similarity ("Deutsche Bank" ↔ "Deutsche Bahn"). Neither approach satisfies MaRisk AT 4.3.4 (data quality) or EU AI Act Art. 12 (record-keeping) requirements.

---

## Why Not Splink Directly?

[Splink](https://github.com/moj-analytical-services/splink) is the recommended probabilistic record linkage framework for production-scale entity resolution — and a natural evolution path for this project. It is not the starting point.

Splink performs best with multiple weakly correlated input columns (name + LEI + sector + address). Applied to a single name column without additional fields, its statistical model cannot cleanly separate match from non-match probabilities. That is precisely the constraint most banking master data reconciliation starts with.

PoC_2 solves the single-column problem first — and in doing so, produces what Splink needs to work well: a curated set of labelled match/non-match decisions from the DQM expert's review process. That ground truth dataset becomes the training input for Splink once the data model is enriched with additional fields in production.

The transition looks like this:

```
PoC_2  — single column (name only)
         → TGFR hybrid matching
         → DQM review decisions → labelled ground truth

Production — multi-column (name + LEI + sector + ...)
         → TGFR FAISS as blocking layer
         → Splink Fellegi-Sunter model trained on PoC_2 ground truth
```

PoC_2 is not replaced by Splink. It delivers the foundation that makes Splink operational in your specific domain.

---

## How It Works

```
Input (CSV/JSON)
      ↓
DAL: Normalisation + Legal Form Extraction
      ↓
BLL: Sentence-Transformer Batch Embedding (local, CPU)
      ↓
BLL: FAISS Similarity Search → Top-K candidates per entry
      ↓
BLL: Fuzzy Re-Rank (Jaro-Winkler + Token-Sort-Ratio)
      ↓
BLL: Legal Form Scoring + Composite Score (weighted ensemble)
      ↓
BLL: Threshold Routing (AUTO_MATCH / REVIEW / NO_MATCH)
      ↓
BLL: 2D Review Priority (Score Zone × Legal Form)
      ↓
Output: JSON + prioritised Review List + JSONL Audit Log
```

Every match decision carries a four-component score vector:

| Component | What it measures |
|---|---|
| `embedding_cosine_score` | Semantic similarity (Sentence-Transformer) |
| `jaro_winkler_score` | Character-level similarity (prefix-weighted) |
| `token_sort_ratio` | Token overlap (order-independent) |
| `legal_form_score` | Legal form alignment (identical / related / conflict) |

The composite score is a configurable weighted ensemble of all four. All weights, thresholds, and the active model are logged per run.

---

## Review Prioritisation

The combination of composite score zone and legal form score produces a structured work list for the DQM expert:

| Composite Zone | Legal Form | Priority | Action |
|---|---|---|---|
| AUTO_MATCH | identical | 0 | No review needed |
| AUTO_MATCH | related | 2 | Optional spot-check |
| AUTO_MATCH | **conflict** | **1** | **Mandatory review** |
| REVIEW | identical | 3 | Low-urgency review |
| REVIEW | related | 2 | Standard review |
| REVIEW | conflict | 1 | Mandatory review |
| NO_MATCH | — | 0 | No review needed |

Priority 1 cases (high composite score despite legal form conflict) flag front-office translation variants (GmbH → Ltd.) and potential false positives — both require human decision before the match is used downstream.

---

## Setup

**Prerequisites:** Python 3.12, pip, Git

```bash
# 1. Clone and install
git clone <repo-url>
cd PoC_2_entity_resolution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download embedding model (once — cached locally afterwards)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Vsevolod/company-names-similarity-sentence-transformer')"

# 3. Optional: enforce offline mode after download
export TRANSFORMERS_OFFLINE=1

# 4. Run
streamlit run gui/streamlit_app.py
```

No API keys required. No external calls during pipeline execution.

---

## Demo

Generate synthetic test data via the Streamlit interface (Faker button), or provide your own CSV:

```
source_a_id,source_a_name     ← CRM export
source_b_id,source_b_name     ← Core banking export
```

After a run, the **Audit Log Viewer** tab shows every scoring decision in structured form — score vectors, configuration parameters, all events. This is the primary auditability demonstration.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full technical breakdown including layer boundaries, data flow, and the production roadmap (FAISS → Qdrant, Hybrid → Splink).

---

## Governance Approach

See [`docs/governance_summary.md`](docs/governance_summary.md) for the regulatory mapping (EU AI Act, MaRisk, DORA, DSGVO) and the compliance maturity model.

---

## Roadmap

| Stage | Description |
|---|---|
| **PoC_2** (this) | Linear TGFR pipeline, full audit trail, Streamlit UI |
| **PoC_3** | LangGraph self-correcting agent for REVIEW list, full HITL approval gate |
| **PoC_4** | Web search + MCP connectors for reference databases (LEI/GLEIF) |
| **Production path** | Qdrant persistent index, Splink probabilistic linking, PII governance wrapper |

---

## Project Structure

```
poc_2_entity_resolution/
├── config/                  # Active config + versioned config archive
├── dal/                     # Data Access Layer (I/O, normalisation, audit log)
├── bll/                     # Business Logic Layer (TGFR pipeline, scoring, routing)
├── governance/              # Cross-cutting audit logger
├── gui/                     # Streamlit pipeline controller + CLI
├── tests/                   # Unit + scoring validation tests
├── docs/                    # Architecture and governance documentation
├── logs/audit/              # JSONL audit logs (gitignored, .gitkeep only)
└── outputs/                 # Pipeline outputs (gitignored, .gitkeep only)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Part of a Banking GenAI PoC series demonstrating governance-first AI architecture for regulated financial institutions.*
