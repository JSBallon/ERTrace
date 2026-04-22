# ERTrace — Entity Resolution Tracer
### Audit-native entity resolution for agentic AI in regulated financial institutions

> **Version:** v0.1.1-alpha
> **Status:** M3 complete — core pipeline, Streamlit GUI, and CLI all functional
> **License:** Apache 2.0
> **Stack:** Python 3.12 · sentence-transformers · FAISS · rapidfuzz · Pydantic v2

---

## Why ERTrace

ERTrace is not primarily a matching library. It is an **audit-native entity resolution skill** — a pipeline that produces structured audit artifacts alongside match results, designed to be called by an agent or agentic workflow in an environment where every automated decision must be explainable and traceable.

The primary output is a **JSONL audit trail** in which every decision is recorded with its full score vector, routing zone, review priority, configuration snapshot, and trace ID. The matching result is a by-product of the audit record.

This distinction matters in banking and other regulated environments where automation thresholds are high. An AI agent that resolves counterparty identities for KYC, regulatory reporting, or master data consolidation cannot simply return `"match"` or `"no match"`. It must return evidence that a human reviewer, internal auditor, or regulator can verify without a data science background. ERTrace delivers that evidence as a first-class output, not as an afterthought.

**Key properties:**

- **Fully local** — no external API calls during pipeline execution; the embedding model runs on CPU after a one-time download. PII stays on the machine — no routing of sensitive data to a third-party service.
- **One-column compatible** — works on name-only data, the realistic starting point for most banking master data use cases. No LEI, no address, no sector code required to get a traceable result.
- **Framework-agnostic** — callable from Streamlit, CLI, or direct Python import. The matching engine has no framework dependencies; it accepts lists and returns typed Pydantic objects.
- **Agent-skill ready** — `run_entity_resolution()` in `bll/app_service.py` is the single entry point. Takes two data source paths and a config, returns `(list[MatchResult], RunSummary)`. Drop it into any orchestration layer.

**The core design principle:** Governance is not an afterthought — it is the architecture.

---

## The Problem

Manual master data reconciliation between banking systems is slow, error-prone, and produces no auditable decision trail. Classical string-distance metrics fail at semantic equivalence (`"BayernLB"` ↔ `"Bayerische Landesbank"`) and produce false positives at lexical similarity (`"Deutsche Bank"` ↔ `"Deutsche Bahn"`). Neither approach satisfies MaRisk AT 4.3.4 (data quality) or EU AI Act Art. 12 (record-keeping) requirements.

---

## Use Cases

| Use Case | Challenge | ERTrace Contribution |
|---|---|---|
| **CRM ↔ Core Banking** | Company names diverge between front-office and settlement systems due to abbreviations, legal form variants, and manual entry | Automated first-pass matching with prioritised review list; full audit trail per decision |
| **Regulatory reporting ↔ GLEIF database** | Reference data is incomplete — entities in the LEI register appear under short names, holding names, or prior legal names; secondary identifiers often absent | One-column semantic matching handles name variants; NO_MATCH surfaced explicitly with score evidence |
| **Agentic AI audit logging** | An agent resolving entity identities autonomously must produce verifiable evidence of each resolution decision for compliance review | Call `run_entity_resolution()` as an agent skill; the returned JSONL audit trail is the compliance artefact |
| **AML / KYC watchlist screening** | Watchlist names are inconsistent, transliterated, or abbreviated | TGFR hybrid catches semantic and syntactic variants; legal form scoring surfaces jurisdiction mismatches |

The GLEIF use case deserves specific mention: the Global LEI register is the authoritative reference for legal entity identification in regulatory reporting (EMIR, MiFID II, SFTR), but its coverage is incomplete and entity names frequently differ from how they appear in internal systems. ERTrace handles the single-column, partial-information problem that makes direct LEI lookup unreliable in practice.

---

## Why Not Splink Directly?

[Splink](https://github.com/moj-analytical-services/splink) is the recommended probabilistic record linkage framework for production-scale entity resolution — and a natural evolution path for this project. It is not the starting point for two reasons.

**Single-column limitation:** Splink's Fellegi-Sunter model performs best with multiple weakly correlated input columns (name + LEI + sector + address). Applied to a single name column, its statistical model cannot cleanly separate match from non-match probabilities. That is precisely the constraint most banking master data reconciliation starts with — particularly against reference databases like GLEIF where secondary identifiers are frequently absent.

**Cold-start problem:** Splink requires labelled training data to calibrate its m- and u-probabilities. Before any labelled data exists, it cannot be trained.

ERTrace solves the single-column problem first — and in doing so, produces what Splink needs: clean normalised inputs, a FAISS-based blocking layer to constrain the candidate space, and a curated set of labelled match/no-match decisions from the DQM expert's review process. That ground truth becomes the training input for Splink once the data model is enriched.

```
ERTrace (this)  — single column, name only
                → TGFR hybrid matching
                → DQM review decisions → labelled ground truth

Production      — multi-column (name + LEI + sector + ...)
                → TGFR FAISS as blocking layer
                → Splink Fellegi-Sunter model trained on ERTrace ground truth
```

ERTrace is not replaced by Splink. It builds the foundation that makes Splink operational in a specific domain.

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
Output: Nested JSON (entry / match / rerank) + Review List + JSONL Audit Log
```

Every match decision carries a four-component score vector:

| Component | What it measures |
|---|---|
| `embedding_cosine_score` | Semantic similarity (Sentence-Transformer) |
| `jaro_winkler_score` | Character-level similarity (prefix-weighted) |
| `token_sort_ratio` | Token overlap (order-independent) |
| `legal_form_score` | Legal form alignment (identical / related / conflict) |

The composite score is a configurable weighted ensemble of all four. All weights, thresholds, and the active model are logged per run.

The output JSON for each entry has three sections: `entry` (what was matched), `match` (the best result with full score vector and routing decision), and `rerank` (all Top-K candidates evaluated, each with their own score vector and routing). This gives the DQM expert and auditor full visibility into not just the winner but every evaluated alternative.

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

## Current State (v0.1.1-alpha)

This is a work-in-progress demo release. The core pipeline is fully implemented and functional. The user-facing interface layer is not yet complete.

| Component | Status |
|---|---|
| DAL: input loading, normalisation, legal form extraction, output writing | ✅ Complete |
| BLL: embedding, FAISS search, fuzzy re-rank, legal form scoring, composite scorer | ✅ Complete |
| BLL: router (routing + 2D priority matrix + FR-LF-05 guardrail) | ✅ Complete |
| BLL: `ERTracePipeline` + `app_service.run_entity_resolution()` | ✅ Complete |
| Governance: JSONL audit logger (run_start / match_result / run_end events) | ✅ Complete |
| Output: nested JSON (`entry / match / rerank`), datetime-prefixed filenames | ✅ Complete |
| `run_manual_test.py` — E2E demo runner (Faker data, German banking names) | ✅ Complete |
| Unit tests: normaliser, legal form extractor, embedder, FAISS, fuzzy, composite, router, output writer, audit logger | ✅ Complete |
| `gui/streamlit_app.py` — Streamlit pipeline controller | ✅ Complete |
| `gui/cli.py` — CLI entry point | ✅ Complete |
| E2E integration tests | ❌ Not written |
| Scoring improvement prototypes (last-token cosine, acronym expansion) | ❌ Planned (M4) |

---

## Quick Start

**Prerequisites:** Python 3.12, pip, Git

```bash
# 1. Clone and install
git clone <repo-url>
cd PoC2_TGFR-Entity-Resolution
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 2. Download embedding model (once — cached locally afterwards)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Vsevolod/company-names-similarity-sentence-transformer')"

# 3. Run the app

# Option A — Streamlit GUI (recommended)
python -m streamlit run gui/streamlit_app.py

# Option B — CLI with your own CSV files
python -m gui.cli --source-a inputs/source_a.csv --source-b inputs/source_b.csv

# Option C — CLI with auto-generated Faker data (no input files needed)
python -m gui.cli --generate
```

No API keys required. No external calls during pipeline execution. Set `TRANSFORMERS_OFFLINE=1` to enforce offline mode after the initial model download.

**Option A** opens the Streamlit pipeline controller in your browser. Upload Source A and Source B CSVs (or generate Faker data in-app), adjust model/threshold/weight parameters via sliders, and download the three output files when the run completes.

**Option B / C** run the pipeline from the terminal and print a summary to stdout. All three output files are written to `outputs/`:

- `output_<datetime>_<run_id>.json` — all results, nested `entry / match / rerank` structure
- `review_<datetime>_<run_id>.json` — REVIEW entries only, sorted by priority (P1 first)
- `audit_<datetime>_<run_id>.jsonl` — append-only audit trail

---

## Architecture

See [`.docs/architecture.md`](.docs/architecture.md) for the full technical breakdown: layer architecture, TGFR algorithm stages, the `ERTracePipeline` / `app_service` hexagonal split, nested output structure, configuration versioning, and the production roadmap (FAISS → Qdrant, ERTrace → Splink).

---

## Governance Approach

See [`.docs/governance_summary.md`](.docs/governance_summary.md) for the regulatory framework mapping (EU AI Act, MaRisk, DORA, DSGVO, NIST AI RMF), the score vector as audit instrument, the FR-LF-05 governance guardrail, and the compliance maturity model.

---

## Project Structure

```
PoC2_TGFR-Entity-Resolution/
├── config/                       # Active config + versioned config archive
│   ├── config.yaml
│   └── versions/v1.0-default.yaml
├── dal/                          # Data Access Layer
│   ├── input_loader.py
│   ├── data_generator.py         # Faker synthetic data generation
│   ├── normalizer.py
│   ├── legal_form_extractor.py
│   ├── sanitizer.py
│   └── output_writer.py          # Nested JSON serialization
├── bll/                          # Business Logic Layer — TGFR matching engine
│   ├── pipeline.py               # ERTracePipeline — pure BLL engine, no I/O
│   ├── app_service.py            # run_entity_resolution() — cross-layer entry point
│   ├── router.py                 # Routing + 2D priority matrix + FR-LF-05 guardrail
│   ├── embedder.py
│   ├── faiss_search.py
│   ├── fuzzy_reranker.py
│   ├── legal_form_scorer.py
│   ├── composite_scorer.py
│   └── schemas.py                # Pydantic: CompanyRecord, ScoreVector, MatchCandidate, MatchResult
├── governance/
│   └── audit_logger.py           # JSONL append-only audit log
├── gui/
│   ├── streamlit_app.py          # Streamlit pipeline controller
│   └── cli.py                    # CLI entry point (4 run modes)
├── tests/                        # Unit + scoring validation tests
├── .docs/                        # Architecture and governance documentation
├── run_manual_test.py            # Minimal E2E smoke runner (Faker data)
├── logs/audit/                   # JSONL audit logs (gitignored)
└── outputs/                      # Pipeline outputs (gitignored)
```

---

## Roadmap

| Stage | Description |
|---|---|
| **v0.1.0-alpha** | Core TGFR pipeline, full audit trail |
| **v0.1.1-alpha** (this) | Streamlit pipeline controller, CLI entry point (M3 complete) |
| **v0.2 — M4** | Scoring improvements (last-token cosine, acronym expansion), reproducibility tests, E2E integration tests |
| **PoC_3** | LangGraph self-correcting agent for REVIEW list, full HITL approval gate |
| **PoC_4** | Web search + MCP connectors for reference databases (LEI/GLEIF) |
| **Production path** | Qdrant persistent index, Splink probabilistic linking, PII governance wrapper |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Part of a Banking GenAI PoC series demonstrating governance-first AI architecture for regulated financial institutions.*
