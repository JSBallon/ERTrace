# Architecture: ERTrace — Entity Resolution Tracer

> Version: v0.1.0-alpha | Part of the Banking GenAI PoC Series
> Design principle: Governance-First, fully local, no LLM in the matching core
> Status: Sections 1–9 reflect the implemented state. Section 6 notes WIP components (Streamlit, CLI).

---

## 1. Design Decision: Why No LLM in the Matching Core

This is the central architectural choice of PoC_2. The matching decision is made entirely without LLM inference. This means:

- **No API calls** during pipeline execution — no latency, no cost, no data privacy concerns
- **Full reproducibility** — identical inputs and configuration produce identical outputs, always
- **Deterministic audit trail** — the score vector is mathematically traceable, not a language model's interpretation
- **No prompt governance overhead** — no prompt versioning, no token logging, no hallucination risk in the matching itself

LLM reasoning is introduced in **PoC_3** (the next stage), where a LangGraph agent handles the ambiguous REVIEW list — a task that genuinely benefits from language understanding. In PoC_2, the matching problem is solved correctly without it.

---

## 2. TGFR Hybrid Architecture

The pipeline follows the **TGFR pattern (Transformer-Gather, Fuzzy-Reconsider)**: a two-stage approach that combines the strengths of semantic embeddings and syntactic string metrics.

### Why a hybrid?

| Approach | Strength | Failure case |
|---|---|---|
| Sentence-Transformer only | Semantic equivalence — "BayernLB" ↔ "Bayerische Landesbank" | Ambiguous abbreviations — "DB" (Deutsche Bank? Deutsche Bahn?) |
| Jaro-Winkler only | Typo detection — "Commerzbnk" ↔ "Commerzbank" | Semantic equivalence with low character overlap |
| **TGFR hybrid** | Both | Neither |

### Stage 1 — Transformer-Gather (Semantic Retrieval)

The embedding model converts all normalised company names into dense vectors. FAISS performs a batch cosine similarity search, returning the Top-K semantically closest candidates from List B for each entry in List A.

```
embeddings_a = model.encode(names_a, normalize_embeddings=True)
embeddings_b = model.encode(names_b, normalize_embeddings=True)
index = faiss.IndexFlatIP(dim)   # Inner Product = Cosine on L2-normalised vectors
index.add(embeddings_b)
scores, indices = index.search(embeddings_a, k=TOP_K)
```

This reduces the comparison space from N×M (all pairs) to N×K (targeted candidates), making the approach scalable to millions of entries.

### Stage 2 — Fuzzy-Reconsider (Syntactic Verification)

On the Top-K candidates per entry, two character-level metrics are applied:

- **Jaro-Winkler similarity** — prefix-weighted edit distance, well-suited for short strings like company names
- **Token-Sort-Ratio** — order-independent token overlap, catches reordered name components

These metrics differentiate cases where the embedding model correctly retrieves candidates but cannot sharply separate them — the "Deutsche Bank" vs. "Deutsche Bahn" problem.

---

## 3. Score Vector

Every matching decision produces a four-component score vector:

```json
{
  "embedding_cosine_score": 0.943,
  "jaro_winkler_score":     0.812,
  "token_sort_ratio":       0.971,
  "legal_form_score":       1.0,
  "composite_score":        0.921,
  "legal_form_relation":    "identical",
  "routing_zone":           "AUTO_MATCH",
  "review_priority":        0
}
```

The composite score is a configurable weighted ensemble:

```
composite = w_emb * embedding_score
          + w_jw  * jaro_winkler_score
          + w_ts  * token_sort_ratio
          + w_lf  * legal_form_score

# Constraint: w_emb + w_jw + w_ts + w_lf = 1.0
# Defaults:    0.50    0.20   0.20   0.10
```

All weights are configurable, versioned, and logged per run. This makes the composite score mathematically traceable — not a black box.

---

## 4. Legal Form Scoring

The legal form is treated as a **score component**, not a hard gate. This is a deliberate governance decision:

A hard gate (GmbH ≠ Ltd. → immediate NO_MATCH) is not auditable — it produces decisions without a score, and it cannot distinguish between a genuine conflict and a front-office translation variant (where a local clerk writes "Ltd." for a German GmbH because that is the equivalent they know).

The legal form score has three configurable levels:

| Relation | Examples | Default Score |
|---|---|---|
| `identical` | GmbH ↔ GmbH, AG ↔ AG | 1.0 |
| `related` | GmbH ↔ AG, SA ↔ NV | 0.5 |
| `conflict` | GmbH ↔ Ltd., AG ↔ Corp. | 0.0 |
| `unknown` | one or both entries missing | 0.5 |

---

## 5. 2D Review Prioritisation

The combination of composite score zone and legal form score produces a structured review priority:

```python
matrix = {
    ("AUTO_MATCH", "identical"): 0,   # never review
    ("AUTO_MATCH", "related"):   2,   # optional
    ("AUTO_MATCH", "conflict"):  1,   # mandatory — high score despite conflict
    ("REVIEW",     "identical"): 3,   # low urgency
    ("REVIEW",     "related"):   2,   # standard
    ("REVIEW",     "conflict"):  1,   # mandatory
    ("NO_MATCH",   "*"):         0,   # no review
}
```

Priority 1 cases (AUTO_MATCH + legal form conflict) are the most important governance signal: a high composite score despite a legal form conflict almost always indicates either a front-office translation variant requiring data correction, or a false positive requiring human rejection.

---

## 6. Layer Architecture

```
┌──────────────────────────────────────────────┐
│  GUI Layer                                   │
│  ⚠ Streamlit pipeline controller — WIP       │
│  ⚠ CLI entry point (gui/cli.py)    — WIP     │
│  Current entry point: run_manual_test.py     │
│  NO imports from BLL/DAL internals           │
└─────────────────────┬────────────────────────┘
                      │ app_service interface only
                      ▼
┌──────────────────────────────────────────────┐
│  Application Service (bll/app_service.py)    │
│  run_entity_resolution() — cross-layer wire  │
│  Config loading → DAL normalisation →        │
│  ERTracePipeline → OutputWriter → AuditLog   │
│  Primary public API for all entry points     │
└──────────┬───────────────────┬───────────────┘
           │ pure BLL          │ DAL interfaces
           ▼                   ▼
┌──────────────────┐  ┌────────────────────────┐
│  ERTracePipeline │  │  DAL Layer             │
│  (ertrace_       │  │  Input loading,        │
│   pipeline.py)   │  │  normalisation,        │
│                  │  │  legal form extraction,│
│  Embedder →      │  │  output writing,       │
│  FAISS →         │  │  JSONL audit log       │
│  Fuzzy Re-Rank → │  │  (append-only)         │
│  Legal Form →    │  └────────────────────────┘
│  Composite →     │
│  Router →        │
│  Priority Matrix │
│                  │
│  NO filesystem,  │
│  NO Streamlit,   │
│  NO config load  │
└──────────────────┘
```

**Hexagonal boundary (`app_service` / `ERTracePipeline`):** The engine (`ERTracePipeline` in `bll/ertrace_pipeline.py`) is the pure domain core — it receives a `RunConfig` and a list of `CompanyRecord` objects and returns a list of `MatchResult` objects. It has zero knowledge of file paths, YAML, or output formats. The application service (`bll/app_service.py`) is the primary adapter: it loads config, normalises inputs via DAL, runs the engine, writes outputs, and logs the audit trail. This boundary is a file boundary in Python — imports make it explicit and testable.

**Framework agnosticism:** `ERTracePipeline` and `app_service` have no dependency on Streamlit or any UI framework. The same `run_entity_resolution()` call is used by `run_manual_test.py` today and will be used by the Streamlit app and CLI when they are implemented.

---

## 7. Normalisation Pipeline

Before embedding, all names go through a four-step normalisation:

```
"Bayerische Landesbank GmbH & Co. KG"
    ↓ 1. Unicode NFC normalisation (Umlaute, diacritics)
    ↓ 2. Whitespace cleaning
    ↓ 3. Legal form extraction → stored as separate field: "GmbH & Co. KG"
    ↓ 4. Legal form stripping (cleanco) + lowercase
"bayerische landesbank"
```

Legal form extraction happens **before** stripping, so the structural information is preserved as a typed field for the legal form scorer — not lost in normalisation.

`cleanco` is a Python library that knows 300+ legal form suffixes from ~60 countries (GmbH, AG, Ltd., SA, NV, Corp., S.p.A., Oy, AS…) and strips them reliably.

---

## 8. Configuration Versioning

All decision parameters are versioned:

```
config/
├── config.yaml                     ← points to active version
└── versions/
    ├── v1.0-default.yaml           ← initial defaults
    └── v1.1-goldstandard.yaml      ← post-calibration (future)
```

Each version file records: all thresholds, all weights, legal form score levels, embedding model, creation date, and rationale. Every run logs the active config version in its `run_start` event — making any historical run fully reconstructable.

---

## 9. Audit Trail Structure

Every run produces three output files, all with datetime-prefixed filenames (`<type>_<YYYYMMDD-HHmm>_<run_id>.<ext>`) for chronological sort order in any file explorer:

**`audit_<datetime>_<run_id>.jsonl`** — append-only event stream (compact, one line per event):

```jsonl
{"event_type": "run_start",    "run_id": "...", "timestamp": "...", "embedding_model": "...", "auto_match_threshold": 0.92, "w_embedding": 0.5, ...}
{"event_type": "match_result", "run_id": "...", "trace_id": "...", "source_a_id": "...", "embedding_cosine_score": 0.943, "composite_score": 0.921, "routing_zone": "AUTO_MATCH", "review_priority": 0, "rerank_count": 5, ...}
{"event_type": "match_result", ...}
{"event_type": "run_end",      "run_id": "...", "count_auto_match": 847, "count_review": 112, "review_quote": 0.117, "total_rerank_candidates": 4235, ...}
```

`rerank_count` in each `match_result` event records how many Top-K candidates were evaluated for that entry (diagnostic counter). `total_rerank_candidates` in `run_end` is the run-level aggregate — auditors can verify `Σ rerank_count == total_rerank_candidates`. The full rerank candidate list is intentionally kept out of the JSONL to maintain compact, scannable event records; it is available in the output JSON (see Section 10).

**`output_<datetime>_<run_id>.json`** — full results in nested human-readable format (see Section 10).

**`review_<datetime>_<run_id>.json`** — REVIEW entries only, sorted by review priority ascending (P1 first), same nested format.

The JSONL is the audit record (compact, machine-readable, append-only). The output JSON is the review document (nested, human-readable, designed for the DQM expert). Both are needed for a complete audit picture — neither is redundant.

The log is human-readable without tooling. A regulator or internal auditor can open the JSONL file and understand every decision — no data science background required.

---

## 10. Nested Output JSON Structure

Every result entry in the output JSON has three sections — a deliberate design that maps to the DQM expert's review workflow:

```json
{
  "run_id": "4006f27e-...",
  "generated_at": "2026-03-30T00:09:22Z",
  "total_entries": 20,
  "results": [
    {
      "entry": {
        "source_a_id": "crm_0002",
        "source_a_name": "Deutsche Hypothekenbank UG (haftungsbeschränkt)",
        "source_a_name_normalized": "deutsche hypothekenbank",
        "source_a_legal_form": "ug"
      },
      "match": {
        "source_b_id": "core_0014",
        "source_b_name": "Deutsche Hypothekenbank KG",
        "source_b_name_normalized": "deutsche hypothekenbank",
        "source_b_legal_form": "kg",
        "score": {
          "embedding_cosine_score": 0.923,
          "jaro_winkler_score": 0.898,
          "token_sort_ratio": 0.657,
          "legal_form_score": 0.5,
          "legal_form_relation": "related",
          "composite_score": 0.823
        },
        "routing_zone": "REVIEW",
        "review_priority": 2,
        "rank": 0
      },
      "rerank": [
        { "source_b_id": "core_0014", "score": { "composite_score": 0.823, ... }, "routing_zone": "REVIEW", "rank": 0 },
        { "source_b_id": "core_0031", "score": { "composite_score": 0.622, ... }, "routing_zone": "REVIEW", "rank": 1 }
      ],
      "trace_id": "4dd7ae62-...",
      "timestamp": "2026-03-30T00:09:21Z"
    }
  ]
}
```

- **`entry`** — Source A identification. What was being matched.
- **`match`** — The best candidate from Source B, with its full score vector and the routing decision. `rank: 0` is an invariant: the pipeline always places the best candidate here.
- **`rerank`** — All Top-K candidates evaluated during the FAISS + fuzzy scoring phase, each with its own `ScoreVector`, `routing_zone`, and `review_priority`. This is the full audit of what was considered and why the winner was selected. Rank-1 and rank-2 candidates are often informative: a rank-1 candidate at REVIEW may indicate a legitimate second candidate the DQM expert should compare against.

For NO_MATCH entries: `match` has all Source B fields as `null`, all scores as `0.0`, `routing_zone: "NO_MATCH"`. `rerank` is an empty list `[]`.

The nesting is a **serialization-only concern** applied by `OutputWriter._to_nested()`. The pipeline and BLL work with flat `MatchResult` objects throughout — the nested structure is the read-model projection (CQRS pattern), optimised for human review. The JSONL audit log uses the flat format, optimised for event streaming.

---

## 11. Embedding Model
**Default:** `Vsevolod/company-names-similarity-sentence-transformer`
Trained specifically for company name similarity. Produces 768-dimensional vectors.

**Alternatives (configurable):**

| Model | Use case |
|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual (DE/EN/FR) dominates |
| `deutsche-telekom/gbert-large-paraphrase-cosine` | German-language optimised |
| `all-MiniLM-L6-v2` | Fastest, smallest (~80 MB), English baseline |

All models are downloaded once at setup and cached locally. No network access during pipeline execution. Set `TRANSFORMERS_OFFLINE=1` to enforce this strictly.

---

## 12. Production Roadmap

| Component | PoC_2 | Production path |
|---|---|---|
| Similarity search | FAISS in-memory (CPU) | Qdrant self-hosted, persistent index |
| Matching logic | TGFR hybrid (deterministic) | Splink (Fellegi-Sunter probabilistic model) |
| Review handling | Prioritised export list | LangChain self-correcting agent (PoC_3) |
| HITL gate | Flagging + export | Approve/Reject workflow in UI (PoC_3) |
| Reference databases | Faker synthetic data | MCP connectors — GLEIF LEI API, Web-Search sub-agents (LangGraph) others (PoC_4) |
| PII handling | Out of scope (synthetic data) | DAL encryption + BLL containerisation + GUI RBAC |

**Production path via Splink:** Splink (built by the UK Ministry of Justice, Fellegi-Sunter probabilistic model, DuckDB backend) is the recommended production path for entity resolution at scale — but it cannot be cold-started. It requires labelled training data to calibrate its m- and u-probabilities, clean normalised inputs, and a blocking strategy to constrain the comparison space.

PoC_2 provides exactly these three prerequisites: the normalisation pipeline (DAL) is reused unchanged; the FAISS Top-K output serves as the blocking layer that pre-qualifies candidate pairs for Splink; and the review decisions made by the DQM expert on PoC_2's output become the labelled ground truth dataset that Splink needs to learn from.

PoC_2 is not replaced by Splink — it builds the foundation that makes Splink operational in a specific domain.