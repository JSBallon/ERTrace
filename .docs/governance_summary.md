# Governance Summary: ERTrace — Entity Resolution Tracer

> Version: v0.1.0-alpha | Regulatory mapping, compliance maturity model, and design rationale
> Audience: Risk & Compliance, Internal Audit, AI Governance, External Reviewers

---

## 1. ERTrace as an Agent Skill: Audit Artifacts for Agentic AI

Banking operates under strict automation thresholds. When an AI agent makes decisions that affect KYC status, regulatory reporting, or master data — decisions a human would traditionally sign off on — the agent must be able to produce evidence of its reasoning that satisfies internal audit and, where applicable, regulatory examination. Returning a confidence score is not sufficient. Returning a match decision with no trace is not acceptable.

ERTrace is designed to meet this requirement as an **agent skill**: a callable unit that an orchestration layer invokes and receives back not just a result but a structured, verifiable audit record. The `run_entity_resolution()` function in `bll/app_service.py` is that skill. It returns a `RunSummary` with paths to three output artefacts:

- **JSONL audit trail** — every decision as a flat, append-only event with the full score vector, routing zone, and review priority. Compact enough to stream; complete enough to reconstruct any decision.
- **Output JSON** — all results in nested `entry / match / rerank` format. Human-readable in a plain editor. The DQM expert or auditor opens this file and sees what was matched, how it was scored, and what alternatives were evaluated.
- **Review JSON** — REVIEW entries only, sorted P1 first. The agent's recommendation list for human follow-up.

**Why local execution matters for agentic use cases:** Because entity resolution in banking often involves data that is subject to banking secrecy, GDPR, or internal data classification policies, the agent skill must be able to operate without routing data to an external service. ERTrace runs entirely on-premise — the embedding model is cached locally, FAISS is in-memory, and no byte leaves the machine during a pipeline execution. This means an agent can invoke the skill on PII-bearing company name data without a separate privacy assessment for external data transfers.

**Customisable one-column review:** All decision parameters — thresholds, composite weights, legal form score levels, Top-K — are configurable and versioned. An agent operator can tune the skill's sensitivity for a specific use case (e.g., more aggressive AUTO_MATCH for a clean reference dataset vs. more conservative REVIEW routing for an incomplete one) and the change is logged in the audit trail automatically.

---

## 2. Core Governance Principle

This system is built on a single architectural premise: **every decision the pipeline makes must be explainable, reproducible, and auditable — without requiring data science expertise to verify it.**

That means:

- No black-box scoring. Every match carries a four-component score vector with named, interpretable dimensions.
- No invisible parameters. Every threshold, weight, and model choice is logged per run with a versioned configuration reference.
- No silent failures. Every guardrail trigger, validation error, and routing override is a typed log event.
- No autonomous action. The pipeline produces structured recommendations. A human makes the final decision on every case in the review list.

---

## 3. What the System Does and Does Not Do

| The system does | The system does not |
|---|---|
| Score every name pair against four explicit criteria | Make autonomous changes to source data |
| Flag legal form conflicts as priority review cases | Resolve ambiguous cases without human input |
| Produce a prioritised review list for the DQM expert | Override the human reviewer's decision |
| Log every decision with full parameter context | Operate without a complete audit trail |
| Support configurable thresholds and weights | Apply hidden or hardcoded decision logic |

---

## 4. Regulatory Framework Mapping

### EU AI Act (2024/1689)

The system is currently operating as a **proof of concept with synthetic data** and is not classified as a high-risk AI system under Annex III. In production use with KYC/AML-relevant master data, a formal risk classification assessment would be required.

| Article | Requirement | Implementation |
|---|---|---|
| Art. 9 — Risk Management | Ongoing risk controls over system lifecycle | Configurable thresholds, review prioritisation, REVIEW-quote monitoring |
| Art. 10 — Data Governance | Data quality and provenance tracking | Normalisation pipeline, source IDs, legal form extraction |
| Art. 11 — Technical Documentation | System documentation | Full documentation series (21.1–21.9) |
| Art. 12 — Record-Keeping | Logging of inputs, outputs, decision logic | JSONL audit trail with score vector, run-start config log |
| Art. 13 — Transparency | Explainable outputs for users | Four-component score vector, legal form relation label, review priority |
| Art. 14 — Human Oversight | Human control and intervention capability | Prioritised review list; full HITL approval gate in PoC_3 |

### MaRisk (BaFin) - Highlights

| Module | Requirement | Implementation |
|---|---|---|
| AT 4.3.4 — Data Quality | Quality and consistency of decision-relevant data | Normalisation, legal form extraction, score as data quality indicator |
| AT 7.2 — Automated Processes | Controls for automated processes, traceability | Audit trail, Pydantic validation, review prioritisation, REVIEW-quote monitoring |
| AT 8.2 — Internal Audit | Auditability of IT-supported processes | JSONL log readable without technical tooling, run-ID correlation, config versioning |
| BTR 3 — Model Risk | Validation, monitoring, and sign-off of models | Config versioning, REVIEW-quote as performance indicator; formal model validation required for production |

### DSGVO (EU 2016/679) - Highlights

> Note: PoC_2 processes exclusively synthetic Faker data. No personal data is in scope.

| Article | Requirement | Implementation / Roadmap |
|---|---|---|
| Art. 5 — Accountability | Documentation of processing | JSONL audit trail + config versioning as accountability record |
| Art. 22 — Automated Decisions | Right to human review of automated decisions | Review prioritisation + human decision required for all flagged cases |
| Art. 25 — Privacy by Design | Data protection by design | DAL isolation enables PII routing without BLL changes (production roadmap) |

### DORA (EU 2022/2554) - Highlights

| Article | Requirement | Implementation |
|---|---|---|
| Art. 9 — ICT Risk Management | Documentation and controls for ICT systems | Audit trail, config versioning |
| Art. 17 — Incident Management | Logging for incident classification | JSONL error events, run-ID for incident tracing |

### NIST AI RMF 1.0 - Highlights

| Function | Implementation |
|---|---|
| GOVERN — Accountability | Role definitions, scope boundaries, agent behavior specification |
| MAP — Risk Identification | Risk-based functional requirements mapped to controls |
| MEASURE — Monitoring | Score vector, REVIEW-quote, run summary metrics |
| MANAGE — Risk Response | Retry logic, error logging, escalation via review prioritisation |

---

## 5. The Score Vector as Audit Instrument

The four-component score vector replaces the "reasoning" field of LLM-based pipelines. Unlike a language model's natural language explanation, the score vector is:

- **Deterministic** — identical inputs and configuration always produce the same scores
- **Mathematically traceable** — the composite score is provably consistent with the component scores and weights
- **Interpretable without data science** — a regulator, auditor, or DQM expert can read `"embedding_cosine_score: 0.94, legal_form_relation: conflict, review_priority: 1"` and understand the decision without a data science background

This is the key governance differentiator from standard GenAI pipelines, which require LLM reasoning logs that are harder to verify and reproduce.

**FR-LF-05 guardrail — a named, visible governance control:**

The most critical single governance decision in the routing logic is enforced as a standalone, explicitly auditable code block in `bll/router.py`:

> *AUTO_MATCH + legal_form_relation = "conflict" → review_priority must be 1 (mandatory review), always, regardless of composite score.*

This is implemented as a hard `if`-block that runs after the priority matrix lookup. If the matrix ever returns a value other than 1 for this combination (e.g., due to a regression or misconfiguration), the guardrail overrides it, logs a `priority_override_FR_LF_05` event to the JSONL audit trail, and continues. The event is a dual-control signal: it fires only when the matrix contradicts the expected value, making any such divergence immediately visible to an auditor.

The guardrail uses Pydantic `model_copy(update={...})` — it never mutates the candidate in place. Every state transition is immutable and traceable.

**Rerank audit trail — all evaluated candidates, not just the winner:**

Every entry in the output JSON includes a `rerank` array containing all Top-K candidates evaluated during the FAISS + fuzzy scoring phase. Each candidate carries its own `ScoreVector`, `routing_zone`, and `review_priority`. This means an auditor or DQM expert can verify:

- Why candidate B was selected over candidate C (compare composite scores)
- Whether a rank-2 candidate might be a legitimate alternative match requiring attention
- That the routing decision applied to the winner is consistent with its individual score

The full rerank list is in the output JSON. The JSONL audit log carries only `rerank_count` (an integer) per `match_result` event — keeping the audit stream compact and scannable. The run-level `total_rerank_candidates` in the `run_end` event enables completeness verification: `Σ rerank_count == total_rerank_candidates`.

---

## 6. The 2D Review Prioritisation as Data Governance Instrument

The review prioritisation is a **data governance instrument** — it structures the quality management process for the master data the system is applied to, not the governance of the AI system itself.

The most important signal is **Priority 1: AUTO_MATCH + legal form conflict.** In banking practice, this pattern arises in two ways:

1. **Front-office translation variant:** A local clerk records "Ltd." for a German GmbH because that is the local equivalent. The match is correct, but the legal form in the source system is wrong and needs correction.
2. **False positive:** Two different entities with similar names in different jurisdictions. The match must be rejected.

Both cases require a human decision. The system correctly identifies them as high-priority without making the decision itself.

---

## 7. Compliance Maturity Model

| Level | Description | Status |
|---|---|---|
| **Level 1 — Basic** | Score vector, routing zones, Pydantic validation, legal form extraction | ✅ Implemented |
| **Level 2 — Operational** | JSONL audit trail, config versioning, review prioritisation, run-ID correlation | ✅ Implemented |
| **Level 3 — Audit-Ready** | Full parameter documentation per run, REVIEW-quote monitoring, FR-LF-05 guardrail, rerank audit trail | ✅ Core pipeline implemented |
| **Level 4 — Production** | Goldstandard calibration, HITL approval gate, formal model validation (BTR 3), PII controls | 🔜 Post-PoC |

**Current level: 3 (core pipeline) — v0.1.0-alpha**

All Level 1–3 governance controls are active in the pipeline core (`ERTracePipeline`, `Router`, `AuditLogger`, `OutputWriter`). The Streamlit pipeline controller and CLI entry point are placeholders in this release — the working demo entry point is `run_manual_test.py`. E2E integration tests are not yet written.

The system demonstrates that regulatory-first AI architecture does not require complexity beyond what the use case demands. All governance controls implemented here are verifiable by reading the source code and the JSONL audit log — no additional tooling required.

---

## 8. What Is Explicitly Out of Scope

The following are documented production requirements, not implemented in PoC_2:

**PII Handling** is a prerequisite for production use, not an optional enhancement. It requires controls at three layers:
- DAL: PII detection before passing data to BLL, encrypted storage, GDPR-compliant deletion
- BLL/Infrastructure: Containerised embedding inference, no PII-bearing vector persistence
- GUI: Authentication and role-based access control

**Goldstandard Calibration** — the current thresholds and weights are documented defaults based on the FuzzyMatching architecture guide. Before production use, they must be calibrated against a curated ground truth dataset (recommended: 500–1,000 manually verified match/non-match pairs from Bundesbank MFI Register + EBA Register of Institutions) and documented as a versioned configuration.

**Formal Model Validation (MaRisk BTR 3)** — required before production deployment. Includes independent validation, re-validation plan, and sign-off by the model risk function.

---

## 9. Transferability

The governance architecture developed here applies directly to:

**LEI validation against GLEIF database** — the strongest single-column, incomplete-data use case. The Global LEI register is the authoritative reference for legal entity identification in EMIR, MiFID II, and SFTR reporting, but its name data is inconsistent with how entities appear in internal systems and its coverage is incomplete. ERTrace's one-column semantic matching with explicit NO_MATCH scoring (not a silent omission) and full decision audit makes it well-suited as a pre-enrichment step: identify which entities can be resolved with high confidence, flag which require manual LEI lookup, and log every decision for regulatory traceability.

**AML watchlist screening** — with a PII governance wrapper for person names (out of scope for v0.1.0-alpha but architecturally prepared via DAL isolation).

**EMIR counterparty matching** — legal entity names in trade reports vs. internal counterparty master data; legal form scoring surfaces jurisdiction conflicts (e.g., a US Corp. matched against a DE GmbH with the same base name).

**Supplier deduplication** — ERP vs. payment systems; same TGFR logic, same audit trail, same review prioritisation pattern.

Any entity resolution use case in a regulated environment requiring a per-decision audit trail with explicit routing and human-readable score evidence.