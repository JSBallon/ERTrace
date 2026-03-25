# Governance Summary: PoC_2 — Entity Resolution

> Regulatory mapping, compliance maturity model, and design rationale
> Audience: Risk & Compliance, Internal Audit, AI Governance, External Reviewers

---

## 1. Core Governance Principle

This system is built on a single architectural premise: **every decision the pipeline makes must be explainable, reproducible, and auditable — without requiring data science expertise to verify it.**

That means:

- No black-box scoring. Every match carries a four-component score vector with named, interpretable dimensions.
- No invisible parameters. Every threshold, weight, and model choice is logged per run with a versioned configuration reference.
- No silent failures. Every guardrail trigger, validation error, and routing override is a typed log event.
- No autonomous action. The pipeline produces structured recommendations. A human makes the final decision on every case in the review list.

---

## 2. What the System Does and Does Not Do

| The system does | The system does not |
|---|---|
| Score every name pair against four explicit criteria | Make autonomous changes to source data |
| Flag legal form conflicts as priority review cases | Resolve ambiguous cases without human input |
| Produce a prioritised review list for the DQM expert | Override the human reviewer's decision |
| Log every decision with full parameter context | Operate without a complete audit trail |
| Support configurable thresholds and weights | Apply hidden or hardcoded decision logic |

---

## 3. Regulatory Framework Mapping

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

## 4. The Score Vector as Audit Instrument

The four-component score vector replaces the "reasoning" field of LLM-based pipelines. Unlike a language model's natural language explanation, the score vector is:

- **Deterministic** — identical inputs and configuration always produce the same scores
- **Mathematically traceable** — the composite score is provably consistent with the component scores and weights
- **Interpretable without data science** — a regulator, auditor, or DQM expert can read "embedding_cosine_score: 0.94, legal_form_relation: conflict, review_priority: 1" and understand the decision

This is the key governance differentiator from standard GenAI pipelines, which require LLM reasoning logs that are harder to verify and reproduce.

---

## 5. The 2D Review Prioritisation as Data Governance Instrument

The review prioritisation is a **data governance instrument** — it structures the quality management process for the master data the system is applied to, not the governance of the AI system itself.

The most important signal is **Priority 1: AUTO_MATCH + legal form conflict.** In banking practice, this pattern arises in two ways:

1. **Front-office translation variant:** A local clerk records "Ltd." for a German GmbH because that is the local equivalent. The match is correct, but the legal form in the source system is wrong and needs correction.
2. **False positive:** Two different entities with similar names in different jurisdictions. The match must be rejected.

Both cases require a human decision. The system correctly identifies them as high-priority without making the decision itself.

---

## 6. Compliance Maturity Model

| Level | Description | Status |
|---|---|---|
| **Level 1 — Basic** | Score vector, routing zones, Pydantic validation, legal form extraction | ✅ Implemented |
| **Level 2 — Operational** | JSONL audit trail, config versioning, review prioritisation, run-ID correlation | ✅ Implemented |
| **Level 3 — Audit-Ready** | Full parameter documentation per run, REVIEW-quote monitoring, reproducibility proof | ✅ Implemented |
| **Level 4 — Production** | Goldstandard calibration, HITL approval gate, formal model validation (BTR 3), PII controls | 🔜 Post-PoC_2 |

**Current level: 3 — Audit-Ready**

The system is fully documented, auditable, and governance-compliant at PoC level. It demonstrates that regulatory-first AI architecture does not require complexity beyond what the use case demands.

---

## 7. What Is Explicitly Out of Scope

The following are documented production requirements, not implemented in PoC_2:

**PII Handling** is a prerequisite for production use, not an optional enhancement. It requires controls at three layers:
- DAL: PII detection before passing data to BLL, encrypted storage, GDPR-compliant deletion
- BLL/Infrastructure: Containerised embedding inference, no PII-bearing vector persistence
- GUI: Authentication and role-based access control

**Goldstandard Calibration** — the current thresholds and weights are documented defaults based on the FuzzyMatching architecture guide. Before production use, they must be calibrated against a curated ground truth dataset (recommended: 500–1,000 manually verified match/non-match pairs from Bundesbank MFI Register + EBA Register of Institutions) and documented as a versioned configuration.

**Formal Model Validation (MaRisk BTR 3)** — required before production deployment. Includes independent validation, re-validation plan, and sign-off by the model risk function.

---

## 8. Transferability

The governance architecture developed here applies directly to:

- LEI validation against GLEIF database
- AML watchlist screening (with PII governance wrapper for person names)
- EMIR counterparty matching
- Supplier deduplication (ERP vs. payment systems)
- Any entity resolution use case in a regulated environment requiring an auditable decision trail