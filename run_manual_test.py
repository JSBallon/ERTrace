# run_manual_test.py — manual demo runner (Faker data, German banking names)
import csv
import os
from pathlib import Path
from dal.data_generator import FakerDataGenerator
from bll.app_service import run_entity_resolution

N_A  = 20   # Source A entries (CRM)
N_B  = 30   # Source B entries (Core Banking)
SEED = 42

gen = FakerDataGenerator(seed=SEED)

# Both sources use German banking names (language="de").
# Source B source_id prefix is rewritten from "crm_" → "src_" so
# CRM entries (crm_0001…) and Core Banking entries (src_0001…) are
# visually distinct in the output JSON without altering the name pool.
records_a = gen.generate_company_list(N_A, "de")
records_b = [
    {**r, "source_id": r["source_id"].replace("crm_", "src_", 1)}
    for r in gen.generate_company_list(N_B, "de")
]

# Write input CSVs (kept on disk after the run for inspection)
def write_csv(path: str, records: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "source_name"])
        writer.writeheader()
        writer.writerows(records)

write_csv("temp_a.csv", records_a)
write_csv("temp_b.csv", records_b)

print(f"Running pipeline: {N_A} Source A (crm_*) × {N_B} Source B (src_*) entries...")
results, summary = run_entity_resolution("temp_a.csv", "temp_b.csv")

print(f"\n--- Run Summary ---")
print(f"Run ID:              {summary.run_id}")
print(f"Total entries:       {summary.total_entries_a}")
print(f"AUTO_MATCH:          {summary.count_auto_match}  ({summary.auto_match_quote:.0%})")
print(f"REVIEW:              {summary.count_review}  ({summary.review_quote:.0%})")
print(f"NO_MATCH:            {summary.count_no_match}  ({summary.no_match_quote:.0%})")
print(f"Total rerank cands:  {summary.total_rerank_candidates}")
if summary.review_quote_warning:
    print(f"  ⚠  REVIEW rate > 30% — threshold calibration recommended")

print(f"\n--- Output Files ---")
print(f"Output JSON:  {summary.output_file_path}")
print(f"Review JSON:  {summary.review_file_path}")
print(f"Audit JSONL:  {summary.audit_log_path}")
print(f"\nOutput folder: {Path(summary.output_file_path).parent.resolve()}")
print(f"  → Drag output_*.json or review_*.json into a JSON viewer for inspection")
print(f"\nInput CSVs (kept for inspection):")
print(f"  temp_a.csv  ({N_A} CRM entries,          source_id prefix: crm_)")
print(f"  temp_b.csv  ({N_B} Core Banking entries, source_id prefix: src_)")

# --- Cleanup ---
# Comment out the lines below to keep temp CSVs for demo inspection.
# os.remove("temp_a.csv")
# os.remove("temp_b.csv")
