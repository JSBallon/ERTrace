# run_faker.py — temporary runner, not committed
import csv, pathlib, tempfile, os
from dal.data_generator import FakerDataGenerator
from bll.app_service import run_entity_resolution

N_A = 20   # Source A entries (CRM)
N_B = 30   # Source B entries (Core Banking)
SEED = 42

gen = FakerDataGenerator(seed=SEED)
records_a = gen.generate_company_list(N_A, "de")
records_b = gen.generate_company_list(N_B, "de")

# Write temp CSVs
def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "source_name"])
        writer.writeheader()
        writer.writerows(records)

write_csv("temp_a.csv", records_a)
write_csv("temp_b.csv", records_b)

print(f"Running pipeline: {N_A} Source A × {N_B} Source B entries...")
results, summary = run_entity_resolution("temp_a.csv", "temp_b.csv")

print(f"\n--- Run Summary ---")
print(f"Run ID:       {summary.run_id}")
print(f"Total:        {summary.total_entries_a}")
print(f"AUTO_MATCH:   {summary.count_auto_match}  ({summary.auto_match_quote:.0%})")
print(f"REVIEW:       {summary.count_review}  ({summary.review_quote:.0%})")
print(f"NO_MATCH:     {summary.count_no_match}  ({summary.no_match_quote:.0%})")
print(f"Output:       {summary.output_file_path}")
print(f"Review:       {summary.review_file_path}")
print(f"Audit log:    {summary.audit_log_path}")

# Cleanup temp files
os.remove("temp_a.csv")
os.remove("temp_b.csv")
