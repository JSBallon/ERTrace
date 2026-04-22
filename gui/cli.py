"""
GUI — CLI Entry Point

Command-line interface for the PoC2 Entity Resolution Pipeline.

Four mutually exclusive modes (ADR-M3-007):

  Explicit    python -m gui.cli --source-a A.csv --source-b B.csv
  Latest      python -m gui.cli --latest
  Generate    python -m gui.cli --generate [--M 100 --N 120 --overlap 80 --noise 30 --typo 30]
  Gen-only    python -m gui.cli --generate-only [--M 100 --N 120 --overlap 80 --noise 30 --typo 30]

Layering rules (ADR-M3-003b):
  - All pipeline execution goes through bll.app_service.run_entity_resolution()
  - All Faker generation goes through dal.data_generator.generate_paired_datasets()
  - No Streamlit imports, no direct DAL or BLL scoring logic
  - This file is the sole owner of argparse, sys.exit, and print logic

Audit equivalence:
  All modes that run the pipeline produce the same run_start / run_end JSONL events
  as the Streamlit UI — the governance record is entry-point-agnostic.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INPUTS_DIR      = Path("inputs")
_PROGRESS_EVERY  = 10          # print progress every N entries
_SEP             = "-" * 57   # horizontal separator line


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_latest_inputs(inputs_dir: Path) -> tuple[str, str]:
    """
    Find the most recently modified source_a_* and source_b_* CSV files.

    Each source type is resolved independently by mtime — the pair may not
    share the same timestamp if multiple generation runs exist. If exact
    pairing is required use --source-a / --source-b explicitly.

    Args:
        inputs_dir: Directory to search (default: inputs/).

    Returns:
        (path_a, path_b) as strings.

    Raises:
        FileNotFoundError: If no matching files exist.
    """
    files_a = sorted(
        inputs_dir.glob("source_a_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    files_b = sorted(
        inputs_dir.glob("source_b_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files_a:
        raise FileNotFoundError(
            f"No source_a_*.csv files found in {inputs_dir}/. "
            "Run --generate-only or --generate first."
        )
    if not files_b:
        raise FileNotFoundError(
            f"No source_b_*.csv files found in {inputs_dir}/. "
            "Run --generate-only or --generate first."
        )
    return str(files_a[0]), str(files_b[0])


def _write_csv(records: list[dict], path: Path) -> None:
    """Write a list of {source_id, source_name} dicts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "source_name"])
        writer.writeheader()
        writer.writerows(records)


def _generate_faker_files(
    m: int,
    n: int,
    overlap: int,
    noise: int,
    typo: int,
) -> tuple[str, str, int]:
    """
    Generate paired Faker datasets and write to inputs/.

    Args:
        m:       Source A entry count.
        n:       Source B entry count.
        overlap: Overlap percentage (0–100).
        noise:   Noise rate percentage (0–100).
        typo:    Typo rate percentage (0–100).

    Returns:
        (path_a, path_b, k_shared) — paths as strings, k_shared as int.
    """
    from dal.data_generator import FakerDataGenerator

    gen = FakerDataGenerator()
    records_a, records_b = gen.generate_paired_datasets(
        n_a=m,
        n_b=n,
        overlap_pct=overlap / 100,
        noise_rate=noise / 100,
        typo_rate=typo / 100,
    )

    k = min(int(m * overlap / 100), n)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    path_a = _INPUTS_DIR / f"source_a_{ts}.csv"
    path_b = _INPUTS_DIR / f"source_b_{ts}.csv"

    _write_csv(records_a, path_a)
    _write_csv(records_b, path_b)

    return str(path_a), str(path_b), k


def _make_progress_callback(total: int):
    """
    Return a progress callback that prints every _PROGRESS_EVERY entries.
    Prints a final line when complete.
    """
    def _cb(completed: int, n: int) -> None:
        if completed % _PROGRESS_EVERY == 0 or completed == n:
            print(f"  Processing {completed:>4} / {n}...")
    return _cb


def _count_csv_rows(path: str) -> int:
    """Count data rows (excluding header) in a CSV file."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def _print_run_summary(summary, path_a: str, path_b: str) -> None:
    """Print the run summary to stdout. REVIEW rate warning to stderr."""
    n_a = _count_csv_rows(path_a)
    n_b = _count_csv_rows(path_b)

    print(_SEP)
    print("PoC2 Entity Resolution — Run Complete")
    print(_SEP)
    print(f"Run ID:       {summary.run_id}")
    print(f"Source A:     {path_a}  ({n_a} entries)")
    print(f"Source B:     {path_b}  ({n_b} entries)")
    print(_SEP)
    print(f"AUTO_MATCH:   {summary.count_auto_match:>5}   ({summary.auto_match_quote:.1%})")
    print(f"REVIEW:       {summary.count_review:>5}   ({summary.review_quote:.1%})")
    print(f"NO_MATCH:     {summary.count_no_match:>5}   ({summary.no_match_quote:.1%})")
    print(f"ERRORS:       {summary.count_error:>5}")
    print(_SEP)
    print(f"Output JSON:  {summary.output_file_path}")
    print(f"Review JSON:  {summary.review_file_path}")
    print(f"Audit JSONL:  {summary.audit_log_path}")
    print(_SEP)

    if summary.review_quote_warning:
        print(
            f"[WARNING] REVIEW rate {summary.review_quote:.1%} exceeds "
            "the warning threshold (30%). Consider recalibrating thresholds.",
            file=sys.stderr,
        )


def _print_generate_summary(path_a: str, path_b: str, m: int, n: int, k: int) -> None:
    """Print the Faker generation summary to stdout."""
    print(_SEP)
    print("PoC2 Faker Data Generation — Complete")
    print(_SEP)
    print(f"Source A:   {path_a}  ({m} entries)")
    print(f"Source B:   {path_b}  ({n} entries)")
    print(f"Shared:     {k} entries  |  A-only: {m-k}  |  B-only: {n-k}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gui.cli",
        description="PoC2 Entity Resolution Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (exactly one required):
  --source-a A --source-b B   Run pipeline on specified CSV/JSON files
  --latest                    Run pipeline on latest source_a_* / source_b_* in inputs/
  --generate                  Generate Faker datasets, then run pipeline
  --generate-only             Generate Faker datasets and exit (no pipeline run)

Examples:
  python -m gui.cli --source-a data/crm.csv --source-b data/core.csv
  python -m gui.cli --latest
  python -m gui.cli --generate --M 100 --N 120 --overlap 80 --noise 30 --typo 30
  python -m gui.cli --generate-only --M 50 --N 60 --overlap 70
        """,
    )

    # --- Input source (explicit paths) ---
    parser.add_argument(
        "--source-a", metavar="PATH",
        help="Path to Source A CSV or JSON file.",
    )
    parser.add_argument(
        "--source-b", metavar="PATH",
        help="Path to Source B CSV or JSON file.",
    )

    # --- Input source (auto-resolve) ---
    parser.add_argument(
        "--latest", action="store_true",
        help="Load the most recently generated source_a_* and source_b_* from inputs/.",
    )

    # --- Generation modes ---
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate Faker datasets then run the pipeline.",
    )
    parser.add_argument(
        "--generate-only", dest="generate_only", action="store_true",
        help="Generate Faker datasets and exit without running the pipeline.",
    )

    # --- Generation parameters ---
    gen_group = parser.add_argument_group("Faker generation parameters")
    gen_group.add_argument("--M", type=int, default=100, metavar="N",
                           help="Source A entry count (default: 100).")
    gen_group.add_argument("--N", type=int, default=120, metavar="N",
                           help="Source B entry count (default: 120).")
    gen_group.add_argument("--overlap", type=int, default=80, metavar="%",
                           help="Intended overlap %% (0–100, default: 80).")
    gen_group.add_argument("--noise", type=int, default=30, metavar="%",
                           help="Noise rate %% (0–100, default: 30).")
    gen_group.add_argument("--typo", type=int, default=30, metavar="%",
                           help="Typo rate %% (0–100, default: 30).")

    # --- Config ---
    parser.add_argument(
        "--config", default="config/config.yaml", metavar="PATH",
        help="Path to config/config.yaml (default: config/config.yaml).",
    )

    return parser


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Post-parse mutual exclusion and consistency checks."""
    modes_active = sum([
        bool(args.source_a or args.source_b),
        args.latest,
        args.generate,
        args.generate_only,
    ])
    if modes_active == 0:
        parser.error(
            "Specify exactly one mode: "
            "--source-a/--source-b, --latest, --generate, or --generate-only."
        )
    if modes_active > 1:
        parser.error(
            "Modes are mutually exclusive. Specify exactly one of: "
            "--source-a/--source-b, --latest, --generate, --generate-only."
        )
    if bool(args.source_a) != bool(args.source_b):
        parser.error("--source-a and --source-b must be specified together.")

    gen_params_used = any([
        args.M != 100, args.N != 120,
        args.overlap != 80, args.noise != 30, args.typo != 30,
    ])
    if gen_params_used and not (args.generate or args.generate_only):
        parser.error(
            "--M, --N, --overlap, --noise, --typo are only valid with "
            "--generate or --generate-only."
        )

    for name, val in [("--M", args.M), ("--N", args.N),
                      ("--overlap", args.overlap),
                      ("--noise", args.noise), ("--typo", args.typo)]:
        if not (0 <= val <= (500 if name in ("--M", "--N") else 100)):
            parser.error(f"{name} must be between 0 and "
                         f"{'500' if name in ('--M', '--N') else '100'}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    _validate_args(args, parser)

    # ------------------------------------------------------------------
    # Mode: generate-only
    # ------------------------------------------------------------------
    if args.generate_only:
        print(f"Generating Faker datasets  (M={args.M}, N={args.N}, "
              f"overlap={args.overlap}%, noise={args.noise}%, typo={args.typo}%)...")
        try:
            path_a, path_b, k = _generate_faker_files(
                args.M, args.N, args.overlap, args.noise, args.typo
            )
        except Exception as exc:
            print(f"[ERROR] Data generation failed: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_generate_summary(path_a, path_b, args.M, args.N, k)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Resolve input file paths for all run modes
    # ------------------------------------------------------------------
    path_a: str
    path_b: str

    if args.generate:
        print(f"Generating Faker datasets  (M={args.M}, N={args.N}, "
              f"overlap={args.overlap}%, noise={args.noise}%, typo={args.typo}%)...")
        try:
            path_a, path_b, k = _generate_faker_files(
                args.M, args.N, args.overlap, args.noise, args.typo
            )
        except Exception as exc:
            print(f"[ERROR] Data generation failed: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_generate_summary(path_a, path_b, args.M, args.N, k)

    elif args.latest:
        try:
            path_a, path_b = _find_latest_inputs(_INPUTS_DIR)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Latest Source A: {path_a}")
        print(f"Latest Source B: {path_b}")

    else:
        # Explicit --source-a / --source-b
        path_a = args.source_a
        path_b = args.source_b

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    from bll.app_service import run_entity_resolution

    n_total_approx = _count_csv_rows(path_a)
    print(f"\nRunning pipeline on {n_total_approx} Source A entries...")

    progress_cb = _make_progress_callback(n_total_approx)

    try:
        results, summary = run_entity_resolution(
            source_a_path=path_a,
            source_b_path=path_b,
            config_path=args.config,
            progress_callback=progress_cb,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] Input file not found: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Pipeline aborted: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_run_summary(summary, path_a, path_b)
    sys.exit(0)


if __name__ == "__main__":
    main()
