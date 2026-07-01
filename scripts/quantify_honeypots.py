"""Quantify honeypot checks across the full 100K candidate pool.

Run from repo root:
    python -m scripts.quantify_honeypots
"""
import csv
import time
from collections import Counter
from pathlib import Path

from src.utils.integrity import verify_data_integrity
from src.ingestion.loader import iter_candidates
from src.validation.honeypots import run_all_checks, ALL_CHECKS
from src.validation.duplicates import find_duplicate_fingerprints


def main() -> None:
    t0 = time.time()

    print("[1/3] Verifying data integrity...")
    verify_data_integrity()
    print("      OK")

    print("[2/3] Computing duplicate fingerprints (batch pass)...")
    duplicate_ids = find_duplicate_fingerprints(iter_candidates())
    print(f"      {len(duplicate_ids):,} candidates share a fingerprint")

    print("[3/3] Streaming 100K candidates through honeypot checks...")
    check_counts = {name: 0 for name, _ in ALL_CHECKS}
    score_distribution = Counter()
    total = 0

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for c in iter_candidates():
        total += 1
        results = run_all_checks(c, duplicate_ids)
        score = sum(1 for tripped, _ in results.values() if tripped)
        score_distribution[score] += 1
        for name, (tripped, _) in results.items():
            if tripped:
                check_counts[name] += 1

    # Write per-check summary
    summary_path = outputs_dir / "honeypot_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["check_name", "tripped_count", "tripped_pct"])
        for name in [n for n, _ in ALL_CHECKS]:
            writer.writerow([name, check_counts[name], f"{check_counts[name] / total * 100:.2f}"])

    # Write score distribution
    score_path = outputs_dir / "honeypot_score_distribution.csv"
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "count", "pct"])
        for score in range(11):
            count = score_distribution.get(score, 0)
            writer.writerow([score, count, f"{count / total * 100:.2f}"])

    # Print summary table
    print("\n=== Per-check summary ===")
    print(f"{'check_name':<35} {'tripped_count':>14} {'tripped_pct':>12}")
    for name in [n for n, _ in ALL_CHECKS]:
        print(f"{name:<35} {check_counts[name]:>14,} {check_counts[name] / total * 100:>11.2f}%")

    print("\n=== Honeypot score distribution ===")
    print(f"{'score':>5} {'count':>14} {'pct':>12}")
    for score in range(11):
        count = score_distribution.get(score, 0)
        print(f"{score:>5} {count:>14,} {count / total * 100:>11.2f}%")

    print(f"\nTotal candidates: {total:,}")
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"Wrote {summary_path} and {score_path}")


if __name__ == "__main__":
    main()
