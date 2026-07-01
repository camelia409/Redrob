"""Precompute silver scores for the full 100K candidate pool.

This is offline precomputation.  The output CSV is used by the ablation harness
so that ranking scripts do not need to run the rubric scorer at ranking time.
"""
import csv
import time
from pathlib import Path

from src.evaluation.rubric_scorer import score_candidate
from src.ingestion.loader import iter_candidates
from src.utils.paths import PROCESSED
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


OUTPUT_PATH = PROCESSED / "silver_scores_full.csv"


def main() -> None:
    print("Loading candidate stream...")
    candidates = list(iter_candidates())
    print(f"Loaded {len(candidates):,} candidates")

    print("Computing duplicate fingerprints...")
    duplicate_ids = find_duplicate_fingerprints(iter(candidates))
    print(f"Found {len(duplicate_ids):,} duplicate fingerprints")

    print("Scoring full population (this may take ~60s)...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "silver_score"])
        for i, c in enumerate(candidates, start=1):
            cid = c.get("candidate_id")
            honeypot_val = honeypot_score_gates_only(c, duplicate_ids)
            silver_score, _ = score_candidate(c, honeypot_val)
            writer.writerow([cid, silver_score])
            if i % 20_000 == 0:
                print(f"  ...{i:,} scored")
    elapsed = time.time() - t0
    print(f"\nWrote {OUTPUT_PATH} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
