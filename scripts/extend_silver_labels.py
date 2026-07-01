"""Extend 30 hand labels to ~500 silver labels using the deterministic rubric scorer.

Streams the full 100K, scores every candidate, then reservoir-samples up to 100
per score bucket (0-4) with seed=42.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from src.evaluation.rubric_scorer import score_candidate
from src.utils.io import iter_jsonl
from src.utils.paths import CHALLENGE, SILVER
from src.utils.integrity import verify_data_integrity
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


CANDIDATES_PATH = CHALLENGE / "candidates.jsonl"
MANUAL_PATH = SILVER / "manual_labels_v1.csv"
OUTPUT_PATH = SILVER / "silver_labels_v1.csv"
TARGET_PER_BUCKET = 100
BUCKETS = [0, 1, 2, 3, 4]
SEED = 42


def _load_manual_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("candidate_id")
            if cid:
                ids.add(cid)
    return ids


def _reservoir_sample(items: List[Dict], k: int, rng: random.Random) -> List[Dict]:
    """Reservoir sample up to k items deterministically."""
    if len(items) <= k:
        return items
    sample = items[:k]
    for i, item in enumerate(items[k:], start=k):
        j = rng.randrange(i + 1)
        if j < k:
            sample[j] = item
    return sample


def main() -> None:
    print("Verifying staged data integrity...")
    verify_data_integrity()
    print("Integrity OK.")

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Candidate pool not found: {CANDIDATES_PATH}")

    print("Computing duplicate fingerprints (title|company|YoE bucket|skills)...")
    duplicate_ids: Set[str] = find_duplicate_fingerprints(iter_jsonl(CANDIDATES_PATH))
    print(f"Found {len(duplicate_ids):,} candidates sharing a fingerprint.")

    buckets: Dict[int, List[Dict]] = defaultdict(list)
    processed = 0

    manual_ids = _load_manual_ids(MANUAL_PATH)
    if manual_ids:
        print(f"Loaded {len(manual_ids)} manual-label IDs for forced inclusion.")

    print("Scoring 100K candidates...")
    for candidate in iter_jsonl(CANDIDATES_PATH):
        cid = candidate.get("candidate_id")
        if cid is None:
            continue
        honeypot_val = honeypot_score_gates_only(candidate, duplicate_ids)
        silver_score, evidence = score_candidate(candidate, honeypot_val)
        buckets[silver_score].append(
            {
                "candidate_id": cid,
                "silver_score": silver_score,
                "top_evidence": " | ".join(evidence[:3]),
                "honeypot_score": honeypot_val,
            }
        )
        processed += 1
        if processed % 20_000 == 0:
            print(f"  ...{processed:,} scored")

    print("\nRaw silver score distribution:")
    for score in range(6):
        print(f"  score {score}: {len(buckets[score]):,}")

    rng = random.Random(SEED)
    selected: List[Dict] = []
    for score in BUCKETS:
        pool = buckets[score]
        manual_in_bucket = [r for r in pool if r["candidate_id"] in manual_ids]
        manual_cids = {r["candidate_id"] for r in manual_in_bucket}
        non_manual = [r for r in pool if r["candidate_id"] not in manual_cids]

        sample_size = min(TARGET_PER_BUCKET - len(manual_in_bucket), len(non_manual))
        if sample_size < 0:
            sample_size = 0
        sample = _reservoir_sample(non_manual, sample_size, rng)
        bucket_selected = manual_in_bucket + sample
        selected.extend(bucket_selected)
        print(
            f"Bucket {score}: selected {len(bucket_selected):,} "
            f"({len(manual_in_bucket)} manual + {len(sample)} sampled) / {len(pool):,}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "silver_score", "top_evidence", "honeypot_score"],
        )
        writer.writeheader()
        writer.writerows(selected)

    print(f"\nWrote {len(selected):,} silver labels to {OUTPUT_PATH}")

    # Full-population distribution + honeypot correlation.
    dist_path = OUTPUT_PATH.parent / "silver_score_distribution_full.csv"
    with open(dist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["silver_score", "count", "mean_honeypot_score"]
        )
        writer.writeheader()
        for score in range(6):
            bucket = buckets.get(score, [])
            count = len(bucket)
            mean_hp = (
                sum(r["honeypot_score"] for r in bucket) / count if count else 0.0
            )
            writer.writerow(
                {
                    "silver_score": score,
                    "count": count,
                    "mean_honeypot_score": round(mean_hp, 3),
                }
            )
    print(f"Wrote full distribution to {dist_path}")


if __name__ == "__main__":
    main()
