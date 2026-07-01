"""Print statistics for the hand-label CSV."""
import statistics
from collections import Counter
from pathlib import Path

from src.utils.io import read_csv_if_exists


def main() -> None:
    path = Path("data/silver/manual_labels_v1.csv")
    rows = read_csv_if_exists(path)

    if not rows:
        print(f"No labels found at {path}")
        return

    scores = Counter()
    lengths = []
    for row in rows:
        score = int(row["rank_score"])
        scores[score] += 1
        lengths.append(len(row.get("rationale", "")))

    print("=== Label statistics ===")
    print(f"Total labeled: {len(rows)}")
    print(f"\nScore distribution:")
    for score in range(6):
        count = scores.get(score, 0)
        print(f"  score {score}: {count:3d} ({count / len(rows) * 100:5.1f}%)")

    print(f"\nMean rationale length: {statistics.mean(lengths):.0f} chars")
    print(f"Min rationale length:  {min(lengths)} chars")
    print(f"Max rationale length:  {max(lengths)} chars")

    warnings = [s for s in range(6) if scores.get(s, 0) < 3]
    if warnings:
        print(f"\nWARNING: score bucket(s) {warnings} have fewer than 3 labels.")
    else:
        print("\nAll score buckets have at least 3 labels.")


if __name__ == "__main__":
    main()
