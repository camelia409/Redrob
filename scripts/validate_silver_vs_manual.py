"""Compare silver labels against hand labels and report agreement."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error

from src.utils.paths import SILVER


MANUAL_PATH = SILVER / "manual_labels_v1.csv"
SILVER_PATH = SILVER / "silver_labels_v1.csv"


def main() -> int:
    if not MANUAL_PATH.exists():
        print(f"Manual labels not found: {MANUAL_PATH}")
        return 1
    if not SILVER_PATH.exists():
        print(f"Silver labels not found: {SILVER_PATH}")
        return 1

    manual = pd.read_csv(MANUAL_PATH)
    silver = pd.read_csv(SILVER_PATH)

    manual = manual.rename(columns={"rank_score": "manual_score"})[
        ["candidate_id", "manual_score", "rationale"]
    ]

    merged = pd.merge(manual, silver, on="candidate_id", how="inner")
    n = len(merged)
    print(f"Candidates in both manual and silver sets: {n}")
    if n == 0:
        print("No overlap; cannot validate.")
        return 1

    exact = (merged["manual_score"] == merged["silver_score"]).sum()
    within_1 = (merged["manual_score"] - merged["silver_score"]).abs().le(1).sum()
    mae = mean_absolute_error(merged["manual_score"], merged["silver_score"])

    print("\nAgreement:")
    print(f"  Exact match: {exact}/{n} ({100 * exact / n:.1f}%)")
    print(f"  Within-1:    {within_1}/{n} ({100 * within_1 / n:.1f}%)")
    print(f"  MAE:         {mae:.2f}")

    labels = list(range(6))
    cm = confusion_matrix(
        merged["manual_score"], merged["silver_score"], labels=labels
    )
    cm_df = pd.DataFrame(cm, index=[f"manual={i}" for i in labels], columns=[f"silver={i}" for i in labels])
    print("\nConfusion matrix (rows = manual, columns = silver):")
    print(cm_df.to_string())

    print("\nPer-candidate comparison:")
    merged["delta"] = merged["silver_score"] - merged["manual_score"]
    display = merged.sort_values("delta", key=lambda s: s.abs(), ascending=False)[
        ["candidate_id", "manual_score", "silver_score", "delta", "top_evidence", "rationale"]
    ]
    for row in display.itertuples(index=False):
        print(
            f"{row.candidate_id}: manual={row.manual_score} silver={row.silver_score} "
            f"delta={row.delta:+d} | {row.top_evidence}"
        )

    within_1_rate = within_1 / n
    print("\n" + "=" * 50)
    if within_1_rate >= 0.60:
        print(f"PASS: Within-1 rate {100 * within_1_rate:.1f}% >= 60%")
        return 0
    else:
        print(f"FAIL: Within-1 rate {100 * within_1_rate:.1f}% < 60%")
        print("Do not commit silver labels. Tune configs/rubric_v1.yaml and re-run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
