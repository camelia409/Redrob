"""Evaluate submission_v3 vs submission_v4 against manual labels."""
import csv
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import mean_average_precision, ndcg_at_k, precision_at_k

ROOT = Path(__file__).resolve().parents[1]
V3_CSV = ROOT / "outputs" / "submission_v3.csv"
V4_CSV = ROOT / "outputs" / "submission_v4.csv"
LABELS_V1 = ROOT / "data" / "silver" / "manual_labels_v1.csv"
LABELS_UNION = ROOT / "data" / "silver" / "manual_labels_top100_union.csv"


def load_all_labels() -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in [LABELS_V1, LABELS_UNION]:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("candidate_id", "").strip()
                if not cid:
                    continue
                score_field = (
                    row.get("score") or row.get("rank_score") or row.get("manual_score")
                )
                try:
                    labels[cid] = int(float(score_field))
                except (TypeError, ValueError):
                    continue
    return labels


def evaluate(csv_path: Path, labels: dict[str, int]) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    df = df.sort_values("rank").reset_index(drop=True)
    top100_ids = df["candidate_id"].head(100).tolist()
    gains = [labels.get(cid, 0) for cid in top100_ids]
    binary = [1 if g >= 3 else 0 for g in gains]

    labeled_top100 = sum(1 for cid in top100_ids if cid in labels)

    return {
        "ndcg@10": ndcg_at_k(gains, 10),
        "ndcg@50": ndcg_at_k(gains, 50),
        "ndcg@100": ndcg_at_k(gains, 100),
        "p@10": precision_at_k(binary, 10),
        "map": mean_average_precision(binary),
        "mean_manual@10": sum(gains[:10]) / 10 if gains else 0.0,
        "mean_manual@50": sum(gains[:50]) / 50 if len(gains) >= 50 else 0.0,
        "mean_manual@100": sum(gains[:100]) / 100 if len(gains) >= 100 else 0.0,
        "labeled_coverage@100": labeled_top100,
    }


def main() -> None:
    labels = load_all_labels()
    print(f"Total labels loaded: {len(labels)}")

    v3 = evaluate(V3_CSV, labels)
    v4 = evaluate(V4_CSV, labels)

    print("\n" + "=" * 78)
    print(
        f"{'Metric':<25} {'v3 (frozen)':<18} {'v4 (cross-encoder)':<20} {'delta':<10}"
    )
    print("-" * 78)
    for k in [
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "p@10",
        "map",
        "mean_manual@10",
        "mean_manual@50",
        "mean_manual@100",
        "labeled_coverage@100",
    ]:
        delta = v4[k] - v3[k]
        marker = " *" if abs(delta) > 0.005 else ""
        print(f"{k:<25} {v3[k]:<18.4f} {v4[k]:<20.4f} {delta:+.4f}{marker}")
    print("=" * 78)
    print("\nDecision rule:")
    print(
        "  - Ship v4 if:  ndcg@10(v4) > ndcg@10(v3)  AND  mean_manual@10(v4) >= mean_manual@10(v3)"
    )
    print("  - Keep v3 if:  either condition fails")
    print()
    ship_v4 = v4["ndcg@10"] > v3["ndcg@10"] and v4["mean_manual@10"] >= v3["mean_manual@10"]
    print(f"Applied: {'SHIP v4' if ship_v4 else 'KEEP v3'}")


if __name__ == "__main__":
    main()
