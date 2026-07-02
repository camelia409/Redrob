"""Failure analysis: which top-100 candidates got the lowest silver score,
and what features carried them there?

This is standard error analysis practice. Any senior reviewer will ask:
'where does your model make its worst decisions, and do you understand why?'
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUB = ROOT / "outputs" / "final_submission.csv"
SILVER = ROOT / "data" / "processed" / "silver_scores_full.csv"
FEATURES = ROOT / "data" / "processed" / "feature_matrix.parquet"


def main() -> None:
    sub = pd.read_csv(FINAL_SUB)
    silver = pd.read_csv(SILVER)
    fm = pd.read_parquet(FEATURES)

    silver_map = dict(zip(silver["candidate_id"], silver["silver_score"]))
    sub["silver_score"] = sub["candidate_id"].map(silver_map).fillna(0).astype(int)

    # Merge features
    merged = sub.merge(fm, on="candidate_id", how="left", suffixes=("", "_feat"))

    # Take the 10 lowest silver scores in top-100, breaking ties by rank (later = weaker)
    worst = (
        merged.sort_values(["silver_score", "rank"], ascending=[True, False])
        .head(10)
        .reset_index(drop=True)
    )

    print("=" * 80)
    print("FAILURE ANALYSIS: 10 lowest-silver candidates in the top-100")
    print("=" * 80)

    for i, row in worst.iterrows():
        cid = row["candidate_id"]
        print(
            f"\n[{i+1}] {cid}  rank={int(row['rank'])}  "
            f"silver={int(row['silver_score'])}  "
            f"reranker_score={float(row['score']):.4f}"
        )
        print(f"    Reasoning: {row.get('reasoning', '')[:120]}")
        # Which features are highest for this candidate?
        feat_cols = [
            c
            for c in fm.columns
            if c
            not in (
                "candidate_id",
                "silver_score",
                "honeypot_score",
                "bm25_score",
                "dense_v2_score",
            )
        ]
        vals = row[feat_cols].astype(float)
        top_feats = vals.sort_values(ascending=False).head(6)
        print("    Top features (values):")
        for f, v in top_feats.items():
            print(f"      {f:<40}  {v:.3f}")
        print(
            f"    Aggregate: bm25={float(row.get('bm25_score', float('nan'))):.4f}  "
            f"dense={float(row.get('dense_v2_score', float('nan'))):.4f}  "
            f"hp={int(row.get('honeypot_score', 0))}"
        )

    # Cross-candidate patterns
    print("\n" + "=" * 80)
    print("CROSS-CANDIDATE PATTERNS across the worst 10")
    print("=" * 80)
    numeric_cols = [c for c in fm.columns if c not in ("candidate_id",)]
    worst_numeric = worst[[c for c in numeric_cols if c in worst.columns]].astype(float)
    means_worst = worst_numeric.mean()
    means_top100 = (
        merged[[c for c in numeric_cols if c in merged.columns]].astype(float).mean()
    )

    print(f"\n{'feature':<42} {'worst-10 mean':>16} {'top-100 mean':>16} {'delta':>12}")
    print("-" * 90)
    for f in means_worst.index:
        if f in means_top100.index:
            wm = means_worst[f]
            tm = means_top100[f]
            delta = wm - tm
            marker = " *" if abs(delta) > 0.1 else ""
            print(f"{f:<42} {wm:>16.3f} {tm:>16.3f} {delta:>+12.3f}{marker}")


if __name__ == "__main__":
    main()
