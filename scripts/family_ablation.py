"""Ablation: score composite metric with each of 7 feature families zeroed out.

Reuses cached feature matrix and full-population silver scores. Each ablation
takes seconds because we only rerun the weighted-sum scoring, not retrieval.
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from src.evaluation.metrics import mean_average_precision, ndcg_at_k, precision_at_k
from src.ranking.weighted_reranker import WeightedReranker

ROOT = Path(__file__).resolve().parents[1]
FEATURE_MATRIX = ROOT / "data" / "processed" / "feature_matrix.parquet"
SILVER = ROOT / "data" / "processed" / "silver_scores_full.csv"
WEIGHTS_YAML = ROOT / "configs" / "reranker_weights_v1.yaml"
FEATURES_YAML = ROOT / "configs" / "features.yaml"
OUT = ROOT / "outputs" / "family_ablation.csv"


def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_weights() -> dict:
    cfg = _load_yaml(WEIGHTS_YAML)
    return cfg.get("weights_v1", cfg)


def _load_family_map() -> dict[str, list[str]]:
    cfg = _load_yaml(FEATURES_YAML)
    return cfg["feature_families"]


def _load_silver_map() -> dict[str, int]:
    df = pd.read_csv(SILVER)
    return dict(zip(df["candidate_id"], df["silver_score"]))


def _evaluate(scored_df: pd.DataFrame, silver_map: dict[str, int]) -> dict:
    """Rank by 'score' column desc, compute metrics against silver."""
    ranked = scored_df.sort_values("score", ascending=False).reset_index(drop=True)
    gains = [silver_map.get(cid, 0) for cid in ranked["candidate_id"]]
    binary = [1 if g >= 3 else 0 for g in gains]
    top100_silver = gains[:100]
    return {
        "ndcg@10": ndcg_at_k(gains, 10),
        "ndcg@50": ndcg_at_k(gains, 50),
        "map": mean_average_precision(binary),
        "p@10": precision_at_k(binary, 10),
        "mean_silver@100": sum(top100_silver) / 100 if top100_silver else 0.0,
        "composite": (
            0.50 * ndcg_at_k(gains, 10)
            + 0.30 * ndcg_at_k(gains, 50)
            + 0.15 * mean_average_precision(binary)
            + 0.05 * precision_at_k(binary, 10)
        ),
    }


def _score_with_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    reranker = WeightedReranker(weights)
    scored = df.copy()
    scored["score"] = reranker.score(scored)
    return scored


def main() -> None:
    df = pd.read_parquet(FEATURE_MATRIX)
    silver_map = _load_silver_map()
    full_weights = _load_weights()
    family_map = _load_family_map()

    results = []

    print("Running full model...")
    scored_full = _score_with_weights(df, full_weights)
    metrics = _evaluate(scored_full, silver_map)
    metrics["variant"] = "full"
    metrics["removed_family"] = "-"
    metrics["n_weights_zeroed"] = 0
    results.append(metrics)
    full_composite = metrics["composite"]

    for family, features in family_map.items():
        print(f"Ablating {family} ({len(features)} features)...")
        ablated_weights = {
            k: (0.0 if k in features else v) for k, v in full_weights.items()
        }
        n_zeroed = sum(1 for k in features if k in full_weights)
        scored = _score_with_weights(df, ablated_weights)
        m = _evaluate(scored, silver_map)
        m["variant"] = f"without_{family}"
        m["removed_family"] = family
        m["n_weights_zeroed"] = n_zeroed
        results.append(m)

    for r in results:
        r["delta_composite_from_full"] = r["composite"] - full_composite

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "removed_family",
        "n_weights_zeroed",
        "ndcg@10",
        "ndcg@50",
        "map",
        "p@10",
        "mean_silver@100",
        "composite",
        "delta_composite_from_full",
    ]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote {OUT}\n")

    print(f"{'variant':<28} {'composite':>10} {'delta':>10} {'ndcg@10':>10} {'mean_silver@100':>16}")
    print("-" * 78)
    sorted_results = sorted(results, key=lambda r: r["composite"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['variant']:<28} {r['composite']:>10.4f} "
            f"{r['delta_composite_from_full']:>+10.4f} {r['ndcg@10']:>10.4f} "
            f"{r['mean_silver@100']:>16.2f}"
        )

    print("\nFamilies ranked by IMPACT (largest composite drop when removed):")
    impacts = sorted(
        [r for r in results if r["variant"] != "full"],
        key=lambda r: r["delta_composite_from_full"],
    )
    for r in impacts:
        print(f"  {r['removed_family']:<24}  drop = {-r['delta_composite_from_full']:.4f}")


if __name__ == "__main__":
    main()
