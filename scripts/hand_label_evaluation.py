"""Evaluate rankers or a submission CSV against the 30 human hand labels."""
import argparse
import csv
from pathlib import Path

import pandas as pd

from src.evaluation.harness import evaluate_full_population, evaluate_labeled_subset
from src.evaluation.metrics import ndcg_at_k
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRanker, DenseRankerV2
from src.ranking.hybrid_ranker import HybridRRFRanker
from src.ranking.weighted_reranker import WeightedReranker
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import CONFIGS, PROCESSED, SILVER
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only

# Reuse the weighted-ranker construction from the v3 ablation script.
from scripts.run_ablation_v3 import _load_weights, _weighted_ranker_output

HP_PENALTY = -1e6

MANUAL_LABELS_PATH = SILVER / "manual_labels_v1.csv"
FEATURE_MATRIX_PATH = PROCESSED / "feature_matrix.parquet"
WEIGHTS_PATH = CONFIGS / "reranker_weights_v1.yaml"
FULL_SCORES_PATH = PROCESSED / "silver_scores_full.csv"


def _load_csv_scores(path: Path) -> dict[str, int]:
    scores = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["candidate_id"]] = int(row["silver_score"])
    return scores


def _load_manual_labels(path: Path) -> dict[str, int]:
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["candidate_id"]] = int(row["rank_score"])
    return labels


def _composite(metrics: dict) -> float:
    """Predicted challenge composite used for ranker selection."""
    return (
        0.50 * metrics["ndcg@10"]
        + 0.30 * metrics["ndcg@50"]
        + 0.15 * metrics["map"]
        + 0.05 * metrics["p@10"]
    )


def _metrics_for_ranker(ranker_output: list[tuple[str, float]], labels: dict[str, int]):
    filtered = [(cid, score) for cid, score in ranker_output if cid in labels]
    gains = [float(labels[cid]) for cid, _ in filtered]
    return {
        "ndcg@10": ndcg_at_k(gains, 10),
        "ndcg@30": ndcg_at_k(gains, 30),
        "mean_manual@10": sum(labels[cid] for cid, _ in filtered[:10]) / 10 if filtered else 0.0,
    }


def evaluate_submission_csv(csv_path: Path, all_candidate_ids: list[str]) -> dict:
    """Evaluate a top-100 submission CSV vs hand labels and silver scores."""
    labels = _load_manual_labels(MANUAL_LABELS_PATH)
    full_scores = _load_csv_scores(FULL_SCORES_PATH)

    candidates = list(iter_candidates())
    dup_ids = find_duplicate_fingerprints(iter(candidates))
    honeypot_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, dup_ids) for c in candidates
    }

    df = pd.read_csv(csv_path)
    top100 = list(zip(df["candidate_id"], df["score"]))
    top100_ids = {cid for cid, _ in top100}
    remaining = [(cid, 0.0) for cid in all_candidate_ids if cid not in top100_ids]
    ranker_output = top100 + remaining

    full_metrics = evaluate_full_population(
        ranker_output, full_scores, honeypot_scores, k_values=[10, 50, 100]
    )
    labeled_metrics = _metrics_for_ranker(ranker_output, labels)
    composite = _composite(full_metrics)

    return {
        "composite": composite,
        **full_metrics,
        **labeled_metrics,
    }


def run_ranker_comparison() -> None:
    print("Loading 100K candidates and manual labels...")
    candidates = list(iter_candidates())
    all_candidate_ids = [c["candidate_id"] for c in candidates]
    labels = _load_manual_labels(MANUAL_LABELS_PATH)
    print(f"  {len(candidates):,} candidates; {len(labels)} hand labels")

    print("Loading embedding index and feature matrix...")
    index = EmbeddingIndex()
    index.load()
    df_features = pd.read_parquet(FEATURE_MATRIX_PATH).copy()
    weights = _load_weights(WEIGHTS_PATH)
    reranker = WeightedReranker(weights, score_col="weighted_score")
    df_features["weighted_score"] = reranker.score(df_features)
    df_features["weighted_score_gated"] = df_features["weighted_score"].copy()
    df_features.loc[df_features["honeypot_score"] >= 3, "weighted_score_gated"] += HP_PENALTY

    print("Running rankers...\n")
    jd_text = get_jd_text()
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    dense_v2_ranker = DenseRankerV2(index=index)
    hybrid_v2_ranker = HybridRRFRanker(bm25_ranker, dense_v2_ranker, k=60, per_ranker_top=2000)

    rankers = [
        ("bm25", bm25_ranker.rank(candidates)),
        ("dense_v2", dense_v2_ranker.rank(candidates)),
        ("hybrid_v2_rrf", hybrid_v2_ranker.rank(candidates)),
        (
            "weighted_reranker_top100",
            _weighted_ranker_output(df_features, gated=True, all_candidate_ids=all_candidate_ids),
        ),
    ]

    results = []
    for name, out in rankers:
        metrics = _metrics_for_ranker(out, labels)
        results.append({"ranker": name, **metrics})
        print(
            f"{name:>30s}  "
            f"NDCG@10={metrics['ndcg@10']:.3f}  "
            f"NDCG@30={metrics['ndcg@30']:.3f}  "
            f"mean_manual@10={metrics['mean_manual@10']:.2f}"
        )

    print("\nHand-label evaluation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rankers or a submission CSV against hand labels."
    )
    parser.add_argument(
        "--submission",
        type=Path,
        help="Path to a top-100 submission CSV to evaluate (instead of running rankers).",
    )
    args = parser.parse_args()

    if args.submission:
        candidates = list(iter_candidates())
        all_candidate_ids = [c["candidate_id"] for c in candidates]
        metrics = evaluate_submission_csv(args.submission, all_candidate_ids)
        print(f"\nEvaluation for: {args.submission}")
        print(
            f"  composite={metrics['composite']:.4f}  "
            f"NDCG@10={metrics['ndcg@10']:.3f}  "
            f"NDCG@50={metrics['ndcg@50']:.3f}  "
            f"MAP={metrics['map']:.3f}  "
            f"P@10={metrics['p@10']:.2f}  "
            f"mean_silver@100={metrics['mean_silver@100']:.2f}  "
            f"HP@100={metrics['honeypot_rate@100']:.1%}"
        )
        print(
            f"  hand-label mean_manual@10={metrics['mean_manual@10']:.2f}  "
            f"hand-label NDCG@10={metrics['ndcg@10']:.3f}  "
            f"hand-label NDCG@30={metrics['ndcg@30']:.3f}"
        )
    else:
        run_ranker_comparison()


if __name__ == "__main__":
    main()
