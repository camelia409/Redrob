"""Evaluate rankers against the 30 human hand labels (pure human signal)."""
import csv
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import ndcg_at_k
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRanker, DenseRankerV2
from src.ranking.hybrid_ranker import HybridRRFRanker
from src.ranking.weighted_reranker import WeightedReranker
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import CONFIGS, PROCESSED, SILVER

# Reuse the weighted-ranker construction from the v3 ablation script.
from scripts.run_ablation_v3 import _load_weights, _weighted_ranker_output

HP_PENALTY = -1e6


MANUAL_LABELS_PATH = SILVER / "manual_labels_v1.csv"
FEATURE_MATRIX_PATH = PROCESSED / "feature_matrix.parquet"
WEIGHTS_PATH = CONFIGS / "reranker_weights_v1.yaml"


def _load_manual_labels(path: Path) -> dict[str, int]:
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["candidate_id"]] = int(row["rank_score"])
    return labels


def _metrics_for_ranker(ranker_output: list[tuple[str, float]], labels: dict[str, int]):
    filtered = [(cid, score) for cid, score in ranker_output if cid in labels]
    gains = [float(labels[cid]) for cid, _ in filtered]
    return {
        "ndcg@10": ndcg_at_k(gains, 10),
        "ndcg@30": ndcg_at_k(gains, 30),
        "mean_manual@10": sum(labels[cid] for cid, _ in filtered[:10]) / 10 if filtered else 0.0,
    }


def main() -> None:
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


if __name__ == "__main__":
    main()
