"""Full-population ablation v3: add weighted re-ranker (gated / no-gate)."""
import csv as _csv
import time
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.harness import evaluate_full_population, evaluate_labeled_subset
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker, RandomRanker, SkillCountRanker
from src.ranking.dense_ranker import DenseRanker, DenseRankerV2
from src.ranking.hybrid_ranker import HybridRRFRanker
from src.ranking.weighted_reranker import WeightedReranker
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import CONFIGS, OUTPUTS, PROCESSED, SILVER
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


SILVER_PATH = SILVER / "silver_labels_v1.csv"
FULL_SCORES_PATH = PROCESSED / "silver_scores_full.csv"
FEATURE_MATRIX_PATH = PROCESSED / "feature_matrix.parquet"
WEIGHTS_PATH = CONFIGS / "reranker_weights_v1.yaml"
HP_PENALTY = -1e6


def _load_csv_scores(path: Path) -> dict:
    scores = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            scores[row["candidate_id"]] = int(row["silver_score"])
    return scores


def _load_weights(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "weights_v1" in cfg:
        return cfg["weights_v1"]
    if "weights" in cfg:
        return cfg["weights"]
    raise ValueError(f"No weights block found in {path}")


def _composite(metrics: dict) -> float:
    """Predicted challenge composite used for ranker selection."""
    return (
        0.50 * metrics["ndcg@10"]
        + 0.30 * metrics["ndcg@50"]
        + 0.15 * metrics["map"]
        + 0.05 * metrics["p@10"]
    )


def _weighted_ranker_output(df: pd.DataFrame, gated: bool, all_candidate_ids: list) -> list[tuple[str, float]]:
    """Return a full 100K ranking from the weighted re-ranker scores.

    Only the retrieval-union candidates have non-zero scores; the rest receive
    0. If ``gated`` is True, candidates with honeypot_score >= 3 are pushed to
    the bottom of the union ranking.
    """
    score_col = "weighted_score_gated" if gated else "weighted_score"

    n = len(df)
    if n > 1:
        df["dense_v2_rank"] = (1.0 - df["dense_v2_rank_within_union"]) * (n - 1) + 1
        df["bm25_rank"] = (1.0 - df["bm25_rank_within_union"]) * (n - 1) + 1
    else:
        df["dense_v2_rank"] = 1
        df["bm25_rank"] = 1

    sorted_df = df.sort_values(
        by=[score_col, "dense_v2_rank", "bm25_rank", "candidate_id"],
        ascending=[False, True, True, True],
    )

    union_order = sorted_df["candidate_id"].tolist()
    union_scores = dict(zip(sorted_df["candidate_id"], sorted_df[score_col]))
    union_set = set(union_order)

    # Append all non-union candidates with a score of 0.
    remaining = [cid for cid in all_candidate_ids if cid not in union_set]
    out = [(cid, union_scores[cid]) for cid in union_order] + [(cid, 0.0) for cid in remaining]
    return out


def main() -> None:
    print("[1/5] Loading 100K candidates...")
    candidates = list(iter_candidates())
    all_candidate_ids = [c["candidate_id"] for c in candidates]
    print(f"      loaded {len(candidates):,} candidates")

    print("[2/5] Computing honeypot scores...")
    duplicate_ids = find_duplicate_fingerprints(iter(candidates))
    honeypot_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, duplicate_ids)
        for c in candidates
    }
    print(f"      computed {len(honeypot_scores):,} honeypot scores")

    print("[3/5] Loading silver labels, weights, and feature matrix...")
    full_scores = _load_csv_scores(FULL_SCORES_PATH)
    labeled_scores = _load_csv_scores(SILVER_PATH)
    print(f"      full-pop scores: {len(full_scores):,}; labeled subset: {len(labeled_scores):,}")

    weights = _load_weights(WEIGHTS_PATH)
    df_features = pd.read_parquet(FEATURE_MATRIX_PATH)
    reranker = WeightedReranker(weights, score_col="weighted_score")
    df_features["weighted_score"] = reranker.score(df_features)
    df_features["weighted_score_gated"] = df_features["weighted_score"].copy()
    df_features.loc[df_features["honeypot_score"] >= 3, "weighted_score_gated"] += HP_PENALTY
    print(f"      feature matrix: {len(df_features):,} rows x {len(df_features.columns)} columns")

    print("[4/5] Instantiating rankers...")
    index = EmbeddingIndex()
    index.load()
    jd_text = get_jd_text()
    random_ranker = RandomRanker(seed=42)
    skill_ranker = SkillCountRanker()
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    dense_ranker = DenseRanker(index=index, jd_text=jd_text)
    dense_v2_ranker = DenseRankerV2(index=index)

    rankers = [
        ("random", random_ranker),
        ("skill_count", skill_ranker),
        ("bm25", bm25_ranker),
        ("dense_v2", dense_v2_ranker),
        ("hybrid_v2_rrf", HybridRRFRanker(bm25_ranker, dense_v2_ranker, k=60, per_ranker_top=2000)),
        (
            "weighted_reranker_top100",
            type("_WeightedGated", (), {
                "rank": lambda _, __: _weighted_ranker_output(df_features, gated=True, all_candidate_ids=all_candidate_ids)
            })(),
        ),
        (
            "weighted_reranker_top100_no_honeypot_gate",
            type("_WeightedNoGate", (), {
                "rank": lambda _, __: _weighted_ranker_output(df_features, gated=False, all_candidate_ids=all_candidate_ids)
            })(),
        ),
    ]

    print("[5/5] Running ablation...\n")
    results = []
    outputs_cache = {}
    for name, ranker in rankers:
        t0 = time.time()
        out = ranker.rank(candidates)
        elapsed = time.time() - t0
        outputs_cache[name] = out

        full_metrics = evaluate_full_population(
            out, full_scores, honeypot_scores, k_values=[10, 50, 100]
        )
        labeled_metrics = evaluate_labeled_subset(out, labeled_scores, k_values=[10])

        composite = _composite(full_metrics)

        row = {
            "ranker": name,
            "runtime_sec": round(elapsed, 1),
            "composite": round(composite, 4),
            **full_metrics,
            **labeled_metrics,
        }
        results.append(row)
        print(
            f"{name:>40s}  "
            f"NDCG@10={full_metrics['ndcg@10']:.3f}  "
            f"NDCG@50={full_metrics['ndcg@50']:.3f}  "
            f"MAP={full_metrics['map']:.3f}  "
            f"P@10={full_metrics['p@10']:.2f}  "
            f"mean_silver@100={full_metrics['mean_silver@100']:.2f}  "
            f"HP@100={full_metrics['honeypot_rate@100']:.1%}  "
            f"composite={composite:.4f}  "
            f"runtime={elapsed:.1f}s"
        )

    # Sort results by predicted composite descending.
    results.sort(key=lambda r: r["composite"], reverse=True)

    print("\nRanked by predicted composite:")
    print(f"{'ranker':>40s}  {'composite':>9s}  {'NDCG@10':>8s}  {'NDCG@50':>8s}  {'MAP':>6s}  {'P@10':>5s}  {'HP@100':>7s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['ranker']:>40s}  "
            f"{r['composite']:9.4f}  "
            f"{r['ndcg@10']:8.3f}  "
            f"{r['ndcg@50']:8.3f}  "
            f"{r['map']:6.3f}  "
            f"{r['p@10']:5.2f}  "
            f"{r['honeypot_rate@100']:7.1%}"
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "ablation_v3.csv"
    fieldnames = [
        "ranker",
        "composite",
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "map",
        "p@10",
        "p@50",
        "p@100",
        "honeypot_rate@100",
        "mean_silver@100",
        "labeled_ndcg@10",
        "labeled_p@10",
        "labeled_map",
        "runtime_sec",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
