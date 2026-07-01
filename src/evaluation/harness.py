"""Evaluation harness: turn a ranker's output into metrics vs. silver labels.

Two modes are provided:

- evaluate_full_population: uses silver scores for *all* 100K candidates. Every
  candidate in the ranker's output contributes its true silver score to gains.
- evaluate_labeled_subset: only considers candidates in the 500-row hand/sampled
  silver set. Useful as a diagnostic but pessimistic for top-K metrics.
"""
from typing import Dict, List, Tuple

from src.evaluation.metrics import mean_average_precision, ndcg_at_k, precision_at_k


def evaluate_full_population(
    ranker_output: List[Tuple[str, float]],
    silver_scores_full: Dict[str, int],
    honeypot_scores: Dict[str, int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute NDCG, P@K, MAP, honeypot rate and mean silver score for a ranking.

    Args:
        ranker_output: list of (candidate_id, score) sorted descending by score.
        silver_scores_full: mapping candidate_id -> silver_score for the full pool.
        honeypot_scores: mapping candidate_id -> honeypot_score_gates_only value.
        k_values: metric cutoffs (default [10, 50, 100]).

    Returns:
        dict with ndcg@K, p@K, map, honeypot_rate@100, mean_silver@100.
    """
    if k_values is None:
        k_values = [10, 50, 100]

    ranked_ids = [cid for cid, _ in ranker_output]
    gains = [float(silver_scores_full.get(cid, 0)) for cid in ranked_ids]
    binary = [1 if g >= 3 else 0 for g in gains]

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(gains, k)
        metrics[f"p@{k}"] = precision_at_k(binary, k)

    metrics["map"] = mean_average_precision(binary)

    top_100_ids = ranked_ids[:100]
    honeypot_hits = sum(
        1 for cid in top_100_ids if honeypot_scores.get(cid, 0) >= 3
    )
    metrics["honeypot_rate@100"] = (
        honeypot_hits / len(top_100_ids) if top_100_ids else 0.0
    )
    metrics["mean_silver@100"] = sum(gains[:100]) / 100 if gains else 0.0

    return metrics


def evaluate_labeled_subset(
    ranker_output: List[Tuple[str, float]],
    labeled: Dict[str, int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute metrics on only the candidates present in a labeled subset.

    This is a diagnostic function; it ignores unlabeled candidates entirely and
    therefore measures ranking quality on known labels, not recall.
    """
    if k_values is None:
        k_values = [10]

    filtered = [(cid, s) for cid, s in ranker_output if cid in labeled]
    filtered.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [cid for cid, _ in filtered]
    gains = [float(labeled[cid]) for cid in ranked_ids]
    binary = [1 if g >= 3 else 0 for g in gains]

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"labeled_ndcg@{k}"] = ndcg_at_k(gains, k) if gains else 0.0
        metrics[f"labeled_p@{k}"] = precision_at_k(binary, k) if binary else 0.0
    metrics["labeled_map"] = mean_average_precision(binary) if binary else 0.0

    return metrics
