"""Evaluation harness: turn a ranker's output into metrics vs. silver labels."""
from typing import Dict, List, Tuple

from src.evaluation.metrics import mean_average_precision, ndcg_at_k, precision_at_k


def evaluate(
    ranker_output: List[Tuple[str, float]],
    silver_labels: Dict[str, int],
    honeypot_scores: Dict[str, int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute NDCG, P@K, MAP and honeypot rate for a ranked candidate list.

    Candidates not present in silver_labels are treated as relevance 0, which
    makes NDCG pessimistic but consistent across rankers.

    Args:
        ranker_output: list of (candidate_id, score) sorted descending by score.
        silver_labels: mapping candidate_id -> silver_score (0-5).
        honeypot_scores: mapping candidate_id -> honeypot_score_gates_only value.
        k_values: metric cutoffs to compute (default [10, 50, 100]).

    Returns:
        dict with ndcg@K, p@K, map, and honeypot_rate@100.
    """
    if k_values is None:
        k_values = [10, 50, 100]

    ranked_ids = [cid for cid, _ in ranker_output]
    gains = [float(silver_labels.get(cid, 0)) for cid in ranked_ids]
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
    metrics["honeypot_rate@100"] = honeypot_hits / len(top_100_ids) if top_100_ids else 0.0

    return metrics


def evaluate_restricted(
    ranker_output: List[Tuple[str, float]],
    silver_labels: Dict[str, int],
    honeypot_scores: Dict[str, int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Same as evaluate, but only over the candidate subset that has silver labels.

    Useful for diagnosing whether rankers can order known-relevant candidates,
    independent of the pessimistic unknown=0 assumption on the full 100K pool.
    """
    if k_values is None:
        k_values = [10, 50, 100]

    labeled_ids = set(silver_labels.keys())
    filtered = [(cid, score) for cid, score in ranker_output if cid in labeled_ids]
    # Re-sort in case filtering changed ordering, though it shouldn't.
    filtered.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [cid for cid, _ in filtered]
    gains = [float(silver_labels[cid]) for cid in ranked_ids]
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
    metrics["honeypot_rate@100"] = honeypot_hits / len(top_100_ids) if top_100_ids else 0.0
    return metrics
