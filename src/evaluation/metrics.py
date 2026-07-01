"""Pure, first-principles ranking metrics.

No external metric libraries here — we implement NDCG, P@K and MAP directly so
later benchmarks can be cross-checked against ranx or sklearn.
"""
import math
from typing import List


def dcg_at_k(gains: List[float], k: int) -> float:
    """Discounted Cumulative Gain using log2(rank+1) discount.

    Rank positions are 0-indexed internally; the divisor uses i+2 so that the
    first position (i=0) corresponds to rank 1 and gets a discount of log2(2)=1.
    """
    if k <= 0:
        return 0.0
    total = 0.0
    for i, g in enumerate(gains[:k]):
        total += g / math.log2(i + 2)
    return total


def ndcg_at_k(gains: List[float], k: int) -> float:
    """Normalized DCG: DCG divided by the ideal DCG for the same gain multiset."""
    if k <= 0:
        return 0.0
    ideal = sorted(gains, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def precision_at_k(binary: List[int], k: int) -> float:
    """Precision @ K for a binary relevance vector."""
    if k <= 0:
        return 0.0
    return sum(binary[:k]) / k


def mean_average_precision(binary: List[int]) -> float:
    """Mean Average Precision over a binary relevance vector."""
    hits = 0
    total = 0.0
    for i, rel in enumerate(binary):
        if rel:
            hits += 1
            total += hits / (i + 1)
    return total / hits if hits else 0.0
