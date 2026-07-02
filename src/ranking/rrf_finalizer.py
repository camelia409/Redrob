"""Late-stage RRF fusion of weighted reranker output with BM25 ranking.

Reciprocal Rank Fusion combines two rankings into one via:
    fused_score(c) = 1/(k + rank_A(c)) + alpha * 1/(k + rank_B(c))

Higher fused_score means better ranked. This module fuses the weighted
reranker's ranking with BM25's ranking to correct known divergence
between silver-label optimization and human judgment.
"""
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def load_rrf_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("rrf_finalizer", cfg)


def fuse_rankings(
    weighted_df: pd.DataFrame,
    bm25_ranking: list[tuple[str, float]],
    alpha: float = 0.7,
    k: int = 60,
) -> pd.DataFrame:
    """
    Fuse weighted reranker output with BM25 ranking via RRF.

    Args:
        weighted_df: DataFrame from weighted reranker, sorted best-first.
                     Must have 'candidate_id' column. Rank = row index + 1.
        bm25_ranking: List of (candidate_id, bm25_score) sorted best-first.
                      Rank = position + 1.
        alpha: Weight for BM25 channel. 0.7 = weighted:BM25 as 1.0:0.7.
        k: RRF constant. 60 is the canonical value from Cormack et al. 2009.

    Returns:
        weighted_df sorted by fused_score descending, with new columns:
        'rrf_score', 'weighted_rank', 'bm25_rank_in_pool', 'fused_score'.
    """
    weighted_df = weighted_df.reset_index(drop=True).copy()
    weighted_df["weighted_rank"] = range(1, len(weighted_df) + 1)

    # BM25 rank for candidates in the weighted pool
    bm25_rank_map = {cid: rank for rank, (cid, _) in enumerate(bm25_ranking, start=1)}

    # Very-far-down default: any candidate not in BM25 top-N gets rank = large
    max_bm25_rank = len(bm25_ranking) + 1000

    weighted_df["bm25_rank_in_pool"] = weighted_df["candidate_id"].map(
        lambda cid: bm25_rank_map.get(cid, max_bm25_rank)
    )

    weighted_df["fused_score"] = (
        1.0 / (k + weighted_df["weighted_rank"])
        + alpha * 1.0 / (k + weighted_df["bm25_rank_in_pool"])
    )

    # Preserve honeypot gate: any candidate with reranker_score massively negative
    # (from -1e6 penalty) stays at the bottom
    HP_MARKER = -1000.0
    is_gated = weighted_df["reranker_score"] < HP_MARKER
    weighted_df.loc[is_gated, "fused_score"] = -1e6

    # Deterministic tiebreak: fused_score desc, weighted_rank asc, candidate_id asc
    return weighted_df.sort_values(
        by=["fused_score", "weighted_rank", "candidate_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
