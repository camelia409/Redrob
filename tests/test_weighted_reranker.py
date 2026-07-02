"""Tests for the transparent weighted-sum re-ranker."""
import pandas as pd
import pytest

from src.ranking.weighted_reranker import WeightedReranker


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004"],
        "f1": [1.0, 0.8, 0.6, 0.4],
        "f2": [0.0, 0.5, 0.5, 1.0],
        "honeypot_score": [0, 0, 3, 0],
        "dense_v2_rank_within_union": [1.0, 0.5, 0.5, 0.0],
        "bm25_rank_within_union": [1.0, 0.5, 0.5, 0.0],
    })


def test_weighted_reranker_is_deterministic():
    df = _df()
    reranker = WeightedReranker({"f1": 1.0, "f2": 0.5})
    scores1 = reranker.score(df)
    scores2 = reranker.score(df)
    pd.testing.assert_series_equal(scores1, scores2)

    ranked1 = reranker.rank(df)
    ranked2 = reranker.rank(df)
    assert ranked1["candidate_id"].tolist() == ranked2["candidate_id"].tolist()


def test_missing_feature_gets_weight_zero():
    df = _df()
    reranker = WeightedReranker({"f1": 1.0, "missing_feature": 99.0})
    scores = reranker.score(df)
    expected = df["f1"] * 1.0
    pd.testing.assert_series_equal(scores, expected, check_names=False)


def test_honeypot_gated_candidates_rank_last():
    df = _df().copy()
    reranker = WeightedReranker({"f1": 1.0, "f2": 0.0}, score_col="raw_score")
    df["raw_score"] = reranker.score(df)

    # Apply the same gate logic used in generate_submission.py.
    HP_PENALTY = -1e6
    df.loc[df["honeypot_score"] >= 3, "raw_score"] += HP_PENALTY

    ranked = df.sort_values(
        by=["raw_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    gated_ids = ranked.loc[ranked["honeypot_score"] >= 3, "candidate_id"].tolist()
    clean_ids = ranked.loc[ranked["honeypot_score"] < 3, "candidate_id"].tolist()

    assert gated_ids == ["CAND_0000003"]
    assert gated_ids not in clean_ids
    # All clean rows must appear before any gated row.
    gated_pos = ranked[ranked["candidate_id"] == "CAND_0000003"].index[0]
    for cid in clean_ids:
        assert ranked[ranked["candidate_id"] == cid].index[0] < gated_pos


def test_tiebreak_order_uses_retrieval_ranks_then_id():
    df = _df().copy()
    # f2 dominates so all scores equal 0.5 * weight; f1 contributes 0.
    reranker = WeightedReranker({"f1": 0.0, "f2": 1.0}, score_col="raw_score")
    df["raw_score"] = reranker.score(df)

    n = len(df)
    df["dense_v2_rank"] = (1.0 - df["dense_v2_rank_within_union"]) * (n - 1) + 1
    df["bm25_rank"] = (1.0 - df["bm25_rank_within_union"]) * (n - 1) + 1

    ranked = reranker.rank(df, tiebreak_cols=["dense_v2_rank", "bm25_rank"])
    ids = ranked["candidate_id"].tolist()

    # CAND_0000004 has f2=1.0 -> highest score, first.
    # CAND_0000002 and CAND_0000003 tie on f2=0.5. Tiebreak: dense_v2_rank asc, then bm25_rank asc,
    # then candidate_id asc -> both ranks equal, so 0002 before 0003.
    # CAND_0000001 has f2=0 -> lowest score, last.
    assert ids == ["CAND_0000004", "CAND_0000002", "CAND_0000003", "CAND_0000001"]
