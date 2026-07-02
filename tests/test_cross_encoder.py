"""Unit tests for cross-encoder reranker.

These tests use the actual model (~90 MB download on first run). If the download
is too slow for CI, mock the CrossEncoder — but for our purposes real invocation
is fine and catches integration bugs.
"""
import numpy as np
import pandas as pd
import pytest

from src.ranking.cross_encoder_ranker import (
    CrossEncoderReranker,
    build_candidate_summary,
    cross_encoder_rerank,
)


def _fake_candidate(
    cid: str, title: str, skills: list[str], desc: str = "shipped ranking system"
) -> dict:
    return {
        "candidate_id": cid,
        "profile": {
            "headline": f"{title} profile",
            "summary": "10 years experience",
            "current_title": title,
            "current_company": "TestCo",
            "years_of_experience": 6.0,
        },
        "skills": [{"name": s} for s in skills],
        "career_history": [{"description": desc, "duration_months": 24}],
    }


def test_build_candidate_summary_truncates():
    c = _fake_candidate(
        "CAND_1", "ML Engineer", ["python", "embeddings"] * 20, desc="x" * 5000
    )
    s = build_candidate_summary(c, max_chars=1200)
    assert len(s) <= 1200
    assert "ML Engineer" in s


def test_cross_encoder_reranker_scores_pairs():
    reranker = CrossEncoderReranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=8
    )
    scores = reranker.score_pairs(
        "Senior ML engineer with retrieval and ranking experience",
        [
            "ML Engineer at Flipkart, shipped ranking system with embeddings",
            "Marketing Manager at consulting firm, wrote content strategy",
        ],
    )
    assert scores.shape == (2,)
    # The ML profile should score higher than the marketing profile
    assert scores[0] > scores[1], f"expected ML profile > marketing: {scores}"


def test_cross_encoder_rerank_preserves_row_count():
    reranker = CrossEncoderReranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=8
    )
    candidates_by_id = {
        "CAND_1": _fake_candidate("CAND_1", "ML Engineer", ["embeddings", "faiss"]),
        "CAND_2": _fake_candidate("CAND_2", "HR Manager", ["excel", "recruitment"]),
        "CAND_3": _fake_candidate("CAND_3", "Data Scientist", ["python", "pytorch"]),
    }
    df = pd.DataFrame(
        {
            "candidate_id": ["CAND_1", "CAND_2", "CAND_3"],
            "fused_score": [0.03, 0.02, 0.01],
        }
    )
    out = cross_encoder_rerank(
        df=df,
        candidates_by_id=candidates_by_id,
        jd_text="Senior ML engineer with retrieval, ranking, embeddings experience",
        reranker=reranker,
        top_n_to_rerank=3,
        beta=0.7,
        prior_score_col="fused_score",
    )
    assert len(out) == 3
    assert set(out["candidate_id"]) == set(df["candidate_id"])
    assert "blended_score" in out.columns
    assert "cross_encoder_score" in out.columns


def test_empty_input_is_safe():
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.score_pairs("test JD", [])
    assert scores.shape == (0,)
