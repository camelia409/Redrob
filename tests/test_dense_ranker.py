"""Tests for the dense semantic ranker."""
import numpy as np
import pytest

from src.ranking.dense_ranker import DenseRanker
from src.retrieval.embeddings import EmbeddingIndex


def _sample_candidates(n: int = 5):
    return [
        {
            "candidate_id": f"CAND_{i:03d}",
            "profile": {
                "headline": f"Engineer {i}",
                "summary": "Machine learning and NLP background.",
                "current_title": "Data Scientist",
            },
            "career_history": [
                {"title": "Data Scientist", "description": "Built ranking models."}
            ],
            "skills": [{"name": "Python"}, {"name": "NLP"}],
        }
        for i in range(n)
    ]


@pytest.fixture
def temp_index(tmp_path, monkeypatch):
    from src.retrieval import embeddings
    monkeypatch.setattr(embeddings, "EMB_PATH", tmp_path / "candidate_embeddings.npy")
    monkeypatch.setattr(embeddings, "IDS_PATH", tmp_path / "candidate_ids.npy")
    idx = EmbeddingIndex()
    idx.build(_sample_candidates(5), batch_size=2)
    return idx


def test_dense_ranker_returns_all_candidate_ids(temp_index, monkeypatch):
    # Prevent ranker from loading the real JD text; a short query is enough.
    ranker = DenseRanker(index=temp_index, jd_text="machine learning")
    candidates = _sample_candidates(5)
    out = ranker.rank(candidates)
    assert len(out) == 5
    returned_ids = {cid for cid, _ in out}
    expected_ids = {c["candidate_id"] for c in candidates}
    assert returned_ids == expected_ids


def test_dense_ranker_is_deterministic(temp_index):
    ranker = DenseRanker(index=temp_index, jd_text="NLP and ranking")
    candidates = _sample_candidates(5)
    out1 = ranker.rank(candidates)
    out2 = ranker.rank(candidates)
    assert out1 == out2
    # Sorted descending.
    scores = [s for _, s in out1]
    assert scores == sorted(scores, reverse=True)
