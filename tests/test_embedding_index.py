"""Tests for the MiniLM embedding index builder/loader/query."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

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
def temp_index_dir(tmp_path, monkeypatch):
    from src.retrieval import embeddings
    monkeypatch.setattr(embeddings, "EMB_PATH", tmp_path / "candidate_embeddings.npy")
    monkeypatch.setattr(embeddings, "IDS_PATH", tmp_path / "candidate_ids.npy")
    return tmp_path


def test_build_saves_embeddings_and_ids(temp_index_dir):
    idx = EmbeddingIndex()
    candidates = _sample_candidates(5)
    idx.build(candidates, batch_size=2)

    assert idx.embeddings is not None
    assert idx.candidate_ids is not None
    assert idx.embeddings.shape == (5, 384)
    assert idx.embeddings.dtype == np.float32
    assert len(idx.candidate_ids) == 5

    # Files exist.
    assert (temp_index_dir / "candidate_embeddings.npy").exists()
    assert (temp_index_dir / "candidate_ids.npy").exists()


def test_load_reads_back_embeddings(temp_index_dir):
    idx = EmbeddingIndex()
    candidates = _sample_candidates(5)
    idx.build(candidates, batch_size=2)

    idx2 = EmbeddingIndex()
    idx2.load()
    np.testing.assert_array_equal(idx.embeddings, idx2.embeddings)
    np.testing.assert_array_equal(idx.candidate_ids, idx2.candidate_ids)


def test_query_returns_cosine_similarity_vector(temp_index_dir):
    idx = EmbeddingIndex()
    candidates = _sample_candidates(5)
    idx.build(candidates, batch_size=2)

    sims = idx.query("machine learning and NLP")
    assert sims.shape == (5,)
    assert sims.dtype == np.float32
    # Normalized embeddings => cosine similarity in [-1, 1].
    assert sims.min() >= -1.0 and sims.max() <= 1.0
