"""Tests for first-principles ranking metrics."""
import math

import pytest

from src.evaluation.metrics import dcg_at_k, mean_average_precision, ndcg_at_k, precision_at_k
from src.ranking.baselines import SkillCountRanker, _tokenize


def test_dcg_at_k_perfect_order():
    gains = [3.0, 2.0, 1.0]
    expected = 3.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k(gains, 3) == pytest.approx(expected, rel=1e-9)


def test_ndcg_at_k_perfect_order_is_one():
    gains = [3.0, 2.0, 1.0]
    assert ndcg_at_k(gains, 3) == pytest.approx(1.0)


def test_ndcg_at_k_reversed_order():
    gains = [1.0, 2.0, 3.0]
    ideal = [3.0, 2.0, 1.0]
    expected = dcg_at_k(gains, 3) / dcg_at_k(ideal, 3)
    assert ndcg_at_k(gains, 3) == pytest.approx(expected)


def test_ndcg_at_k_zero_gains_returns_zero():
    assert ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_ndcg_at_k_k_larger_than_list():
    gains = [3.0, 2.0, 1.0]
    assert ndcg_at_k(gains, 10) == pytest.approx(1.0)


def test_precision_at_k_basic():
    binary = [1, 0, 1, 0, 1]
    assert precision_at_k(binary, 5) == pytest.approx(0.6)
    assert precision_at_k(binary, 3) == pytest.approx(2 / 3)
    assert precision_at_k(binary, 0) == 0.0


def test_mean_average_precision_basic():
    # rel at ranks 1 and 3
    binary = [1, 0, 1, 0]
    expected = (1 / 1 + 2 / 3) / 2
    assert mean_average_precision(binary) == pytest.approx(expected)


def test_mean_average_precision_all_zero():
    assert mean_average_precision([0, 0, 0]) == 0.0


def test_mean_average_precision_perfect():
    binary = [1, 1, 1]
    assert mean_average_precision(binary) == pytest.approx(1.0)


def test_ndcg_binary_relevance():
    # Ideal order would put both 1s first.
    gains = [0.0, 1.0, 0.0, 1.0]
    ideal = [1.0, 1.0, 0.0, 0.0]
    expected = dcg_at_k(gains, 4) / dcg_at_k(ideal, 4)
    assert ndcg_at_k(gains, 4) == pytest.approx(expected)


def test_bm25_tokenizer_lowercase_alphanumeric():
    assert _tokenize("BM25 + NLP/retrieval, LLMs!") == ["bm25", "nlp", "retrieval", "llms"]


def test_skill_count_ranker_is_deterministic():
    candidates = [
        {"candidate_id": "A", "skills": [{"name": "Python"}, {"name": "NLP"}]},
        {"candidate_id": "B", "skills": [{"name": "Python"}]},
    ]
    ranker = SkillCountRanker()
    out1 = ranker.rank(candidates)
    out2 = ranker.rank(candidates)
    assert out1 == out2
    assert out1[0][0] == "A"
