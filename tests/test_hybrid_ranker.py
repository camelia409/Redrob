"""Tests for the RRF hybrid ranker."""
from src.ranking.hybrid_ranker import HybridRRFRanker


class _MockRanker:
    def __init__(self, ordered_ids):
        self.ordered_ids = ordered_ids

    def rank(self, candidates):
        return [(cid, float(len(self.ordered_ids) - i)) for i, cid in enumerate(self.ordered_ids)]


def test_hybrid_combines_two_rankers_with_rrf():
    bm25 = _MockRanker(["A", "B", "C"])
    dense = _MockRanker(["C", "B", "D"])
    hybrid = HybridRRFRanker(bm25, dense, k=60, per_ranker_top=10)
    out = hybrid.rank([{}] * 10)

    # C appears in both lists at rank 3 and rank 1 → highest RRF.
    ids = [cid for cid, _ in out]
    assert ids[0] == "C"
    # All four IDs should be present.
    assert set(ids) == {"A", "B", "C", "D"}
    # Scores are sorted descending.
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_respects_per_ranker_top():
    ids = [f"ID_{i:02d}" for i in range(100)]
    bm25 = _MockRanker(ids)
    dense = _MockRanker(list(reversed(ids)))
    hybrid = HybridRRFRanker(bm25, dense, k=60, per_ranker_top=5)
    out = hybrid.rank([{}] * 100)
    # Only the top-5 from each mock ranker should be fused.
    returned = {cid for cid, _ in out}
    expected = set(ids[:5]) | set(ids[-5:])
    assert returned == expected


def test_hybrid_returns_nonempty_for_identical_rankers():
    bm25 = _MockRanker(["X", "Y", "Z"])
    dense = _MockRanker(["X", "Y", "Z"])
    hybrid = HybridRRFRanker(bm25, dense, k=0, per_ranker_top=10)
    out = hybrid.rank([{}] * 3)
    assert [cid for cid, _ in out] == ["X", "Y", "Z"]
    # With k=0, X gets 1/1 + 1/1 = 2, Y gets 1/2 + 1/2 = 1, Z gets 1/3 + 1/3 ≈ 0.67.
    scores = [s for _, s in out]
    assert scores[0] > scores[1] > scores[2]
