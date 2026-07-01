"""Hybrid ranker fusing BM25 and dense similarities via Reciprocal Rank Fusion."""
from typing import List, Tuple


class HybridRRFRanker:
    """RRF fusion of two ranker outputs.

    score(c) = Σ 1/(k + rank_i(c))   (Cormack et al., SIGIR 2009)
    """

    def __init__(
        self,
        bm25_ranker,
        dense_ranker,
        k: int = 60,
        per_ranker_top: int = 2000,
    ):
        self.bm25_ranker = bm25_ranker
        self.dense_ranker = dense_ranker
        self.k = k
        self.per_ranker_top = per_ranker_top

    def rank(self, candidates: List[dict]) -> List[Tuple[str, float]]:
        bm25_out = self.bm25_ranker.rank(candidates)[: self.per_ranker_top]
        dense_out = self.dense_ranker.rank(candidates)[: self.per_ranker_top]

        rrf: dict = {}
        for rank, (cid, _) in enumerate(bm25_out, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.k + rank)
        for rank, (cid, _) in enumerate(dense_out, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.k + rank)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)
