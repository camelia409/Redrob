"""Dense semantic ranker using a pre-built MiniLM embedding index."""
from typing import List, Tuple

from src.ingestion.jd import get_jd_text
from src.retrieval.embeddings import EmbeddingIndex, load_jd_query_v2


class DenseRanker:
    """Rank candidates by cosine similarity between JD and candidate embeddings."""

    def __init__(self, index: EmbeddingIndex | None = None, jd_text: str | None = None):
        self.index = index or EmbeddingIndex()
        self.index.load()
        self._jd_text = jd_text

    def _get_jd_query(self) -> str:
        """Return the JD text to encode. Subclasses may override."""
        return self._jd_text if self._jd_text is not None else get_jd_text()

    def rank(self, candidates: List[dict]) -> List[Tuple[str, float]]:
        similarities = self.index.query(self._get_jd_query())
        ids = self.index.candidate_ids
        if ids is None or len(ids) != len(similarities):
            raise ValueError("Embedding index not loaded or mismatched with candidate list")
        pairs = [(str(cid), float(score)) for cid, score in zip(ids, similarities)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


class DenseRankerV2(DenseRanker):
    """Dense ranker using the hand-distilled technical JD query."""

    def __init__(self, index: EmbeddingIndex | None = None):
        super().__init__(index=index, jd_text=None)

    def _get_jd_query(self) -> str:
        return load_jd_query_v2()
