"""Dense semantic ranker using a pre-built MiniLM embedding index."""
from typing import List, Tuple

from src.ingestion.jd import get_jd_text
from src.retrieval.embeddings import EmbeddingIndex


class DenseRanker:
    """Rank candidates by cosine similarity between JD and candidate embeddings."""

    def __init__(self, index: EmbeddingIndex | None = None, jd_text: str | None = None):
        self.index = index or EmbeddingIndex()
        self.index.load()
        self.jd_text = jd_text or get_jd_text()

    def rank(self, candidates: List[dict]) -> List[Tuple[str, float]]:
        similarities = self.index.query(self.jd_text)
        ids = self.index.candidate_ids
        if ids is None or len(ids) != len(similarities):
            raise ValueError("Embedding index not loaded or mismatched with candidate list")
        pairs = [(str(cid), float(score)) for cid, score in zip(ids, similarities)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
