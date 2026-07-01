"""Dense embedding index using sentence-transformers/all-MiniLM-L6-v2.

This module is intentionally offline/precomputation: `build()` encodes the full
100K pool once and saves float32 embeddings to disk.  Ranking scripts then call
`load()` / `query()` without re-encoding candidates.
"""
from pathlib import Path
from typing import List

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.paths import CONFIGS, PROCESSED


EMB_PATH = PROCESSED / "candidate_embeddings.npy"
IDS_PATH = PROCESSED / "candidate_ids.npy"


def load_jd_query_v2() -> str:
    """Return the hand-distilled technical JD query from configs/jd_query.yaml."""
    with open(CONFIGS / "jd_query.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["jd_query_v2"]


class EmbeddingIndex:
    """L2-normalized MiniLM embeddings for the candidate pool."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings: np.ndarray | None = None
        self.candidate_ids: np.ndarray | None = None

    def _lazy_load_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device="cpu")

    def _candidate_text(self, candidate: dict) -> str:
        """Concatenate all textual fields from a candidate."""
        profile = candidate.get("profile", {})
        parts = [
            profile.get("headline", ""),
            profile.get("summary", ""),
            profile.get("current_title", ""),
        ]
        for job in candidate.get("career_history", []):
            parts.append(job.get("description", ""))
            parts.append(job.get("title", ""))
        for skill in candidate.get("skills", []):
            parts.append(skill.get("name", ""))
        text = " ".join(str(p) for p in parts if p)
        return text[:4000]

    def build(self, candidates: List[dict], batch_size: int = 64) -> None:
        """Encode all candidates and persist embeddings + IDs to disk."""
        self._lazy_load_model()
        texts = [self._candidate_text(c) for c in candidates]
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        ids = np.array([c["candidate_id"] for c in candidates])

        PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(EMB_PATH, embs)
        np.save(IDS_PATH, ids)
        self.embeddings = embs
        self.candidate_ids = ids

    def load(self) -> None:
        """Load pre-computed embeddings from disk."""
        if not EMB_PATH.exists() or not IDS_PATH.exists():
            raise FileNotFoundError(
                f"Run scripts/prebuild_index.py first — missing {EMB_PATH} or {IDS_PATH}"
            )
        self.embeddings = np.load(EMB_PATH)
        self.candidate_ids = np.load(IDS_PATH)

    def query(self, jd_text: str) -> np.ndarray:
        """Return cosine-similarity vector (shape [n_candidates]) for a JD."""
        self._lazy_load_model()
        if self.embeddings is None:
            self.load()
        query_emb = self.model.encode(
            [jd_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        # Both vectors are L2-normalized, so dot product == cosine similarity.
        return (self.embeddings @ query_emb.T).flatten()
