"""FeatureExtractor: flatten the 7 families into a single feature vector."""
from typing import Dict, List

import pandas as pd

from src.features.families import (
    behavioral,
    career_trajectory,
    integrity_risk,
    logistics,
    production_evidence,
    semantic_jd_fit,
    skill_evidence,
)


class FeatureExtractor:
    """Extract normalized features for the learned re-ranker."""

    def extract(
        self,
        candidate: dict,
        bm25_score: float,
        dense_score: float,
        honeypot_score_val: int | float,
        bm25_max: float | None = None,
        rank_bm25: int | None = None,
        rank_dense: int | None = None,
        n_candidates: int = 1,
    ) -> Dict[str, float]:
        """Return a flat dict of features for a single candidate."""
        if bm25_max and bm25_max > 0:
            bm25_norm = max(0.0, bm25_score) / bm25_max
        else:
            bm25_norm = 0.0

        features: Dict[str, float] = {}
        features.update(
            semantic_jd_fit(
                candidate,
                dense_score,
                bm25_score_normalized=bm25_norm,
                rank_bm25=rank_bm25,
                rank_dense=rank_dense,
                n_candidates=n_candidates,
            )
        )
        features.update(skill_evidence(candidate))
        features.update(career_trajectory(candidate))
        features.update(production_evidence(candidate))
        features.update(behavioral(candidate))
        features.update(logistics(candidate))
        features.update(integrity_risk(candidate, honeypot_score_val))

        return features

    def extract_batch(
        self,
        candidates: List[dict],
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        hp_scores: Dict[str, int | float],
    ) -> pd.DataFrame:
        """Return a DataFrame of features for a batch of candidates.

        BM25 scores are normalized per-batch (clip negatives, divide by max).
        Rank-agreement features are computed from descending rank positions.
        """
        n = len(candidates)

        # Per-batch BM25 normalization.
        bm25_values = [max(0.0, bm25_scores.get(c["candidate_id"], 0.0)) for c in candidates]
        bm25_max = max(bm25_values) if bm25_values else 1.0

        # Rank maps: lower rank = better.
        bm25_sorted = sorted(
            bm25_scores.items(), key=lambda x: x[1], reverse=True
        )
        dense_sorted = sorted(
            dense_scores.items(), key=lambda x: x[1], reverse=True
        )
        bm25_rank = {cid: idx for idx, (cid, _) in enumerate(bm25_sorted, start=1)}
        dense_rank = {cid: idx for idx, (cid, _) in enumerate(dense_sorted, start=1)}

        rows = []
        for candidate in candidates:
            cid = candidate["candidate_id"]
            features = self.extract(
                candidate,
                bm25_score=bm25_scores.get(cid, 0.0),
                dense_score=dense_scores.get(cid, 0.0),
                honeypot_score_val=hp_scores.get(cid, 0),
                bm25_max=bm25_max,
                rank_bm25=bm25_rank.get(cid),
                rank_dense=dense_rank.get(cid),
                n_candidates=n,
            )
            rows.append({"candidate_id": cid, **features})

        return pd.DataFrame(rows)
