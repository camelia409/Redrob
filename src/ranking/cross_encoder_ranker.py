"""Cross-encoder rerank stage.

Takes a pandas DataFrame representing the top-N candidates from earlier
pipeline stages and re-orders them by cross-encoder relevance between the
distilled JD query and a compact candidate summary.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default. CPU-only.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import CrossEncoder


def load_cross_encoder_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("cross_encoder", cfg)


def build_candidate_summary(candidate: dict[str, Any], max_chars: int = 1200) -> str:
    """Build a compact text summary of the candidate for cross-encoder scoring.

    Keeps: headline, summary, current title, top 6 skills, first career bullet.
    Truncates to max_chars to fit cross-encoder token budget.
    """
    profile = candidate.get("profile", {})
    parts: list[str] = []

    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    if title:
        parts.append(f"Currently {title} at {company} with {yoe} years experience.")

    skills = candidate.get("skills", [])[:6]
    if skills:
        skill_names = ", ".join(s.get("name", "") for s in skills if s.get("name"))
        if skill_names:
            parts.append(f"Skills: {skill_names}.")

    career = candidate.get("career_history", [])
    if career:
        first = career[0]
        first_desc = first.get("description", "")
        if first_desc:
            parts.append(first_desc)

    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


class CrossEncoderReranker:
    """Wrap the sentence-transformers CrossEncoder for our pipeline."""

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device="cpu")

    def score_pairs(self, jd_text: str, candidate_texts: list[str]) -> np.ndarray:
        """Score each (jd_text, candidate_text) pair. Returns array of scores."""
        self._lazy_load()
        if not candidate_texts:
            return np.array([], dtype=np.float32)
        pairs = [[jd_text, ct] for ct in candidate_texts]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return scores.astype(np.float32)


def cross_encoder_rerank(
    df: pd.DataFrame,
    candidates_by_id: dict[str, dict],
    jd_text: str,
    reranker: CrossEncoderReranker,
    top_n_to_rerank: int = 500,
    beta: float = 0.7,
    prior_score_col: str = "fused_score",
) -> pd.DataFrame:
    """Rerank the top_n_to_rerank rows of df with cross-encoder.

    df must already be sorted best-first and contain 'candidate_id' and prior_score_col.
    Rows beyond top_n_to_rerank are appended unchanged after the reranked head.
    """
    df = df.reset_index(drop=True).copy()
    n = min(top_n_to_rerank, len(df))
    head = df.iloc[:n].copy()
    tail = df.iloc[n:].copy()

    cids = head["candidate_id"].tolist()
    texts = [build_candidate_summary(candidates_by_id[cid]) for cid in cids]

    ce_scores = reranker.score_pairs(jd_text, texts)
    head["cross_encoder_score"] = ce_scores

    # Preserve honeypot gate: any row with prior_score massively negative stays at bottom
    HP_MARKER = -1000.0
    is_gated = head[prior_score_col] < HP_MARKER

    # Normalize both signals to [0, 1] within the head only
    ce_min, ce_max = head["cross_encoder_score"].min(), head["cross_encoder_score"].max()
    prior_min, prior_max = head[prior_score_col].min(), head[prior_score_col].max()

    def _safe_norm(s, lo, hi):
        if hi - lo < 1e-9:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    ce_norm = _safe_norm(head["cross_encoder_score"], ce_min, ce_max)
    prior_norm = _safe_norm(head[prior_score_col], prior_min, prior_max)

    head["blended_score"] = beta * ce_norm + (1.0 - beta) * prior_norm
    head.loc[is_gated, "blended_score"] = -1e6

    # Tail rows don't have cross_encoder_score; use a placeholder 0.0 and set
    # blended_score to a value <= 0 so they stay below the reranked head.
    tail["cross_encoder_score"] = 0.0
    tail["blended_score"] = tail[prior_score_col].apply(lambda x: min(x, 0.0))

    # Sort head by blended, keep tail order (they didn't reach the reranker)
    head = head.sort_values(
        by=["blended_score", prior_score_col, "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    if tail.empty:
        return head
    return pd.concat([head, tail], ignore_index=True)
