"""Transparent weighted-sum re-ranker over the 7-family feature matrix.

No hidden model — the score is a plain dot product between feature values and
hand-designed (or later learned) weights. Missing features are treated as 0 so
that ablations can intentionally drop features without crashing.
"""
from typing import Dict, List, Optional

import pandas as pd


class WeightedReranker:
    """Score candidates via a weighted sum of normalized features.

    Args:
        weights: mapping {feature_name: weight}. Weights are NOT normalized
            to sum to 1; the caller controls the scale.
        score_col: name of the output score column produced by ``rank``.
    """

    def __init__(self, weights: Dict[str, float], score_col: str = "score"):
        self.weights = {k: float(v) for k, v in weights.items()}
        self.score_col = score_col

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return weighted-sum scores for each row.

        Features present in ``df`` contribute ``df[feature] * weight``.
        Missing features contribute 0.
        """
        total = pd.Series(0.0, index=df.index)
        for feature, weight in self.weights.items():
            if feature in df.columns:
                total = total + df[feature].astype(float) * weight
        return total

    def rank(
        self,
        df: pd.DataFrame,
        tiebreak_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return ``df`` sorted descending by the weighted sum.

        A stable deterministic tiebreak uses ``candidate_id`` ascending by
        default; callers can supply additional columns (e.g. retrieval ranks).
        """
        df = df.copy()
        df[self.score_col] = self.score(df)

        ascending = [False]
        by = [self.score_col]

        if tiebreak_cols:
            by.extend(tiebreak_cols)
            ascending.extend([True] * len(tiebreak_cols))

        if "candidate_id" in df.columns:
            if "candidate_id" not in by:
                by.append("candidate_id")
                ascending.append(True)

        return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
