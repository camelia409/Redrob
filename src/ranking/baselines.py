"""Simple baseline rankers: random, skill-count, BM25.

Each ranker exposes `rank(candidates) -> list[tuple[str, float]]` sorted by
score descending.
"""
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from rank_bm25 import BM25Okapi


_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "configs" / "rubric_v1.yaml"
with open(_RUBRIC_PATH, "r", encoding="utf-8") as f:
    RUBRIC = yaml.safe_load(f)


def _tokenize(text: str) -> List[str]:
    """Simple BM25 tokenizer: lowercase, keep alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _candidate_text(candidate: Dict) -> str:
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
    return " ".join(str(p) for p in parts if p)


class RandomRanker:
    """Deterministic random baseline."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def rank(self, candidates: List[Dict]) -> List[Tuple[str, float]]:
        scores = [self.rng.random() for _ in candidates]
        pairs = [(c["candidate_id"], s) for c, s in zip(candidates, scores)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


class SkillCountRanker:
    """Score = count of broad ML/data/systems skills from the rubric."""

    def __init__(self):
        self.skill_set = {s.lower() for s in RUBRIC["ml_skills_broad"]}

    def rank(self, candidates: List[Dict]) -> List[Tuple[str, float]]:
        pairs = []
        for c in candidates:
            count = 0
            for skill in c.get("skills", []):
                name = str(skill.get("name", "")).strip().lower()
                if name in self.skill_set:
                    count += 1
            pairs.append((c["candidate_id"], float(count)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


class BM25Ranker:
    """BM25 over the candidate's full text profile vs. the JD text."""

    def __init__(self, jd_text: str):
        self.jd_tokens = _tokenize(jd_text)

    def rank(self, candidates: List[Dict]) -> List[Tuple[str, float]]:
        corpus = [_tokenize(_candidate_text(c)) for c in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(self.jd_tokens)
        pairs = [(c["candidate_id"], float(s)) for c, s in zip(candidates, scores)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
