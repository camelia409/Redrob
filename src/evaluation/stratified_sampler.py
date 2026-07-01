"""Stratified candidate sampler for hand-labeling silver labels."""
import random
from typing import Dict, List, Set

from src.ingestion.loader import iter_candidates
from src.validation.honeypots import honeypot_score


ML_TITLES = {
    "ml engineer",
    "ai research engineer",
    "data scientist",
    "senior data engineer",
    "data engineer",
    "analytics engineer",
    "backend engineer",
    "senior software engineer",
}

NON_TECH_TITLES = {
    "hr manager",
    "marketing manager",
    "accountant",
    "sales executive",
    "content writer",
    "customer support",
    "operations manager",
    "business analyst",
    "project manager",
    "mechanical engineer",
    "civil engineer",
    "graphic designer",
}

ML_SKILLS = {
    "embeddings",
    "retrieval",
    "ranking",
    "llms",
    "fine-tuning",
    "vector search",
    "prompt engineering",
    "transformers",
    "nlp",
    "image classification",
    "speech recognition",
    "information retrieval",
    "feature engineering",
    "bentoml",
    "llamaindex",
    "langchain",
    "openai",
    "hugging face",
    "tensorflow",
    "pytorch",
    "kubeflow",
    "mlflow",
    "weaviate",
    "qdrant",
    "faiss",
    "elastic",
    "spark",
    "kafka",
    "airflow",
    "etl",
    "bigquery",
    "databricks",
    "aws",
}

TECH_SKILLS = {
    "python",
    "sql",
    "java",
    "go",
    "typescript",
    "javascript",
    "docker",
    "kubernetes",
    "ci/cd",
    "flask",
    "next.js",
    "angular",
    "react",
    "html",
    "css",
    "terraform",
    "apache beam",
    "redis",
    "mongodb",
    "postgres",
    "mysql",
}


def _count_matching_skills(candidate: Dict, skill_set: Set[str]) -> int:
    names = {s.get("name", "").strip().lower() for s in candidate.get("skills", []) if s.get("name")}
    return len(names & skill_set)


def _reservoir_sample(items: List[str], k: int, rng: random.Random) -> List[str]:
    """Reservoir sample up to k items from a list."""
    if len(items) <= k:
        return items
    sample = items[:k]
    for i, item in enumerate(items[k:], start=k):
        j = rng.randint(0, i)
        if j < k:
            sample[j] = item
    return sample


def sample_30_for_labeling(seed: int = 42) -> List[str]:
    """Return 30 candidate IDs stratified into high/mid/low relevance buckets."""
    rng = random.Random(seed)
    high_bucket: List[str] = []
    mid_bucket: List[str] = []
    low_bucket: List[str] = []

    for c in iter_candidates():
        prof = c.get("profile", {})
        title = str(prof.get("current_title", "")).strip().lower()
        yoe = prof.get("years_of_experience")
        country = str(prof.get("country", "")).strip().lower()
        ml_matches = _count_matching_skills(c, ML_SKILLS)
        tech_matches = _count_matching_skills(c, TECH_SKILLS)
        score = honeypot_score(c)

        # High: likely 4–5
        if title in ML_TITLES and isinstance(yoe, (int, float)) and 4 <= yoe <= 10 and ml_matches >= 2:
            high_bucket.append(c["candidate_id"])
        # Low: likely 0–1 (non-tech title OR honeypot score >= 2)
        elif title in NON_TECH_TITLES or score >= 2:
            low_bucket.append(c["candidate_id"])
        # Mid: likely 2–3
        elif isinstance(yoe, (int, float)) and 3 <= yoe <= 11 and tech_matches >= 1 and country == "india":
            mid_bucket.append(c["candidate_id"])

    selected = []
    selected.extend(_reservoir_sample(high_bucket, 10, rng))
    selected.extend(_reservoir_sample(mid_bucket, 10, rng))
    selected.extend(_reservoir_sample(low_bucket, 10, rng))
    return selected
