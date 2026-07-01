"""Manual-inspection helper: compare BM25, Dense-v1, and Dense-v2 top-20.

Prints one compact line per candidate so a human can quickly rate plausibility,
keyword-stuffing, or hidden signal.
"""
import csv
from typing import Dict, List, Tuple

from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRanker, DenseRankerV2
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import CONFIGS, PROCESSED


FULL_SCORES_PATH = PROCESSED / "silver_scores_full.csv"
JD_TXT_PATH = CONFIGS / "jd.txt"


def _load_full_scores() -> Dict[str, int]:
    scores: Dict[str, int] = {}
    with open(FULL_SCORES_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["candidate_id"]] = int(row["silver_score"])
    return scores


def _top_skills(candidate: dict, n: int = 3) -> str:
    skills = candidate.get("skills", [])
    if not skills:
        return "n/a"
    # Use endorsement count as a proxy for signal strength.
    ordered = sorted(
        skills,
        key=lambda s: int(s.get("endorsements", 0) or 0),
        reverse=True,
    )
    return ", ".join(str(s.get("name", "")).strip().lower() for s in ordered[:n])


def _fmt_candidate(
    rank: int,
    candidate_id: str,
    candidate: dict,
    silver_score: int,
) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "n/a") or "n/a"
    yoe = profile.get("years_of_experience", 0.0) or 0.0
    headline = str(profile.get("headline", "") or "")[:60]
    skills = _top_skills(candidate)
    return (
        f"{rank:>2d}  {candidate_id:<14s}  {title:<30s}  {yoe:>4.1f}  "
        f"{skills:<40s}  silver={silver_score:<2d}  \"{headline}\""
    )


def _print_table(
    name: str,
    ranking: List[Tuple[str, float]],
    candidates_by_id: Dict[str, dict],
    full_scores: Dict[str, int],
    limit: int = 20,
) -> None:
    print(f"\n{'=' * 120}")
    print(f"{name} top-{limit}")
    print(f"{'=' * 120}")
    for rank, (candidate_id, _) in enumerate(ranking[:limit], start=1):
        candidate = candidates_by_id.get(candidate_id, {})
        silver_score = full_scores.get(candidate_id, 0)
        print(_fmt_candidate(rank, candidate_id, candidate, silver_score))


def main() -> None:
    print("Loading candidates and silver scores...")
    candidates = list(iter_candidates())
    candidates_by_id = {c["candidate_id"]: c for c in candidates}
    full_scores = _load_full_scores()
    print(f"  {len(candidates):,} candidates, {len(full_scores):,} scores")

    jd_text = get_jd_text()
    index = EmbeddingIndex()
    index.load()

    print("Ranking with BM25 (full JD)...")
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    bm25_out = bm25_ranker.rank(candidates)

    print("Ranking with Dense-v1 (full JD)...")
    dense_v1 = DenseRanker(index=index, jd_text=jd_text)
    dense_v1_out = dense_v1.rank(candidates)

    print("Ranking with Dense-v2 (distilled JD)...")
    dense_v2 = DenseRankerV2(index=index)
    dense_v2_out = dense_v2.rank(candidates)

    _print_table("BM25", bm25_out, candidates_by_id, full_scores)
    _print_table("Dense-v1 (full JD)", dense_v1_out, candidates_by_id, full_scores)
    _print_table("Dense-v2 (distilled JD)", dense_v2_out, candidates_by_id, full_scores)

    # Optional quick numeric overlap.
    bm25_set = {cid for cid, _ in bm25_out[:100]}
    dense_v1_set = {cid for cid, _ in dense_v1_out[:100]}
    dense_v2_set = {cid for cid, _ in dense_v2_out[:100]}
    print("\nTop-100 Jaccard diagnostics")
    print(f"  Jaccard(BM25, Dense-v1) = {len(bm25_set & dense_v1_set) / len(bm25_set | dense_v1_set):.3f}")
    print(f"  Jaccard(BM25, Dense-v2) = {len(bm25_set & dense_v2_set) / len(bm25_set | dense_v2_set):.3f}")
    print(f"  Jaccard(Dense-v1, Dense-v2) = {len(dense_v1_set & dense_v2_set) / len(dense_v1_set | dense_v2_set):.3f}")


if __name__ == "__main__":
    main()
