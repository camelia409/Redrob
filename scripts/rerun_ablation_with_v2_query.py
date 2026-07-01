"""Re-run ablation adding Dense-v2 (distilled JD) and Hybrid-v2 (BM25 + Dense-v2)."""
import csv as _csv
import time
from pathlib import Path

from src.evaluation.harness import evaluate_full_population, evaluate_labeled_subset
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker, RandomRanker, SkillCountRanker
from src.ranking.dense_ranker import DenseRanker, DenseRankerV2
from src.ranking.hybrid_ranker import HybridRRFRanker
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import OUTPUTS, PROCESSED, SILVER
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


SILVER_PATH = SILVER / "silver_labels_v1.csv"
FULL_SCORES_PATH = PROCESSED / "silver_scores_full.csv"


def _load_csv_scores(path: Path) -> dict:
    scores = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            scores[row["candidate_id"]] = int(row["silver_score"])
    return scores


def main() -> None:
    print("[1/4] Loading 100K candidates...")
    candidates = list(iter_candidates())
    print(f"      loaded {len(candidates):,} candidates")

    print("[2/4] Computing honeypot scores...")
    duplicate_ids = find_duplicate_fingerprints(iter(candidates))
    honeypot_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, duplicate_ids)
        for c in candidates
    }
    print(f"      computed {len(honeypot_scores):,} honeypot scores")

    print("[3/4] Loading silver labels and embedding index...")
    full_scores = _load_csv_scores(FULL_SCORES_PATH)
    labeled_scores = _load_csv_scores(SILVER_PATH)
    print(f"      full-pop scores: {len(full_scores):,}; labeled subset: {len(labeled_scores):,}")
    index = EmbeddingIndex()
    index.load()

    print("[4/4] Running ablation with distilled JD...\n")
    jd_text = get_jd_text()
    random_ranker = RandomRanker(seed=42)
    skill_ranker = SkillCountRanker()
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    dense_ranker = DenseRanker(index=index, jd_text=jd_text)
    dense_v2_ranker = DenseRankerV2(index=index)

    rankers = [
        ("random", random_ranker),
        ("skill_count", skill_ranker),
        ("bm25", bm25_ranker),
        ("dense", dense_ranker),
        ("dense_v2", dense_v2_ranker),
        ("hybrid_rrf", HybridRRFRanker(bm25_ranker, dense_ranker, k=60, per_ranker_top=2000)),
        ("hybrid_v2", HybridRRFRanker(bm25_ranker, dense_v2_ranker, k=60, per_ranker_top=2000)),
    ]

    results = []
    outputs_cache = {}
    for name, ranker in rankers:
        t0 = time.time()
        out = ranker.rank(candidates)
        elapsed = time.time() - t0
        outputs_cache[name] = out

        full_metrics = evaluate_full_population(
            out, full_scores, honeypot_scores, k_values=[10, 50, 100]
        )
        labeled_metrics = evaluate_labeled_subset(out, labeled_scores, k_values=[10])

        row = {
            "ranker": name,
            "runtime_sec": round(elapsed, 1),
            **full_metrics,
            **labeled_metrics,
        }
        results.append(row)
        print(
            f"{name:>12s}  "
            f"full-NDCG@10={full_metrics['ndcg@10']:.3f}  "
            f"full-NDCG@50={full_metrics['ndcg@50']:.3f}  "
            f"full-MAP={full_metrics['map']:.3f}  "
            f"full-P@10={full_metrics['p@10']:.2f}  "
            f"mean_silver@100={full_metrics['mean_silver@100']:.2f}  "
            f"HP@100={full_metrics['honeypot_rate@100']:.1%}  "
            f"labeled-NDCG@10={labeled_metrics['labeled_ndcg@10']:.3f}  "
            f"runtime={elapsed:.1f}s"
        )

    bm25_top100 = {cid for cid, _ in outputs_cache["bm25"][:100]}
    dense_v2_top100 = {cid for cid, _ in outputs_cache["dense_v2"][:100]}
    intersection = len(bm25_top100 & dense_v2_top100)
    union = len(bm25_top100 | dense_v2_top100)
    jaccard = intersection / union if union else 0.0
    print(f"\nTop-100 Jaccard(BM25, Dense-v2) = {jaccard:.3f} ({intersection}/{union})")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "ablation_v2_with_distilled_jd.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "ranker",
            "ndcg@10",
            "ndcg@50",
            "ndcg@100",
            "map",
            "p@10",
            "p@50",
            "p@100",
            "honeypot_rate@100",
            "mean_silver@100",
            "labeled_ndcg@10",
            "labeled_p@10",
            "labeled_map",
            "runtime_sec",
        ]
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
