"""Run all 5 rankers and produce the full-population + labeled-subset ablation."""
import csv as _csv
import time
from pathlib import Path

from src.evaluation.harness import evaluate_full_population, evaluate_labeled_subset
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker, RandomRanker, SkillCountRanker
from src.ranking.dense_ranker import DenseRanker
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
    print("[1/5] Loading 100K candidates...")
    candidates = list(iter_candidates())
    print(f"      loaded {len(candidates):,} candidates")

    print("[2/5] Computing honeypot scores...")
    duplicate_ids = find_duplicate_fingerprints(iter(candidates))
    honeypot_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, duplicate_ids)
        for c in candidates
    }
    print(f"      computed {len(honeypot_scores):,} honeypot scores")

    print("[3/5] Loading silver labels...")
    full_scores = _load_csv_scores(FULL_SCORES_PATH)
    labeled_scores = _load_csv_scores(SILVER_PATH)
    print(f"      full-pop scores: {len(full_scores):,}; labeled subset: {len(labeled_scores):,}")

    print("[4/5] Instantiating rankers...")
    jd_text = get_jd_text()
    random_ranker = RandomRanker(seed=42)
    skill_ranker = SkillCountRanker()
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    dense_ranker = DenseRanker(jd_text=jd_text)

    rankers = [
        ("random", random_ranker),
        ("skill_count", skill_ranker),
        ("bm25", bm25_ranker),
        ("dense", dense_ranker),
        ("hybrid_rrf", HybridRRFRanker(bm25_ranker, dense_ranker, k=60, per_ranker_top=2000)),
    ]

    print("[5/5] Running ablation...\n")
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

    # Jaccard diagnostic between BM25 and Dense top-100 sets.
    bm25_top100 = {cid for cid, _ in outputs_cache["bm25"][:100]}
    dense_top100 = {cid for cid, _ in outputs_cache["dense"][:100]}
    intersection = len(bm25_top100 & dense_top100)
    union = len(bm25_top100 | dense_top100)
    jaccard = intersection / union if union else 0.0
    print(f"\nTop-100 Jaccard(BM25, Dense) = {jaccard:.3f} ({intersection}/{union})")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "ablation_v2.csv"
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
