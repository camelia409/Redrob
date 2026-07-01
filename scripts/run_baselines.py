"""Run random / skill-count / BM25 baselines and measure against silver labels."""
import csv as _csv
import time
from pathlib import Path

from src.evaluation.harness import evaluate, evaluate_restricted
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker, RandomRanker, SkillCountRanker
from src.utils.paths import OUTPUTS, SILVER
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


CANDIDATES_GZ = Path(__file__).resolve().parents[1] / "data" / "challenge" / "candidates.jsonl"
SILVER_PATH = SILVER / "silver_labels_v1.csv"


def _load_silver_labels(path: Path) -> dict:
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            labels[row["candidate_id"]] = int(row["silver_score"])
    return labels


def main() -> None:
    print("[1/4] Loading 100K candidates (materialize for repeated ranking)...")
    candidates = list(iter_candidates())
    print(f"      loaded {len(candidates):,} candidates")

    print("[2/4] Computing duplicate fingerprints + honeypot scores...")
    duplicate_ids = find_duplicate_fingerprints(iter(candidates))
    honeypot_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, duplicate_ids)
        for c in candidates
    }
    print(f"      computed honeypot scores for {len(honeypot_scores):,} candidates")

    print("[3/4] Loading silver labels...")
    labels = _load_silver_labels(SILVER_PATH)
    print(f"      loaded {len(labels):,} silver labels")

    print("[4/4] Running baselines...\n")
    rankers = {
        "random": RandomRanker(seed=42),
        "skill_count": SkillCountRanker(),
        "bm25": BM25Ranker(jd_text=get_jd_text()),
    }

    results = []
    for name, ranker in rankers.items():
        t0 = time.time()
        out = ranker.rank(candidates)
        elapsed = time.time() - t0
        metrics = evaluate(out, labels, honeypot_scores, k_values=[10, 50, 100])
        metrics["ranker"] = name
        metrics["runtime_sec"] = round(elapsed, 1)
        results.append(metrics)
        print(
            f"{name:>12s}  "
            f"NDCG@10={metrics['ndcg@10']:.3f}  "
            f"NDCG@50={metrics['ndcg@50']:.3f}  "
            f"NDCG@100={metrics['ndcg@100']:.3f}  "
            f"MAP={metrics['map']:.3f}  "
            f"P@10={metrics['p@10']:.2f}  "
            f"P@100={metrics['p@100']:.2f}  "
            f"HP@100={metrics['honeypot_rate@100']:.1%}  "
            f"runtime={elapsed:.1f}s"
        )

        labeled_in_top = {
            k: sum(1 for cid, _ in out[:k] if cid in labels) for k in (10, 50, 100)
        }
        print(
            f"              labeled in top 10/50/100: "
            f"{labeled_in_top[10]}/{labeled_in_top[50]}/{labeled_in_top[100]}"
        )

        restricted = evaluate_restricted(
            out, labels, honeypot_scores, k_values=[10, 50, 100]
        )
        print(
            f"              restricted  NDCG@10={restricted['ndcg@10']:.3f}  "
            f"NDCG@50={restricted['ndcg@50']:.3f}  "
            f"MAP={restricted['map']:.3f}  "
            f"P@10={restricted['p@10']:.2f}"
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "ablation_baselines.csv"
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
            "runtime_sec",
        ]
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
