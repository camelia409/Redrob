"""Run the pipeline with instrumented per-stage timing, write latency budget CSV.

This does NOT modify the pipeline. It runs generate_submission.py's internals
inline with explicit time.perf_counter() calls around each stage.
"""
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.features.extractor import FeatureExtractor
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRankerV2
from src.ranking.rrf_finalizer import fuse_rankings, load_rrf_config
from src.ranking.weighted_reranker import WeightedReranker
from src.reasoning.generator import generate
from src.reasoning.grounding import assert_grounded
from src.utils.integrity import verify_data_integrity
from src.validation.honeypots import honeypot_score_gates_only, run_all_checks

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "configs" / "reranker_weights_v1.yaml"
RRF_CFG = ROOT / "configs" / "rrf_finalizer.yaml"
OUT = ROOT / "outputs" / "latency_budget.csv"

BUDGET_SEC = 300.0  # from challenge spec


def _time() -> float:
    return time.perf_counter()


def main() -> None:
    stages: list[tuple[str, float]] = []
    total_t0 = _time()

    # Data integrity check
    t0 = _time()
    verify_data_integrity()
    stages.append(("data_integrity_check", _time() - t0))

    # Load candidates
    t0 = _time()
    candidates = list(iter_candidates())
    stages.append(("load_candidates", _time() - t0))

    # BM25 retrieval
    t0 = _time()
    bm25 = BM25Ranker(jd_text=get_jd_text())
    bm25_out = bm25.rank(candidates)
    stages.append(("bm25_retrieval", _time() - t0))

    # Dense retrieval
    t0 = _time()
    dense = DenseRankerV2()
    dense_out = dense.rank(candidates)
    stages.append(("dense_retrieval", _time() - t0))

    # Union + feature extraction
    t0 = _time()
    bm25_top = {cid for cid, _ in bm25_out[:1500]}
    dense_top = {cid for cid, _ in dense_out[:1500]}
    union_ids = bm25_top | dense_top
    union = [c for c in candidates if c["candidate_id"] in union_ids]
    hp_scores = {c["candidate_id"]: honeypot_score_gates_only(c) for c in union}
    bm25_map = dict(bm25_out)
    dense_map = dict(dense_out)
    fe = FeatureExtractor()
    df = fe.extract_batch(union, bm25_map, dense_map, hp_scores)
    stages.append(("union_and_feature_extraction", _time() - t0))

    # Weighted rerank
    t0 = _time()
    with open(WEIGHTS, encoding="utf-8") as f:
        w_cfg = yaml.safe_load(f)
    weights = w_cfg.get("weights_v1", w_cfg)
    reranker = WeightedReranker(weights)
    df["reranker_score"] = reranker.score(df)
    df["honeypot_score"] = df["candidate_id"].map(hp_scores)
    df.loc[df["honeypot_score"] >= 3, "reranker_score"] -= 1e6
    stages.append(("weighted_rerank", _time() - t0))

    # RRF fusion
    t0 = _time()
    rrf_cfg = load_rrf_config(RRF_CFG)
    if rrf_cfg.get("enabled", True):
        df = fuse_rankings(df, bm25_out, alpha=rrf_cfg["alpha"], k=rrf_cfg["k"])
    stages.append(("rrf_fusion", _time() - t0))

    # Reasoning generation on top 100
    t0 = _time()
    sort_col = "fused_score" if "fused_score" in df.columns else "reranker_score"
    df_sorted = df.sort_values(
        by=[sort_col, "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    top100 = df_sorted.head(100).copy()
    top100["rank"] = range(1, 101)
    cands_by_id = {c["candidate_id"]: c for c in candidates}
    for _, row in top100.iterrows():
        cid = row["candidate_id"]
        c = cands_by_id[cid]
        flags = [k for k, (tripped, _) in run_all_checks(c).items() if tripped]
        reason = generate(
            c,
            int(row["rank"]),
            float(row.get("fused_score", row["reranker_score"])),
            row.to_dict(),
            flags,
        )
        assert_grounded(c, reason)
    stages.append(("reasoning_generation_and_grounding", _time() - t0))

    total_sec = _time() - total_t0
    stages.append(("TOTAL", total_sec))

    # Write CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "seconds", "pct_of_pipeline", "pct_of_5min_budget"])
        for name, sec in stages:
            pct_pipe = sec / total_sec * 100 if total_sec else 0
            pct_budget = sec / BUDGET_SEC * 100
            w.writerow([name, f"{sec:.2f}", f"{pct_pipe:.1f}", f"{pct_budget:.1f}"])

    # Print
    print("\n" + "=" * 78)
    print(
        f"{'stage':<40} {'seconds':>10} {'% pipeline':>12} {'% 5-min budget':>16}"
    )
    print("-" * 78)
    for name, sec in stages:
        pct_pipe = sec / total_sec * 100 if total_sec else 0
        pct_budget = sec / BUDGET_SEC * 100
        marker = " *" if name == "TOTAL" else ""
        print(
            f"{name:<40} {sec:>10.2f} {pct_pipe:>11.1f}% {pct_budget:>15.1f}%{marker}"
        )
    print("=" * 78)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
