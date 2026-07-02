"""End-to-end pipeline: produce outputs/submission_v1.csv."""
import time
from pathlib import Path

import pandas as pd
import yaml

from src.features.extractor import FeatureExtractor
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRankerV2
from src.ranking.weighted_reranker import WeightedReranker
from src.reasoning.generator import generate
from src.reasoning.grounding import assert_grounded
from src.utils import integrity
from src.utils.paths import CONFIGS, OUTPUTS
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only, run_all_checks


TOP_K = 1500
WEIGHTS_PATH = CONFIGS / "reranker_weights_v1.yaml"
OUTPUT_PATH = OUTPUTS / "submission_v2.csv"
HP_PENALTY = -1e6


def load_weights(path: Path) -> dict[str, float]:
    """Load weights from YAML, accepting either ``weights_v1`` or ``weights``."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "weights_v1" in cfg:
        return cfg["weights_v1"]
    if "weights" in cfg:
        return cfg["weights"]
    raise ValueError(f"No weights block found in {path}")


def main() -> None:
    integrity.verify_data_integrity()
    t0 = time.time()

    candidates = list(iter_candidates())
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    print("Running BM25 (full JD)...")
    t1 = time.time()
    bm25 = BM25Ranker(get_jd_text())
    bm25_out = bm25.rank(candidates)
    print(f"  BM25 done in {time.time()-t1:.1f}s")

    print("Running Dense-v2 (distilled JD)...")
    t1 = time.time()
    dense_v2 = DenseRankerV2()
    dense_out = dense_v2.rank(candidates)
    print(f"  Dense-v2 done in {time.time()-t1:.1f}s")

    bm25_top_ids = {cid for cid, _ in bm25_out[:TOP_K]}
    dense_top_ids = {cid for cid, _ in dense_out[:TOP_K]}
    union_ids = bm25_top_ids | dense_top_ids
    print(f"Retrieval union size: {len(union_ids)} (BM25={len(bm25_top_ids)}, Dense={len(dense_top_ids)})")

    union_candidates = [c for c in candidates if c["candidate_id"] in union_ids]

    bm25_map = dict(bm25_out)
    dense_map = dict(dense_out)

    print("Computing honeypot scores for union...")
    dup_ids = find_duplicate_fingerprints(iter(candidates))
    hp_scores = {
        c["candidate_id"]: honeypot_score_gates_only(c, dup_ids)
        for c in union_candidates
    }

    print("Building feature matrix on union...")
    t1 = time.time()
    fe = FeatureExtractor()
    df = fe.extract_batch(union_candidates, bm25_map, dense_map, hp_scores)
    df["honeypot_score"] = df["candidate_id"].map(hp_scores)
    n = len(df)
    print(f"  feature matrix: {n} rows x {len(df.columns)} columns in {time.time()-t1:.1f}s")

    weights = load_weights(WEIGHTS_PATH)
    reranker = WeightedReranker(weights, score_col="reranker_score")
    df["reranker_score"] = reranker.score(df)

    # Honeypot gate: candidates with >=3 gate trips are forced to the end.
    gated_count = (df["honeypot_score"] >= 3).sum()
    df.loc[df["honeypot_score"] >= 3, "reranker_score"] += HP_PENALTY
    print(f"Honeypot gate applied to {gated_count} candidate(s)")

    # Deterministic tiebreak: original retrieval rank ascending, then id.
    if n > 1:
        df["dense_v2_rank"] = (1.0 - df["dense_v2_rank_within_union"]) * (n - 1) + 1
        df["bm25_rank"] = (1.0 - df["bm25_rank_within_union"]) * (n - 1) + 1
    else:
        df["dense_v2_rank"] = 1
        df["bm25_rank"] = 1

    df = df.sort_values(
        by=["reranker_score", "dense_v2_rank", "bm25_rank", "candidate_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    top100 = df.head(100).copy()
    top100["rank"] = range(1, 101)

    print("\nGenerating grounded reasonings for top 100...")
    candidate_map = {c["candidate_id"]: c for c in candidates}
    reasonings = []
    for _, row in top100.iterrows():
        cid = row["candidate_id"]
        c = candidate_map[cid]
        all_checks = run_all_checks(c)
        hp_flags = [name for name, (tripped, _) in all_checks.items() if tripped]
        features_row = row.to_dict()
        reasoning = generate(
            c,
            rank=int(row["rank"]),
            score=float(row["reranker_score"]),
            features_row=features_row,
            honeypot_flags=hp_flags,
        )
        assert_grounded(c, reasoning)
        reasonings.append(reasoning)
    top100["reasoning"] = reasonings
    print(f"  generated and grounded {len(reasonings)} reasonings")

    top100[["candidate_id", "rank", "reranker_score", "reasoning"]].rename(
        columns={"reranker_score": "score"}
    ).to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\nWrote {OUTPUT_PATH} in {elapsed:.1f}s")

    print("\nTop 5 candidates:")
    for _, row in top100.head(5).iterrows():
        cid = row["candidate_id"]
        c = candidate_map[cid]
        reasoning_preview = row["reasoning"][:90] + "..." if len(row["reasoning"]) > 90 else row["reasoning"]
        print(
            f"  #{row['rank']:>3d} {cid}  "
            f"{c['profile'].get('current_title', ''):<35s} "
            f"score={row['reranker_score']:.4f} hp={row['honeypot_score']}\n"
            f"      reasoning: {reasoning_preview}"
        )

    hp_in_top100 = (top100["honeypot_score"] >= 3).sum()
    print(f"\nHP@100 = {hp_in_top100}/100 ({hp_in_top100:.0%})")
    print(f"Total end-to-end runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
