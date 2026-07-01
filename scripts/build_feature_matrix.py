"""Build the feature matrix for the top-1500 union of BM25 and Dense-v2."""
import csv
import time
from pathlib import Path

from src.features.extractor import FeatureExtractor
from src.ingestion.jd import get_jd_text
from src.ingestion.loader import iter_candidates
from src.ranking.baselines import BM25Ranker
from src.ranking.dense_ranker import DenseRankerV2
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import PROCESSED
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only


FULL_SCORES_PATH = PROCESSED / "silver_scores_full.csv"
TOP_K = 1500


def _load_csv_scores(path: Path) -> dict:
    scores = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["candidate_id"]] = int(row["silver_score"])
    return scores


def main() -> None:
    t0 = time.time()

    print("Loading 100K candidates...")
    candidates = list(iter_candidates())
    print(f"  loaded {len(candidates):,} candidates")

    print("Ranking with BM25 (full JD)...")
    jd_text = get_jd_text()
    bm25_ranker = BM25Ranker(jd_text=jd_text)
    bm25_out = bm25_ranker.rank(candidates)

    print("Ranking with Dense-v2 (distilled JD)...")
    index = EmbeddingIndex()
    index.load()
    dense_ranker = DenseRankerV2(index=index)
    dense_out = dense_ranker.rank(candidates)

    bm25_top_ids = [cid for cid, _ in bm25_out[:TOP_K]]
    dense_top_ids = [cid for cid, _ in dense_out[:TOP_K]]
    union_ids = set(bm25_top_ids) | set(dense_top_ids)
    print(f"  BM25 top-{TOP_K}: {len(bm25_top_ids)}, Dense-v2 top-{TOP_K}: {len(dense_top_ids)}")
    print(f"  Union size: {len(union_ids)}")

    bm25_score_map = dict(bm25_out)
    dense_score_map = dict(dense_out)
    silver_score_map = _load_csv_scores(FULL_SCORES_PATH)

    print("Computing honeypot scores for union...")
    dup_ids = find_duplicate_fingerprints(iter(candidates))
    hp_score_map = {
        c["candidate_id"]: honeypot_score_gates_only(c, dup_ids)
        for c in candidates
        if c["candidate_id"] in union_ids
    }

    union_candidates = [c for c in candidates if c["candidate_id"] in union_ids]

    print("Extracting features...")
    extractor = FeatureExtractor()
    df = extractor.extract_batch(
        union_candidates,
        bm25_scores={cid: bm25_score_map[cid] for cid in union_ids},
        dense_scores={cid: dense_score_map[cid] for cid in union_ids},
        hp_scores=hp_score_map,
    )

    df["silver_score"] = df["candidate_id"].map(silver_score_map)
    df["honeypot_score"] = df["candidate_id"].map(hp_score_map)
    df["bm25_score"] = df["candidate_id"].map(bm25_score_map)
    df["dense_score"] = df["candidate_id"].map(dense_score_map)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "feature_matrix.parquet"
    df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nWrote {out_path}")
    print(f"  rows={len(df):,}, columns={len(df.columns)}")
    print(f"  runtime={elapsed:.1f}s")


if __name__ == "__main__":
    main()
