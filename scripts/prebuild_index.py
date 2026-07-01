"""One-shot offline builder for the dense candidate embedding index."""
import time
from pathlib import Path

from src.ingestion.loader import iter_candidates
from src.retrieval.embeddings import EmbeddingIndex
from src.utils.paths import PROCESSED


def main() -> None:
    print("Streaming candidates from candidates.jsonl...")
    candidates = list(iter_candidates())
    print(f"Loaded {len(candidates):,} candidates")

    print("Building MiniLM embedding index (this is precomputation, not ranking)...")
    index = EmbeddingIndex(model_name="sentence-transformers/all-MiniLM-L6-v2")
    t0 = time.time()
    index.build(candidates, batch_size=64)
    elapsed = time.time() - t0

    emb_path = PROCESSED / "candidate_embeddings.npy"
    ids_path = PROCESSED / "candidate_ids.npy"
    emb_bytes = emb_path.stat().st_size
    ids_bytes = ids_path.stat().st_size

    print("\n=== Index build complete ===")
    print(f"Embeddings shape : {index.embeddings.shape}")
    print(f"Embeddings dtype : {index.embeddings.dtype}")
    print(f"Embeddings file  : {emb_path} ({emb_bytes / 1e6:.1f} MB)")
    print(f"IDs file         : {ids_path} ({ids_bytes / 1e3:.1f} KB)")
    print(f"Build time       : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
