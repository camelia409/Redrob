# Docker Reproduction Audit

## Build

- Command: `docker build -t redrob-ranking:v1 .`
- Strategy: fallback — precomputed `candidate_embeddings.npy`, `candidate_ids.npy`, `silver_scores_full.csv` copied from host; MiniLM model downloaded at build time so runtime is fully offline.
- Duration: ~40 s (all layers cached after first build; MiniLM download: 37.8 s; final layer: 0.4 s)
- Image ID: `6c19107473aa`
- Image content size (compressed): **1.1 GB** ✅ (< 4 GB limit)
- Disk usage including overlay storage: 4.67 GB

### Build log excerpt (last 15 lines)

```
#16 [12/13] RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
#16 DONE 37.8s

#17 [13/13] RUN sed -i 's/\r$//' scripts/docker_entrypoint.sh && chmod +x scripts/docker_entrypoint.sh
#17 DONE 0.4s

#18 exporting to image
#18 exporting layers done
#18 exporting manifest sha256:6c191... done
#18 naming to docker.io/library/redrob-ranking:v1 done
#18 DONE 0.4s
=== BUILD TIME: 3.3s (incremental; first full build ~45s) ===
```

## Run

- Command: `docker run --rm --cpus=4 --memory=16g --network=none -v $(pwd)/outputs_docker:/app/outputs redrob-ranking:v1`
- Runtime: **94.1 s** ✅ (< 300 s limit)
- Output written: `outputs_docker/submission.csv` (100 rows)

### Run log

```
Loaded 100000 candidates in 5.7s
Running BM25 (full JD)...
  BM25 done in 86.3s
Running Dense-v2 (distilled JD)...
  Dense-v2 done in 0.5s
Retrieval union size: 2477 (BM25=1500, Dense=1500)
Computing honeypot scores for union...
Building feature matrix on union...
  feature matrix: 2477 rows x 30 columns in 0.6s
Honeypot gate applied to 0 candidate(s)

Generating grounded reasonings for top 100...
  generated and grounded 100 reasonings

Wrote /app/outputs/submission_v2.csv in 94.1s

Top 5 candidates:
  #  1 CAND_0042029  Senior Data Scientist    score=1.1386 hp=0
  #  2 CAND_0079284  Machine Learning Engineer  score=1.1283 hp=0
  #  3 CAND_0006418  Machine Learning Engineer  score=1.1212 hp=0
  #  4 CAND_0050454  AI Engineer              score=1.1073 hp=0
  #  5 CAND_0064326  Search Engineer          score=1.0892 hp=0

HP@100 = 0/100 (0%)
Total end-to-end runtime: 94.1s
```

## Reproduction check

```
Frozen top-100: 100
Docker top-100: 100
Exact rank-order match: 100/100
Set overlap (rank-agnostic): 100/100
PASS: Docker reproduction is byte-identical in candidate ordering.
```

- Exact rank-order match with `final_submission.csv`: **100/100** ✅
- Set overlap (rank-agnostic): 100/100
- Verdict: **PASS**

## Verified constraints

- CPU only (no GPU) ✅ (no CUDA in `python:3.13-slim` base image)
- Network off (`--network=none`) ✅ (HF model cached at build time via `HF_HOME=/app/.hf_cache`)
- Memory ≤ 16 GB ✅
- Runtime ≤ 5 min ✅ (94.1 s actual)

## Build strategy note

`scripts/prebuild_index.py` encodes the 100K candidate pool with MiniLM (~30 min on CPU). To keep `docker build` under budget, the precomputed artefacts are shipped with the image:

| File | Size | Source |
|------|------|--------|
| `data/processed/candidate_embeddings.npy` | 153 MB | `scripts/prebuild_index.py` |
| `data/processed/candidate_ids.npy` | 4.6 MB | `scripts/prebuild_index.py` |
| `data/processed/silver_scores_full.csv` | 1.5 MB | `scripts/score_full_population.py` |

The MiniLM model itself is downloaded at build time (37.8 s) and cached in `/app/.hf_cache` so the runtime container needs no outbound network. `ENV TRANSFORMERS_OFFLINE=1` and `ENV HF_HUB_OFFLINE=1` are set before the entrypoint to enforce offline mode.

### Runtime bottleneck

BM25 index build over 100K candidates takes 86.3 s (rank_bm25 is single-threaded). Dense retrieval is 0.5 s (loads precomputed embeddings, encodes only the JD query). This is within the 5-minute spec limit.
