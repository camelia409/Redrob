# Redrob Candidate Ranking

Submission for the India Runs Data and AI Challenge 2026, hosted by Redrob.

The challenge asks us to rank 100,000 synthetic candidate profiles against a Senior AI Engineer job description and return the top 100 with a short reason for each ranking.

## What the system does

Given the job description in `data/challenge/job_description.docx` and the candidate pool in `data/challenge/candidates.jsonl`, our pipeline produces `outputs/final_submission.csv` and `outputs/final_submission.xlsx`. Both list 100 candidates, ranked from best fit at rank 1 to weakest fit at rank 100. Every row has a one-sentence reason grounded in facts from the candidate's own profile.

The pipeline runs on a laptop with 16 GB of RAM, uses CPU only, and does not call any external service. Total runtime is 91 seconds against the challenge budget of 300 seconds.

## Key finding that shaped the design

We measured the correlation between every one of our engineered features and our silver relevance labels on the 2,477-candidate retrieval union. The strongest correlation is negative:

> Raw count of strict AI keywords on a candidate profile has a correlation of **-0.376** with our silver relevance labels.

Keyword stuffers accumulate JD terms without genuine career fit. The job description itself warns about this trap. We use raw keyword count as a small negative weight in the reranker rather than a positive signal. Every design decision falls out of this measurement. See `docs/family_ablation.md`.

## How the pipeline works

The pipeline has eight stages.

1. **Data integrity check.** Verify SHA-256 checksums on every input file. Refuses to run on tampered data.
2. **Candidate loading.** Stream the 100,000-candidate JSONL line by line.
3. **BM25 retrieval.** Score all 100,000 candidates against the job description using BM25. Keep the top 1,500.
4. **Dense semantic retrieval.** Encode a hand-distilled 200-word version of the job description with MiniLM sentence embeddings. Score all 100,000 candidates by cosine similarity to the JD embedding. Keep the top 1,500.
5. **Union and feature extraction.** Take the union of the two top-1,500 lists, which gives about 2,477 candidates. Compute 30 features across 7 orthogonal families: semantic fit, skill evidence, career trajectory, production evidence, behavioral signals, location and logistics, and integrity risk.
6. **Weighted rerank.** Combine the 30 features into a single score using hand-designed weights in `configs/reranker_weights_v1.yaml`. Apply a honeypot filter that pushes any candidate with three or more integrity flags to the bottom.
7. **Reciprocal Rank Fusion.** Fuse the weighted-reranker ranking with the BM25 ranking using RRF (Cormack et al. SIGIR 2009). This corrects for silver-label bias where the weighted reranker over-fits rubric-aligned signals.
8. **Grounded reasoning generation.** Write a one-sentence reason for each of the top 100 using a template that pulls facts directly from the candidate JSON. Every fact stated in a reason is verified against the source data. If any fact would be hallucinated, the pipeline fails loudly.

## Numbers

Silver-label evaluation on the 2,477 retrieval union candidates:

| Metric | Value |
|---|---:|
| **Composite** | **0.858 ± 0.035** (95% CI, bootstrap n=100) |
| NDCG@10 | 0.796 |
| NDCG@50 | 0.899 |
| MAP | 0.939 |
| P@10 | 1.00 |
| Mean silver score in top 100 | 3.78 |

Composite formula (from the challenge spec): `0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10`.

Integrity and reproducibility:

| Metric | Value |
|---|---:|
| Honeypot rate in top 100 | 0 / 100 |
| Grounded reasoning pass rate | 100 / 100 |
| Honeypot detector precision (10-sample audit) | 10 / 10 |
| Adversarial red-team pass rate (5 crafted attacks) | 5 / 5 |
| Docker reproduction match | 100 / 100 (byte-identical rank order) |
| End-to-end runtime | 91 seconds |
| BM25 share of runtime | 88.6% |
| Total unit tests | 105 |

Human-label spot check (48 candidates):

| Metric | Value |
|---|---:|
| Top-10 mean human rating (Milestone 13) | 4.7 / 5 |
| v3 vs v4 dropped candidates mean | 4.75 / 5 |
| v3 vs v4 promoted candidates mean | 4.50 / 5 |

## Reproduce locally

You need Python 3.13 and about 5 GB of free disk.

```bash
python -m venv .venv
source .venv/Scripts/activate       # on Windows use .venv\Scripts\activate
pip install torch==2.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prebuild_index.py           # one time, about 30 minutes
python scripts/score_full_population.py    # one time, about 5 seconds
python scripts/generate_submission.py      # about 91 seconds
```

Verify the output matches the frozen submission:

```bash
python scripts/verify_docker_reproduction.py
```

## Reproduce with Docker

Docker is the recommended path for judges. The image caches the MiniLM model and precomputed embeddings so the ranking step runs with no network.

```bash
docker build -t redrob-ranking:v1 .
docker run --rm --cpus=4 --memory=16g --network=none \
  -v $(pwd)/outputs:/app/outputs \
  redrob-ranking:v1
```

The container reproduces `outputs/final_submission.csv` byte-identically in candidate ordering in about 94 seconds.

## Try the live demo

Open the Colab notebook in a browser:

https://colab.research.google.com/github/camelia409/Redrob/blob/main/app/demo_colab.ipynb

Click Runtime, then Run all. The notebook installs a few small packages, clones the repo, and ranks 50 sample candidates with grounded reasoning. Total runtime after dependencies install is about 5 seconds.

The Colab demo uses BM25 plus the weighted reranker and honeypot filter. It skips MiniLM to keep the notebook small.

## Documentation

Every design decision, audit, and evaluation is documented. Open any of these:

**Design and decisions**
- `docs/MODEL_CARD.md` — model card in Mitchell et al. 2019 format
- `DECISIONS.md` — full architectural decision log (9 entries)
- `docs/relevance_rubric_v1.md` — the relevance rubric we scored against
- `docs/honeypot_hypotheses.md` — the 10 integrity checks we designed

**Evaluation**
- `outputs/composite_ci.txt` — bootstrap confidence intervals on composite
- `outputs/family_ablation.csv` and `docs/family_ablation.md` — per-family ablation
- `outputs/latency_budget.csv` and `docs/latency_budget.md` — per-stage runtime

**Audits**
- `docs/honeypot_audit.md` — precision + recall sanity check on the detector
- `docs/adversarial_red_team.md` — 5 hand-crafted attacks, all caught
- `docs/fairness_audit.md` — concentration and structural audit of the top 100
- `docs/failure_analysis.md` — the 10 lowest-silver candidates in our top 100
- `docs/pre_freeze_audit.md` — reproducibility + top-10/bottom-10 hand-rated
- `docs/docker_reproduction_audit.md` — Docker byte-identical reproduction

**EDA and rubric**
- `docs/eda_report.md` — dataset findings
- `notebooks/01_eda.ipynb` — executable EDA

## What we deliberately did not use

**No vector database.** At 100,000 vectors of 384 dimensions, the embedding index is 153 MB. It fits in memory. A single matrix multiplication finds nearest neighbors in about 50 milliseconds. Adding a hosted service like Pinecone or Weaviate would add network dependencies and reproducibility risk without any accuracy gain at this scale. For a production Redrob system with millions of candidates and streaming inserts, we would use pgvector or Qdrant with HNSW indexing.

**No large language model in the ranking or reasoning code.** The challenge rules forbid hosted LLM calls during ranking. A local LLM cannot run per candidate within the 91-second budget. Reasoning is template-based with a grounding assertion, giving us deterministic, provably-fact-checked output.

**No fine-tuning.** Off-the-shelf `sentence-transformers/all-MiniLM-L6-v2`. Fine-tuning without real relevance labels would over-fit our own rubric.

**No cross-encoder in production.** We tested a cross-encoder rerank stage (`cross-encoder/ms-marco-MiniLM-L-6-v2`). It underperformed on human labels (mean rating 4.50 vs 4.75 for the candidates it demoted). Disabled and archived in the codebase, not shipped. See `DECISIONS.md`.

**No learned reranker.** We considered LightGBM. Silver labels partially overlap with reranker features, so a learned model would fit our own rubric rather than genuine recruiter preference.

## Honest limitations

**Rubric inbreeding.** Our silver labels come from a rule-based scorer we wrote from the job description. They partially overlap with our reranker's features. Our internal composite is optimistic by an unknown amount.

**Small human-label set.** Only 48 candidates have real human ratings. The sample is too small to statistically distinguish small ranking improvements at 95% confidence.

**Career-trajectory dominates.** Per-family ablation shows removing `career_trajectory` collapses composite from 0.858 to 0.741, while removing every other family has near-zero effect. Two interpretations are possible: silver-label leakage into career features, or feature over-engineering elsewhere. See `docs/family_ablation.md`.

**One failure in top-100.** Our failure analysis identified one candidate at rank 64 with silver score 1 — a genuine ranking miss driven by JD-vocabulary presence without matching career shape. See `docs/failure_analysis.md`.

**Synthetic dataset.** The 100,000 candidates are synthetic. About 36% show `identical_career_descriptions` as a data-generation artifact. We handle this by excluding it from the honeypot gate.

## Repository layout

```
data/challenge/          The 10 official challenge files (read only).
data/processed/          Embeddings and score caches (not in git).
data/silver/             Hand-labeled and rubric-labeled evaluation data.
configs/                 YAML configuration for weights, features, honeypots.
src/                     Python source, one module per pipeline stage.
scripts/                 Command-line entry points and audit scripts.
tests/                   pytest suite (105 tests).
app/                     Colab notebook demo.
outputs/                 Final submission CSV and XLSX, audit CSVs.
docs/                    Model card, decisions, audits, and reports.
Dockerfile               Reproducible container.
requirements.txt         Pinned Python dependencies.
DECISIONS.md             Full architectural decision log.
```

## License and team

Hackathon submission. Not licensed for production hiring use.

Team: to be filled at submission time.
