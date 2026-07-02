# Redrob Candidate Ranking

This repository is our submission to the India Runs Data and AI Challenge 2026, hosted by Redrob.

The challenge asks us to rank 100,000 synthetic candidate profiles against a single job description for a Senior AI Engineer role, and to return the top 100 candidates with a short reason for each ranking.

## What the system does

Given the job description in `data/challenge/job_description.docx` and the candidate pool in `data/challenge/candidates.jsonl`, our pipeline produces `outputs/final_submission.csv`. The CSV lists 100 candidates, ranked from best fit at rank 1 to weakest fit at rank 100. Every row has a one-sentence reason grounded in facts from the candidate's own profile.

The pipeline runs on a laptop with 16 GB of RAM, uses only CPU, and does not call any external service. Total runtime is about 90 seconds.

## How the pipeline works

The pipeline runs in five stages.

1. Retrieve. We run two independent searches over all 100,000 candidates. The first is BM25, a classic keyword-matching search. The second is a dense semantic search using MiniLM sentence embeddings against a distilled 200-word version of the job description. We keep the top 1,500 from each search and take their union, which gives us about 2,477 candidates.

2. Score features. For each of these 2,477 candidates, we compute 30 features across 7 families: semantic fit, skill evidence, career trajectory, production evidence, behavioral signals, location and logistics, and integrity risk. Every feature is a number between 0 and 1.

3. Rank. We combine the 30 features into a single score using hand-designed weights stored in `configs/reranker_weights_v1.yaml`. One weight is intentionally negative: raw count of strict AI keywords on a profile is a small penalty. We measured that this feature correlates negatively with true relevance, because keyword stuffers accumulate them without genuine fit.

4. Filter honeypots. We run 10 integrity checks on each candidate, such as impossible tenure lengths, expert claims with zero endorsements, and identical career descriptions across roles. Candidates who trip 3 or more checks are pushed to the bottom.

5. Generate reasons. For each of the top 100, we write a one-sentence reason using a template that pulls facts directly from the candidate's JSON. Every fact stated in a reason is checked against the source data. A candidate cannot be described with a skill they don't have or a company they didn't work at.

## Repository layout

```
data/challenge/          The 10 official challenge files (read only).
data/processed/          Embeddings and score caches (not in git).
data/silver/             Hand-labeled and rubric-labeled training data.
configs/                 YAML configuration for weights, features, honeypots.
src/                     Python source, one module per pipeline stage.
scripts/                 Command-line entry points.
tests/                   pytest suite (96 tests).
app/                     Colab notebook demo.
outputs/                 Final submission CSV.
docs/                    Design decisions, audits, and reports.
Dockerfile               Reproducible container.
requirements.txt         Pinned Python dependencies.
DECISIONS.md             Log of every architectural decision with reasoning.
```

## Reproduce the submission

You need Python 3.13 and about 5 GB of free disk.

```bash
python -m venv .venv
source .venv/Scripts/activate       # on Windows use .venv\Scripts\activate
pip install torch==2.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prebuild_index.py           # one time, about 30 minutes
python scripts/score_full_population.py    # one time, about 5 seconds
python scripts/generate_submission.py      # about 90 seconds
```

The output is `outputs/submission_v2.csv`. Compare against the frozen `outputs/final_submission.csv` using:

```bash
python scripts/verify_docker_reproduction.py
```

## Reproduce with Docker

Docker is the recommended path for judges. The image bakes in the MiniLM model and precomputed embeddings so the ranking step runs offline.

```bash
docker build -t redrob-ranking:v1 .
docker run --rm --cpus=4 --memory=16g --network=none \
  -v $(pwd)/outputs:/app/outputs \
  redrob-ranking:v1
```

The container writes the submission to `outputs/submission_v2.csv` in about 94 seconds and produces the same 100 candidates in the same order as the frozen file.

## Try the live demo

Open the Colab notebook in your browser:

https://colab.research.google.com/github/camelia409/Redrob/blob/main/app/demo_colab.ipynb

Click Runtime, then Run all. The notebook installs a few small packages, clones the repo, and ranks 50 sample candidates. Expected runtime is about 30 seconds. The last cell prints the top 10 candidates with their scores and reasons.

The Colab demo skips the dense embedding stage to stay small. The full pipeline in the repo does use dense embeddings.

## Numbers to know

Our own evaluation, using rubric-based silver labels on the full 100,000 pool:

| Metric | Value |
|---|---|
| Predicted composite score | 0.7762 |
| NDCG at 10 | 0.793 |
| NDCG at 50 | 0.893 |
| MAP | 0.412 |
| Precision at 10 | 1.00 |
| Honeypot rate in top 100 | 0 percent |
| Reasoning grounding pass rate | 100 out of 100 |
| End to end runtime | 90 seconds |

The composite metric weights are set by the challenge itself: 0.50 times NDCG at 10, plus 0.30 times NDCG at 50, plus 0.15 times MAP, plus 0.05 times precision at 10.

## What we deliberately did not use

We did not use a vector database. At 100,000 candidates and 384 dimensions, our embedding index is 153 MB. It fits in memory and a single matrix multiplication finds nearest neighbors in about 50 milliseconds. Adding a service like Pinecone or Weaviate would add network dependencies and reproducibility risk without any accuracy gain at this scale.

We did not use a large language model in the ranking or reasoning code. Reasons are template-based and every fact is checked against the source JSON. The challenge rules do not allow hosted LLM calls in the ranking step, and generating reasons with a local LLM would take longer than 5 minutes.

We did not fine-tune the embedding model. Off-the-shelf MiniLM is enough for the retrieval stage. Fine-tuning without real relevance labels would over-fit to our own rubric.

## Honest limitations

Our silver labels come from a rule-based scorer we wrote from the job description. They partially overlap with the features our ranker uses. This means our internal composite score is slightly optimistic. On 30 candidates that we labeled by hand, plain BM25 actually gives a slightly higher average manual rating than our weighted ranker. We disclose this rather than tune it away. The weighted ranker still wins on the composite metric that the judges use, and it wins by a wide margin on NDCG at 50.

## Team

To be filled in on submission.
