# Sandbox / Live Demo Link

**URL:** https://huggingface.co/spaces/YOUR_USERNAME/redrob-candidate-ranker

## What it does
- Runs a simplified version of our ranking pipeline (BM25 + 7-family features + weighted rerank + grounded reasoning) on a small candidate sample.
- Supports either a bundled 50-sample or a JSONL upload (max 100 candidates per §10.5).
- Produces a top-10 list with reasonings, honeypot flags, and a downloadable CSV.

## What it does NOT do
- Does not use MiniLM dense embeddings (would exceed HF Spaces free-tier build limits).
- Does not run the full 100K pipeline (see the Docker instructions in `README.md` for that).

## Full pipeline reproduction
See `README.md > Reproduce with Docker` for the full 100K pipeline that produced `outputs/final_submission.csv`.
