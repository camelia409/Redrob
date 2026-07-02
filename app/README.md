---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: streamlit_app.py
pinned: false
---

# Redrob Candidate Ranker — Live Demo

India Runs Data & AI Challenge 2026 submission.

Runs a simplified BM25 + 7-family feature reranker + grounded reasoning pipeline
on a small candidate sample. Full 100K pipeline is available in the GitHub repository.

## How to use
1. Choose the bundled 50-sample or upload your own `candidates.jsonl` (max 100).
2. Click "Rank candidates."
3. See top-10 with grounded reasonings and honeypot flags.

## Full architecture
See the GitHub README for the full 100K pipeline including MiniLM dense embeddings,
hybrid RRF retrieval, and Docker reproduction.
