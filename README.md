# Redrob Candidate Ranking — Team _TBD_

Intelligent Candidate Discovery & Ranking Challenge — India Runs Data & AI 2026.

## Overview

End-to-end candidate ranking pipeline: BM25 + dense (MiniLM) retrieval → weighted re-ranker → grounded reasoning generation. Produces a top-100 ranked submission from a 100K candidate pool.

## Try the live demo

🎯 **[https://huggingface.co/spaces/YOUR_USERNAME/redrob-candidate-ranker](https://huggingface.co/spaces/YOUR_USERNAME/redrob-candidate-ranker)**

Upload up to 100 candidates as JSONL, or use the bundled 50-sample, and see the top-10 with grounded reasoning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce with Docker

Prerequisites: Docker installed, 16 GB RAM, 5 GB free disk.

```bash
docker build -t redrob-ranking:v1 .

docker run --rm \
  --cpus=4 --memory=16g --network=none \
  -v $(pwd)/outputs:/app/outputs \
  redrob-ranking:v1
```

The output `outputs/submission.csv` should match `outputs/final_submission.csv` in candidate ordering.

### Compute environment (author's machine)

- Windows 11 Home Single Language (10.0.26200), running Docker Desktop 29.4.3
- CPU: x86-64 (4 cores allocated to container)
- RAM: 16 GB (container limit)
- Python: 3.13 (inside container, `python:3.13-slim` base)
- Total pipeline runtime: ~90 s

### Verify reproduction

```bash
mkdir -p outputs_docker
docker run --rm \
  --cpus=4 --memory=16g --network=none \
  -v $(pwd)/outputs_docker:/app/outputs \
  redrob-ranking:v1

python scripts/verify_docker_reproduction.py
```

## Team
_TBD_

## License
Hackathon submission. Not for production hiring use.
