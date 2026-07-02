# Reproducible CPU-only ranking image for the Redrob hackathon.
# Build-time precomputation fallback: embeddings, IDs, and silver scores are
# copied from the host because encoding the full 100K pool takes ~30 min.
# The MiniLM model is still downloaded at build time so runtime runs with
# --network=none.
FROM python:3.13-slim AS base

# System dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only torch is already pinned in requirements.txt).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source, scripts, configs, and staged challenge data.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/challenge/ ./data/challenge/

# Precomputed offline artifacts (fallback for build speed).
# These are produced by scripts/prebuild_index.py and scripts/score_full_population.py.
RUN mkdir -p data/processed outputs
COPY data/processed/candidate_embeddings.npy data/processed/candidate_ids.npy \
     data/processed/silver_scores_full.csv ./data/processed/

# Cache the sentence-transformer model at build time so runtime needs no network.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Runtime is fully offline.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Entrypoint wrapper generates the ranked submission and exposes it as submission.csv.
RUN sed -i 's/\r$//' scripts/docker_entrypoint.sh && chmod +x scripts/docker_entrypoint.sh
ENTRYPOINT ["./scripts/docker_entrypoint.sh"]
