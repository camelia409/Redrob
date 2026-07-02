# Sandbox and live demo

## Public link

https://colab.research.google.com/github/camelia409/Redrob/blob/main/app/demo_colab.ipynb

## How to use it

Open the link in any browser. Sign in with a Google account if Colab asks. Choose Runtime, then Run all. The notebook does the following, in order:

1. Installs five small Python packages. Takes about 20 seconds.
2. Clones the GitHub repo into the Colab environment.
3. Loads the job description and 50 sample candidates.
4. Runs BM25 keyword retrieval.
5. Extracts 30 features across 7 families.
6. Applies the weighted rerank and the honeypot filter.
7. Prints the top 10 candidates with a reason for each ranking.

Total runtime after dependencies install is about 5 seconds.

## What the demo shows

The demo shows the same code paths as the full pipeline, but on a small sample so the notebook stays fast and does not need heavy dependencies. Specifically, it uses:

- BM25 keyword retrieval against the full job description.
- All 30 engineered features across the 7 families.
- The same weighted rerank with the same weights from `configs/reranker_weights_v1.yaml`.
- The same honeypot detector with 10 integrity checks.
- The same template-based reasoning generator with the grounding assertion.

## What the demo does not do

To keep the notebook small and fast, the demo skips two things:

- It does not use MiniLM dense embeddings. Downloading and running the model would add about 200 MB and 30 seconds. The demo uses BM25 rank as a substitute for the dense score in the feature matrix. This is a small approximation only, and it does not change the shape of the top 10 for the sample candidates.
- It runs on 50 candidates from `data/challenge/sample_candidates.json`, not the full 100,000. Colab is meant for small samples per the challenge spec.

## Full pipeline

The full 100,000-candidate pipeline, including MiniLM dense embeddings, is in the main repository and runs under Docker. See the "Reproduce with Docker" section of the top-level README.
