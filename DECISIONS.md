# Architectural Decisions

## 2026-07-01 — Milestone 1: Initial scaffold + data staging
- Created `final/` project skeleton.
- Copied (did not move) 10 official challenge files into `final/data/challenge/`.
- Committed SHA-256 checksums to guarantee data integrity across the project lifetime.
- Python 3.11 was not available on the host; only Python 3.13.6 is installed. The originally pinned versions (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, lightgbm 4.3.0) do not ship pre-built wheels for Python 3.13 and fail to build from source without a C compiler. After confirming with the user, updated pins to Python 3.13-compatible versions: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, sentence-transformers 3.4.1, lightgbm 4.6.0, pytest 8.3.5, streamlit 1.41.1, tqdm 4.67.1, orjson 3.10.15, pyyaml 6.0.2.
- Excluded `candidates.jsonl` from git (large file); all other challenge files are committed to keep the repo self-contained for judges.

## 2026-07-01 — Relocation
- Project moved from C:\Users\ABINANIDA\Downloads\[PUB]...\Final to E:\redrob-hackathon\final due to C: drive space exhaustion during venv install.
- Official challenge source folder on C: remains READ-ONLY and untouched (verified by before/after diff).
- Installed torch 2.12.1+cpu from the PyTorch CPU wheel index (Python 3.13.6 requires torch >= 2.12; the requested torch 2.3.0 does not provide cp313 wheels).
