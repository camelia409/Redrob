#!/usr/bin/env bash
set -euo pipefail

# Generate the ranked submission inside the container and expose it with the
# canonical filename expected by the reproduction verifier.
python scripts/generate_submission.py

cp outputs/submission_v2.csv outputs/submission.csv
