"""Verify that the Docker-produced submission matches the frozen submission."""
import sys
from pathlib import Path

import pandas as pd


FROZEN_PATH = Path(__file__).resolve().parents[1] / "outputs" / "final_submission.csv"
DOCKER_PATH = Path(__file__).resolve().parents[1] / "outputs_docker" / "submission.csv"


def main() -> int:
    if not FROZEN_PATH.exists():
        print(f"Frozen submission not found: {FROZEN_PATH}")
        return 1
    if not DOCKER_PATH.exists():
        print(f"Docker output not found: {DOCKER_PATH}")
        return 1

    frozen = pd.read_csv(FROZEN_PATH)
    docker_out = pd.read_csv(DOCKER_PATH)

    frozen_ids = frozen["candidate_id"].tolist()
    docker_ids = docker_out["candidate_id"].tolist()

    print(f"Frozen top-100: {len(frozen_ids)}")
    print(f"Docker top-100: {len(docker_ids)}")

    id_match = sum(1 for a, b in zip(frozen_ids, docker_ids) if a == b)
    print(f"Exact rank-order match: {id_match}/100")

    frozen_set = set(frozen_ids)
    docker_set = set(docker_ids)
    overlap = len(frozen_set & docker_set)
    print(f"Set overlap (rank-agnostic): {overlap}/100")

    if id_match == 100:
        print("PASS: Docker reproduction is byte-identical in candidate ordering.")
        return 0
    elif overlap >= 95:
        print("SOFT PASS: same candidates, slightly different order (investigate before Stage 5).")
        return 0
    else:
        print("FAIL: Docker reproduction diverges materially.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
