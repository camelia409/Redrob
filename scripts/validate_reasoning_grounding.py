"""Post-hoc audit: verify every reasoning in submission_v2.csv is grounded."""
import csv
import sys
from pathlib import Path

from src.ingestion.loader import iter_candidates
from src.reasoning.grounding import HallucinationError, assert_grounded
from src.utils.paths import OUTPUTS


SUBMISSION_PATH = OUTPUTS / "submission_v2.csv"


def main() -> None:
    if not SUBMISSION_PATH.exists():
        print(f"Submission not found: {SUBMISSION_PATH}")
        sys.exit(1)

    # Read submission rows.
    rows = []
    with open(SUBMISSION_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    target_ids = {row["candidate_id"] for row in rows}
    candidates = {
        c["candidate_id"]: c
        for c in iter_candidates()
        if c["candidate_id"] in target_ids
    }

    failures = []
    for row in rows:
        cid = row["candidate_id"]
        reasoning = row.get("reasoning", "")
        candidate = candidates.get(cid)
        if candidate is None:
            failures.append((cid, "candidate not found"))
            continue
        try:
            assert_grounded(candidate, reasoning)
        except HallucinationError as exc:
            failures.append((cid, str(exc)))

    if failures:
        print(f"FAIL: {len(failures)}/{len(rows)} reasoning(s) failed grounding")
        for cid, msg in failures[:20]:
            print(f"  {cid}: {msg}")
        sys.exit(1)

    print(f"PASS: {len(rows)}/{len(rows)} reasonings are grounded.")


if __name__ == "__main__":
    main()
