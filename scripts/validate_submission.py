"""Wrapper around the challenge's submission validator."""
import sys
from pathlib import Path

# Make the challenge validator importable.
CHALLENGE_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge"
sys.path.insert(0, str(CHALLENGE_DIR))

from validate_submission import validate_submission  # noqa: E402


SUBMISSION_PATH = Path(__file__).resolve().parents[1] / "outputs" / "submission_v1.csv"


def main() -> None:
    path = SUBMISSION_PATH
    errors = validate_submission(path)
    if errors:
        print(f"Validation FAILED ({len(errors)} issue(s)):\n")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    print("Submission is valid.")


if __name__ == "__main__":
    main()
