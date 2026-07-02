"""Pretty-print candidate profiles for a submission CSV (top-N and bottom-N)."""
import csv
import sys
from pathlib import Path

from src.ingestion.loader import iter_candidates


SUBMISSION_PATH = Path(__file__).resolve().parents[1] / "outputs" / "submission_v2.csv"


def _summarize(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    lines = [
        f"  candidate_id: {candidate.get('candidate_id')}",
        f"  title: {profile.get('current_title')} at {profile.get('current_company')}",
        f"  YoE: {profile.get('years_of_experience')} | location: {profile.get('location')}, {profile.get('country')}",
        f"  skills ({len(candidate.get('skills', []))}): "
        + ", ".join(
            f"{s['name']} ({s['proficiency']})" for s in candidate.get("skills", [])[:8]
        ),
        "  career_history:",
    ]
    for job in candidate.get("career_history", [])[:4]:
        lines.append(
            f"    - {job.get('title')} at {job.get('company')} ({job.get('duration_months')} mo)"
        )
    return "\n".join(lines)


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    bottom_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    target_ids = {r["candidate_id"] for r in rows[:top_n] + rows[-bottom_n:]}
    candidates = {
        c["candidate_id"]: c
        for c in iter_candidates()
        if c["candidate_id"] in target_ids
    }

    print(f"=== Top {top_n} ===\n")
    for row in rows[:top_n]:
        cid = row["candidate_id"]
        print(f"#{row['rank']} {cid}  score={row['score']}")
        print(f"reasoning: {row['reasoning']}")
        print(_summarize(candidates[cid]))
        print()

    print(f"=== Bottom {bottom_n} ===\n")
    for row in rows[-bottom_n:]:
        cid = row["candidate_id"]
        print(f"#{row['rank']} {cid}  score={row['score']}")
        print(f"reasoning: {row['reasoning']}")
        print(_summarize(candidates[cid]))
        print()


if __name__ == "__main__":
    main()
