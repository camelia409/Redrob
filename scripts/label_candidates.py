"""Interactive CLI for hand-labeling candidates against the v1 rubric.

Run from repo root:
    python -m scripts.label_candidates

Resumable: skips candidate_ids already present in data/silver/manual_labels_v1.csv.
"""
import csv
import datetime
import os
import sys
from pathlib import Path

from src.utils.io import read_csv_if_exists
from src.ingestion.loader import iter_candidates
from src.validation.honeypots import run_all_checks
from src.evaluation.stratified_sampler import sample_30_for_labeling


LABELER = "human_labeler_v1"
CSV_PATH = Path("data/silver/manual_labels_v1.csv")
JD_PREVIEW_LEN = 80


def _clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def _skill_line(skill: dict) -> str:
    name = skill.get("name", "?")
    prof = skill.get("proficiency", "?")
    dur = skill.get("duration_months", "?")
    end = skill.get("endorsements", "?")
    return f"{name:22s} ({prof}, {dur}mo, {end} endorsements)"


def _job_line(job: dict) -> str:
    title = job.get("title", "?")
    company = job.get("company", "?")
    dur = job.get("duration_months", "?")
    return f"[{title} @ {company}, {dur}mo]"


def _format_candidate(c: dict, idx: int, total: int, flags: dict) -> str:
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    lines = [
        "=" * 64,
        f"{c.get('candidate_id')}  (labeling {idx + 1} of {total})",
        "-" * 64,
        f"Title       : {prof.get('current_title')} @ {prof.get('current_company')}  ({prof.get('years_of_experience')} YoE)",
        f"Location    : {prof.get('location')}, {prof.get('country')}",
        "-" * 64,
        "Top skills  :",
    ]
    for s in c.get("skills", [])[:6]:
        lines.append(f"  - {_skill_line(s)}")
    lines.append("-" * 64)
    lines.append("Recent career:")
    for job in c.get("career_history", [])[:2]:
        lines.append(f"  {_job_line(job)}")
        desc = job.get("description", "")
        if desc:
            snippet = desc[:160].replace("\n", " ")
            lines.append(f"    {snippet}{'...' if len(desc) > 160 else ''}")
    lines.append("-" * 64)
    lines.append(
        f"Signals: resp_rate={sig.get('recruiter_response_rate')}  "
        f"notice={sig.get('notice_period_days')}d  "
        f"github={sig.get('github_activity_score')}  "
        f"compl={sig.get('profile_completeness_score')}"
    )
    lines.append(
        f"         interviews={sig.get('interview_completion_rate')}  "
        f"last_active={sig.get('last_active_date')}  "
        f"relocate={sig.get('willing_to_relocate')}"
    )
    tripped = [name for name, (t, _) in flags.items() if t]
    lines.append(f"Honeypot flags: {tripped if tripped else '[none]'}")
    lines.append("=" * 64)
    return "\n".join(lines)


def _prompt_score() -> str:
    print("Rubric: 5=perfect  4=strong  3=reasonable  2=marginal  1=weak  0=none")
    while True:
        raw = input("Score (0-5, q to quit): ").strip().lower()
        if raw == "q":
            return "q"
        if raw in {"0", "1", "2", "3", "4", "5"}:
            return raw
        print("Invalid input. Enter 0-5 or q.")


def _prompt_rationale() -> str:
    while True:
        raw = input("Rationale (one sentence, at least 15 chars): ").strip()
        if len(raw) >= 15:
            return raw
        print("Rationale too short; please explain the score.")


def main() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Determine which candidates still need labels
    existing = read_csv_if_exists(CSV_PATH)
    labeled_ids = {row["candidate_id"] for row in existing}
    print(f"Found {len(labeled_ids)} existing label(s).")

    selected_ids = sample_30_for_labeling(seed=42)
    remaining = [cid for cid in selected_ids if cid not in labeled_ids]
    print(f"{len(remaining)} of 30 candidates still need labels.")

    if not remaining:
        print("All 30 candidates are already labeled.")
        return

    # Build lookup for selected candidates
    candidate_map = {}
    for c in iter_candidates():
        cid = c.get("candidate_id")
        if cid in remaining:
            candidate_map[cid] = c
        if len(candidate_map) == len(remaining):
            break

    # Open CSV in append mode; write header if new
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["candidate_id", "rank_score", "rationale", "labeled_at", "labeler"])
            f.flush()

        for i, cid in enumerate(remaining):
            c = candidate_map[cid]
            flags = run_all_checks(c)
            _clear_screen()
            print(_format_candidate(c, i, len(remaining), flags))

            score = _prompt_score()
            if score == "q":
                print("Quitting. Progress saved.")
                break

            rationale = _prompt_rationale()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            writer.writerow([cid, score, rationale, now, LABELER])
            f.flush()

    print(f"\nLabels saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
