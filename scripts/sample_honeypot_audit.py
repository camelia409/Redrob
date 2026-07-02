"""Sample 10 flagged and 10 top-100 candidates for the honeypot audit."""
import os
import sys
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.ingestion.loader import iter_candidates
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score, run_all_checks

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUB = ROOT / "outputs" / "final_submission.csv"

random.seed(42)

# Audit uses the full honeypot score (all 10 checks), matching the "415 flagged"
# count in the spec. The pipeline gate uses honeypot_score_gates_only, which is
# stricter; the audit intentionally looks at all tripped checks.
DUP_IDS = find_duplicate_fingerprints(iter_candidates())


def _print(candidate, hp_score, group, position, total):
    p = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    tripped = [k for k, (t, e) in run_all_checks(candidate).items() if t]

    print("=" * 82)
    print(f"[{group} {position}/{total}]  {candidate['candidate_id']}  hp_score={hp_score}")
    print("-" * 82)
    print(
        f"Title    : {p.get('current_title', '?')} @ {p.get('current_company', '?')}  "
        f"({p.get('years_of_experience', 0):.1f} YoE)"
    )
    print(f"Location : {p.get('location', '?')}, {p.get('country', '?')}")
    print(f"Headline : {(p.get('headline') or '')[:130]}")
    print()
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    print(f"Career total: {total_career_months} months = {total_career_months/12:.1f} years")
    for j in career[:3]:
        desc = (j.get("description") or "")[:120]
        print(
            f"  [{j.get('title', '?')} @ {j.get('company', '?')}, "
            f"{j.get('duration_months', 0)}mo]  {desc}"
        )
    print()
    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    expert_zero_endor = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0
    )
    print(
        f"Skills: {len(skills)} total, "
        f"{sum(1 for s in skills if s.get('proficiency') == 'expert')} expert, "
        f"{expert_zero} expert+0dur, {expert_zero_endor} expert+0endor"
    )
    for s in skills[:6]:
        print(
            f"  - {s.get('name', '?'):<30} {s.get('proficiency', '?'):<12} "
            f"{s.get('duration_months', 0)}mo  {s.get('endorsements', 0)} endor"
        )
    print()
    print(
        f"Signals: compl={signals.get('profile_completeness_score', 0)}  "
        f"resp={signals.get('recruiter_response_rate', 0):.2f}  "
        f"int_rate={signals.get('interview_completion_rate', 0):.2f}  "
        f"ver_email={signals.get('verified_email', False)}  "
        f"ver_phone={signals.get('verified_phone', False)}"
    )
    print(f"Tripped checks: {', '.join(tripped) if tripped else 'none'}")
    print("=" * 82)
    print()


def main() -> None:
    top100_ids = set(pd.read_csv(FINAL_SUB)["candidate_id"].tolist())

    flagged_reservoir: list[tuple[dict, int]] = []
    top100_reservoir: list[tuple[dict, int]] = []

    for c in iter_candidates():
        cid = c["candidate_id"]
        hp_score = honeypot_score(c, DUP_IDS)
        if hp_score >= 3:
            flagged_reservoir.append((c, hp_score))
        if cid in top100_ids:
            top100_reservoir.append((c, hp_score))

    flagged_sample = random.sample(
        flagged_reservoir, k=min(10, len(flagged_reservoir))
    )
    top100_sample = random.sample(
        top100_reservoir, k=min(10, len(top100_reservoir))
    )

    print("\n" + "#" * 82)
    print("# GROUP A — 10 candidates flagged with hp_score >= 3 (precision check)")
    print("# For each: is this profile ACTUALLY implausible?  (yes / no / uncertain)")
    print("#" * 82 + "\n")
    for i, (c, hp) in enumerate(flagged_sample, start=1):
        _print(c, hp, "A", i, 10)

    print("\n" + "#" * 82)
    print("# GROUP B — 10 candidates from our final top-100 (recall sanity check)")
    print("# For each: any red flags a recruiter would notice?  (none / minor / major)")
    print("#" * 82 + "\n")
    for i, (c, hp) in enumerate(top100_sample, start=1):
        _print(c, hp, "B", i, 10)


if __name__ == "__main__":
    main()
