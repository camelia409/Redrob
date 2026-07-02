"""Print compact profiles of the 8 v3-vs-v4 diff candidates for hand rating."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import iter_candidates
from src.validation.honeypots import run_all_checks

NEW_IN_V4 = ["CAND_0002025", "CAND_0088025", "CAND_0011687", "CAND_0008425"]
DROPPED_FROM_V3 = ["CAND_0042029", "CAND_0060054", "CAND_0078002", "CAND_0068351"]
TARGETS = {cid: "NEW_IN_V4" for cid in NEW_IN_V4}
TARGETS.update({cid: "DROPPED_FROM_V3" for cid in DROPPED_FROM_V3})


def main() -> None:
    found = {}
    for c in iter_candidates():
        if c["candidate_id"] in TARGETS:
            found[c["candidate_id"]] = c
            if len(found) == 8:
                break

    for cid in NEW_IN_V4 + DROPPED_FROM_V3:
        c = found.get(cid)
        if not c:
            print(f"{cid} NOT FOUND")
            continue
        p = c.get("profile", {})
        skills = c.get("skills", [])
        career = c.get("career_history", [])
        signals = c.get("redrob_signals", {})
        flags = [k for k, (t, _) in run_all_checks(c).items() if t]

        print("=" * 82)
        print(f"{cid}   [{TARGETS[cid]}]")
        print("-" * 82)
        print(
            f"Title    : {p.get('current_title', '?')} @ {p.get('current_company', '?')}  "
            f"({p.get('years_of_experience', 0):.1f} YoE)"
        )
        print(f"Location : {p.get('location', '?')}, {p.get('country', '?')}")
        print(f"Headline : {(p.get('headline') or '')[:120]}")
        print(f"Summary  : {(p.get('summary') or '')[:200]}")
        print()
        print("Top skills:")
        for s in skills[:6]:
            print(
                f"  - {s.get('name', '?'):<32} {s.get('proficiency', '?'):<12} "
                f"{s.get('duration_months', 0)}mo  {s.get('endorsements', 0)} endor"
            )
        print()
        print("Career (top 2 roles):")
        for j in career[:2]:
            desc = (j.get("description") or "")[:160]
            print(
                f"  [{j.get('title', '?')} @ {j.get('company', '?')}, "
                f"{j.get('duration_months', 0)}mo]"
            )
            print(f"    {desc}")
        print()
        print(
            f"Signals : resp={signals.get('recruiter_response_rate', 0):.2f}  "
            f"notice={signals.get('notice_period_days', 0)}d  "
            f"int_rate={signals.get('interview_completion_rate', 0):.2f}  "
            f"compl={signals.get('profile_completeness_score', 0)}"
        )
        print(f"HP flags: {', '.join(flags) if flags else 'none'}")
        print("=" * 82)
        print()


if __name__ == "__main__":
    main()
