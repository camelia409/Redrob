"""Interactive resumable CLI for labeling the union of two submission CSVs' top-100.

Usage:
    python -m scripts.label_submission_union
    python -m scripts.label_submission_union --auto

Reads outputs/submission_v3.csv and outputs/submission_v4.csv, computes their
top-100 union, reuses any existing labels from data/silver/manual_labels_v1.csv
and data/silver/manual_labels_top100_union.csv, and prompts for the rest (or
auto-labels with the rubric scorer when --auto is passed).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.rubric_scorer import score_candidate
from src.ingestion.loader import iter_candidates
from src.validation.duplicates import find_duplicate_fingerprints
from src.validation.honeypots import honeypot_score_gates_only, run_all_checks

ROOT = Path(__file__).resolve().parents[1]
V3_CSV = ROOT / "outputs" / "submission_v3.csv"
V4_CSV = ROOT / "outputs" / "submission_v4.csv"
LABELS_OUT = ROOT / "data" / "silver" / "manual_labels_top100_union.csv"
LABELS_V1 = ROOT / "data" / "silver" / "manual_labels_v1.csv"


def load_existing_labels() -> dict[str, tuple[int, str]]:
    """Return {candidate_id: (score, rationale)} from all prior label files."""
    labels: dict[str, tuple[int, str]] = {}
    for path in [LABELS_V1, LABELS_OUT]:
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("candidate_id", "").strip()
                    if not cid:
                        continue
                    score_field = (
                        row.get("score")
                        or row.get("rank_score")
                        or row.get("manual_score")
                    )
                    try:
                        score = int(float(score_field))
                    except (TypeError, ValueError):
                        continue
                    rationale = row.get("rationale", "").strip()
                    labels[cid] = (score, rationale)
    return labels


def get_union_ids() -> list[str]:
    v3 = pd.read_csv(V3_CSV)
    v4 = pd.read_csv(V4_CSV)
    return sorted(
        set(v3["candidate_id"].tolist()) | set(v4["candidate_id"].tolist())
    )


def load_candidates_for_ids(target_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for c in iter_candidates():
        cid = c["candidate_id"]
        if cid in target_ids:
            found[cid] = c
            if len(found) == len(target_ids):
                break
    return found


def show_candidate(
    cid: str,
    c: dict,
    position: int,
    total: int,
    v3_rank: int | None,
    v4_rank: int | None,
) -> None:
    p = c.get("profile", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})

    print("=" * 78)
    print(f"[{position}/{total}]  {cid}")
    print(f"v3 rank: {v3_rank if v3_rank else '-'}   v4 rank: {v4_rank if v4_rank else '-'}")
    print("-" * 78)
    print(
        f"Title    : {p.get('current_title', '?')} @ {p.get('current_company', '?')} "
        f"({p.get('years_of_experience', 0):.1f} YoE)"
    )
    print(f"Location : {p.get('location', '?')}, {p.get('country', '?')}")
    print(f"Headline : {(p.get('headline') or '')[:120]}")
    print()
    print("Top skills:")
    for s in skills[:6]:
        print(
            f"  - {s.get('name', '?'):<30} {s.get('proficiency', '?')}, "
            f"{s.get('duration_months', 0)}mo, {s.get('endorsements', 0)} endor"
        )
    print()
    print("Career (top 2):")
    for j in career[:2]:
        desc = (j.get("description") or "")[:130]
        print(
            f"  [{j.get('title', '?')} @ {j.get('company', '?')}, "
            f"{j.get('duration_months', 0)}mo]  {desc}"
        )
    print()
    flags = [k for k, (t, _) in run_all_checks(c).items() if t]
    print(
        f"Signals : resp={signals.get('recruiter_response_rate', 0):.2f}  "
        f"notice={signals.get('notice_period_days', 0)}d  "
        f"int_rate={signals.get('interview_completion_rate', 0):.2f}  "
        f"compl={signals.get('profile_completeness_score', 0)}"
    )
    print(f"HP flags: {', '.join(flags) if flags else 'none'}")
    print("=" * 78)
    print("Rubric: 5=perfect  4=strong  3=reasonable  2=marginal  1=weak  0=none")


def auto_label_all(
    to_label: list[str],
    candidates: dict[str, dict],
) -> list[tuple[str, int, str]]:
    """Score all unlabeled candidates with the deterministic rubric scorer."""
    dup_ids = find_duplicate_fingerprints(iter(candidates.values()))
    hp_scores = {
        cid: honeypot_score_gates_only(c, dup_ids)
        for cid, c in candidates.items()
    }
    rows = []
    for cid in to_label:
        c = candidates.get(cid)
        if c is None:
            print(f"WARN: candidate {cid} not found; skipping.")
            continue
        score, evidence = score_candidate(c, hp_scores.get(cid, 0))
        rationale = "; ".join(evidence) if evidence else "auto-rubric"
        rows.append((cid, score, rationale))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label the union of v3 and v4 top-100 submissions."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use the deterministic rubric scorer instead of interactive prompts.",
    )
    args = parser.parse_args()

    union_ids = get_union_ids()
    v3 = pd.read_csv(V3_CSV)
    v4 = pd.read_csv(V4_CSV)
    v3_rank_map = dict(zip(v3["candidate_id"], v3["rank"]))
    v4_rank_map = dict(zip(v4["candidate_id"], v4["rank"]))

    existing = load_existing_labels()
    to_label = [cid for cid in union_ids if cid not in existing]

    print(
        f"Union size: {len(union_ids)}  |  "
        f"Already labeled: {len(union_ids) - len(to_label)}  |  "
        f"Remaining: {len(to_label)}"
    )
    if not to_label:
        print("Nothing to label. Existing labels cover the full union.")
        return

    print("Loading candidate profiles from JSONL (one-time streaming)...")
    cands = load_candidates_for_ids(set(to_label))

    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not LABELS_OUT.exists()
    f = open(LABELS_OUT, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new_file:
        writer.writerow(["candidate_id", "score", "rationale", "labeled_at", "labeler"])
        f.flush()

    t0 = time.time()

    if args.auto:
        print("Auto-labeling remaining candidates with rubric scorer...")
        rows = auto_label_all(to_label, cands)
        for cid, score, rationale in rows:
            writer.writerow(
                [cid, score, rationale, time.strftime("%Y-%m-%dT%H:%M:%S"), "auto-rubric"]
            )
        f.flush()
        elapsed = time.time() - t0
        print(f"Auto-labeled {len(rows)} candidates in {elapsed:.1f}s.")
    else:
        for i, cid in enumerate(to_label, start=1):
            c = cands.get(cid)
            if c is None:
                print(f"WARN: candidate {cid} not found in JSONL; skipping.")
                continue
            show_candidate(
                cid, c, i, len(to_label), v3_rank_map.get(cid), v4_rank_map.get(cid)
            )

            raw = input("Score (0-5, q to quit): ").strip().lower()
            if raw == "q":
                print(f"\nSaved {i - 1} labels. Resume by running the script again.")
                break
            if raw not in {"0", "1", "2", "3", "4", "5"}:
                print("Invalid input, skipping this candidate.")
                continue
            score = int(raw)
            rationale = input("Rationale (>= 10 chars): ").strip()
            if len(rationale) < 10:
                print("Rationale too short, using placeholder.")
                rationale = "quick label"
            writer.writerow(
                [cid, score, rationale, time.strftime("%Y-%m-%dT%H:%M:%S"), "human"]
            )
            f.flush()

            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(to_label) - i)
            print(
                f"[time: {elapsed:.0f}s elapsed, {rate:.1f}s/candidate, ETA {eta:.0f}s]\n"
            )

    f.close()
    print("Done labeling. Next: run scripts/eval_submissions_on_labels.py")


if __name__ == "__main__":
    main()
