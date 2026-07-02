"""Template-based, grounded reasoning generator.

No LLM is used. Text is assembled from candidate JSON fields and a small
phrase bank. Variation is deterministic per candidate_id.
"""
from typing import Dict, List

from src.reasoning.phrase_bank import (
    CONCERN_CLAUSES,
    LEAD_PHRASES_11_30,
    LEAD_PHRASES_31_70,
    LEAD_PHRASES_71_100,
    LEAD_PHRASES_TOP10,
    NEUTRAL_MID_TIER_PHRASES,
    REASON_CLAUSES,
    STRENGTH_CLAUSES,
    candidate_seed,
    concern_phrases_for,
    reason_phrases_for,
)


_MIN_REASONING_LEN = 60
_MAX_REASONING_LEN = 200


def _proficiency_value(proficiency: str) -> float:
    mapping = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}
    return mapping.get(str(proficiency).strip().lower(), 0.0)


def _top_skills(candidate: Dict, n: int = 3) -> List[str]:
    """Return the top-N skill names by proficiency then duration."""
    skills = candidate.get("skills", [])
    scored = []
    for skill in skills:
        name = skill.get("name", "").strip()
        if not name:
            continue
        prof = _proficiency_value(skill.get("proficiency"))
        duration = int(skill.get("duration_months", 0) or 0)
        scored.append((name, prof, duration))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [name for name, _, _ in scored[:n]]


def _skill_kwargs(skill_names: List[str]) -> Dict[str, str]:
    """Map a variable-length skill list to skill1/skill2/skill3 placeholders."""
    kwargs: Dict[str, str] = {}
    for i in range(3):
        kwargs[f"skill{i + 1}"] = skill_names[i] if i < len(skill_names) else ""
    return kwargs


def _skill_summary(skill_names: List[str]) -> str:
    """Return a short grounded skill summary phrase."""
    if not skill_names:
        return "relevant experience"
    if len(skill_names) == 1:
        return f"skills include {skill_names[0]}"
    if len(skill_names) == 2:
        return f"skills include {skill_names[0]} and {skill_names[1]}"
    return f"skills include {skill_names[0]}, {skill_names[1]}, and {skill_names[2]}"


def _title_appears_ml_aligned(title: str) -> bool:
    """Return True if the current title already signals ML relevance."""
    t = str(title).lower()
    ml_hints = [
        "ml",
        "ai",
        "machine learning",
        "data scien",
        "research",
        "applied",
        "recommendation",
        "search",
        "nlp",
    ]
    return any(h in t for h in ml_hints)


def _choose(clauses: List[str], seed: int) -> str:
    return clauses[seed % len(clauses)]


def _filter_concerns(concerns: List[str], title: str) -> List[str]:
    """Drop self-contradicting title concerns when the title is already ML-aligned."""
    if not _title_appears_ml_aligned(title):
        return concerns
    filtered = []
    for c in concerns:
        cl = c.lower()
        if any(bad in cl for bad in ("not ml-focused", "weaker match for the role", "not a strong fit")):
            continue
        filtered.append(c)
    return filtered


def _filter_reasons(reasons: List[str], title: str) -> List[str]:
    """Drop self-contradicting title reasons when the title is already ML-aligned."""
    if not _title_appears_ml_aligned(title):
        return reasons
    filtered = []
    for r in reasons:
        rl = r.lower()
        if "current title" in rl and "not a strong fit" in rl:
            continue
        filtered.append(r)
    return filtered


def generate(
    candidate: Dict,
    rank: int,
    score: float,
    features_row: Dict,
    honeypot_flags: List[str],
) -> str:
    """Return a grounded, rank-aware reasoning string for one candidate."""
    profile = candidate.get("profile", {})
    candidate_id = candidate.get("candidate_id", "")
    title = profile.get("current_title", "?")
    company = profile.get("current_company", "?")
    yoe = float(profile.get("years_of_experience", 0.0) or 0.0)
    seed = candidate_seed(candidate_id)

    top_skills = _top_skills(candidate, 3)
    skills = _skill_kwargs(top_skills)
    skill_summary = _skill_summary(top_skills)

    def fmt(text: str) -> str:
        return text.format(
            title=title, company=company, yoe=yoe, skill_summary=skill_summary, **skills
        )

    if rank <= 10:
        lead = fmt(_choose(LEAD_PHRASES_TOP10, seed))
        text = lead
        if len(top_skills) >= 2:
            text += "; " + fmt(_choose(STRENGTH_CLAUSES, seed + 1))

    elif rank <= 30:
        lead = fmt(_choose(LEAD_PHRASES_11_30, seed))
        text = lead
        if len(top_skills) >= 1:
            text += "; " + fmt(_choose(STRENGTH_CLAUSES, seed + 2))

    elif rank <= 70:
        concerns = _filter_concerns(
            concern_phrases_for(candidate, honeypot_flags, features_row), title
        )
        if concerns:
            lead = fmt(_choose(LEAD_PHRASES_31_70, seed))
            text = lead
            if len(top_skills) >= 1:
                text += "; " + fmt(_choose(STRENGTH_CLAUSES, seed + 3))
            text += "; " + _choose(CONCERN_CLAUSES, seed).format(concern1=concerns[0])
        else:
            # Title is ML-aligned and no other concern; use neutral mid-tier language.
            text = fmt(_choose(NEUTRAL_MID_TIER_PHRASES, seed))

    else:
        lead = fmt(_choose(LEAD_PHRASES_71_100, seed))
        text = lead
        concerns = _filter_concerns(
            concern_phrases_for(candidate, honeypot_flags, features_row), title
        )
        reasons = _filter_reasons(
            reason_phrases_for(candidate, features_row), title
        )
        if concerns:
            text += "; " + _choose(CONCERN_CLAUSES, seed).format(concern1=concerns[0])
        if reasons:
            text += "; " + _choose(REASON_CLAUSES, seed).format(reason1=reasons[0])

    # Trim long reasonings; never truncate mid-word if possible.
    if len(text) > _MAX_REASONING_LEN:
        text = text[: _MAX_REASONING_LEN - 3].rsplit(" ", 1)[0] + "..."

    # If somehow too short, append a generic grounded clause.
    if len(text) < _MIN_REASONING_LEN:
        short_pad = f" Current profile: {title} at {company}, {yoe:.1f} years experience."
        text += short_pad
        if len(text) > _MAX_REASONING_LEN:
            text = text[: _MAX_REASONING_LEN - 3].rsplit(" ", 1)[0] + "..."

    return text
