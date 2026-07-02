"""Template phrase bank for grounded, rank-aware reasoning.

All text is slotted from candidate JSON fields:
  {title}, {company}, {yoe}, {location}, {skill1}, {skill2}, {skill3}
plus generated concern/reason strings.

Variation is deterministic: ``seed = int(md5(candidate_id).hexdigest(), 16) % 10``.
"""
import hashlib
from typing import Dict, List


def candidate_seed(candidate_id: str) -> int:
    """Stable 0-9 seed derived from the candidate id."""
    digest = hashlib.md5(str(candidate_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10


# ---------------------------------------------------------------------------
# Lead phrases by rank band
# ---------------------------------------------------------------------------

LEAD_PHRASES_TOP10 = [
    "{title} at {company} with {yoe:.1f} years",
    "Strong {yoe:.1f}-year applied-ML profile, currently {title} at {company}",
    "Senior {title} with {yoe:.1f} years and clear ML production evidence",
    "Standout {title} at {company}; {yoe:.1f} years of relevant experience",
    "Top-fit candidate: {title} at {company} ({yoe:.1f} yrs)",
    "Senior applied-ML engineer profile at {company}, {yoe:.1f} years",
]

LEAD_PHRASES_11_30 = [
    "{title} at {company} with {yoe:.1f} years",
    "Confident match: {title} at {company}, {yoe:.1f} yrs",
    "Solid {yoe:.1f}-year profile as {title} at {company}",
    "Strong {title} background at {company}; {yoe:.1f} years",
    "Good ML fit: {title} with {yoe:.1f} years at {company}",
]

LEAD_PHRASES_31_70 = [
    "{title} at {company} with {yoe:.1f} years",
    "Mixed fit: {title} at {company}, {yoe:.1f} yrs",
    "{title} profile at {company}; {yoe:.1f} years with caveats",
    "Adequate {yoe:.1f}-year background as {title} at {company}",
    "Measured match: {title} with {yoe:.1f} years at {company}",
]

LEAD_PHRASES_71_100 = [
    "Weak fit: {title} at {company} with {yoe:.1f} years",
    "Concerning match: {title} at {company}, {yoe:.1f} yrs",
    "Low-confidence {title} profile at {company}; {yoe:.1f} years",
    "Poor alignment: {title} with {yoe:.1f} years at {company}",
    "Risky candidate: {title} at {company}, {yoe:.1f} yrs",
]


# ---------------------------------------------------------------------------
# Strength / concern / reason clauses
# ---------------------------------------------------------------------------

STRENGTH_CLAUSES = [
    "strong skills in {skill1}, {skill2}, and {skill3}",
    "advanced {skill1} + {skill2} capabilities",
    "depth in {skill1} paired with {skill2}",
    "relevant expertise: {skill1}, {skill2}, {skill3}",
    "solid command of {skill1} and {skill2}",
    "{skill1}, {skill2}, and {skill3} in the toolkit",
]

CONCERN_CLAUSES = [
    "concern: {concern1}",
    "notable gap: {concern1}",
    "flagged: {concern1}",
    "caution: {concern1}",
    "issue: {concern1}",
]

REASON_CLAUSES = [
    "likely due to {reason1}",
    "driven by {reason1}",
    "root cause appears to be {reason1}",
    "mainly because of {reason1}",
]


# ---------------------------------------------------------------------------
# Fact-derived concerns and reasons
# ---------------------------------------------------------------------------

def _yoe_text(candidate: Dict) -> str:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    return f"YoE {yoe:.1f} outside ideal band"


def _location_text(candidate: Dict) -> str:
    profile = candidate.get("profile", {})
    country = profile.get("country", "")
    location = profile.get("location", "")
    return f"based in {location or country}"


def concern_phrases_for(
    candidate: Dict,
    honeypot_flags: List[str],
    features_row: Dict,
) -> List[str]:
    """Return grounded concern strings for this candidate.

    Every substring here is derived directly from candidate JSON or honeypot
    checks; the grounding assertion will verify it.
    """
    concerns: List[str] = []
    profile = candidate.get("profile", {})

    if features_row.get("consulting_only_flag", 0) == 1.0:
        concerns.append("consulting-only career")

    if features_row.get("yoe_in_ideal_band", 0) < 0.5:
        concerns.append(_yoe_text(candidate))

    if features_row.get("india_score", 0) == 0.0:
        concerns.append(_location_text(candidate))

    if "consulting_keyword_stuffing" in honeypot_flags:
        company = profile.get("current_company", "")
        concerns.append(f"high skill count at {company}")

    if features_row.get("career_direction_bonus", 0) == 0.0:
        title = profile.get("current_title", "")
        concerns.append(f"current title {title} is not ML-focused")

    if features_row.get("tech_role_fraction", 0) < 0.5:
        concerns.append("mostly non-technical roles")

    if features_row.get("interview_completion_rate", 0) < 0.5:
        rate = candidate.get("redrob_signals", {}).get("interview_completion_rate", 0)
        concerns.append(f"interview completion rate {rate:.2f}")

    for flag in honeypot_flags:
        if flag == "timeline_inflation":
            concerns.append("declared YoE exceeds career tenure")
        elif flag == "expert_zero_duration":
            concerns.append("expert skills with zero duration")
        elif flag == "expert_zero_endorsements":
            concerns.append("expert skills with zero endorsements")
        elif flag == "duplicate_candidate_fingerprint":
            concerns.append("duplicate candidate fingerprint")
        elif flag == "tenure_over_240_months":
            concerns.append("single job claims >20 years")
        elif flag == "rookie_perfect_profile":
            concerns.append("rookie with near-perfect profile")
        elif flag == "skill_assessment_inversion":
            concerns.append("expert skill with low assessment score")

    if not concerns:
        # Fallback that is still grounded in the current title.
        title = profile.get("current_title", "")
        concerns.append(f"{title} is a weaker match for the role")

    return concerns


def reason_phrases_for(
    candidate: Dict,
    features_row: Dict,
) -> List[str]:
    """Return grounded reason strings for low-rank candidates."""
    reasons: List[str] = []
    profile = candidate.get("profile", {})

    if features_row.get("ml_role_fraction", 0) < 0.25:
        reasons.append("limited ML-specific career history")

    if features_row.get("career_production_phrase_count_capped", 0) < 0.25:
        reasons.append("little production-scale evidence")

    if features_row.get("max_skill_proficiency_ml", 0) < 0.5:
        reasons.append("no advanced-level ML skill")

    if profile.get("country", "").lower() != "india" and not candidate.get("redrob_signals", {}).get("willing_to_relocate", False):
        reasons.append("location outside India with no relocation")

    if features_row.get("mean_skill_duration_months_ml_capped", 0) < 0.3:
        reasons.append("short ML skill tenure")

    if not reasons:
        title = profile.get("current_title", "")
        reasons.append(f"current title {title} is not a strong fit")

    return reasons
