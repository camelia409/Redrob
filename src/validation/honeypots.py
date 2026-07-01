"""Per-candidate honeypot checks. All thresholds live in configs/honeypot_rules.yaml."""
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "honeypot_rules.yaml"
with open(_RULES_PATH, "r", encoding="utf-8") as f:
    RULES = yaml.safe_load(f)


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key, default)
        if obj is None:
            return default
    return obj


def _format_float(value: float) -> str:
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# 1. timeline_inflation
# ---------------------------------------------------------------------------
def check_timeline_inflation(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Declared YoE vs cumulative career tenure."""
    yoe = _safe_get(candidate, "profile", "years_of_experience")
    career = _safe_get(candidate, "career_history", default=[])
    if yoe is None or not isinstance(career, list):
        return False, "insufficient data"
    total_months = sum((job.get("duration_months") or 0) for job in career)
    career_years = total_months / 12.0
    if career_years <= 0:
        return False, "insufficient data"
    ratio = yoe / career_years
    threshold = RULES["timeline_inflation"]["ratio_threshold"]
    if ratio > threshold:
        return True, f"yoe={yoe}, career_years={_format_float(career_years)}, ratio={_format_float(ratio)}"
    return False, f"ratio={_format_float(ratio)} below threshold {threshold}"


# ---------------------------------------------------------------------------
# 2. expert_zero_duration
# ---------------------------------------------------------------------------
def check_expert_zero_duration(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Many expert skills with zero declared duration."""
    skills = _safe_get(candidate, "skills", default=[])
    if not isinstance(skills, list) or not skills:
        return False, "insufficient data"
    count = sum(
        1
        for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and (s.get("duration_months") == 0 or s.get("duration_months") is None)
    )
    threshold = RULES["expert_zero_duration"]["min_expert_skills"]
    if count >= threshold:
        return True, f"expert skills with zero duration = {count}"
    return False, f"expert skills with zero duration = {count}"


# ---------------------------------------------------------------------------
# 3. expert_zero_endorsements
# ---------------------------------------------------------------------------
def check_expert_zero_endorsements(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Many expert skills with zero endorsements."""
    skills = _safe_get(candidate, "skills", default=[])
    if not isinstance(skills, list) or not skills:
        return False, "insufficient data"
    count = sum(
        1
        for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and (s.get("endorsements") == 0 or s.get("endorsements") is None)
    )
    threshold = RULES["expert_zero_endorsements"]["min_expert_skills"]
    if count >= threshold:
        return True, f"expert skills with zero endorsements = {count}"
    return False, f"expert skills with zero endorsements = {count}"


# ---------------------------------------------------------------------------
# 4. identical_career_descriptions
# ---------------------------------------------------------------------------
def check_identical_career_descriptions(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Two or more career entries share the exact same description."""
    career = _safe_get(candidate, "career_history", default=[])
    if not isinstance(career, list) or len(career) < 2:
        return False, "insufficient data"
    descriptions = [
        job.get("description", "").strip() for job in career if job.get("description")
    ]
    if len(descriptions) != len(set(descriptions)):
        return True, "duplicate descriptions found across career entries"
    return False, "all descriptions unique"


# ---------------------------------------------------------------------------
# 5. tenure_over_240_months
# ---------------------------------------------------------------------------
def check_tenure_over_240_months(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Any single job claims >20 years."""
    career = _safe_get(candidate, "career_history", default=[])
    if not isinstance(career, list) or not career:
        return False, "insufficient data"
    max_months = RULES["tenure_over_240_months"]["max_months"]
    for job in career:
        dur = job.get("duration_months") or 0
        if dur > max_months:
            return True, f"job '{job.get('title')}' at '{job.get('company')}' claims {dur} months"
    return False, f"no job exceeds {max_months} months"


# ---------------------------------------------------------------------------
# 6. consulting_keyword_stuffing
# ---------------------------------------------------------------------------
def _is_consulting_firm(company: str, firms: list) -> bool:
    company = company.lower()
    return any(firm in company for firm in firms)


def check_consulting_keyword_stuffing(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Consulting/IT shop + very long skill list."""
    company = _safe_get(candidate, "profile", "current_company", default="")
    skills = _safe_get(candidate, "skills", default=[])
    if not company or not isinstance(skills, list):
        return False, "insufficient data"
    cfg = RULES["consulting_keyword_stuffing"]
    if not _is_consulting_firm(company, cfg["firms"]):
        return False, f"company '{company}' not in consulting list"
    if len(skills) > cfg["min_skills"]:
        return True, f"consulting firm '{company}' with {len(skills)} skills"
    return False, f"consulting firm '{company}' with only {len(skills)} skills"


# ---------------------------------------------------------------------------
# 7. rookie_perfect_profile
# ---------------------------------------------------------------------------
def check_rookie_perfect_profile(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Very low YoE but near-perfect completeness score."""
    yoe = _safe_get(candidate, "profile", "years_of_experience")
    score = _safe_get(candidate, "redrob_signals", "profile_completeness_score")
    if yoe is None or score is None:
        return False, "insufficient data"
    cfg = RULES["rookie_perfect_profile"]
    if yoe <= cfg["max_yoe"] and score >= cfg["min_completeness"]:
        return True, f"yoe={yoe}, completeness={score}"
    return False, f"yoe={yoe}, completeness={score}"


# ---------------------------------------------------------------------------
# 8. stale_high_activity
# ---------------------------------------------------------------------------
def check_stale_high_activity(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Last active long ago but recent 30-day counters are non-zero."""
    sig = _safe_get(candidate, "redrob_signals", default={})
    if not isinstance(sig, dict):
        return False, "insufficient data"
    last_active = sig.get("last_active_date")
    if not last_active:
        return False, "insufficient data"
    cfg = RULES["stale_high_activity"]
    ref = datetime.date.fromisoformat(RULES["reference_date"])
    days = (ref - datetime.date.fromisoformat(last_active)).days
    if days < cfg["min_days_since_active"]:
        return False, f"last active {days}d ago"
    non_zero = []
    for counter in cfg["suspicious_counters"]:
        value = sig.get(counter)
        if value and value > 0:
            non_zero.append(f"{counter}={value}")
    if non_zero:
        return True, f"last active {days}d ago but {', '.join(non_zero)}"
    return False, f"last active {days}d ago with no contradictory recent counters"


# ---------------------------------------------------------------------------
# 9. skill_assessment_inversion
# ---------------------------------------------------------------------------
def check_skill_assessment_inversion(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Skill marked expert but assessment score is very low."""
    skills = _safe_get(candidate, "skills", default=[])
    sig = _safe_get(candidate, "redrob_signals", default={})
    scores = sig.get("skill_assessment_scores", {}) if isinstance(sig, dict) else {}
    if not isinstance(skills, list) or not isinstance(scores, dict):
        return False, "insufficient data"
    threshold = RULES["skill_assessment_inversion"]["expert_skill_max_assessment"]
    offenders = []
    for s in skills:
        if s.get("proficiency", "").lower() != "expert":
            continue
        name = s.get("name", "").strip()
        score = scores.get(name)
        if score is not None and score < threshold:
            offenders.append(f"{name}={score}")
    if offenders:
        return True, "; ".join(offenders)
    return False, "no expert skill with low assessment score"


# ---------------------------------------------------------------------------
# 10. duplicate_candidate_fingerprint
# ---------------------------------------------------------------------------
def check_duplicate_candidate_fingerprint(
    candidate: Dict[str, Any], duplicate_ids: set
) -> Tuple[bool, str]:
    """Candidate shares a deterministic fingerprint with another candidate."""
    cid = candidate.get("candidate_id")
    if cid is None:
        return False, "insufficient data"
    if cid in duplicate_ids:
        return True, f"fingerprint collision for {cid}"
    return False, "no fingerprint collision"


# ---------------------------------------------------------------------------
# Dispatcher + aggregator
# ---------------------------------------------------------------------------
ALL_CHECKS = [
    ("timeline_inflation", check_timeline_inflation),
    ("expert_zero_duration", check_expert_zero_duration),
    ("expert_zero_endorsements", check_expert_zero_endorsements),
    ("identical_career_descriptions", check_identical_career_descriptions),
    ("tenure_over_240_months", check_tenure_over_240_months),
    ("consulting_keyword_stuffing", check_consulting_keyword_stuffing),
    ("rookie_perfect_profile", check_rookie_perfect_profile),
    ("stale_high_activity", check_stale_high_activity),
    ("skill_assessment_inversion", check_skill_assessment_inversion),
    ("duplicate_candidate_fingerprint", check_duplicate_candidate_fingerprint),
]


def run_all_checks(
    candidate: Dict[str, Any], duplicate_ids: set = None
) -> Dict[str, Tuple[bool, str]]:
    """Run all honeypot checks on a single candidate."""
    duplicate_ids = duplicate_ids or set()
    results = {}
    for name, fn in ALL_CHECKS:
        if name == "duplicate_candidate_fingerprint":
            results[name] = fn(candidate, duplicate_ids)
        else:
            results[name] = fn(candidate)
    return results


def honeypot_score(candidate: Dict[str, Any], duplicate_ids: set = None) -> int:
    """Count of tripped checks (0–10)."""
    return sum(1 for tripped, _ in run_all_checks(candidate, duplicate_ids).values() if tripped)
