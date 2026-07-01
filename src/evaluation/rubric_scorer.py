"""Deterministic, rule-based relevance scorer from docs/relevance_rubric_v1.md.

score_candidate(candidate, honeypot_val) -> (score_0_to_5, evidence_list)

No LLM, no learned weights.  Every rule is explicit and inspectable.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "configs" / "rubric_v1.yaml"
with open(_RUBRIC_PATH, "r", encoding="utf-8") as f:
    RUBRIC = yaml.safe_load(f)


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key, default)
        if obj is None:
            return default
    return obj


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _normalized_title(candidate: Dict[str, Any]) -> str:
    title = _normalize(_safe_get(candidate, "profile", "current_title"))
    if "@" in title:
        title = title.split("@")[0].strip()
    return title


def _title_matches(title: str, keywords: List[str]) -> bool:
    return any(kw in title for kw in keywords)


def _ml_title(title: str) -> bool:
    return _title_matches(title, RUBRIC["ml_titles"])


def _technical_adjacent_title(title: str) -> bool:
    return _title_matches(title, RUBRIC["technical_adjacent_titles"])


def _non_technical_title(title: str) -> bool:
    return _title_matches(title, RUBRIC["non_technical_titles"])


def _proficiency_level(proficiency: str) -> int:
    return _safe_get(
        RUBRIC, "proficiency_levels", _normalize(proficiency), default=0
    )


def _ml_skill_evidence(
    candidate: Dict[str, Any],
    skill_names: List[str],
    min_proficiency: int,
    min_duration: int,
) -> int:
    """Count distinct skills from skill_names that meet proficiency+duration."""
    skill_set = {s.strip().lower() for s in skill_names}
    count = 0
    seen = set()
    for skill in _safe_get(candidate, "skills", default=[]):
        name = _normalize(skill.get("name"))
        if not name or name in seen:
            continue
        if name not in skill_set:
            continue
        if _proficiency_level(skill.get("proficiency", "")) < min_proficiency:
            continue
        dur = skill.get("duration_months") or 0
        if dur < min_duration:
            continue
        seen.add(name)
        count += 1
    return count


def _career_texts(candidate: Dict[str, Any]) -> List[str]:
    texts = []
    for entry in _safe_get(candidate, "career_history", default=[]):
        desc = _normalize(entry.get("description"))
        if desc:
            texts.append(desc)
        title = _normalize(entry.get("title"))
        if title:
            texts.append(title)
    return texts


def _career_mentions_any(candidate: Dict[str, Any], phrases: List[str]) -> bool:
    texts = _career_texts(candidate)
    return any(phrase in text for text in texts for phrase in phrases)


def _career_bullet_mentions_ml_system(candidate: Dict[str, Any]) -> bool:
    return _career_mentions_any(candidate, RUBRIC["career_ml_system_phrases"])


def _career_shows_production_ml(candidate: Dict[str, Any]) -> bool:
    return _career_mentions_any(candidate, RUBRIC["career_production_phrases"])


def _career_has_software_data_engineering(candidate: Dict[str, Any]) -> bool:
    return _career_mentions_any(candidate, RUBRIC["career_software_data_phrases"])


def _india_based_or_relocating(candidate: Dict[str, Any]) -> bool:
    location = _normalize(_safe_get(candidate, "profile", "location"))
    country = _normalize(_safe_get(candidate, "profile", "country"))
    in_india = "india" in location or "india" in country
    relocate = bool(_safe_get(candidate, "redrob_signals", "willing_to_relocate"))
    return in_india or relocate


def _completeness(candidate: Dict[str, Any]) -> float:
    score = _safe_get(candidate, "redrob_signals", "profile_completeness_score")
    if score is None:
        return 0.0
    return float(score)


def _interview_completion_rate(candidate: Dict[str, Any]) -> float:
    rate = _safe_get(candidate, "redrob_signals", "interview_completion_rate")
    if rate is None:
        return 0.0
    return float(rate)


def _yoe(candidate: Dict[str, Any]) -> Any:
    return _safe_get(candidate, "profile", "years_of_experience")


def _has_any_skill_from(candidate: Dict[str, Any], skill_names: List[str]) -> bool:
    skill_set = {s.strip().lower() for s in skill_names}
    for skill in _safe_get(candidate, "skills", default=[]):
        if _normalize(skill.get("name")) in skill_set:
            return True
    return False


def _has_technical_skills(candidate: Dict[str, Any]) -> bool:
    return _has_any_skill_from(
        candidate, RUBRIC["ml_skills_broad"] + RUBRIC["overlapping_skills"]
    )


# ---------------------------------------------------------------------------
# Hard disqualifiers (Score 0)
# ---------------------------------------------------------------------------
def _is_hard_disqualifier(candidate: Dict[str, Any], evidence: List[str]) -> bool:
    yoe = _yoe(candidate)
    cfg = RUBRIC["thresholds"]["hard_zero"]

    if isinstance(yoe, (int, float)):
        if yoe < 0 or yoe > cfg["max_yoe"]:
            evidence.append(f"yoe={yoe} outside valid range")
            return True

    career = _safe_get(candidate, "career_history", default=[])
    if (not career or not isinstance(career, list)) and isinstance(yoe, (int, float)) and yoe > 0:
        evidence.append("empty_career_with_positive_yoe")
        return True

    title = _normalized_title(candidate)
    if _non_technical_title(title) and not _has_technical_skills(candidate):
        evidence.append("non_technical_title_without_technical_skills")
        return True

    verified_email = bool(_safe_get(candidate, "redrob_signals", "verified_email"))
    verified_phone = bool(_safe_get(candidate, "redrob_signals", "verified_phone"))
    completeness = _completeness(candidate)
    if not verified_email and not verified_phone and completeness < cfg["min_profile_completeness_unverified"]:
        evidence.append("unverified_and_low_completeness")
        return True

    return False


# ---------------------------------------------------------------------------
# Score-level predicates
# ---------------------------------------------------------------------------
def _meets_score_5(candidate: Dict[str, Any], evidence: List[str]) -> bool:
    cfg = RUBRIC["thresholds"]["score_5"]
    title = _normalized_title(candidate)

    if not _ml_title(title):
        return False

    yoe = _yoe(candidate)
    if not (isinstance(yoe, (int, float)) and cfg["min_yoe"] <= yoe <= cfg["max_yoe"]):
        return False

    strict_count = _ml_skill_evidence(
        candidate,
        RUBRIC["ml_skills_strict"],
        min_proficiency=_proficiency_level("advanced"),
        min_duration=cfg["min_skill_duration_months"],
    )
    if strict_count < cfg["min_strict_skills"]:
        return False

    if not _career_bullet_mentions_ml_system(candidate):
        return False

    if _interview_completion_rate(candidate) < cfg["min_interview_completion_rate"]:
        return False

    evidence.append(f"score_5: ml_title='{title}' yoe={yoe}")
    evidence.append(f"score_5: {strict_count} strict ML skills >= {cfg['min_strict_skills']} months")
    evidence.append("score_5: career mentions ranking/retrieval/LLM system")
    evidence.append(f"score_5: interview_completion_rate >= {cfg['min_interview_completion_rate']}")
    return True


def _meets_score_4(candidate: Dict[str, Any], evidence: List[str]) -> bool:
    cfg = RUBRIC["thresholds"]["score_4"]
    title = _normalized_title(candidate)

    if not (_ml_title(title) or _technical_adjacent_title(title)):
        return False

    yoe = _yoe(candidate)
    if not (isinstance(yoe, (int, float)) and cfg["min_yoe"] <= yoe <= cfg["max_yoe"]):
        return False

    broad_count = _ml_skill_evidence(
        candidate,
        RUBRIC["ml_skills_broad"],
        min_proficiency=_proficiency_level("advanced"),
        min_duration=cfg["min_skill_duration_months"],
    )
    if broad_count < cfg["min_advanced_skills"]:
        return False

    if not _career_shows_production_ml(candidate):
        return False

    if _completeness(candidate) < cfg["min_profile_completeness"]:
        return False

    evidence.append(f"score_4: adjacent_title='{title}' yoe={yoe}")
    evidence.append(f"score_4: {broad_count} broad ML/systems skills >= {cfg['min_advanced_skills']}")
    evidence.append("score_4: career shows production-scale data/ML systems")
    evidence.append(f"score_4: completeness >= {cfg['min_profile_completeness']}")
    return True


def _meets_score_3(candidate: Dict[str, Any], evidence: List[str]) -> bool:
    cfg = RUBRIC["thresholds"]["score_3"]

    yoe = _yoe(candidate)
    if not (isinstance(yoe, (int, float)) and cfg["min_yoe"] <= yoe <= cfg["max_yoe"]):
        return False

    broad_count = _ml_skill_evidence(
        candidate,
        RUBRIC["ml_skills_broad"],
        min_proficiency=_proficiency_level("intermediate"),
        min_duration=0,
    )
    if broad_count < cfg["min_broad_skills"]:
        return False

    if not _career_has_software_data_engineering(candidate):
        return False

    if not _india_based_or_relocating(candidate):
        return False

    evidence.append(f"score_3: yoe={yoe}")
    evidence.append(f"score_3: {broad_count} broad skill(s) at intermediate+")
    evidence.append("score_3: career includes software/data engineering")
    evidence.append("score_3: India-based or willing to relocate")
    return True


def _meets_score_2(candidate: Dict[str, Any], evidence: List[str]) -> bool:
    cfg = RUBRIC["thresholds"]["score_2"]

    yoe = _yoe(candidate)
    if not (isinstance(yoe, (int, float)) and cfg["min_yoe"] <= yoe <= cfg["max_yoe"]):
        return False

    # Path A: generic engineering overlap but no explicit ML/ranking evidence.
    has_overlap = _has_any_skill_from(candidate, RUBRIC["overlapping_skills"])
    meets_score_3_skill = (
        _ml_skill_evidence(
            candidate,
            RUBRIC["ml_skills_broad"],
            min_proficiency=_proficiency_level("intermediate"),
            min_duration=0,
        )
        >= 1
    )

    if has_overlap and not meets_score_3_skill:
        evidence.append(f"score_2: yoe={yoe}")
        evidence.append("score_2: overlapping engineering skills, no explicit ML/ranking")
        return True

    # Path B: YoE outside the 5-9 sweet spot but strong adjacent signals.
    title = _normalized_title(candidate)
    strong_adjacent = (
        (_ml_title(title) or _technical_adjacent_title(title))
        and _ml_skill_evidence(
            candidate,
            RUBRIC["ml_skills_broad"],
            min_proficiency=_proficiency_level("advanced"),
            min_duration=6,
        )
        >= 1
    )
    if strong_adjacent:
        evidence.append(f"score_2: yoe={yoe} outside 5-9 but strong adjacent signals")
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_candidate(
    candidate: Dict[str, Any], honeypot_val: int
) -> Tuple[int, List[str]]:
    """Return a 0-5 silver score and a list of human-readable evidence strings."""
    evidence: List[str] = []

    # 0. Honeypot gate (high-signal checks only).
    if honeypot_val >= 3:
        return 0, ["honeypot_gate_triggered"]

    # 0. Hard disqualifiers.
    if _is_hard_disqualifier(candidate, evidence):
        return 0, evidence

    # 5. Perfect fit (requires zero honeypot flags).
    if honeypot_val == 0 and _meets_score_5(candidate, evidence):
        return 5, evidence

    # 4. Strong fit.
    if _meets_score_4(candidate, evidence):
        return 4, evidence

    # 3. Reasonable fit.
    if _meets_score_3(candidate, evidence):
        return 3, evidence

    # 2. Marginal.
    if _meets_score_2(candidate, evidence):
        return 2, evidence

    # 1. Weak (default).
    yoe = _yoe(candidate)
    if isinstance(yoe, (int, float)) and yoe < RUBRIC["thresholds"]["score_1"]["max_yoe"]:
        evidence.append("score_1: very junior")
    elif _completeness(candidate) < RUBRIC["thresholds"]["score_1"]["min_profile_completeness"]:
        evidence.append("score_1: low completeness")
    else:
        evidence.append("score_1: default_weak")
    return 1, evidence
