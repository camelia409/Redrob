"""7 orthogonal feature families for the learned re-ranker.

Every function returns a dict[str, float] with values normalized to [0, 1].
"""
import re
from datetime import date, datetime
from typing import Any, Dict, List

import yaml

from src.utils.paths import CONFIGS


_RUBRIC_PATH = CONFIGS / "rubric_v1.yaml"
_FEATURES_PATH = CONFIGS / "features.yaml"

with open(_RUBRIC_PATH, "r", encoding="utf-8") as f:
    _RUBRIC = yaml.safe_load(f)

with open(_FEATURES_PATH, "r", encoding="utf-8") as f:
    _FEATURES_CFG = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def _today() -> date:
    ref = _FEATURES_CFG.get("reference_date", "2026-07-01")
    return datetime.strptime(ref, "%Y-%m-%d").date()


def _proficiency_value(proficiency: str) -> float:
    mapping = {
        "expert": 1.0,
        "advanced": 0.75,
        "intermediate": 0.5,
        "beginner": 0.25,
    }
    return mapping.get(str(proficiency).strip().lower(), 0.0)


def _ml_skills_set() -> set:
    return {s.lower() for s in _RUBRIC["ml_skills_strict"]} | {
        s.lower() for s in _RUBRIC["ml_skills_broad"]
    }


_ML_SKILLS = _ml_skills_set()
_ML_TITLES = {t.lower() for t in _RUBRIC["ml_titles"]}
_TECH_ADJACENT_TITLES = {t.lower() for t in _RUBRIC["technical_adjacent_titles"]}
_ML_SYSTEM_PHRASES = _RUBRIC["career_ml_system_phrases"]
_PRODUCTION_PHRASES = _RUBRIC["career_production_phrases"]
_PRODUCT_COMPANIES = {c.lower() for c in _FEATURES_CFG["product_companies"]}
_CONSULTING_FIRMS = {c.lower() for c in _FEATURES_CFG["consulting_firms"]}
_PREFERRED_CITIES = {c.lower() for c in _FEATURES_CFG["preferred_cities"]}
_TIER1_CITIES = {c.lower() for c in _FEATURES_CFG["tier1_indian_cities"]}


# ---------------------------------------------------------------------------
# Family 1: Semantic JD fit
# ---------------------------------------------------------------------------


def semantic_jd_fit(
    candidate: dict,
    dense_score: float,
    bm25_score_normalized: float = 0.0,
    rank_bm25: int | None = None,
    rank_dense: int | None = None,
    n_candidates: int = 1,
) -> Dict[str, float]:
    """Features measuring how well the candidate matches the JD query."""
    dense_score = max(0.0, min(1.0, float(dense_score)))
    bm25_norm = max(0.0, min(1.0, float(bm25_score_normalized)))

    agreement = 0.0
    if (
        rank_bm25 is not None
        and rank_dense is not None
        and n_candidates is not None
        and n_candidates > 1
    ):
        agreement = 1.0 - abs(rank_bm25 - rank_dense) / (n_candidates - 1)
        agreement = max(0.0, min(1.0, agreement))

    bm25_rank_within_union = 0.0
    dense_v2_rank_within_union = 0.0
    if n_candidates is not None and n_candidates > 1:
        if rank_bm25 is not None:
            bm25_rank_within_union = 1.0 - (rank_bm25 - 1) / (n_candidates - 1)
            bm25_rank_within_union = max(0.0, min(1.0, bm25_rank_within_union))
        if rank_dense is not None:
            dense_v2_rank_within_union = 1.0 - (rank_dense - 1) / (n_candidates - 1)
            dense_v2_rank_within_union = max(0.0, min(1.0, dense_v2_rank_within_union))

    return {
        "dense_v2_score": dense_score,
        "bm25_score_normalized": bm25_norm,
        "bm25_rank_within_union": bm25_rank_within_union,
        "dense_v2_rank_within_union": dense_v2_rank_within_union,
        "bm25_dense_rank_agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Family 2: Skill evidence
# ---------------------------------------------------------------------------


def skill_evidence(candidate: dict) -> Dict[str, float]:
    """Features based on skills that match the JD/rubric."""
    strict_set = {s.lower() for s in _RUBRIC["ml_skills_strict"]}
    broad_set = {s.lower() for s in _RUBRIC["ml_skills_broad"]}
    caps = _FEATURES_CFG["feature_caps"]

    skills = candidate.get("skills", [])
    strict_count = 0
    broad_count = 0
    max_proficiency = 0.0
    durations = []

    for skill in skills:
        name = str(skill.get("name", "")).strip().lower()
        if not name:
            continue
        if name in strict_set:
            strict_count += 1
        if name in broad_set or name in strict_set:
            broad_count += 1
            prof = _proficiency_value(skill.get("proficiency"))
            max_proficiency = max(max_proficiency, prof)
            dur = skill.get("duration_months", 0) or 0
            durations.append(dur)

    strict_count = min(strict_count, caps["strict_ml_skills"])
    broad_count = min(broad_count, caps["broad_ml_skills"])
    mean_duration = sum(durations) / len(durations) if durations else 0.0
    mean_duration = min(mean_duration, caps["skill_duration_months"])

    return {
        "strict_ml_skill_count_capped": strict_count / caps["strict_ml_skills"],
        "broad_ml_skill_count_capped": broad_count / caps["broad_ml_skills"],
        "max_skill_proficiency_ml": max_proficiency,
        "mean_skill_duration_months_ml_capped": mean_duration
        / caps["skill_duration_months"],
    }


# ---------------------------------------------------------------------------
# Family 3: Career trajectory
# ---------------------------------------------------------------------------


def career_trajectory(candidate: dict) -> Dict[str, float]:
    """Features based on years of experience and career history."""
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0.0) or 0.0)

    # Ideal band 5-9; linear falloff to 0 at yoe=3 and yoe=13.
    if 5.0 <= yoe <= 9.0:
        yoe_score = 1.0
    elif yoe < 5.0:
        yoe_score = max(0.0, (yoe - 3.0) / 2.0)
    else:
        yoe_score = max(0.0, (13.0 - yoe) / 4.0)

    history = candidate.get("career_history", [])
    caps = _FEATURES_CFG["feature_caps"]

    ml_count = 0
    tech_count = 0
    tenures = []
    for job in history:
        title = str(job.get("title", "")).strip().lower()
        if title in _ML_TITLES:
            ml_count += 1
        if title in _ML_TITLES or title in _TECH_ADJACENT_TITLES:
            tech_count += 1
        dur = job.get("duration_months", 0) or 0
        if dur:
            tenures.append(dur)

    total_jobs = len(history)
    ml_role_fraction = (ml_count / total_jobs) if total_jobs else 0.0
    tech_role_fraction = (tech_count / total_jobs) if total_jobs else 0.0

    # Career direction bonus: 1.0 if most recent role is ML and previous wasn't,
    # 0.5 if most recent is ML and any prior is ML, 0 otherwise.
    career_direction = 0.0
    if history:
        recent_title = str(history[0].get("title", "")).strip().lower()
        recent_is_ml = recent_title in _ML_TITLES
        prior_ml = any(
            str(job.get("title", "")).strip().lower() in _ML_TITLES
            for job in history[1:]
        )
        if recent_is_ml and not prior_ml:
            career_direction = 1.0
        elif recent_is_ml and prior_ml:
            career_direction = 0.5

    mean_tenure = sum(tenures) / len(tenures) if tenures else 0.0
    mean_tenure = min(mean_tenure, caps["mean_tenure_months"])

    return {
        "yoe_in_ideal_band": yoe_score,
        "ml_role_fraction": ml_role_fraction,
        "tech_role_fraction": tech_role_fraction,
        "career_direction_bonus": career_direction,
        "mean_tenure_months_capped": mean_tenure / caps["mean_tenure_months"],
    }


# ---------------------------------------------------------------------------
# Family 4: Production evidence
# ---------------------------------------------------------------------------


def production_evidence(candidate: dict) -> Dict[str, float]:
    """Features indicating production-scale ML/systems experience."""
    caps = _FEATURES_CFG["feature_caps"]
    history = candidate.get("career_history", [])

    all_text = " ".join(
        _normalize(job.get("description", "")) + " " + _normalize(job.get("title", ""))
        for job in history
    )

    ml_hits = sum(all_text.count(_normalize(p)) for p in _ML_SYSTEM_PHRASES)
    prod_hits = sum(all_text.count(_normalize(p)) for p in _PRODUCTION_PHRASES)
    ml_hits = min(ml_hits, caps["career_ml_phrases"])
    prod_hits = min(prod_hits, caps["career_production_phrases"])

    companies = [
        str(job.get("company", "")).strip().lower()
        for job in history
        if job.get("company")
    ]
    product_flag = 0.0
    consulting_flag = 0.0
    if companies:
        for company in companies:
            if any(pc in company for pc in _PRODUCT_COMPANIES):
                product_flag = 1.0
                break
        if all(any(cf in company for cf in _CONSULTING_FIRMS) for company in companies):
            consulting_flag = 1.0

    return {
        "career_ml_system_phrase_count_capped": ml_hits / caps["career_ml_phrases"],
        "career_production_phrase_count_capped": prod_hits
        / caps["career_production_phrases"],
        "product_company_flag": product_flag,
        "consulting_only_flag": consulting_flag,
    }


# ---------------------------------------------------------------------------
# Family 5: Behavioral
# ---------------------------------------------------------------------------


def behavioral(candidate: dict) -> Dict[str, float]:
    """Features from platform behavior signals."""
    signals = candidate.get("redrob_signals", {})

    # Recency score based on last_active_date.
    recency = 0.05
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last = datetime.strptime(str(last_active), "%Y-%m-%d").date()
            days = (_today() - last).days
            if days <= 30:
                recency = 1.0
            elif days <= 90:
                recency = 0.5
            elif days <= 180:
                recency = 0.25
            else:
                recency = 0.05
        except ValueError:
            recency = 0.05

    response_rate = max(0.0, min(1.0, float(signals.get("recruiter_response_rate", 0.0) or 0.0)))

    github = signals.get("github_activity_score", -1)
    if github is None or github == -1:
        github_score = 0.3
    else:
        github_score = min(1.0, max(0.0, float(github)) / 100.0)

    verified = [
        bool(signals.get("verified_email", False)),
        bool(signals.get("verified_phone", False)),
        bool(signals.get("linkedin_connected", False)),
    ]
    verified_score = sum(verified) / len(verified)

    interview_rate = max(
        0.0,
        min(1.0, float(signals.get("interview_completion_rate", 0.0) or 0.0)),
    )

    return {
        "recency_score": recency,
        "response_rate_score": response_rate,
        "github_score": github_score,
        "verified_score": verified_score,
        "interview_completion_rate": interview_rate,
    }


# ---------------------------------------------------------------------------
# Family 6: Logistics
# ---------------------------------------------------------------------------


def logistics(candidate: dict) -> Dict[str, float]:
    """Location, relocation, and notice-period features."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = str(profile.get("country", "")).strip().lower()
    location = str(profile.get("location", "")).strip().lower()

    india_score = 0.0
    if country == "india":
        india_score = 1.0
    elif bool(signals.get("willing_to_relocate", False)):
        india_score = 0.5

    location_lower = location.lower()
    if any(pc in location_lower for pc in _PREFERRED_CITIES):
        city_score = 1.0
    elif any(t1 in location_lower for t1 in _TIER1_CITIES):
        city_score = 0.7
    else:
        city_score = 0.3

    notice = int(signals.get("notice_period_days", 999) or 999)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.15

    return {
        "india_score": india_score,
        "preferred_city_bonus": city_score,
        "notice_period_score": notice_score,
    }


# ---------------------------------------------------------------------------
# Family 7: Integrity risk (inverted)
# ---------------------------------------------------------------------------


def integrity_risk(candidate: dict, honeypot_score_val: int | float) -> Dict[str, float]:
    """Integrity-related features; higher values mean lower risk."""
    hp = float(honeypot_score_val or 0)
    integrity_score = 1.0 - min(1.0, hp / 5.0)
    return {
        "integrity_score": integrity_score,
        "has_zero_gates": 1.0 if hp == 0 else 0.0,
    }
