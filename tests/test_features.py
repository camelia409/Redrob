"""Tests for the 7 feature families.

Each family has one "strong candidate" test expecting high feature values and
one "weak candidate" test expecting low feature values.
"""
from datetime import date, timedelta

import pytest

from src.features.families import (
    behavioral,
    career_trajectory,
    integrity_risk,
    logistics,
    production_evidence,
    semantic_jd_fit,
    skill_evidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> date:
    return date(2026, 7, 1)


def _skill(name: str, proficiency: str = "intermediate", duration: int = 24):
    return {"name": name, "proficiency": proficiency, "duration_months": duration}


def _job(title: str, company: str, duration: int = 24, description: str = ""):
    return {
        "title": title,
        "company": company,
        "duration_months": duration,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Family 1: Semantic JD fit
# ---------------------------------------------------------------------------


def test_semantic_jd_fit_high():
    candidate = {"candidate_id": "CAND_HIGH"}
    feats = semantic_jd_fit(
        candidate,
        dense_score=0.95,
        bm25_score_normalized=0.90,
        rank_bm25=1,
        rank_dense=1,
        n_candidates=100,
    )
    assert feats["dense_v2_score"] == pytest.approx(0.95, abs=1e-6)
    assert feats["bm25_score_normalized"] == pytest.approx(0.90, abs=1e-6)
    assert feats["bm25_dense_rank_agreement"] == pytest.approx(1.0, abs=1e-6)


def test_semantic_jd_fit_low():
    candidate = {"candidate_id": "CAND_LOW"}
    feats = semantic_jd_fit(
        candidate,
        dense_score=0.10,
        bm25_score_normalized=0.05,
        rank_bm25=1,
        rank_dense=100,
        n_candidates=100,
    )
    assert feats["dense_v2_score"] == pytest.approx(0.10, abs=1e-6)
    assert feats["bm25_score_normalized"] == pytest.approx(0.05, abs=1e-6)
    assert feats["bm25_dense_rank_agreement"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 2: Skill evidence
# ---------------------------------------------------------------------------


def test_skill_evidence_high():
    candidate = {
        "skills": [
            _skill("embeddings", "expert", 36),
            _skill("retrieval", "expert", 36),
            _skill("ranking", "expert", 36),
            _skill("llms", "expert", 36),
            _skill("vector search", "expert", 36),
            _skill("python", "advanced", 36),
            _skill("nlp", "advanced", 36),
            _skill("deep learning", "advanced", 36),
            _skill("pytorch", "advanced", 36),
            _skill("rag", "advanced", 36),
        ]
    }
    feats = skill_evidence(candidate)
    assert feats["strict_ml_skill_count_capped"] == pytest.approx(1.0, abs=1e-6)
    assert feats["broad_ml_skill_count_capped"] == pytest.approx(1.0, abs=1e-6)
    assert feats["max_skill_proficiency_ml"] == pytest.approx(1.0, abs=1e-6)
    assert feats["mean_skill_duration_months_ml_capped"] == pytest.approx(1.0, abs=1e-6)


def test_skill_evidence_low():
    candidate = {
        "skills": [
            _skill("javascript", "beginner", 6),
            _skill("photoshop", "beginner", 6),
        ]
    }
    feats = skill_evidence(candidate)
    assert feats["strict_ml_skill_count_capped"] == pytest.approx(0.0, abs=1e-6)
    assert feats["broad_ml_skill_count_capped"] == pytest.approx(0.0, abs=1e-6)
    assert feats["max_skill_proficiency_ml"] == pytest.approx(0.0, abs=1e-6)
    assert feats["mean_skill_duration_months_ml_capped"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 3: Career trajectory
# ---------------------------------------------------------------------------


def test_career_trajectory_high():
    candidate = {
        "profile": {"years_of_experience": 7.0},
        "career_history": [
            _job("Machine Learning Engineer", "Google", 36),
            _job("AI Engineer", "Flipkart", 36),
        ],
    }
    feats = career_trajectory(candidate)
    assert feats["yoe_in_ideal_band"] == pytest.approx(1.0, abs=1e-6)
    assert feats["ml_role_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert feats["career_direction_bonus"] == pytest.approx(0.5, abs=1e-6)
    assert feats["mean_tenure_months_capped"] == pytest.approx(1.0, abs=1e-6)


def test_career_trajectory_low():
    candidate = {
        "profile": {"years_of_experience": 1.0},
        "career_history": [
            _job("HR Manager", "Infosys", 12),
            _job("Marketing Manager", "TCS", 12),
        ],
    }
    feats = career_trajectory(candidate)
    assert feats["yoe_in_ideal_band"] == pytest.approx(0.0, abs=1e-6)
    assert feats["ml_role_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert feats["career_direction_bonus"] == pytest.approx(0.0, abs=1e-6)
    assert feats["mean_tenure_months_capped"] == pytest.approx(12 / 36, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 4: Production evidence
# ---------------------------------------------------------------------------


def test_production_evidence_high():
    candidate = {
        "career_history": [
            _job(
                "Senior Machine Learning Engineer",
                "Google",
                36,
                description="Built production ranking and retrieval systems at scale. "
                "Deployed vector search and fine-tuning pipelines.",
            ),
            _job(
                "ML Engineer",
                "Flipkart",
                36,
                description="Production recommendation system serving billion requests.",
            ),
        ]
    }
    feats = production_evidence(candidate)
    assert feats["career_ml_system_phrase_count_capped"] > 0.5
    assert feats["career_production_phrase_count_capped"] > 0.5
    assert feats["product_company_flag"] == pytest.approx(1.0, abs=1e-6)
    assert feats["consulting_only_flag"] == pytest.approx(0.0, abs=1e-6)


def test_production_evidence_low():
    candidate = {
        "career_history": [
            _job(
                "Business Analyst",
                "Infosys",
                12,
                description="Created spreadsheets and attended meetings.",
            ),
            _job(
                "Project Manager",
                "Accenture",
                12,
                description="Managed client deliverables and timelines.",
            ),
        ]
    }
    feats = production_evidence(candidate)
    assert feats["career_ml_system_phrase_count_capped"] == pytest.approx(0.0, abs=1e-6)
    assert feats["career_production_phrase_count_capped"] == pytest.approx(0.0, abs=1e-6)
    assert feats["product_company_flag"] == pytest.approx(0.0, abs=1e-6)
    assert feats["consulting_only_flag"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 5: Behavioral
# ---------------------------------------------------------------------------


def test_behavioral_high():
    recent = (_today() - timedelta(days=10)).isoformat()
    candidate = {
        "redrob_signals": {
            "last_active_date": recent,
            "recruiter_response_rate": 0.95,
            "github_activity_score": 90,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "interview_completion_rate": 0.95,
        }
    }
    feats = behavioral(candidate)
    assert feats["recency_score"] == pytest.approx(1.0, abs=1e-6)
    assert feats["response_rate_score"] == pytest.approx(0.95, abs=1e-6)
    assert feats["github_score"] == pytest.approx(0.90, abs=1e-6)
    assert feats["verified_score"] == pytest.approx(1.0, abs=1e-6)
    assert feats["interview_completion_rate"] == pytest.approx(0.95, abs=1e-6)


def test_behavioral_low():
    old = (_today() - timedelta(days=365)).isoformat()
    candidate = {
        "redrob_signals": {
            "last_active_date": old,
            "recruiter_response_rate": 0.0,
            "github_activity_score": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "interview_completion_rate": 0.0,
        }
    }
    feats = behavioral(candidate)
    assert feats["recency_score"] == pytest.approx(0.05, abs=1e-6)
    assert feats["response_rate_score"] == pytest.approx(0.0, abs=1e-6)
    assert feats["github_score"] == pytest.approx(0.3, abs=1e-6)
    assert feats["verified_score"] == pytest.approx(0.0, abs=1e-6)
    assert feats["interview_completion_rate"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 6: Logistics
# ---------------------------------------------------------------------------


def test_logistics_high():
    candidate = {
        "profile": {"country": "India", "location": "Pune"},
        "redrob_signals": {"notice_period_days": 30, "willing_to_relocate": False},
    }
    feats = logistics(candidate)
    assert feats["india_score"] == pytest.approx(1.0, abs=1e-6)
    assert feats["preferred_city_bonus"] == pytest.approx(1.0, abs=1e-6)
    assert feats["notice_period_score"] == pytest.approx(1.0, abs=1e-6)


def test_logistics_low():
    candidate = {
        "profile": {"country": "Canada", "location": "Toronto"},
        "redrob_signals": {"notice_period_days": 120, "willing_to_relocate": False},
    }
    feats = logistics(candidate)
    assert feats["india_score"] == pytest.approx(0.0, abs=1e-6)
    assert feats["preferred_city_bonus"] == pytest.approx(0.3, abs=1e-6)
    assert feats["notice_period_score"] == pytest.approx(0.15, abs=1e-6)


# ---------------------------------------------------------------------------
# Family 7: Integrity risk
# ---------------------------------------------------------------------------


def test_integrity_risk_high():
    candidate = {"candidate_id": "CAND_CLEAN"}
    feats = integrity_risk(candidate, honeypot_score_val=0)
    assert feats["integrity_score"] == pytest.approx(1.0, abs=1e-6)
    assert feats["has_zero_gates"] == pytest.approx(1.0, abs=1e-6)


def test_integrity_risk_low():
    candidate = {"candidate_id": "CAND_RISKY"}
    feats = integrity_risk(candidate, honeypot_score_val=5)
    assert feats["integrity_score"] == pytest.approx(0.0, abs=1e-6)
    assert feats["has_zero_gates"] == pytest.approx(0.0, abs=1e-6)
