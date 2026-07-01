"""Tests for honeypot checks and duplicate detector."""
import pytest

from src.validation.honeypots import (
    check_timeline_inflation,
    check_expert_zero_duration,
    check_expert_zero_endorsements,
    check_identical_career_descriptions,
    check_tenure_over_240_months,
    check_consulting_keyword_stuffing,
    check_rookie_perfect_profile,
    check_stale_high_activity,
    check_skill_assessment_inversion,
    check_duplicate_candidate_fingerprint,
    run_all_checks,
    honeypot_score,
)
from src.validation.duplicates import find_duplicate_fingerprints


def _candidate(**kwargs):
    """Minimal candidate builder."""
    base = {
        "candidate_id": kwargs.get("candidate_id", "CAND_TEST"),
        "profile": {},
        "career_history": [],
        "skills": [],
        "redrob_signals": {},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. timeline_inflation
# ---------------------------------------------------------------------------
def test_timeline_inflation_positive():
    c = _candidate(
        profile={"years_of_experience": 16},
        career_history=[{"duration_months": 60}, {"duration_months": 30}],
    )
    tripped, evidence = check_timeline_inflation(c)
    assert tripped
    assert "ratio=" in evidence


def test_timeline_inflation_negative():
    c = _candidate(
        profile={"years_of_experience": 6},
        career_history=[{"duration_months": 60}, {"duration_months": 30}],
    )
    tripped, _ = check_timeline_inflation(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 2. expert_zero_duration
# ---------------------------------------------------------------------------
def test_expert_zero_duration_positive():
    c = _candidate(
        skills=[
            {"name": "X", "proficiency": "expert", "duration_months": 0},
            {"name": "Y", "proficiency": "expert", "duration_months": 0},
            {"name": "Z", "proficiency": "expert", "duration_months": 0},
        ]
    )
    tripped, _ = check_expert_zero_duration(c)
    assert tripped


def test_expert_zero_duration_negative():
    c = _candidate(
        skills=[
            {"name": "X", "proficiency": "expert", "duration_months": 12},
            {"name": "Y", "proficiency": "intermediate", "duration_months": 0},
        ]
    )
    tripped, _ = check_expert_zero_duration(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 3. expert_zero_endorsements
# ---------------------------------------------------------------------------
def test_expert_zero_endorsements_positive():
    c = _candidate(
        skills=[
            {"name": "X", "proficiency": "expert", "endorsements": 0},
            {"name": "Y", "proficiency": "expert", "endorsements": 0},
            {"name": "Z", "proficiency": "expert", "endorsements": 0},
        ]
    )
    tripped, _ = check_expert_zero_endorsements(c)
    assert tripped


def test_expert_zero_endorsements_negative():
    c = _candidate(
        skills=[
            {"name": "X", "proficiency": "expert", "endorsements": 5},
            {"name": "Y", "proficiency": "expert", "endorsements": 1},
        ]
    )
    tripped, _ = check_expert_zero_endorsements(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 4. identical_career_descriptions
# ---------------------------------------------------------------------------
def test_identical_descriptions_positive():
    c = _candidate(
        career_history=[
            {"description": "Built ML pipelines"},
            {"description": "Built ML pipelines"},
        ]
    )
    tripped, _ = check_identical_career_descriptions(c)
    assert tripped


def test_identical_descriptions_negative():
    c = _candidate(
        career_history=[
            {"description": "Built ML pipelines"},
            {"description": "Led backend team"},
        ]
    )
    tripped, _ = check_identical_career_descriptions(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 5. tenure_over_240_months
# ---------------------------------------------------------------------------
def test_tenure_over_240_positive():
    c = _candidate(
        career_history=[{"title": "CEO", "company": "X", "duration_months": 300}]
    )
    tripped, _ = check_tenure_over_240_months(c)
    assert tripped


def test_tenure_over_240_negative():
    c = _candidate(
        career_history=[{"title": "Eng", "company": "X", "duration_months": 36}]
    )
    tripped, _ = check_tenure_over_240_months(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 6. consulting_keyword_stuffing
# ---------------------------------------------------------------------------
def test_consulting_stuffing_positive():
    c = _candidate(
        profile={"current_company": "Infosys Ltd"},
        skills=[{"name": f"skill_{i}"} for i in range(15)],
    )
    tripped, _ = check_consulting_keyword_stuffing(c)
    assert tripped


def test_consulting_stuffing_negative():
    c = _candidate(
        profile={"current_company": "Infosys Ltd"},
        skills=[{"name": f"skill_{i}"} for i in range(5)],
    )
    tripped, _ = check_consulting_keyword_stuffing(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 7. rookie_perfect_profile
# ---------------------------------------------------------------------------
def test_rookie_perfect_profile_positive():
    c = _candidate(
        profile={"years_of_experience": 0.5},
        redrob_signals={"profile_completeness_score": 99.5},
    )
    tripped, _ = check_rookie_perfect_profile(c)
    assert tripped


def test_rookie_perfect_profile_negative():
    c = _candidate(
        profile={"years_of_experience": 5},
        redrob_signals={"profile_completeness_score": 99.5},
    )
    tripped, _ = check_rookie_perfect_profile(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 8. stale_high_activity
# ---------------------------------------------------------------------------
def test_stale_high_activity_positive():
    c = _candidate(
        redrob_signals={
            "last_active_date": "2025-12-01",
            "applications_submitted_30d": 5,
            "profile_views_received_30d": 0,
        }
    )
    tripped, evidence = check_stale_high_activity(c)
    assert tripped
    assert "applications_submitted_30d=5" in evidence


def test_stale_high_activity_negative():
    c = _candidate(
        redrob_signals={
            "last_active_date": "2026-06-20",
            "applications_submitted_30d": 5,
        }
    )
    tripped, _ = check_stale_high_activity(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 9. skill_assessment_inversion
# ---------------------------------------------------------------------------
def test_skill_assessment_inversion_positive():
    c = _candidate(
        skills=[{"name": "NLP", "proficiency": "expert"}],
        redrob_signals={"skill_assessment_scores": {"NLP": 15.0}},
    )
    tripped, _ = check_skill_assessment_inversion(c)
    assert tripped


def test_skill_assessment_inversion_negative():
    c = _candidate(
        skills=[{"name": "NLP", "proficiency": "expert"}],
        redrob_signals={"skill_assessment_scores": {"NLP": 85.0}},
    )
    tripped, _ = check_skill_assessment_inversion(c)
    assert not tripped


# ---------------------------------------------------------------------------
# 10. duplicate_candidate_fingerprint
# ---------------------------------------------------------------------------
def test_duplicate_fingerprint_positive():
    c = _candidate(candidate_id="CAND_A")
    tripped, _ = check_duplicate_candidate_fingerprint(c, {"CAND_A", "CAND_B"})
    assert tripped


def test_duplicate_fingerprint_negative():
    c = _candidate(candidate_id="CAND_C")
    tripped, _ = check_duplicate_candidate_fingerprint(c, {"CAND_A", "CAND_B"})
    assert not tripped


# ---------------------------------------------------------------------------
# Missing-field handling
# ---------------------------------------------------------------------------
def test_check_handles_missing_fields_gracefully():
    c = _candidate(profile={})  # no years_of_experience, no career_history key
    assert check_timeline_inflation(c) == (False, "insufficient data")
    assert check_expert_zero_duration(c) == (False, "insufficient data")
    assert check_tenure_over_240_months(c) == (False, "insufficient data")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def test_honeypot_score_counts_tripped_checks():
    c = _candidate(
        candidate_id="CAND_X",
        profile={"years_of_experience": 15},
        career_history=[{"duration_months": 60}],
    )
    assert honeypot_score(c) >= 1


# ---------------------------------------------------------------------------
# Duplicate detector batch
# ---------------------------------------------------------------------------
def test_find_duplicate_fingerprints():
    c1 = {
        "candidate_id": "A",
        "profile": {"current_title": "ML Engineer", "current_company": "X", "years_of_experience": 5},
        "skills": [{"name": "Python"}, {"name": "NLP"}],
    }
    c2 = {
        "candidate_id": "B",
        "profile": {"current_title": "ML Engineer", "current_company": "X", "years_of_experience": 5},
        "skills": [{"name": "Python"}, {"name": "NLP"}],
    }
    c3 = {
        "candidate_id": "C",
        "profile": {"current_title": "Data Scientist", "current_company": "Y", "years_of_experience": 5},
        "skills": [{"name": "Python"}],
    }
    dups = find_duplicate_fingerprints(iter([c1, c2, c3]))
    assert dups == {"A", "B"}
