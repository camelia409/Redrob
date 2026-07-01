"""Tests for the deterministic rubric scorer."""
import pytest

from src.evaluation.rubric_scorer import score_candidate


def _candidate(**kwargs):
    base = {
        "candidate_id": "CAND_TEST",
        "profile": {
            "current_title": kwargs.pop("title", "ML Engineer"),
            "years_of_experience": kwargs.pop("yoe", 6.0),
            "location": kwargs.pop("location", "Bangalore"),
            "country": kwargs.pop("country", "India"),
        },
        "career_history": kwargs.pop(
            "career",
            [
                {
                    "title": "ML Engineer",
                    "company": "X",
                    "description": "Built a ranking and retrieval system in production.",
                    "duration_months": 36,
                }
            ],
        ),
        "skills": kwargs.pop("skills", []),
        "redrob_signals": kwargs.pop(
            "signals",
            {
                "profile_completeness_score": 80.0,
                "interview_completion_rate": 0.75,
                "willing_to_relocate": True,
                "verified_email": True,
                "verified_phone": True,
            },
        ),
    }
    assert not kwargs, f"Unexpected kwargs: {kwargs}"
    return base


def _skill(name, proficiency, duration=24, endorsements=5):
    return {
        "name": name,
        "proficiency": proficiency,
        "duration_months": duration,
        "endorsements": endorsements,
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_scorer_is_deterministic():
    c = _candidate(
        title="Senior Data Engineer",
        yoe=6.5,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
            _skill("Fine-tuning LLMs", "advanced", 30),
        ],
    )
    s1, e1 = score_candidate(c, honeypot_val=0)
    s2, e2 = score_candidate(c, honeypot_val=0)
    assert s1 == s2
    assert e1 == e2


# ---------------------------------------------------------------------------
# Score 5 clauses
# ---------------------------------------------------------------------------
def test_score_5_requires_ml_title():
    c = _candidate(
        title="Backend Engineer",
        yoe=6,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
            _skill("Fine-tuning LLMs", "advanced", 30),
        ],
    )
    score, _ = score_candidate(c, honeypot_val=0)
    assert score < 5


def test_score_5_requires_three_strict_skills():
    c = _candidate(
        title="ML Engineer",
        yoe=6,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
        ],
    )
    score, _ = score_candidate(c, honeypot_val=0)
    assert score < 5


def test_score_5_requires_career_ml_system_mention():
    c = _candidate(
        title="ML Engineer",
        yoe=6,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
            _skill("Fine-tuning LLMs", "advanced", 30),
        ],
        career=[
            {
                "title": "ML Engineer",
                "description": "Built backend APIs and data pipelines.",
            }
        ],
    )
    score, _ = score_candidate(c, honeypot_val=0)
    assert score < 5


def test_score_5_requires_interview_rate():
    c = _candidate(
        title="ML Engineer",
        yoe=6,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
            _skill("Fine-tuning LLMs", "advanced", 30),
        ],
        signals={
            "profile_completeness_score": 80.0,
            "interview_completion_rate": 0.55,
            "willing_to_relocate": True,
            "verified_email": True,
            "verified_phone": True,
        },
    )
    score, _ = score_candidate(c, honeypot_val=0)
    assert score < 5


def test_score_5_achieved_when_all_clauses_met():
    c = _candidate(
        title="ML Engineer",
        yoe=6,
        skills=[
            _skill("Vector Search", "advanced", 18),
            _skill("Retrieval", "expert", 24),
            _skill("Fine-tuning LLMs", "advanced", 30),
        ],
    )
    score, evidence = score_candidate(c, honeypot_val=0)
    assert score == 5
    assert any("ml_title" in e for e in evidence)
    assert any("strict ML skills" in e for e in evidence)


# ---------------------------------------------------------------------------
# Score 0 gates
# ---------------------------------------------------------------------------
def test_honeypot_gate_forces_zero():
    c = _candidate()
    score, evidence = score_candidate(c, honeypot_val=3)
    assert score == 0
    assert evidence == ["honeypot_gate_triggered"]


def test_negative_yoe_forces_zero():
    c = _candidate(yoe=-1)
    score, _ = score_candidate(c, honeypot_val=0)
    assert score == 0


def test_empty_career_with_positive_yoe_forces_zero():
    c = _candidate(yoe=5, career=[])
    score, evidence = score_candidate(c, honeypot_val=0)
    assert score == 0
    assert any("empty_career" in e for e in evidence)


def test_non_technical_title_without_skills_forces_zero():
    c = _candidate(
        title="HR Manager",
        yoe=5,
        skills=[],
        career=[{"title": "HR Manager", "description": "Managed HR operations."}],
    )
    score, evidence = score_candidate(c, honeypot_val=0)
    assert score == 0
    assert any("non_technical" in e for e in evidence)


# ---------------------------------------------------------------------------
# Cascading scores
# ---------------------------------------------------------------------------
def test_score_4_for_strong_adjacent_profile():
    c = _candidate(
        title="Senior Data Engineer",
        yoe=6,
        skills=[
            _skill("Computer Vision", "advanced", 24),
            _skill("NLP", "expert", 30),
        ],
    )
    score, evidence = score_candidate(c, honeypot_val=0)
    assert score == 4
    assert any("score_4" in e for e in evidence)


def test_score_3_for_reasonable_profile():
    c = _candidate(
        title="Data Analyst",
        yoe=5,
        skills=[_skill("Python", "intermediate", 24)],
        career=[{"title": "Data Analyst", "description": "Built SQL reports and dashboards."}],
    )
    score, _ = score_candidate(c, honeypot_val=0)
    assert score == 3


def test_score_1_for_very_junior():
    c = _candidate(yoe=1, skills=[_skill("Python", "beginner", 6)])
    score, evidence = score_candidate(c, honeypot_val=0)
    assert score == 1
    assert any("very junior" in e for e in evidence)
