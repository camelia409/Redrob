"""Adversarial red-team tests for the honeypot detector.

Each test crafts a synthetic candidate profile that emulates a specific attack
pattern a bad actor might use to game a candidate ranking system. All 5 attacks
must trip the honeypot gate (>= 3 integrity flags) using
honeypot_score_gates_only.

Attack patterns covered:
1. Timeline inflation      -- declared YoE double the career-history sum.
2. Expert-zero-duration    -- expert proficiency on many skills with 0 months.
3. Consulting stuffing     -- non-technical title at consulting firm + many skills.
4. Impossible tenure       -- single job with 25+ years continuous tenure.
5. Rookie perfect profile  -- 1 YoE + 100% completeness + many expert skills.
"""
import pytest

from src.validation.honeypots import honeypot_score_gates_only, run_all_checks


def _base_candidate(cid: str = "CAND_ADVERSARIAL", **overrides):
    """Minimal viable candidate skeleton. Callers override fields."""
    c = {
        "candidate_id": cid,
        "profile": {
            "current_title": "Software Engineer",
            "current_company": "Neutral Co",
            "years_of_experience": 5.0,
            "location": "Bangalore",
            "country": "India",
            "headline": "SWE",
            "summary": "10 years experience",
        },
        "career_history": [
            {"title": "Engineer", "company": "Neutral Co", "description": "wrote code", "duration_months": 60}
        ],
        "skills": [
            {"name": "python", "proficiency": "advanced", "duration_months": 24, "endorsements": 5}
        ],
        "redrob_signals": {
            "profile_completeness_score": 60,
            "recruiter_response_rate": 0.4,
            "interview_completion_rate": 0.7,
        },
        "education": [],
        "certifications": [],
        "languages": [],
    }
    for k, v in overrides.items():
        c[k] = v
    return c


def _assert_caught(c):
    """Assert that a synthetic candidate trips the honeypot gate."""
    score = honeypot_score_gates_only(c)
    tripped = [k for k, (t, _) in run_all_checks(c).items() if t]
    assert score >= 3, f"expected >= 3, got {score}; tripped={tripped}"
    return score, tripped


def test_attack_1_timeline_inflation():
    """Attack: claim 15 YoE but list only 5 years of actual career history."""
    c = _base_candidate("CAND_ATTACK_1")
    c["profile"]["years_of_experience"] = 15.0
    c["career_history"] = [
        {"title": "ML Engineer", "company": "Startup", "description": "shipped ranker", "duration_months": 24},
        {"title": "SWE", "company": "Startup", "description": "shipped api", "duration_months": 36},
    ]
    c["skills"] = [
        {"name": "python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        {"name": "pytorch", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        {"name": "embeddings", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        {"name": "ranking", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
    ]
    score, tripped = _assert_caught(c)
    print(f"attack_1 score={score} tripped={tripped}")


def test_attack_2_expert_zero_duration_stacking():
    """Attack: claim expert-level in many skills with zero duration and zero endorsements, plus inflated YoE."""
    c = _base_candidate("CAND_ATTACK_2")
    # Inflate YoE above actual career history to also trip timeline_inflation.
    c["profile"]["years_of_experience"] = 15.0
    c["skills"] = [
        {"name": skill, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
        for skill in ["python", "pytorch", "embeddings", "vector search", "ranking",
                      "retrieval", "llms", "fine-tuning llms", "transformers",
                      "prompt engineering", "faiss", "pinecone"]
    ]
    score, tripped = _assert_caught(c)
    print(f"attack_2 score={score} tripped={tripped}")


def test_attack_3_consulting_stuffing_with_nontech_title():
    """Attack: non-technical role at a consulting firm with a keyword-stuffed skill list."""
    c = _base_candidate("CAND_ATTACK_3")
    c["profile"]["current_title"] = "HR Manager"
    c["profile"]["current_company"] = "Infosys"
    c["profile"]["headline"] = "HR Manager | GenAI explorer"
    c["career_history"] = [
        {"title": "HR Manager", "company": "Infosys", "description": "employee onboarding and payroll", "duration_months": 36},
        {"title": "HR Executive", "company": "Wipro", "description": "recruitment and reviews", "duration_months": 30},
    ]
    c["skills"] = [
        {"name": s, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
        for s in ["python", "pytorch", "tensorflow", "embeddings", "vector search",
                  "faiss", "pinecone", "qdrant", "llms", "fine-tuning llms",
                  "transformers", "ranking", "rag", "prompt engineering", "bert"]
    ]
    score, tripped = _assert_caught(c)
    print(f"attack_3 score={score} tripped={tripped}")


def test_attack_4_impossible_tenure():
    """Attack: claim a single continuous role of 30+ years."""
    c = _base_candidate("CAND_ATTACK_4")
    c["profile"]["years_of_experience"] = 30.0
    c["career_history"] = [
        {"title": "Senior Engineer", "company": "AncientCo",
         "description": "did everything for 30 years", "duration_months": 360}
    ]
    c["skills"] = [
        {"name": "python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        {"name": "embeddings", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        {"name": "vector search", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
    ]
    score, tripped = _assert_caught(c)
    print(f"attack_4 score={score} tripped={tripped}")


def test_attack_5_rookie_perfect_profile():
    """Attack: 1 year of experience but 100% profile completeness and many expert skills."""
    c = _base_candidate("CAND_ATTACK_5")
    c["profile"]["years_of_experience"] = 1.0
    c["profile"]["current_title"] = "Marketing Manager"
    c["profile"]["current_company"] = "TCS"
    c["career_history"] = [
        {"title": "Marketing Intern", "company": "SmallCo", "description": "wrote blog posts", "duration_months": 12}
    ]
    c["skills"] = [
        {"name": s, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
        for s in ["python", "embeddings", "vector search", "ranking", "llms",
                  "fine-tuning llms", "transformers", "prompt engineering",
                  "faiss", "pinecone", "qdrant", "milvus"]
    ]
    c["redrob_signals"]["profile_completeness_score"] = 100
    score, tripped = _assert_caught(c)
    print(f"attack_5 score={score} tripped={tripped}")
