# Relevance Rubric — Senior AI Engineer @ Redrob (v1)

## Source of truth

Extracted JD passages driving this rubric:

> "Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning."

> "Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is 'obviously suboptimal,' because we need to learn from real users before we know what to actually optimize for."

> "Experience Required: 5–9 years"

> "The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles."

> "Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking."

## 0–5 relevance scale

| Score | Label | Criteria (all must hold, unless stated) |
|-------|-------|------------------------------------------|
| 5 | Perfect fit | Current title is in `{ML Engineer, AI Research Engineer, Senior Data Engineer, Data Scientist}` **AND** `profile.years_of_experience` is in 5–9 **AND** ≥3 skills from `{embeddings, retrieval, ranking, LLMs, fine-tuning, vector search, prompt engineering, transformers}` with `proficiency` = advanced/expert and `duration_months` ≥ 12 **AND** at least one career entry mentions ranking/retrieval/matching/embeddings/LLMs in `description` **AND** `redrob_signals.interview_completion_rate` ≥ 0.6 **AND** no honeypot flags. |
| 4 | Strong fit | Current title is in `{Data Engineer, Backend Engineer, Analytics Engineer, Senior Software Engineer}` **AND** `profile.years_of_experience` is in 4–10 **AND** ≥2 relevant ML/systems skills with advanced/expert proficiency and duration ≥ 12 months **AND** career history shows production-scale data/ML systems **AND** `redrob_signals.profile_completeness_score` ≥ 50. |
| 3 | Reasonable fit | `profile.years_of_experience` is in 3–11 **AND** ≥1 relevant skill (e.g., Python, NLP, vector DB, LLM) with intermediate+ proficiency **AND** career history includes software/data engineering **AND** located in India or open to relocation (`redrob_signals.willing_to_relocate` = True). |
| 2 | Marginal | `profile.years_of_experience` is 2–12 **AND** has some overlapping skills (e.g., AWS, Python, SQL) but no explicit ML/ranking exposure **OR** YoE outside 5–9 but with strong adjacent signals. |
| 1 | Weak | Very junior (`profile.years_of_experience` < 2) **OR** only generic business/HR/support titles with no technical skills **OR** `redrob_signals.profile_completeness_score` < 30. |
| 0 | Not relevant / honeypot / adversarial | Fails any integrity check **OR** honeypot score above threshold **OR** title is non-technical and no relevant skills (e.g., HR Manager, Content Writer) **OR** `redrob_signals.verified_email`/`verified_phone` are both False and profile score < 20. |

## Instant disqualifiers

- Any tampering detected by `verify_data_integrity()`.
- `honeypot_score` ≥ 3 from `docs/honeypot_hypotheses_v1.md` checks.
- `profile.years_of_experience` is negative or > 50.
- `career_history` is empty while `profile.years_of_experience` > 0.

## Explicit tie-breakers between adjacent scores

- **5 vs 4:** Candidate has shipped a ranking/retrieval/recommender system in production (evidence in `career_history[].description`).
- **4 vs 3:** Candidate has >1 advanced skill directly from the JD quote list (`embeddings`, `retrieval`, `ranking`, `LLMs`, `fine-tuning`).
- **3 vs 2:** Candidate is currently employed in a technical role (`current_title` contains engineer/developer/scientist/analyst) vs. a transitional/non-technical role.
- **2 vs 1:** Candidate has at least one skill with `proficiency` = advanced/expert and `duration_months` ≥ 6.

## What this rubric will NOT consider (responsible AI)

- `profile.anonymized_name` (gender/ethnicity proxy).
- `profile.location` beyond work-authorization/relocation fit.
- `profile.country` beyond willingness/ability to work in India.
- Any inferred demographics, caste, religion, marital status, or age beyond the explicit YoE range.

## Candidate JSON fields cited

- `profile.current_title`, `profile.years_of_experience`, `profile.location`, `profile.country`
- `skills[].name`, `skills[].proficiency`, `skills[].duration_months`, `skills[].endorsements`
- `career_history[].title`, `career_history[].company`, `career_history[].description`, `career_history[].duration_months`
- `redrob_signals.profile_completeness_score`, `redrob_signals.interview_completion_rate`, `redrob_signals.recruiter_response_rate`, `redrob_signals.notice_period_days`, `redrob_signals.willing_to_relocate`, `redrob_signals.verified_email`, `redrob_signals.verified_phone`
