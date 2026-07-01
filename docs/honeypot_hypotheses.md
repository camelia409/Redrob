# Honeypot Hypotheses — v1

Each check measures a different data-quality or adversarial failure mode. They are intentionally orthogonal: a candidate may trip multiple checks, but each check isolates one suspicious pattern.

| # | Name | Fields inspected | Rule (plain English) |
|---|------|------------------|----------------------|
| 1 | timeline_inflation | `profile.years_of_experience` vs `sum(career_history[].duration_months) / 12` | Declared YoE is more than 2× the cumulative tenure in career history. |
| 2 | expert_zero_duration | `skills[].proficiency`, `skills[].duration_months` | ≥3 skills marked `expert` but with `duration_months == 0` or missing. |
| 3 | expert_zero_endorsements | `skills[].proficiency`, `skills[].endorsements` | ≥3 skills marked `expert` but with `endorsements == 0`. |
| 4 | identical_career_descriptions | `career_history[].description` | Two or more distinct career entries share the exact same description text. |
| 5 | tenure_over_240_months | `career_history[].duration_months` | Any single job claims >240 months (20 years) of continuous tenure. |
| 6 | consulting_keyword_stuffing | `profile.current_company`, `skills[].name` | Candidate is at a mass-recruitment consulting/IT shop **and** lists >12 skills, suggesting keyword stuffing. |
| 7 | rookie_perfect_profile | `profile.years_of_experience`, `redrob_signals.profile_completeness_score` | `years_of_experience` ≤ 1 but `profile_completeness_score` ≥ 99 — unlikely organic combination. |
| 8 | stale_high_activity | `redrob_signals.last_active_date`, `redrob_signals.applications_submitted_30d`, `redrob_signals.profile_views_received_30d` | Last active >180 days ago but recent 30-day activity counters are non-zero (stale profile with contradictory signals). |
| 9 | skill_assessment_inversion | `skills[].proficiency`, `redrob_signals.skill_assessment_scores` | A skill is marked `expert` but its `skill_assessment_score` is <30 (if assessment exists for that skill). |
| 10 | duplicate_candidate_fingerprint | `profile.current_title`, `profile.current_company`, `profile.years_of_experience`, `skills[].name` | Two or more candidates share the exact same (title, company, YoE, sorted skill list) tuple — possible duplicate/synthetic clone. |

## Notes

- **consulting_keyword_stuffing** uses the firm list observed in EDA: Infosys, TCS, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mphasis, IBM, EY, PwC, KPMG, Deloitte.
- **stale_high_activity** uses the 2026-07-01 reference date from the recency analysis.
- Checks 7, 8, 9, and 10 go beyond the standard experience/tenure checks and target synthetic-profile artifacts.
