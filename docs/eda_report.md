# EDA Report — v1

Dataset: **100,000 candidates** in `candidates.jsonl`, integrity-verified against SHA-256.

## Structural findings

- Each candidate has 8 top-level keys (`candidate_id`, `profile`, `career_history`, `education`, `skills`, `certifications`, `languages`, `redrob_signals`) — no schema drift detected in a 100-record reservoir sample.
- Averages per candidate: **9.6 skills**, **3.0 career entries**, **1.4 education entries**.
- The 50-record `sample_candidates.json` is structurally identical to the full file and can be used for fast tests.

## Distributions

- **Titles:** The pool is dominated by generic roles. Top 3 are *business analyst* (5,833), *hr manager* (5,830), and *mechanical engineer* (5,791). Only ~1,115 candidates hold explicitly ML/AI titles (*ml engineer*, *ai research engineer*, *data scientist* combined).
- **Skills:** 133 unique skill names. Top 3 are *html* (12,246), *databricks* (12,244), and *redux* (12,222). The distribution is suspiciously flat (~12% prevalence each for the top 30), suggesting synthetic skill assignment rather than organic specialization.
- **Companies:** Top 3 are *infosys* (7,590), *wayne enterprises* (7,571), and *wipro* (7,566). Roughly **28.8%** of candidates are at recognized consulting/IT services firms (Infosys, TCS, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mphasis, IBM, EY, PwC, KPMG, Deloitte).
- **Locations:** **75.1%** are in India. Top Indian cities include Bhubaneswar (4,321), Noida (4,283), Hyderabad (4,283), and Bangalore (4,238). Non-Indian hubs are Sydney (2,579) and San Francisco (2,536).
- **Experience:** Median YoE is **6.8 years** (mean 7.2). Only **34.4%** fall in the JD’s stated 5–9 year band.
- **Behavioral signals:** Median recruiter_response_rate = 0.44; median notice_period_days = 90; median github_activity_score = -1 (missing/value encoded as -1); median profile_completeness_score = 56.8; median interview_completion_rate = 0.62.
- **Recency:** Median days since last active = **140**. **0.0%** active within 30 days; **26.7%** active within 90 days. The entire pool is stale relative to the 2026-07-01 reference date.

## Read-through observations (from 15 candidates)

- Strong-ML candidates exist but are rare and scattered (e.g., ML Engineer @ Aganitha, Data Engineer @ Ola, Backend Engineer @ Mindtree with LLM side projects).
- Several candidates claim advanced ML skills (*Fine-tuning LLMs*, *Vector Search*, *Prompt Engineering*) while holding non-ML titles, indicating possible keyword stuffing or aspirational tagging.
- Career descriptions sometimes repeat verbatim across roles within the same candidate — a clear data-quality red flag.

## Implications for ranking

1. **Scarcity:** With only ~1% holding core ML/AI titles, title matching alone is insufficient; we must score skills, career descriptions, and project evidence.
2. **Skill noise:** The flat, high-frequency skill list means raw skill-count features will be weak. Duration, proficiency, endorsements, and assessment scores must be used to distinguish signal from noise.
3. **Recency penalty:** Because no candidate is active within 30 days, recency should be a soft signal, not a hard filter.
4. **Honeypots:** Consulting-firm concentration and repeated descriptions suggest adversarial/generated profiles; integrity/quality checks should feed into the final score.
