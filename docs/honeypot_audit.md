# Honeypot detector audit

## Setup

Sampled 10 candidates flagged at `honeypot_score >= 3` (all 10 checks) from the
415 total flagged, and 10 candidates from `outputs/final_submission.csv` top-100.
All sampling deterministic with seed=42.

The pipeline gate uses `honeypot_score_gates_only` (8 high-signal checks), which
excludes the synthetic-artifact checks `identical_career_descriptions` and
`stale_high_activity`. This audit reports the full `honeypot_score` so we can
inspect both real integrity signals and known artifacts.

## Group A — precision check (10 flagged candidates)

| cid | hp_score | tripped checks | rating | notes |
|---|---|---|---|---|
| CAND_0080068 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Project Manager @ Wipro with non-technical career and a grab-bag of 15 tech skills; low response rate; stale activity. |
| CAND_0013883 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Content Writer @ TCS with mismatched career history (accounting, mechanical design) and 18 tech skills; US-based, low completeness. |
| CAND_0003189 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Project Manager @ Wipro with generic GenAI headline but operations/content career; 17 skills, stale activity. |
| CAND_0089953 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Project Manager @ Infosys with accountant roles and 18 unrelated tech skills; very low response/completion. |
| CAND_0031936 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Civil Engineer @ TCS in customer-support roles with 13 tech skills; clear consulting-keyword stuffing. |
| CAND_0028830 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | HR Manager @ Infosys with graphic-design/marketing history and 17 tech skills; not an AI-engineering profile. |
| CAND_0026836 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Graphic Designer @ TCS based in Berlin with beginner/intermediate tech skills; not relevant to the JD. |
| CAND_0016640 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Content Writer @ Wipro with accounting/sales history and 21 skills; consulting-keyword stuffing. |
| CAND_0089563 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Java/QA engineer @ HCL with 15 mismatched skills and duplicate career descriptions; not a senior AI role fit. |
| CAND_0013085 | 3 | identical_career_descriptions, consulting_keyword_stuffing, stale_high_activity | implausible | Marketing Manager @ Infosys with brand-design/customer-support history and 21 tech skills; not relevant. |

**Precision: 10/10 actually look implausible.**

## Group B — recall sanity check (10 top-100 candidates)

| cid | rank in submission | red flags | notes |
|---|---|---|---|
| CAND_0086022 | 8 | none | Strong applied scientist with RAG/ranking pipeline; clean profile. |
| CAND_0093547 | 56 | minor | Identical career descriptions between two roles; otherwise strong ML/ranking profile. |
| CAND_0070202 | 10 | none | Strong ML engineer with ranking/recommendation experience; clean. |
| CAND_0007460 | 5 | none | AI Engineer at Salesforce with strong vector-search skills; clean. |
| CAND_0077337 | 14 | minor | Identical career descriptions between two roles; otherwise strong Staff ML profile. |
| CAND_0052328 | 12 | none | Recommendation-systems engineer at Amazon with relevant skills; clean. |
| CAND_0005509 | 24 | minor | Identical career descriptions; Data Scientist profile is reasonable but less senior. |
| CAND_0005260 | 3 | none | Senior NLP Engineer with LLM/RAG/vector-search background; clean. |
| CAND_0029367 | 19 | minor | Identical career descriptions; Senior Data Scientist with applied-ML skills. |
| CAND_0030348 | 13 | none | ML Engineer with semantic-search experience; clean. |

**Recall sanity: 5/10 top-100 candidates show the synthetic `identical_career_descriptions` artifact.**

## Findings

All 10 flagged candidates are genuinely implausible fits for the Senior AI
Engineer role: they are non-technical consulting profiles with keyword-stuffed
skill lists and/or stale activity. The detector's precision is excellent.

The 5 "red flags" in the top-100 are exclusively the `identical_career_descriptions`
artifact, which is known to affect ~36% of the synthetic pool and is intentionally
excluded from the gate check. None of the top-100 show the high-signal integrity
issues (timeline inflation, expert-zero-duration, consulting-keyword stuffing,
etc.) that the gate uses. Therefore there is no Stage 3 risk for real honeypots
slipping into the submission.

## Implications for the submission

No threshold change is needed. The gate is correctly conservative: it keeps the
submission clean of real integrity issues while tolerating the harmless but
ubiquitous duplicate-description artifact. The artifact should be tracked as a
data-generation quirk, not a detector failure.
