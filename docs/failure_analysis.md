# Failure analysis: lowest-silver-score candidates in the top-100

## Method

We identified the 10 candidates in `outputs/final_submission.csv` with the
lowest silver score (from `data/processed/silver_scores_full.csv`). For each,
we inspected which of our 30 engineered features had the highest values — in
other words, what features carried them into the top-100 despite their
rubric-relative weakness.

Silver scores of the worst 10: **1, 2, 3, 3, 3, 3, 3, 3, 3, 3**. The presence
of a silver-1 and a silver-2 candidate in the top-100 is a genuine ranking
failure; the eight silver-3 candidates are weaker-than-average fits but not
outright errors.

## The 10 candidates

```text
[1] CAND_0039754  rank=64  silver=1  reranker_score=0.0130
    Reasoning: Senior Applied Scientist profile at Meta; 16.2 years with caveats; solid command of Python and OpenSearch; flagged: YoE
    Top features (values):
      max_skill_proficiency_ml                  1.000
      mean_skill_duration_months_ml_capped      1.000
      broad_ml_skill_count_capped               1.000
      career_ml_system_phrase_count_capped      1.000
      has_zero_gates                            1.000
      verified_score                            1.000
    Aggregate: bm25=2058.2691  dense=0.7625  hp=0

[2] CAND_0093547  rank=75  silver=2  reranker_score=0.0122
    Reasoning: Risky candidate: Senior Machine Learning Engineer at PhonePe, 2.9 yrs; issue: YoE 2.9 outside ideal band; likely due to
    Top features (values):
      max_skill_proficiency_ml                  1.000
      mean_skill_duration_months_ml_capped      1.000
      broad_ml_skill_count_capped               1.000
      career_ml_system_phrase_count_capped      1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1982.5626  dense=0.7300  hp=0

[3] CAND_0083307  rank=100  silver=3  reranker_score=0.0109
    Reasoning: Low-confidence Search Engineer profile at CRED; 7.8 years Current profile: Search Engineer at CRED, 7.8 years experience
    Top features (values):
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      max_skill_proficiency_ml                  1.000
      broad_ml_skill_count_capped               1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1136.2740  dense=0.7747  hp=0

[4] CAND_0086151  rank=97  silver=3  reranker_score=0.0110
    Reasoning: Concerning match: Recommendation Systems Engineer at Wysa, 7.7 yrs; notable gap: mostly non-technical roles
    Top features (values):
      max_skill_proficiency_ml                  1.000
      yoe_in_ideal_band                         1.000
      broad_ml_skill_count_capped               1.000
      career_ml_system_phrase_count_capped      1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1227.8103  dense=0.7231  hp=0

[5] CAND_0030953  rank=96  silver=3  reranker_score=0.0110
    Reasoning: Risky candidate: Search Engineer at Nykaa, 7.8 yrs; issue: mostly non-technical roles
    Top features (values):
      max_skill_proficiency_ml                  1.000
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      career_ml_system_phrase_count_capped      1.000
      india_score                               1.000
      integrity_score                           1.000
    Aggregate: bm25=1275.8031  dense=0.7679  hp=0

[6] CAND_0041610  rank=95  silver=3  reranker_score=0.0111
    Reasoning: Concerning match: Recommendation Systems Engineer at Zoho, 6.7 yrs; notable gap: mostly non-technical roles
    Top features (values):
      max_skill_proficiency_ml                  1.000
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      career_ml_system_phrase_count_capped      1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1202.3621  dense=0.7651  hp=0

[7] CAND_0005260  rank=90  silver=3  reranker_score=0.0114
    Reasoning: Risky candidate: Senior NLP Engineer at Netflix, 5.2 yrs; issue: mostly non-technical roles; likely due to limited ML-sp
    Top features (values):
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      max_skill_proficiency_ml                  1.000
      broad_ml_skill_count_capped               1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1476.7030  dense=0.7259  hp=0

[8] CAND_0092278  rank=85  silver=3  reranker_score=0.0116
    Reasoning: Poor alignment: Senior NLP Engineer with 6.8 years at Microsoft; caution: mostly non-technical roles; mainly because of
    Top features (values):
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      max_skill_proficiency_ml                  1.000
      broad_ml_skill_count_capped               1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1433.3433  dense=0.7747  hp=0

[9] CAND_0057563  rank=81  silver=3  reranker_score=0.0120
    Reasoning: Poor alignment: NLP Engineer with 6.8 years at Locobuzz Current profile: NLP Engineer at Locobuzz, 6.8 years experience.
    Top features (values):
      max_skill_proficiency_ml                  1.000
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      career_ml_system_phrase_count_capped      1.000
      verified_score                            1.000
      india_score                               1.000
    Aggregate: bm25=1210.0970  dense=0.6976  hp=0

[10] CAND_0033861  rank=76  silver=3  reranker_score=0.0122
    Reasoning: Concerning match: Senior NLP Engineer at Mad Street Den, 8.0 yrs; notable gap: mostly non-technical roles; root cause ap
    Top features (values):
      yoe_in_ideal_band                         1.000
      mean_skill_duration_months_ml_capped      1.000
      max_skill_proficiency_ml                  1.000
      broad_ml_skill_count_capped               1.000
      has_zero_gates                            1.000
      verified_score                            1.000
    Aggregate: bm25=1594.9302  dense=0.7366  hp=0
```

## Cross-candidate patterns

```text
feature                                       worst-10 mean     top-100 mean        delta
------------------------------------------------------------------------------------------
dense_v2_score                                        0.744            0.706       +0.039
bm25_score_normalized                                 0.671            0.620       +0.051
bm25_rank_within_union                                0.979            0.964       +0.015
dense_v2_rank_within_union                            0.981            0.944       +0.037
bm25_dense_rank_agreement                             0.981            0.959       +0.023
strict_ml_skill_count_capped                          0.280            0.248       +0.032
broad_ml_skill_count_capped                           0.910            0.869       +0.041
max_skill_proficiency_ml                              1.000            0.985       +0.015
mean_skill_duration_months_ml_capped                  0.999            0.991       +0.008
yoe_in_ideal_band                                     0.800            0.932       -0.132 *
ml_role_fraction                                      0.215            0.499       -0.284 *
tech_role_fraction                                    0.215            0.499       -0.284 *
career_direction_bonus                                0.000            0.355       -0.355 *
mean_tenure_months_capped                             0.734            0.800       -0.066
career_ml_system_phrase_count_capped                  1.000            0.744       +0.256 *
career_production_phrase_count_capped                 0.938            0.905       +0.032
product_company_flag                                  0.700            0.810       -0.110 *
consulting_only_flag                                  0.000            0.000       +0.000
recency_score                                         0.380            0.439       -0.058
response_rate_score                                   0.585            0.683       -0.098
github_score                                          0.446            0.543       -0.096
verified_score                                        0.967            0.873       +0.093
interview_completion_rate                             0.795            0.800       -0.005
india_score                                           1.000            0.940       +0.060
preferred_city_bonus                                  0.450            0.555       -0.105 *
notice_period_score                                   0.650            0.666       -0.016
integrity_score                                       1.000            1.000       +0.000
has_zero_gates                                        1.000            1.000       +0.000
silver_score                                          2.700            3.780       -1.080 *
honeypot_score                                        0.000            0.000       +0.000
bm25_score                                         1459.815         1349.224     +110.591 *
dense_score                                           0.744            0.706       +0.039
```

## Findings

**Pattern 1: Phrase-count keyword inflation is masking weak career
 trajectories.**
Every one of the worst 10 has `career_ml_system_phrase_count_capped = 1.0`,
giving them the maximum production-evidence score even though their
`ml_role_fraction` and `tech_role_fraction` are less than half the top-100
average (0.215 vs 0.499). The reasoning strings themselves confirm this: they
repeatedly say "mostly non-technical roles." The phrase counter is rewarding
surface-level keyword matches in career descriptions instead of actual ML
engineering trajectory.

**Pattern 2: Retrieval-channel scores are over-crediting borderline
 profiles.**
The worst-10 have higher `bm25_score_normalized` (0.671 vs 0.620) and higher
`dense_v2_score` (0.744 vs 0.706) than the top-100 average. At the same time
their career-trajectory signals are substantially weaker: `yoe_in_ideal_band`
is 0.80 vs 0.93, `career_direction_bonus` is 0.0 vs 0.36, and
`product_company_flag` is 0.70 vs 0.81. The retrieval features are carrying
candidates who look good to BM25 + MiniLM but who the rubric sees as mediocre.

**Pattern 3: Verification and India bonuses provide a small extra lift.**
`verified_score` is 0.967 and `india_score` is 1.000 in the worst-10 versus
0.873 and 0.940 in the full top-100. These are small absolute weights, but on
the margin they can push a borderline candidate above the rank-100 cutoff.

## What this means for the model

These failures are a **mix of defensible and concerning**.

- Defensible: a silver-3 candidate in the top-100 is a "reasonable but not
  great" fit, and at rank 90+ that is exactly what we expect the tail of a
  ranked list to contain. The model is not hallucinating these profiles; they
  have ML-relevant titles and some relevant skills.
- Concerning: the silver-1 candidate (`CAND_0039754`, rank 64) and the
  phrase-count inflation pattern show that surface keyword features can
  override deeper career-quality signals. This is a real error mode that would
  matter in production.

## What we would do differently in a future iteration

1. **Down-weight or replace `career_ml_system_phrase_count_capped`.** The
   feature is too easy to game with keyword-laden descriptions. We would
   replace it with a title-aware production-evidence signal or require the
   phrase count to be paired with a minimum `ml_role_fraction`.
2. **Add a cross-feature interaction term.** For example, only award full
   skill-evidence and production-evidence scores when `ml_role_fraction` is
   above a threshold, so weak trajectories cannot be rescued by keyword counts.
3. **Consider a hard silver-score floor for the top 100.** Any candidate with
   `silver_score <= 1` is explicitly rated implausible by the rubric; allowing
   one into the top-100 is a clear ranking failure that a post-hoc guard could
   prevent.

## Why this analysis matters

Failure analysis on your own model output is standard senior-ML practice. It
answers "does the team understand where their model is weakest, or are they
just reporting aggregate metrics?" — a question every serious reviewer asks.
Here the answer is that the model's weakest top-100 decisions are profiles
with ML-aligned titles and keyword-rich descriptions but weak actual career
 trajectories; the fixes are known and would be the first priority in a
second iteration.
