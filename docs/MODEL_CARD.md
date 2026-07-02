# Model Card — Redrob Candidate Ranker

Model card following Mitchell et al. 2019 conventions, as used at Google
Research and Anthropic for documenting production ML systems.

## 1. Model details

- **Name:** Redrob Candidate Ranker
- **Version:** submission-v3-frozen (current commit on main; tag to be applied at
  final submission time)
- **Date:** 2026-07-02
- **Type:** Two-stage retrieve-then-rerank pipeline with hand-designed weights
- **Architecture:**
  1. BM25 keyword retrieval (top 1,500 from 100,000)
  2. MiniLM sentence-embeddings dense retrieval (top 1,500 from 100,000)
  3. Union merge (~2,477 candidates)
  4. 25-feature extraction across 7 orthogonal families
  5. Weighted-sum rerank with YAML-configured weights
  6. Reciprocal Rank Fusion with BM25 rank as late-stage tiebreaker
  7. Honeypot integrity gate (10 checks, gate threshold >= 3)
  8. Template-based grounded reasoning generation
- **Codebase:** (team to fill at submission time)
- **License:** Hackathon submission, not licensed for production use.

## 2. Intended use

- **Primary intended use:** Rank 100,000 synthetic candidate profiles against a
  single Senior AI Engineer job description for the India Runs Data and AI
  Challenge 2026.
- **Primary intended users:** Redrob hackathon judges and the ranked-candidate
  reviewer for demonstration purposes.
- **Out-of-scope use cases:**
  - Real-world hiring decisions without human-in-the-loop review.
  - Automated candidate rejection.
  - Ranking against job descriptions other than the one distilled in
    `configs/jd_query.yaml`.
  - Deployment in jurisdictions with automated hiring regulations (NYC Local
    Law 144, EU AI Act) without additional compliance work.

## 3. Factors

- Language: profile text is English.
- Geography: JD is India-focused; ranking is calibrated toward India-based
  candidates and Indian metro locations.
- Experience level: JD targets 5-9 years of applied ML/AI experience.
- Role type: Senior AI Engineer with retrieval, ranking, and LLM production
  experience.

## 4. Metrics

Composite formula (from challenge spec):
`composite = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10`

Silver-label evaluation on the 2,477-candidate retrieval union for the shipped
v3 ranking (`outputs/final_submission.csv`):

| Metric | Value | Notes |
|---|---:|---|
| Composite | **0.858 ± 0.035** | 95% CI, bootstrap n=100 (see `outputs/composite_ci.txt`) |
| NDCG@10 | 0.796 | Silver labels |
| NDCG@50 | 0.899 | Silver labels |
| MAP | 0.939 | Silver labels |
| P@10 | 1.00 | Silver labels |
| Mean silver score in top 100 | 3.78 | Pool average is ~2.0 |

The v2 weighted-only variant has composite **0.857 ± 0.031**; the 95% CIs
overlap, so v2 and v3 are statistically indistinguishable on silver-score
composite. The ship/keep-v3 decision rests on human-label evidence, not the
silver-score delta.

Human-label evaluation (see `docs/pre_freeze_audit.md` and `DECISIONS.md`):

| Metric | Value |
|---|---:|
| Top-10 mean human rating | 4.7 / 5 |
| v3 vs v4 dropped-candidate mean | 4.75 / 5 |
| v3 vs v4 promoted-candidate mean | 4.50 / 5 |

Integrity metrics (see `docs/honeypot_audit.md` and `docs/adversarial_red_team.md`):

| Metric | Value |
|---|---:|
| Honeypot rate in top 100 (`honeypot_score_gates_only >= 3`) | 0 / 100 |
| Grounded reasoning pass rate | 100 / 100 |
| Honeypot detector precision (10-sample audit) | 10 / 10 |
| Adversarial red-team pass rate (5 crafted attacks) | 5 / 5 |

Runtime (see `docs/pre_freeze_audit.md`):

| Stage | Time |
|---|---:|
| BM25 retrieval | ~75 s |
| Dense retrieval | ~5 s |
| Feature extraction | ~0.5 s |
| Weighted rerank + RRF | ~0.1 s |
| Reasoning generation | ~9 s |
| **Total end-to-end** | **90.5 s** (under 300 s budget) |

## 5. Evaluation data

- **Silver labels:** 100,000 candidates scored by `src/evaluation/rubric_scorer.py`
  using rules distilled from the job description. Rubric documented in
  `docs/relevance_rubric_v1.md`. Known limitation: rubric features partially
  overlap with reranker features, so internal composite is optimistic by an
  unknown amount.
- **Hand labels:** 30 stratified candidates (Milestone 5) + top-10 of a
  candidate submission (`docs/pre_freeze_audit.md`) + 8-candidate v3/v4 diff
  (`DECISIONS.md`) = 48 candidates with real human ratings.
- **Feature matrix:** 2,477 candidates from the retrieval union, 25 features
  each, cached at `data/processed/feature_matrix.parquet`.

## 6. Training data

- The reranker is **not trained** on labels. Weights are hand-designed based on
  correlation analysis of features with silver labels (see
  `docs/family_ablation.md` and `outputs/family_ablation.csv`).
- The dense embedding model is off-the-shelf `sentence-transformers/all-MiniLM-L6-v2`
  (no fine-tuning).
- The cross-encoder (evaluated but not shipped) is off-the-shelf
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## 7. Ethical considerations

- **Protected attributes are not used** in the ranking pipeline: name, gender,
  caste, religion, marital status, disability, age (beyond the JD's stated
  experience range).
- **Structural filters are used and disclosed:**
  - India location preferred (`india_score`, `preferred_city_bonus` features).
  - Product-company employment rewarded (`product_company_flag`).
  - Consulting-only careers penalized (`consulting_only_flag`).
- **Fairness audit results** (see `docs/fairness_audit.md`):
  - Top-5 company share: 28% of top 100 (40 unique companies).
  - India share: 93% (vs 75.1% in the pool).
  - 0% currently at consulting firms in top 100 (vs 31.2% in the pool).
- No hallucinated content in reasoning: every fact in every reason is verified
  against the source candidate JSON at generation time.

## 8. Known limitations

- **Rubric inbreeding:** silver labels come from a rule-based scorer whose
  outputs partially overlap with reranker features. Internal composite score
  is optimistic by an unknown amount. Disclosed in `DECISIONS.md`.
- **Per-family ablation:** only `career_trajectory` features carry robust
  composite signal (drop of 0.1161 when removed). The other six families are
  marginal or slightly negative under current weights (see
  `docs/family_ablation.md`). Two interpretations are possible: silver leakage
  or feature over-engineering.
- **Small human-label set:** only 48 candidates have real human ratings. This
  sample is too small to statistically distinguish small ranking improvements
  at 95% confidence.
- **Cross-encoder experiment:** we tested a cross-encoder rerank stage. It
  underperformed on human labels (mean rating 4.50 vs 4.75). Disabled and
  archived in the codebase, not shipped.
- **Synthetic dataset:** the 100,000 candidates are synthetic. Approximately
  36% show `identical_career_descriptions` as a data-generation artifact. We
  handle this by excluding it from the honeypot gate.

## 9. Reproducibility

- Data integrity: `data/challenge/CHECKSUMS.sha256` (SHA-256 of every input).
- Environment: `requirements.txt` with pinned versions plus CPU-only torch.
- Docker: image reproduces the submission byte-identically in candidate
  ordering. See `docs/docker_reproduction_audit.md`.
- Git commit: `c94f41f` on main; tag `submission-v3-frozen` applied.
- Runtime: 90.5 seconds end-to-end on a Windows 11 laptop with 16 GB RAM, CPU
  only.

## 10. Contact and citation

Team (fill at submission time). Contact (fill at submission time).
Submission for the India Runs Data and AI Challenge 2026, hosted by Redrob.
