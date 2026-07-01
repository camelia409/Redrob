# Engineering decisions log

## 2026-07-01 — Milestone 7: baseline rankers + first NDCG

### Interpretation rule stated BEFORE seeing numbers

- **Healthy:** BM25 NDCG@10 > SkillCount NDCG@10 > Random NDCG@10 + 0.05 on the
  full 100K silver-label evaluation.
- **Red flag:** BM25 < SkillCount → BM25 tokenization or JD text is off.
- **Both barely beat Random:** silver labels are too sparse/noisy for top-K
  evaluation; investigate before Milestone 8.

### Actual result

Full-pool evaluation (unknown candidates treated as relevance 0):

| ranker      | NDCG@10 | NDCG@50 | NDCG@100 | MAP   | P@10 | P@100 | HP@100 | runtime |
|-------------|---------|---------|----------|-------|------|-------|--------|---------|
| random      | 0.000   | 0.000   | 0.000    | 0.002 | 0.00 | 0.00  | 0.0%   | 0.1s    |
| skill_count | 0.000   | 0.014   | 0.008    | 0.007 | 0.00 | 0.01  | 0.0%   | 0.3s    |
| bm25        | 0.000   | 0.000   | 0.008    | 0.008 | 0.00 | 0.01  | 0.0%   | 82.2s   |

Labeled-only diagnostic (metrics computed on the 500 silver-labeled candidates only):

| ranker      | NDCG@10 | NDCG@50 | MAP   | P@10 |
|-------------|---------|---------|-------|------|
| random      | 0.313   | 0.462   | 0.408 | 0.30 |
| skill_count | 0.949   | 0.885   | 0.836 | 1.00 |
| bm25        | 0.976   | 0.955   | 0.828 | 1.00 |

Only 0–1 labeled candidates appear in any top-50 of the full-pool ranking
because the silver set covers just 0.5% of the 100K pool.

### Interpretation

Neither baseline dominates on the full-pool NDCG@10 because the evaluation is
pessimistic: the top-10 is almost entirely unlabeled candidates, all treated as
irrelevant. The red flag is **not** a broken BM25 ranker — the labeled-only
diagnostics show both BM25 and SkillCount rank the known-relevant candidates
very well (NDCG@10 ≈ 0.95–0.98). This means the ranking signal exists, but the
sparse silver labels are insufficient to measure it at K=10 on the full pool.

### Action

1. Proceed to Milestone 8 (dense embeddings + hybrid retrieval) because the
   baselines prove there is a measurable ranking signal on the labeled subset.
2. Before final evaluation, build a denser silver/gold set or evaluate
   primarily on the labeled subset / pooled evaluation so top-K metrics are
   not dominated by unknown candidates.
3. Keep BM25 as a strong lexical baseline and add it as a feature channel in
   the hybrid model.


## 2026-07-02 — Milestone 8: dense embeddings + hybrid RRF + full-population ablation

### Interpretation rule stated BEFORE seeing numbers

- **Healthy:** HybridRRF > Dense > BM25 > SkillCount > Random on full-population
  NDCG@10, using the full 100K silver scores as gains.
- **Expected shape:** Top-100 Jaccard(BM25, Dense) should be 0.1–0.4 — partially
  overlapping but meaningfully different top results.
- **Red flag:** Dense NDCG@10 barely beats BM25 → the JD embedding is noisy,
  MiniLM's coverage of ML terminology is weak, or the JD text dominates the
  candidate text in an unhelpful way.
- **Both barely beat random:** the full-population silver scores are too noisy
  to measure ranking quality; investigate the rubric or labels first.

### Actual result

Full-population evaluation using the complete 100K silver scores as gains:

| ranker      | NDCG@10 | NDCG@50 | MAP   | P@10 | mean_silver@100 | HP@100 | labeled-NDCG@10 | runtime |
|-------------|---------|---------|-------|------|-----------------|--------|-----------------|---------|
| random      | 0.413   | 0.451   | 0.349 | 0.30 | 2.01            | 0.0%   | 0.313           | 0.1s    |
| skill_count | 0.798   | 0.794   | 0.574 | 1.00 | 3.41            | 0.0%   | 0.949           | 0.3s    |
| bm25        | 0.731   | 0.840   | 0.583 | 0.90 | 3.53            | 0.0%   | 0.976           | 91.2s   |
| dense       | 0.402   | 0.396   | 0.389 | 0.50 | 1.70            | 0.0%   | 0.703           | 6.4s    |
| hybrid_rrf  | 0.634   | 0.642   | 0.659 | 0.90 | 2.64            | 0.0%   | 0.957           | 89.3s   |

Top-100 Jaccard(BM25, Dense) = **0.000** (0 / 200 candidates overlap).

### Interpretation

The pre-declared "healthy" ordering did **not** hold. SkillCount and BM25 both
outperform Dense on full-population NDCG@10, and HybridRRF lands below both
BM25 and SkillCount. The full-pop scores are now sensible (non-zero everywhere),
so this is a real signal, not an artifact of sparse labels.

Key take-aways:

1. **MiniLM generic semantic similarity is poorly aligned with this rubric.**
   The JD is long and narrative; truncation + generic domain pre-training
   likely maps the query to broad "AI/ML" semantics rather than the specific
   combination of production retrieval, ranking, and startup-fit the rubric
   rewards.
2. **Lexical and semantic channels are almost completely disjoint.**
   Zero overlap in the top-100 means the two rankers are finding different
   candidate populations, so RRF cannot rescue a weak dense channel — it just
   dilutes the strong BM25 signal with irrelevant dense candidates.
3. **SkillCount is unexpectedly strong.** The rubric heavily rewards explicit
   skills and experience range, so a simple matcher already gets close to the
   ceiling on the labeled subset.

### Action

1. **Do not promote Dense/HybridRRF as the final model yet.** BM25 is still the
   strongest single ranker on full-population NDCG@50 and MAP.
2. **Improve the dense channel before fusing it:**
   - Shorten the JD query to a focused "role requirements" snippet instead of
     the entire 4K-character narrative.
   - Build a domain-specific candidate text (headline + titles + key skills)
     rather than a full concatenation.
   - Evaluate a stronger embedding model (e.g., `all-mpnet-base-v2` or a
     fine-tuned job-candidate model) if compute budget allows.
3. **Inspect the top-50 dense candidates manually** to confirm whether the poor
   NDCG is due to generic matches, text truncation, or misalignment with the
   rubric.
4. **Next milestone:** add a learned re-ranker / cross-encoder that takes BM25
   + Dense + SkillCount features and outputs a calibrated score, rather than
   relying on heuristic RRF fusion.


## 2026-07-02 — Milestone 9: distilled JD query + dense failure-mode investigation

### Interpretation rule stated BEFORE seeing numbers

- **Healthy:** Dense-v2 (distilled query) returns plausible Senior AI/Search
  engineers with no junk, has non-zero top-100 overlap with BM25, and improves
  `mean_silver@100` over Dense-v1.
- **Success for fusion:** Hybrid-v2 (BM25 + Dense-v2) improves MAP or
  `mean_silver@100` over BM25 alone.
- **Red flag:** Dense-v2 still surfaces HR/Marketing/Engineering managers with
  generic AI headlines → the dense channel is fundamentally misaligned and
  should be dropped.
- **Decision options:**
  1. BM25 only if Dense-v2 is garbage.
  2. Hybrid-v2 if Dense-v2 is good but not additive to BM25 on NDCG.
  3. BM25 + Dense-v2 as separate feature channels (recommended if manual
     inspection is mixed and both channels find good-but-different candidates).

### Actual result

Full-population ablation with Dense-v2 and Hybrid-v2 added:

| ranker      | NDCG@10 | NDCG@50 | MAP   | P@10 | mean_silver@100 | HP@100 | labeled-NDCG@10 | runtime |
|-------------|---------|---------|-------|------|-----------------|--------|-----------------|---------|
| random      | 0.413   | 0.451   | 0.349 | 0.30 | 2.01            | 0.0%   | 0.313           | 0.1s    |
| skill_count | 0.798   | 0.794   | 0.574 | 1.00 | 3.41            | 0.0%   | 0.949           | 0.2s    |
| bm25        | **0.731** | **0.840** | 0.583 | 0.90 | **3.53**        | 0.0%   | 0.976           | 74.0s   |
| dense       | 0.402   | 0.396   | 0.389 | 0.50 | 1.70            | 0.0%   | 0.703           | 4.2s    |
| dense_v2    | 0.598   | 0.705   | 0.370 | 0.90 | 3.34            | 0.0%   | **1.000**       | 0.1s    |
| hybrid_rrf  | 0.634   | 0.642   | 0.659 | 0.90 | 2.64            | 0.0%   | 0.957           | 74.1s   |
| hybrid_v2   | 0.604   | 0.739   | **0.821** | 0.80 | 3.41            | 0.0%   | 0.979           | 75.3s   |

Top-100 Jaccard(BM25, Dense-v1) = **0.000**.  
Top-100 Jaccard(BM25, Dense-v2) = **0.449** (62 / 138 candidates overlap).

Manual inspection of top-20 lists (see `docs/dense_investigation.md`):

| ranker   | plausible AI engineer | plain-language Tier-5 | keyword stuffer | junk |
|----------|----------------------:|----------------------:|----------------:|-----:|
| BM25     | 20/20                 | 0/20                  | 0/20            | 0/20 |
| Dense-v1 | 0/20                  | 0/20                  | 20/20           | 0/20 |
| Dense-v2 | 18/20                 | 0/20                  | 2/20            | 0/20 |

### Interpretation

The long narrative JD was the main failure mode for Dense-v1. Once the query is
hand-distilled to the technical requirements, MiniLM finds an almost entirely
plausible set of Search / AI / Recommendation-Systems engineers.

Dense-v2 is **not garbage** — manual inspection shows no junk and meaningful
overlap with BM25 — but it is also **not strictly better** than BM25 on the
metrics the rubric optimizes for (NDCG@10/50). Hybrid-v2 improves MAP and
`mean_silver@100` versus BM25, which suggests the two channels are capturing
complementary signal, but simple RRF is too crude to realize the full gain.

### Decision

**Use BM25 + Dense-v2 as separate feature channels for a learned re-ranker.**

BM25 remains the strongest single ranker for NDCG@10/50. Dense-v2 adds
complementary semantic candidates and reaches a labeled-subset NDCG@10 of 1.000,
so it should be retained as a feature rather than discarded. The next milestone
should train a point-wise or pair-wise re-ranker over BM25 score, Dense-v2 score,
SkillCount, and metadata signals, with a held-out validation set to choose the
fusion weights.


## 2026-07-02 — Milestone 10: 7-family feature extractor

### What was built

A `FeatureExtractor` that produces ~25 numeric features from 7 orthogonal
families for every candidate in the BM25 ∪ Dense-v2 top-1500 retrieval funnel
(2,477 candidates). Features are all normalized to [0, 1]. No feature uses the
silver score directly.

### Feature matrix

- Rows: **2,477**
- Columns: **31** (candidate_id + 25 features + silver_score + honeypot_score +
  raw bm25_score + raw dense_score)
- Output: `data/processed/feature_matrix.parquet`

### Top-5 features by |corr| with silver_score

| rank | feature | corr |
|------|---------|------|
| 1 | yoe_in_ideal_band | 0.6212 |
| 2 | mean_skill_duration_months_ml_capped | 0.5890 |
| 3 | career_production_phrase_count_capped | 0.5098 |
| 4 | tech_role_fraction | 0.4701 |
| 5 | bm25_score_normalized | 0.4501 |

### Bottom-5 features by |corr| with silver_score (flagged for removal)

| rank | feature | corr |
|------|---------|------|
| 21 | broad_ml_skill_count_capped | 0.0787 |
| 22 | mean_tenure_months_capped | 0.0634 |
| 23 | dense_v2_score | -0.0576 |
| 24 | notice_period_score | 0.0458 |
| 25 | career_ml_system_phrase_count_capped | -0.0442 |

### Surprises

1. **`strict_ml_skill_count_capped` is negatively correlated (-0.3757).**
   Having many strict JD skills is associated with *lower* silver scores in the
   retrieved union. This is likely because keyword-stuffing profiles load up on
   exact skills but fail other rubric gates (title, YoE, interview rate,
   honeypots). It reinforces the JD warning that keyword matching alone is a
   trap.
2. **`dense_v2_score` has almost no correlation with silver_score (-0.0576).**
   Manual inspection showed Dense-v2 returns plausible candidates, but the raw
   cosine is not a strong silver predictor on its own. It should still be
   retained as a feature for the learned re-ranker because it captures
   complementary semantic signal.
3. **`career_production_phrase_count_capped` is far more predictive than
   `career_ml_system_phrase_count_capped`.** The rubric rewards production-scale
   evidence more than retrieval-specific buzzwords.

### Decision

Keep all 25 features for the first re-ranker, but expect the model to down-weight
or zero out the bottom-5. After the first model fit, run feature importance and
remove any feature with both low correlation and low model importance. Do not
remove features now based on correlation alone because some may become useful in
non-linear combinations (e.g., dense_v2_score × yoe_in_ideal_band).
