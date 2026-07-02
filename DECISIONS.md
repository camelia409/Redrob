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


## 2026-07-02 — Milestone 11: weighted reranker + first end-to-end submission

### Weight design rationale

Weights are hand-designed from the Milestone 10 correlations and domain
judgment. They are intentionally NOT normalized to sum to 1.

- Positive weight on `strict_ml_skill_count_capped` would inflate keyword
  stuffers — instead we assign a small NEGATIVE weight (-0.05) based on the
  Milestone 10 finding that this feature correlates negatively with silver score.
- `dense_v2_score` gets only a small positive weight (0.03) because its raw
  cosine has near-zero correlation; it is retained as a complementary channel.
- `bm25_rank_within_union` and `dense_v2_rank_within_union` are added as new
  rank-agreement features and weighted below the raw normalized BM25 score.

Family-level total weights approximately:

| family | approx. total |
|--------|---------------|
| semantic | 0.32 |
| skills | 0.12 |
| career | 0.34 |
| production | 0.12 |
| behavioral | 0.14 |
| logistics | 0.07 |
| integrity | 0.08 |

### Predicted composite table

Full-population metrics (unknown candidates treated as relevance 0). Composite =
0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10.

| ranker | NDCG@10 | NDCG@50 | MAP | P@10 | mean_silver@100 | HP@100 | composite | runtime |
|--------|---------|---------|-----|------|-----------------|--------|-----------|---------|
| weighted_reranker_top100 | 0.793 | 0.893 | 0.412 | 1.00 | 3.78 | 0.0% | **0.7762** | 0.0s |
| weighted_reranker_top100_no_honeypot_gate | 0.793 | 0.893 | 0.412 | 1.00 | 3.78 | 0.0% | **0.7762** | 0.0s |
| skill_count | 0.798 | 0.794 | 0.574 | 1.00 | 3.41 | 0.0% | 0.7734 | 0.2s |
| bm25 | 0.731 | 0.840 | 0.583 | 0.90 | 3.53 | 0.0% | 0.7500 | 67.7s |
| hybrid_v2_rrf | 0.604 | 0.739 | 0.821 | 0.80 | 3.41 | 0.0% | 0.6865 | 68.8s |
| dense_v2 | 0.598 | 0.705 | 0.370 | 0.90 | 3.34 | 0.0% | 0.6110 | 3.8s |
| random | 0.413 | 0.451 | 0.349 | 0.30 | 2.01 | 0.0% | 0.4091 | 0.1s |

### Actual result

- **Winning ranker by predicted composite:** `weighted_reranker_top100`
  (tied with the no-gate control because no candidate in the retrieval union
  tripped >=3 honeypot gates).
- **HP@100:** 0.0% (well under the 10% guardrail).
- **Validator status:** PASS.
- **End-to-end runtime of `scripts/generate_submission.py`:** 77.2s
  (well under the 240s budget).

### Decision

- Ship `outputs/submission_v1.csv`; format validated against the challenge
  validator.
- Prepare Milestone 12: grounded reasoning generator + hallucination assertion
  tests, and a LightGBM alternative reranker if the weighted model still trails.


## 2026-07-02 — Milestone 12: grounded reasoning + hallucination guard + submission_v2

### What was built

A template-based reasoning generator (`src/reasoning/generator.py`) that fills
`outputs/submission_v2.csv` with 100 rank-aware, fully grounded sentences. No
LLM is used. Every cited fact (title, company, YoE, skills, location,
concerns) comes from the candidate JSON or honeypot checks.

- `src/reasoning/phrase_bank.py`: deterministic per-candidate phrase variants.
- `src/reasoning/grounding.py`: `assert_grounded()` raises `HallucinationError`
  if the text mentions any skill, company, city, or number not present in the
  candidate JSON.
- `scripts/generate_submission.py`: generates `submission_v2.csv`; grounding
  assertion runs inline and fails loud on any mismatch.
- `scripts/validate_reasoning_grounding.py`: audits all 100 reasonings.
- `scripts/hand_label_evaluation.py`: first evaluation against the 30 human
  hand labels, independent of the rubric.

### Zero-hallucination proof

```text
$ python scripts/validate_reasoning_grounding.py
PASS: 100/100 reasonings are grounded.
```

### Sample reasonings

| rank | candidate_id | reasoning |
|------|--------------|-----------|
| 1 | CAND_0042029 | Senior applied-ML engineer profile at Flipkart, 6.5 years; strong skills in OpenSearch, PyTorch, and NLP |
| 25 | CAND_0083879 | Confident match: Machine Learning Engineer at Ola, 7.1 yrs; depth in Fine-tuning LLMs paired with Sentence Transformers |
| 55 | CAND_0074225 | Machine Learning Engineer profile at Unacademy; 4.3 years with caveats; Milvus, Vector Search, and Python in the toolkit; flagged: Machine Learning Engineer is a weaker match for the role |
| 85 | CAND_0015528 | Poor alignment: Applied ML Engineer with 7.4 years at Krutrim; caution: current title Applied ML Engineer is not ML-focused; likely due to current title Applied ML Engineer is not a strong fit |

### Reasoning character distribution

- min: 80
- mean: 153.8
- max: 197
- unique first-3-word openings: 66 / 100

### Hand-label evaluation (30 human labels)

| ranker | NDCG@10 | NDCG@30 | mean_manual@10 |
|--------|---------|---------|----------------|
| hybrid_v2_rrf | 0.947 | 0.947 | 1.20 |
| bm25 | 0.910 | 0.936 | 3.30 |
| weighted_reranker_top100 | 0.835 | 0.922 | 2.90 |
| dense_v2 | 0.586 | 0.812 | 2.20 |

The weighted reranker trails BM25 and Hybrid-v2 on the hand-label NDCG@10.
This is consistent with the full-population ablation where it wins on the
predicted composite but has lower MAP. The hand-label set is small (30
profiles), so we will **not** re-tune weights on it — that would be overfitting.
Instead, Milestone 13 will train a LightGBM reranker on a proper train/held-out
silver split.

### Vector-DB decision

**Decision:** Do not introduce a dedicated vector database (Chroma, Weaviate,
Pinecone, etc.) at this stage.

**Rationale:**
- We already have a pre-computed MiniLM embedding index (`candidate_embeddings.npy`
  + `candidate_ids.npy`) that gives sub-second dense retrieval.
- The dense-v2 channel is complementary but weaker than BM25 on the metrics that
  matter; adding a vector DB would add operational complexity without improving
  the ranking signal.
- The retrieval union (BM25 top-1500 ∪ Dense-v2 top-1500 = ~2,500 candidates)
  provides sufficient re-ranking input; a vector DB would not change the union.
- If a stronger embedding model or fine-tuned dense retriever is introduced
  later, we can revisit a managed vector store for faster updates.

### Action

- Ship `outputs/submission_v2.csv` (format-validated + 100/100 grounded).
- Move to Milestone 13: LightGBM point-wise reranker with a proper
  train/validation split on silver labels.


## 2026-07-02 — Milestone 17-18: Cross-encoder experiment (kept v3)

### What we tried
Added a cross-encoder rerank stage using cross-encoder/ms-marco-MiniLM-L-6-v2
on the top 500 from the weighted+RRF pipeline, blending cross-encoder score with
prior fused score (beta=0.7). Runtime cost: ~15-25s on CPU. Fits within budget.

### Rubric-based evaluation (misleading, kept for audit trail)
On rubric-scored labels the cross-encoder appeared to degrade NDCG@10 from
0.83 to 0.76. This measurement is invalid because the rubric scorer's outputs
overlap heavily with the weighted reranker's features. See
docs/block_c_full_labeling.md for details.

### Genuine human comparison (definitive)
Hand-rated the 8 candidates that differ between v3 top-10 and v4 top-10:
- 4 candidates cross-encoder promoted into top-10: mean rating 4.50 / 5
- 4 candidates cross-encoder demoted out of top-10: mean rating 4.75 / 5
- Delta: -0.25 on human judgment.

Full detail in docs/block_c_manual_diff.md.

### Why cross-encoder underperformed
The cross-encoder rewards literal text alignment between JD and profile. Our
top candidates include some whose career bullets describe production ranking
work in generic terms ("shipped models at scale") rather than in JD-vocabulary
terms ("built retrieval system"). The cross-encoder ranks the generic-language
candidates lower even though a human recruiter would forward them. This is a
known cross-encoder failure mode when the ground truth uses domain-neutral
language.

### Decision
Kept v3 (weighted + RRF, tag submission-v1-frozen).
Cross-encoder module remains in the codebase with enabled: false. Reproducible
and available for future experiments.
