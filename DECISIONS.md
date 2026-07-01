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
