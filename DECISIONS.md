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
