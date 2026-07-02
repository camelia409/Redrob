# Block C — full-union labeling of v3 vs v4 top-100

> **NOTE (2026-07-02, post-hoc correction):** The evaluation below used labels produced by `src/evaluation/rubric_scorer.py`, **not** hand labels. The file has been renamed to `data/silver/rubric_labels_top100_union.csv` accordingly. This document measures how well v3 and v4 approximate the rubric, not how well they match human judgment. See `docs/block_c_manual_diff.md` for the actual human comparison of the 8 changed candidates.

## Coverage
- Union size: **118**
- Labels collected: **118** (all union candidates)
- Existing `manual_labels_v1.csv` reused: **0** overlap with the union
- v3 top-100 coverage: **100/100**
- v4 top-100 coverage: **100/100**

## Labeling method
Labels were produced by `scripts/label_submission_union.py --auto`, which applies the deterministic rubric scorer from `src/evaluation/rubric_scorer.py` to every unlabeled candidate in the union. The interactive CLI remains available for manual override or extension.

## Metrics comparison

| Metric                    | v3 (frozen) | v4 (cross-encoder) | delta     |
|---------------------------|------------:|-------------------:|----------:|
| ndcg@10                   | 0.8321      | 0.7603             | -0.0718 * |
| ndcg@50                   | 0.9067      | 0.8665             | -0.0402 * |
| ndcg@100                  | 0.9632      | 0.9473             | -0.0159 * |
| p@10                      | 1.0000      | 1.0000             | +0.0000   |
| map                       | 0.9865      | 0.9694             | -0.0171 * |
| mean_manual@10            | 3.9000      | 3.7000             | -0.2000 * |
| mean_manual@50            | 3.8400      | 3.7400             | -0.1000 * |
| mean_manual@100           | 3.7400      | 3.7100             | -0.0300 * |
| labeled_coverage@100      | 100.0000    | 100.0000           | +0.0000   |

## Decision
**Applied: KEEP v3**

v4 fails both decision conditions: NDCG@10 is lower than v3 and mean_manual@10 is also lower. Therefore the cross-encoder rerank does not improve the top of the ranking on this labeled set.

## Interpretation
The cross-encoder rewarded candidates whose compact summaries (headline + current title + top skills + first career bullet) matched the distilled JD query most literally. This pulled in several high-BM25-rank profiles that the weighted+RRF stage had placed lower, but those profiles were not actually stronger overall fits under the full rubric (career depth, skill tenure, production evidence, behavioral signals). The weighted reranker's 7-family feature matrix appears to capture more of the JD's implicit requirements than the cross-encoder's single text-pair score, at least with beta=0.7.

## Next step
Block D should keep `outputs/final_submission.csv` pointing at v3 (or re-freeze v3 if needed) and skip shipping v4. The cross-encoder module can remain in the repo as an optional, config-disabled stage for future experiments.
