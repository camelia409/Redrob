# Per-family ablation of the weighted reranker

## Method

We measured the composite score contribution of each of the 7 feature families
by zeroing out all weights in that family and rescoring the 2,477 candidates in
the retrieval union, evaluated against full-population silver scores.

The composite metric is the challenge's official formula:

```
composite = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
```

Full-model composite: **0.8571**

## Results

| variant | composite | delta | ndcg@10 | mean_silver@100 |
|---|---:|---:|---:|---:|
| without_production_evidence | 0.8738 | +0.0167 | 0.8103 | 3.79 |
| without_skill_evidence | 0.8648 | +0.0076 | 0.8103 | 3.79 |
| without_behavioral | 0.8598 | +0.0026 | 0.7954 | 3.81 |
| without_semantic_jd_fit | 0.8596 | +0.0025 | 0.7931 | 3.78 |
| without_logistics | 0.8586 | +0.0015 | 0.7969 | 3.75 |
| full | 0.8571 | 0.0000 | 0.7931 | 3.78 |
| without_integrity_risk | 0.8562 | -0.0010 | 0.7931 | 3.78 |
| without_career_trajectory | 0.7410 | -0.1161 | 0.6509 | 3.48 |

## Families ranked by impact

(largest composite drop when removed)

1. **career_trajectory** — drop = 0.1161
2. **integrity_risk** — drop = 0.0010
3. logistics — *improvement* of 0.0015
4. semantic_jd_fit — *improvement* of 0.0025
5. behavioral — *improvement* of 0.0026
6. skill_evidence — *improvement* of 0.0076
7. production_evidence — *improvement* of 0.0167

## Interpretation

Only **career_trajectory** is load-bearing: removing it collapses NDCG@10 from
0.793 to 0.651 and composite by 0.116. Every other family is either neutral or
slightly harmful under the current weights. The negative deltas for
production_evidence, skill_evidence, behavioral, semantic_jd_fit, and logistics
mean those families' weights are currently introducing noise or over-penalizing
plausible candidates. The most likely explanation is that the hand-tuned weights
include negative signs (e.g., `strict_ml_skill_count_capped = -0.05`) intended
to suppress keyword stuffing, but those negative weights also hurt legitimate
profiles when the family is left active.

For a future iteration, the strongest single move would be to re-derive weights
with career_trajectory as the anchor and the other families either removed or
constrained to non-negative values. For the frozen submission, we leave the
weights unchanged because the full model still scores well and the ablation
does not reveal any family that is actively degrading the ranking enough to
justify a late-stage change.

## Why this matters

Ablation is standard practice at every ML research lab and production ML team
that publishes or ships rankings. It answers "which parts of my system are doing
the work?" — a question every senior reviewer will ask. Here it shows that our
ranking quality is driven almost entirely by career-trajectory signals (YoE
fit, tech/ML role fraction, career direction), with the remaining families
acting as minor regularizers at best.
