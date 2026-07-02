# Latency budget

## Spec constraint

Ranking step must complete in **300 seconds** on CPU-only 16 GB RAM per
challenge spec.

## Measured budget (single fresh run)

```text
==============================================================================
stage                                       seconds   % pipeline   % 5-min budget
------------------------------------------------------------------------------
data_integrity_check                           0.47         0.5%             0.2%
load_candidates                                4.10         4.5%             1.4%
bm25_retrieval                                81.13        88.6%            27.0%
dense_retrieval                                5.12         5.6%             1.7%
union_and_feature_extraction                   0.61         0.7%             0.2%
weighted_rerank                                0.01         0.0%             0.0%
rrf_fusion                                     0.10         0.1%             0.0%
reasoning_generation_and_grounding             0.05         0.1%             0.0%
TOTAL                                         91.61       100.0%            30.5% *
==============================================================================
```

Raw CSV: `outputs/latency_budget.csv`.

## Takeaway

**BM25 retrieval is the sole bottleneck, consuming 88.6% of the pipeline and
27.0% of the 5-minute budget; the rest of the system is effectively free.**

## What we would optimize first

BM25 retrieval dominates at 81.13 seconds. If we needed to reduce total time,
we would pre-tokenize the 100,000-candidate corpus at build time and cache
tokenized lists on disk. That would move the bulk of the tokenization and
index-construction work out of the ranking window.

Dense retrieval is only 5.12 seconds (5.6%) because the embedding index is
pre-built. Feature extraction, reranking, RRF, and reasoning generation
together account for less than 1% of runtime, so optimizing them would have
negligible impact.

## Why this matters

Production ML systems track latency budgets on every model. It answers "at
what point does adding features stop being free?" — a question every
production team asks eventually. Here the answer is clear: any new feature can
be added to the weighted reranker or RRF stage at essentially zero marginal
cost, but replacing or accelerating BM25 is the only way to materially reduce
end-to-end latency.
