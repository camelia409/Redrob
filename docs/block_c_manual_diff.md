# Block C — manual v3 vs v4 diff (8 candidates)

## Rubric reminder
5 = strong forward | 4 = would forward with note | 3 = reasonable | 2 = marginal | 1 = weak | 0 = decline

## NEW in v4 top-10 (cross-encoder promoted these)

| cid                       | rating (0-5) | one-line rationale |
|---------------------------|-------------:|--------------------|
| CAND_0002025 (v4 rank 4)  | 5 | Strong AI Engineer at Apple with 5.9 YoE; shipped production recommendation system with hybrid retrieval and LLM fine-tuning; 30-day notice. |
| CAND_0088025 (v4 rank 8)  | 4 | Staff ML Engineer with ranking pipeline and vector-DB skills but identical career descriptions and 90-day notice reduce confidence. |
| CAND_0011687 (v4 rank 9)  | 5 | Senior NLP Engineer with search/ranking pipeline, embeddings, PEFT, and strong behavioral signals; product-company background. |
| CAND_0008425 (v4 rank 10) | 4 | Senior NLP Engineer at Ola with semantic-search and LTR expertise but lower response rate and 90-day notice. |

**mean_NEW = 4.50**

## DROPPED from v3 top-10 (cross-encoder demoted these out of top-10)

| cid                          | rating (0-5) | one-line rationale |
|------------------------------|-------------:|--------------------|
| CAND_0042029 (was v3 rank 1) | 5 | Senior Data Scientist at Flipkart with proven ranking-model shipping experience and RAG; strong product-company fit. |
| CAND_0060054 (was v3 rank 4) | 4 | AI Engineer with semantic search and recommendation experience at product companies; solid but not standout. |
| CAND_0078002 (was v3 rank 6) | 5 | ML Engineer at Meta with ranking/NLP and strong behavioral signals; duplicate descriptions are a minor synthetic artifact. |
| CAND_0068351 (was v3 rank 8) | 5 | Lead AI Engineer with end-to-end search/discovery ownership, multiple vector DBs, and excellent behavioral signals. |

**mean_DROPPED = 4.75**

## Decision rule
- **Ship v4** if mean_NEW > mean_DROPPED (cross-encoder objectively improves top-10 quality on human signal).
- **Keep v3** if mean_NEW <= mean_DROPPED.

## Applied decision
**delta = mean_NEW - mean_DROPPED = -0.25**

**Applied: KEEP v3**

## One-sentence interpretation
The four candidates the cross-encoder promoted are strong, but the four it pushed out of the top-10 are slightly stronger on average, so v4's re-ordering does not improve human-perceived top-10 quality.
