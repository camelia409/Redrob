# Portal submission metadata

Copy-paste ready. Fill only where marked (fill).

## Team information
- **Team name:** (fill)
- **Primary contact name:** (fill)
- **Primary contact email:** (fill)
- **Primary contact phone:** (fill)
- **Team members:** (fill - name and email for each)

## Repository and demo
- **GitHub repository URL:** https://github.com/camelia409/Redrob
- **Sandbox / demo link:** https://colab.research.google.com/github/camelia409/Redrob/blob/main/app/demo_colab.ipynb
- **Video walkthrough URL:** (fill from teammate - YouTube unlisted or Loom)

## Compute environment
- **Environment summary:** Windows 11, Python 3.13.6, 16 GB RAM, CPU only, single machine

## AI tools declaration
- **Declared:** Claude (used as engineering mentor and code reviewer during development)
- Other tools: (fill any others you used)

## Methodology summary (≤200 words)

Two-stage retrieve-then-rerank. BM25 over the full job description and MiniLM
dense embeddings over a hand-distilled 200-word JD query each retrieve top 1,500
from 100,000 candidates, giving a union of about 2,477. A weighted reranker over
30 features across 7 orthogonal families produces the ranking. A late-stage
Reciprocal Rank Fusion with BM25 is applied to the top of the list. A honeypot
gate with 10 integrity checks pushes profiles with 3 or more flags to the bottom.
Template-based reasoning generation with an assertion that verifies every fact
in the reason against the source JSON gives us zero hallucination.

Key finding: raw count of strict AI keywords on candidate profiles correlates
NEGATIVELY (-0.376) with our silver relevance labels. Keyword stuffers
accumulate JD terms without genuine career fit. We use this as a small negative
weight in the reranker.

Composite = 0.858 ± 0.035 (95% CI, bootstrap n=100). Runtime 91 seconds.
Honeypot rate in top 100: 0/100. Grounded reasoning: 100/100. Docker reproduces
byte-identically. Silver labels partially overlap with reranker features
(disclosed). On 30 hand-labeled candidates BM25 slightly outperforms our
reranker; on 8 diff candidates our reranker slightly underperforms cross-encoder-
promoted candidates. Both disclosed in DECISIONS.md rather than tuned away.
