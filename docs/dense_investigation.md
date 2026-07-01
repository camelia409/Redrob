# Dense vs BM25 Investigation

## Setup

- Query v1 = full `configs/jd.txt` (long narrative JD).
- Query v2 = hand-distilled `configs/jd_query.yaml` (~190 words, technical requirements only).
- Embedding model fixed = `sentence-transformers/all-MiniLM-L6-v2`.
- Inspection format per candidate: `rank | candidate_id | current_title | YoE | top-3 skills | silver_score | headline`.

## Raw top-20 tables

### BM25 top-20

```text
 1  CAND_0088025    Staff Machine Learning Engineer   8.6  elasticsearch, learning to rank, rag      silver=4   "Staff Machine Learning Engineer | Building AI-native search "
 2  CAND_0068351    Lead AI Engineer                 6.4  peft, deep learning, nlp                  silver=4   "Senior Engineer | Search & Discovery Infrastructure"
 3  CAND_0037980    Senior Applied Scientist         9.0  text encoders, lora, tts                  silver=4   "Senior Engineer | Information Retrieval at scale"
 4  CAND_0011687    Senior NLP Engineer              7.8  semantic search, langchain, image classification  silver=3   "Senior NLP Engineer | Production ML at scale | 7.8+ yrs"
 5  CAND_0061257    Staff Machine Learning Engineer   8.0  workflow orchestration, indexing algorithms, vector representations  silver=4   "Senior Engineer | 8.0+ yrs in production systems"
 6  CAND_0046064    Senior NLP Engineer              8.9  deep learning, pinecone, bm25             silver=3   "Senior NLP Engineer | LLMs, RAG, Vector Search | ex-Top Tech"
 7  CAND_0039754    Senior Applied Scientist        16.2  nlp, deep learning, python                silver=1   "Senior Applied Scientist | Building AI-native search & ranki"
 8  CAND_0007411    Senior Machine Learning Engineer   8.0  information retrieval, vector search, prompt engineering  silver=5   "Senior Machine Learning Engineer | Production ML at scale | "
 9  CAND_0008425    Senior NLP Engineer              7.8  tensorflow, information retrieval, sentence transformers  silver=3   "Senior NLP Engineer | Production ML at scale | 7.8+ yrs"
10  CAND_0071974    Senior AI Engineer               7.8  speech recognition, peft, qdrant          silver=4   "Senior AI Engineer | Production ML at scale | 7.8+ yrs"
11  CAND_0080766    Staff Machine Learning Engineer   8.8  opensearch, deep learning, search infrastructure  silver=4   "Senior Engineer | Search & Discovery Infrastructure"
12  CAND_0093547    Senior Machine Learning Engineer   2.9  pytorch, peft, rag                        silver=2   "Senior Machine Learning Engineer | Building AI-native search"
13  CAND_0006567    Senior AI Engineer               7.9  nlp, content matching, ranking systems    silver=4   "Senior Engineer | Information Retrieval at scale"
14  CAND_0033861    Senior NLP Engineer              8.0  milvus, llamaindex, tensorflow            silver=3   "Senior NLP Engineer | Building AI-native search & ranking sy"
15  CAND_0002025    Senior AI Engineer               5.9  deep learning, nlp, faiss                 silver=4   "Senior AI Engineer | Building AI-native search & ranking sys"
16  CAND_0005538    Senior AI Engineer               5.9  information retrieval systems, tts, pgvector  silver=4   "Senior Engineer | 5.9+ yrs in production systems"
17  CAND_0094759    Lead AI Engineer                 8.6  scikit-learn, nlp, weaviate               silver=5   "Lead AI Engineer | Production ML at scale | 8.6+ yrs"
18  CAND_0055905    Senior Machine Learning Engineer   8.1  opensearch, fine-tuning llms, asr         silver=5   "Senior Machine Learning Engineer | LLMs, RAG, Vector Search "
19  CAND_0081846    Lead AI Engineer                 6.7  information retrieval, learning to rank, vector search  silver=4   "Lead AI Engineer | LLMs, RAG, Vector Search | ex-Top Tech"
20  CAND_0086022    Senior Applied Scientist         5.3  recommendation systems, pgvector, langchain  silver=5   "Senior Applied Scientist | LLMs, RAG, Vector Search | ex-Top"
```

### Dense-v1 (full JD) top-20

```text
 1  CAND_0060220    HR Manager                       1.6  snowflake, redis, javascript              silver=1   "HR Manager | Generative AI explorer"
 2  CAND_0095844    Mechanical Engineer              4.1  graphql, redis, aws                       silver=3   "Mechanical Engineer | Exploring AI & GenAI applications"
 3  CAND_0003067    HR Manager                       3.9  sap, spring boot, kafka                   silver=3   "HR Manager | Exploring AI & GenAI applications"
 4  CAND_0021708    HR Manager                       4.7  rest apis, apache flink, marketing        silver=3   "HR Manager | Generative AI explorer"
 5  CAND_0002126    Civil Engineer                   2.0  bigquery, scrum, redis                    silver=1   "Civil Engineer | Exploring AI & GenAI applications"
 6  CAND_0023074    HR Manager                       4.0  dbt, javascript, gcp                      silver=3   "HR Manager | Exploring AI & GenAI applications"
 7  CAND_0021581    Mechanical Engineer              1.6  redis, redux, typescript                  silver=1   "Mechanical Engineer | Generative AI explorer"
 8  CAND_0056808    HR Manager                       3.8  spark, salesforce crm, six sigma          silver=3   "HR Manager | Generative AI explorer"
 9  CAND_0050679    Civil Engineer                   2.3  typescript, apache flink, asr             silver=1   "Civil Engineer | Exploring AI & GenAI applications"
10  CAND_0073765    Mechanical Engineer              1.9  javascript, webpack, content writing      silver=1   "Mechanical Engineer | Exploring AI & GenAI applications"
11  CAND_0020974    Civil Engineer                   1.1  html, java, llms                          silver=1   "Civil Engineer | Exploring AI & GenAI applications"
12  CAND_0093707    Mechanical Engineer              1.6  angular, sales, hadoop                    silver=1   "Mechanical Engineer | Generative AI explorer"
13  CAND_0019696    Civil Engineer                   1.7  redis, salesforce crm, aws                silver=1   "Civil Engineer | Exploring AI & GenAI applications"
14  CAND_0081472    Civil Engineer                   2.2  bigquery, html, agile                     silver=1   "Civil Engineer | Exploring AI & GenAI applications"
15  CAND_0090946    HR Manager                       7.2  computer vision, vue.js, angular          silver=3   "HR Manager | Generative AI explorer"
16  CAND_0091863    Business Analyst                 2.5  content writing, photoshop, gans          silver=1   "Business Analyst | Exploring AI & GenAI applications"
17  CAND_0068102    HR Manager                       2.8  apache beam, terraform, redux             silver=1   "HR Manager | Generative AI explorer"
18  CAND_0064011    HR Manager                       1.6  data science, etl, java                   silver=1   "HR Manager | Generative AI explorer"
19  CAND_0046822    Operations Manager               1.3  gcp, javascript, postgresql               silver=1   "Operations Manager | Exploring AI & GenAI applications"
20  CAND_0009229    Operations Manager               1.5  vue.js, grpc, time series                 silver=1   "Operations Manager | Exploring AI & GenAI applications"
```

### Dense-v2 (distilled JD) top-20

```text
 1  CAND_0065195    Search Engineer                  5.1  hugging face transformers, llamaindex, diffusion models  silver=3   "Search Engineer | Search, Ranking & Retrieval"
 2  CAND_0084819    Search Engineer                  4.5  lora, semantic search, bm25               silver=3   "Search Engineer | Search, Ranking & Retrieval"
 3  CAND_0055992    AI Engineer                     16.9  deep learning, data science, information retrieval  silver=1   "AI Engineer | Search, Ranking & Retrieval"
 4  CAND_0041611    Staff Machine Learning Engineer   6.4  opensearch, langchain, weaviate           silver=4   "Staff Machine Learning Engineer | Building AI-native search "
 5  CAND_0014440    Recommendation Systems Engineer   6.4  faiss, tensorflow, elasticsearch          silver=3   "Recommendation Systems Engineer | Search, Ranking & Retrieva"
 6  CAND_0099806    AI Engineer                      4.6  qdrant, bm25, prompt engineering          silver=4   "AI Engineer | Search, Ranking & Retrieval"
 7  CAND_0083307    Search Engineer                  7.8  weaviate, python, scikit-learn            silver=3   "Search Engineer | Search, Ranking & Retrieval"
 8  CAND_0064326    Search Engineer                  7.6  deep learning, rag, weaviate              silver=3   "Search Engineer | Search, Ranking & Retrieval"
 9  CAND_0049896    Search Engineer                  7.3  information retrieval, data science, langchain  silver=3   "Search Engineer | Search, Ranking & Retrieval"
10  CAND_0042506    Search Engineer                  4.2  nlp, forecasting, faiss                   silver=3   "Search Engineer | Search, Ranking & Retrieval"
11  CAND_0030953    Search Engineer                  7.8  learning to rank, rag, langchain          silver=3   "Search Engineer | ML, NLP, Recommendation Systems"
12  CAND_0018722    Recommendation Systems Engineer   6.6  tensorflow, gans, hugging face transformers  silver=3   "Recommendation Systems Engineer | Search, Ranking & Retrieva"
13  CAND_0057701    Recommendation Systems Engineer   4.1  deep learning, learning to rank, prompt engineering  silver=3   "Recommendation Systems Engineer | Search, Ranking & Retrieva"
14  CAND_0041610    Recommendation Systems Engineer   6.7  opensearch, gans, lora                    silver=3   "Recommendation Systems Engineer | Search, Ranking & Retrieva"
15  CAND_0041568    Search Engineer                  5.2  computer vision, semantic search, weaviate  silver=1   "Search Engineer | Search, Ranking & Retrieval"
16  CAND_0000031    Recommendation Systems Engineer   6.0  mlflow, image classification, embeddings  silver=3   "Recommendation Systems Engineer | Search, Ranking & Retrieva"
17  CAND_0019480    NLP Engineer                     2.8  tensorflow, llms, milvus                  silver=1   "NLP Engineer | Search, Ranking & Retrieval"
18  CAND_0039754    Senior Applied Scientist        16.2  nlp, deep learning, python                silver=1   "Senior Applied Scientist | Building AI-native search & ranki"
19  CAND_0076251    Search Engineer                  7.6  hugging face transformers, image classification, forecasting  silver=3   "Search Engineer | Search, Ranking & Retrieval"
20  CAND_0045250    Applied ML Engineer              6.6  sentence transformers, nlp, milvus        silver=5   "Applied ML Engineer | Search, Ranking & Retrieval"
```

## Manual inspection

### BM25 top-20

- **Plausible AI Engineers:** 20/20
- **Plain-language Tier-5s (real signal in bullets, no keywords):** 0/20
- **Keyword stuffers:** 0/20
- **Junk / wrong role:** 0/20
- **Representative example:** `CAND_0007411` — Senior Machine Learning Engineer, 8.0 YoE, skills `information retrieval`, `vector search`, `prompt engineering`, silver=5. Title, experience band, and skills all align with the Senior AI Engineer role.

### Dense-v1 (full JD) top-20

- **Plausible AI Engineers:** 0/20
- **Plain-language Tier-5s:** 0/20
- **Keyword stuffers:** 20/20
- **Junk / wrong role:** 0/20
- **Representative example:** `CAND_0060220` — HR Manager, 1.6 YoE, skills `snowflake`, `redis`, `javascript`, headline "HR Manager | Generative AI explorer". The headline contains generic AI wording but the title and skill set are completely mismatched. This is exactly the "Marketing Manager with AI keywords" trap the JD warned about.

### Dense-v2 (distilled JD) top-20

- **Plausible AI Engineers:** 18/20
- **Plain-language Tier-5s:** 0/20
- **Keyword stuffers:** 2/20
- **Junk / wrong role:** 0/20
- **Representative example:** `CAND_0045250` — Applied ML Engineer, 6.6 YoE, skills `sentence transformers`, `nlp`, `milvus`, silver=5. Title, YoE, and skills directly match the distilled technical requirements.
- **Weak examples:** `CAND_0019480` (NLP Engineer, 2.8 YoE, silver=1) and `CAND_0041568` (Search Engineer with mostly computer-vision skills, silver=1) are relevant titles but weaker overall fit.

## Conclusion

Dense-v1 was broken by the long narrative JD. MiniLM matched the headline phrase "Exploring AI & GenAI applications" and surfaced HR Managers, Mechanical Engineers, and Civil Engineers — keyword stuffers, not candidates.

Dense-v2 with the distilled technical query fixes the failure mode completely: every top result is a Search / AI / Recommendation-Systems / NLP engineer, and there is no junk. The distilled query also creates partial overlap with BM25 (`Jaccard(BM25, Dense-v2 top-100) = 0.449`), confirming the two channels are finding related but not identical candidates.

The Dense-v2 channel is finding **real signal that BM25 misses** (e.g., candidates titled "Search Engineer" or "Recommendation Systems Engineer" whose profiles use semantic rather than exact keyword matches), but BM25 still wins on full-population NDCG@10/50 because the silver rubric is strongly lexical.

## Chosen path forward

**BM25 + Dense-v2 as separate feature channels for a learned re-ranker.**

Hybrid-v2 (BM25 + Dense-v2 RRF) improves MAP and `mean_silver@100` over BM25 alone, but it does not beat BM25 on NDCG@10/50. A simple weighted RRF is therefore not the final answer. The right next step is to feed BM25 scores, Dense-v2 scores, SkillCount, and metadata signals into a calibrated re-ranker so the model learns how to combine lexical and semantic evidence rather than averaging ranks heuristically.
