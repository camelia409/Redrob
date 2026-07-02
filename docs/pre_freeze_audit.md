# Pre-freeze audit — Milestone 13

Date: 2026-07-02  
Candidate submission: `outputs/submission_v2.csv`  
Frozen submission: `outputs/final_submission.csv`

---

## 1. Reproducibility audit

### Procedure

```bash
rm -f data/processed/feature_matrix.parquet
PYTHONPATH=. python scripts/build_feature_matrix.py
PYTHONPATH=. python scripts/generate_submission.py
diff outputs/submission_v2_committed.csv outputs/submission_v2.csv
```

### Result

- Feature matrix rebuilt deterministically: 2,477 rows × 33 columns.
- End-to-end generation completed in **90.5 seconds** (well under the 5-minute limit).
- `diff` output was empty — the rerun produced a **byte-identical** `submission_v2.csv`.

### Reproducibility hash

```text
sha256(outputs/submission_v2.csv) = 27c5f63e2da57b43ea220abde11baebaa3e5a5a8e65b0d9bbae6af954b1dead3
bytes: 15518
```

**Verdict:** ✅ Deterministic and reproducible.

---

## 2. Reasoning tone-bug fix

### Defect

Ranks 40–80 generated self-contradicting reasonings such as:

> "Machine Learning Engineer profile at Unacademy; 4.3 years with caveats; ... flagged: Machine Learning Engineer is a weaker match for the role"

### Fix

- Added `_title_appears_ml_aligned()` and `_filter_concerns()` helpers in
  `src/reasoning/generator.py`.
- Added neutral mid-tier phrases in `src/reasoning/phrase_bank.py`.
- When a candidate already holds an ML-aligned title, the generator now suppresses
  "not ML-focused" / "weaker match" / "not a strong fit" concerns and falls back
  to neutral language.
- Applied the same filter to low-rank reason clauses to avoid contradictions
  at ranks 71–100.
- Added `test_ml_titled_candidate_at_mid_rank_not_flagged_as_non_ml`.

### Verification

Sample reasonings after the fix:

```text
#50: Machine Learning Engineer at PharmEasy with 5.1 years; relevant expertise: Semantic Search, PyTorch, Elasticsearch; concern: based in Seattle
#55: credible fit: Machine Learning Engineer at Unacademy; 4.3 yrs of relevant experience; skills include Milvus, Vector Search, and Python
#60: solid mid-tier match: Senior Data Scientist at Rephrase.ai, 5.7 yrs; skills include Sentence Transformers, FAISS, and RAG
#70: credible fit: AI Research Engineer at PhonePe; 6.0 yrs of relevant experience; skills include NLP, Elasticsearch, and BM25
```

No rank 40–80 reasoning now contains "current title X is not ML-focused" or
"X is not a strong fit" for an ML-titled candidate.

**Verdict:** ✅ Tone bug fixed and regression-tested.

---

## 3. Eyeball audit

### Top-10

| rank | candidate_id | title @ company | rating (1–5) | notes |
|------|--------------|-----------------|--------------|-------|
| 1 | CAND_0042029 | Senior Data Scientist @ Flipkart | 5 | Prior ML Engineer role; expert PyTorch/NLP/RAG; strong product-company trajectory. |
| 2 | CAND_0079284 | Machine Learning Engineer @ Google | 5 | Clear ML title; expert recommendation/vector-search skills; top-tier companies. |
| 3 | CAND_0006418 | Machine Learning Engineer @ Verloop.io | 5 | Strong retrieval stack (Semantic Search, Weaviate, Embeddings); prior AI Engineer at Flipkart. |
| 4 | CAND_0050454 | AI Engineer @ Rephrase.ai | 5 | Deep LLM-fine-tuning & QLoRA; prior ML Engineer at Uber/Adobe. |
| 5 | CAND_0064326 | Search Engineer @ Sarvam AI | 4 | Title is search, but history is solid ML and skills are retrieval-heavy; still forwardable. |
| 6 | CAND_0078002 | Machine Learning Engineer @ Meta | 5 | Strong ML title + product company + relevant skills. |
| 7 | CAND_0051630 | Machine Learning Engineer @ Razorpay | 5 | Expert-level skills in retrieval/recommendation/Embeddings. |
| 8 | CAND_0053591 | AI Engineer @ Ola | 4 | Strong skills, but located in Toronto (not India); still a credible candidate. |
| 9 | CAND_0037566 | Machine Learning Engineer @ LinkedIn | 5 | Excellent retrieval/LLM skill set; strong product-company history. |
| 10 | CAND_0030031 | AI Engineer @ Microsoft | 5 | Prior Amazon/Google; strong PyTorch/RAG/NLP. |

**Mean top-10 rating: 4.7 / 5**

No top-10 candidate would be declined for forwarding.

### Bottom-10 (ranks 91–100)

| rank | candidate_id | title @ company | rating (1–5) | notes |
|------|--------------|-----------------|--------------|-------|
| 91 | CAND_0018722 | Recommendation Systems Engineer @ Saarthi.ai | 3 | Good skills and prior ML roles; weakened by Toronto location and non-technical-role fraction. |
| 92 | CAND_0041625 | AI Research Engineer @ Infosys | 2 | Decent title but intermediate-level skills and consulting-firm context. |
| 93 | CAND_0098952 | AI Research Engineer @ CRED | 2 | CRED is strong, but skill proficiencies are mostly intermediate. |
| 94 | CAND_0078262 | ML Engineer @ Yellow.ai | 2 | Title is good, but prior role is Data Scientist at Wipro and skill depth is mixed. |
| 95 | CAND_0035653 | Data Scientist @ Swiggy | 2 | Good company, but title is data-scientist-centric and the trajectory is mixed. |
| 96 | CAND_0010603 | ML Engineer @ BYJU'S | 2 | Consulting-ish trajectory; moderate skill depth. |
| 97 | CAND_0058688 | AI Engineer @ Vedantu | 2 | Strong title but located in Berlin and skills are more CV than retrieval/LLM. |
| 98 | CAND_0040677 | AI Research Engineer @ Swiggy | 2 | Junior-ish tenure; prior ML Engineer role but limited depth. |
| 99 | CAND_0011162 | Recommendation Systems Engineer @ upGrad | 3 | Strong skills and prior ML Engineer at Google; pulled down by role-fraction features. |
| 100 | CAND_0041669 | Recommendation Systems Engineer @ CRED | 3 | Strong skills and prior Search/NLP roles; still weaker than top tier. |

**Mean bottom-10 rating: 2.3 / 5**

Bottom-10 candidates are weaker fits but not garbage or honeypots; they are plausible profiles with some misalignment on title trajectory, location, or skill depth.

**Verdict:** ✅ Eyeball audit passes.

---

## 4. Gate audit

### Format validator

```text
$ python data/challenge/validate_submission.py outputs/submission_v2.csv
Submission is valid.
```

### Honeypot rate in top-100

```text
Honeypot rate: 0/100 = 0%
```

### Reasoning grounding

```text
$ python scripts/validate_reasoning_grounding.py
PASS: 100/100 reasonings are grounded.
```

### Runtime

```text
Total end-to-end runtime: 90.5s
```

Well under the 5-minute (300s) limit.

**Verdict:** ✅ All gates pass.

---

## 5. Decision

- `outputs/submission_v2.csv` is the audited candidate.
- `outputs/final_submission.csv` is a byte-identical freeze of v2.
- v1 remains in `outputs/submission_v1.csv` for the audit trail.
- The submission is ready for Docker packaging in the next milestone.

**Freeze approved.**
