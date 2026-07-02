#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd)"

echo "============================================================"
echo "FINAL CHECK - all 7 gates must pass before submission"
echo "============================================================"

echo ""
echo "[Gate 1/7] Data integrity"
python -c "from src.utils.integrity import verify_data_integrity; verify_data_integrity(); print('  OK')"

echo ""
echo "[Gate 2/7] Format validator"
python data/challenge/validate_submission.py outputs/final_submission.csv

echo ""
echo "[Gate 3/7] Reasoning grounding"
python scripts/validate_reasoning_grounding.py

echo ""
echo "[Gate 4/7] Honeypot rate in top 100"
python -c "
import pandas as pd
from src.ingestion.loader import iter_candidates
from src.validation.honeypots import honeypot_score_gates_only
sub = pd.read_csv('outputs/final_submission.csv')
top100_ids = set(sub['candidate_id'].tolist())
cands = {c['candidate_id']: c for c in iter_candidates() if c['candidate_id'] in top100_ids}
gated = sum(1 for cid in top100_ids if honeypot_score_gates_only(cands[cid]) >= 3)
print(f'  Honeypot rate: {gated}/100 = {gated}%')
assert gated < 10, 'FAIL: honeypot rate exceeds 10 percent'
"

echo ""
echo "[Gate 5/7] Row count and rank uniqueness"
python -c "
import pandas as pd
sub = pd.read_csv('outputs/final_submission.csv')
assert len(sub) == 100, f'Expected 100 rows, got {len(sub)}'
assert sorted(sub['rank'].tolist()) == list(range(1, 101)), 'Ranks not 1-100 unique'
assert len(sub['candidate_id'].unique()) == 100, 'Duplicate candidate_ids'
print('  100 rows, unique ranks 1-100, unique candidate_ids  OK')
"

echo ""
echo "[Gate 6/7] Score monotonicity"
python -c "
import pandas as pd
sub = pd.read_csv('outputs/final_submission.csv').sort_values('rank')
scores = sub['score'].tolist()
for i in range(1, len(scores)):
    assert scores[i] <= scores[i-1] + 1e-9, f'Score not monotone at rank {i+1}: {scores[i]} > {scores[i-1]}'
print('  Scores monotone non-increasing  OK')
"

echo ""
echo "[Gate 7/7] Test suite"
pytest tests/ -q

echo ""
echo "============================================================"
echo "ALL 7 GATES PASSED"
echo "============================================================"
echo ""
echo "Git status:"
git status --short
echo ""
echo "Last 3 commits:"
git log --oneline -3
echo ""
echo "Tags:"
git tag
