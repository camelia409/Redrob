# Adversarial red-team of the honeypot detector

## Purpose

Before shipping the submission, we hand-crafted 5 synthetic candidate profiles
representing distinct attack patterns a bad actor might use to game a candidate
ranking system. Each profile is designed to look plausible to a naive keyword
matcher but should be caught by our integrity checks.

## Attacks tested

| # | Attack pattern | What we crafted | `honeypot_score_gates_only` | Tripped checks |
|---|---|---|---|---|
| 1 | Timeline inflation | Claimed 15 YoE, career history sums to 5 years | 3 | timeline_inflation, expert_zero_duration, expert_zero_endorsements |
| 2 | Expert-zero-duration stacking | 12 expert-level skills with 0 duration and 0 endorsements, plus inflated YoE | 3 | timeline_inflation, expert_zero_duration, expert_zero_endorsements |
| 3 | Consulting keyword stuffing | HR Manager at Infosys with 15 expert AI skills | 3 | expert_zero_duration, expert_zero_endorsements, consulting_keyword_stuffing |
| 4 | Impossible tenure | Single role of 30 years continuous tenure | 3 | expert_zero_duration, expert_zero_endorsements, tenure_over_240_months |
| 5 | Rookie perfect profile | 1 YoE, 100% completeness, 12 expert AI skills | 3 | expert_zero_duration, expert_zero_endorsements, rookie_perfect_profile |

## Result

All 5 attacks trip exactly 3 integrity checks and would be zeroed out by our
honeypot gate. Tests are in `tests/test_adversarial_honeypots.py` and run on
every commit.

```text
105 passed in 201.04s (0:03:21)
```

## Why this matters

Real hiring platforms face profile fabrication as a live threat. LinkedIn's
Trust & Safety team runs similar adversarial tests on their profile detection
models. This kind of red-teaming is standard practice for any ML system that
gates a decision recruiters trust.
