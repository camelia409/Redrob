# Fairness and concentration audit of top-100

## Scope

We audit structural properties of our frozen `outputs/final_submission.csv` top
100: company concentration, geographic concentration, experience distribution,
notice-period distribution, product vs consulting employment, and behavioral
signal averages.

We do **not** audit protected attributes (caste, religion, gender, marital
status, disability, ethnicity, age beyond the stated JD range). Our ranking
pipeline does not use these attributes; auditing them would give a false sense
of safety and inferring them from names would itself introduce bias.

## Findings

```text
========================================================================
FAIRNESS AND CONCENTRATION AUDIT — TOP 100
========================================================================

1. Company concentration
   Unique companies in top 100: 40
   Top-5 companies share: 28/100 = 28%
     - Ola                                        6
     - Rephrase.ai                                6
     - Mad Street Den                             6
     - Netflix                                    5
     - Niramai                                    5
   Pool context: 63 unique companies across 100000 candidates

2. Location concentration
   Unique locations in top 100: 22
   Top-3 locations share: 30/100 = 30%
     - Noida, Uttar Pradesh                      11
     - Kolkata, West Bengal                      10
     - Delhi, Delhi                               9
   Pool context: 28 unique locations across 100000 candidates

3. Country distribution
   India: 93/100 = 93%
     - India                                     93
     - Singapore                                  2
     - Canada                                     2
     - USA                                        1
     - Germany                                    1
   Pool context: India = 75.1% of full pool

4. Years of experience
   Mean: 6.41    Std: 1.65    Min: 2.9    Max: 16.2
   Inside JD ideal band [5,9]: 82/100 = 82%
   Pool context: mean=7.17, std=3.82

5. Notice period distribution
   <= 30 days       37
   31-60 days       35
   61-90 days       17
   > 90 days        11

6. Product vs consulting employment
   Currently at consulting firm : 0/100
   Has consulting history       : 2/100
   Pool context: currently consulting = 31.2%

7. Behavioral signal averages in top 100
   Mean recruiter_response_rate : 0.683
   Mean interview_completion    : 0.800

========================================================================
PROTECTED ATTRIBUTES NOT AUDITED
========================================================================
The following were deliberately NOT used in ranking or in this audit:
  - candidate name (any gender/ethnicity/caste inference)
  - religion / marital status / disability
  - age beyond the JD's stated experience range (5-9 years)

This is a hackathon synthetic-data prototype, not a production hiring
system. Any production deployment would require additional bias auditing,
human-in-the-loop review, and legal compliance (NYC Local Law 144,
EU AI Act, EEOC guidance).
```

## Interpretation

**Company concentration is moderate.** The top 100 spans 40 unique companies,
with the top five employers accounting for 28% of the shortlist. No single
company dominates, and the leaders (Ola, Rephrase.ai, Mad Street Den, Netflix,
Niramai) are a mix of Indian startups/product companies and one global
streaming company. This suggests the model is not merely returning candidates
from a single well-represented employer in the pool.

**Geographic concentration is plausible for an India-focused role.** The top
three cities (Noida, Kolkata, Delhi) account for 30% of the shortlist, and 93%
of candidates are based in India versus 75.1% in the full pool. The India
over-representation is expected because the JD features include an
`india_score` and `preferred_city_bonus`, and the role is framed for the Indian
market. In a production system this would need to be checked against the
employer's actual hiring geography; if the role were genuinely global, 93%
India would be a red flag.

**Experience distribution aligns well with the JD.** The mean YoE is 6.41 years
(standard deviation 1.65), lower than the pool mean of 7.17, and 82% fall
inside the stated 5–9 year ideal band. The model is therefore filtering toward
the requested seniority rather than simply ranking the most senior candidates
highest. The maximum of 16.2 years shows that a few very senior profiles still
make the cut, likely due to strong retrieval/rubric alignment, but they are not
representative of the list.

**Availability is practical.** Seventy-two percent of the top 100 have a notice
period of 60 days or less, and only 11% require more than 90 days. Because
`notice_period_score` is a positive feature, this concentration is intentional
and operationally useful for a hiring manager.

**Consulting representation is essentially zero by design.** Zero candidates
are currently at consulting firms and only two have any consulting history,
compared with 31.2% of the full pool currently in consulting. This is a direct
consequence of the `consulting_only_flag` penalty and the domain judgment that
the target role prefers product-company experience. It is not a protected-class
bias, but it is a strong structural filter that should be disclosed.

**Behavioral signals are high in the top 100.** Mean recruiter response rate is
0.683 and mean interview completion rate is 0.800. These features are positively
weighted, so the top of the ranking naturally selects candidates who are more
responsive and more likely to complete an interview process.

## Limitations of this audit

- This is a synthetic dataset. Real-world hiring systems would need audit
  against actual observed hiring outcomes and legal frameworks.
- We cannot audit protected attributes because the schema does not include them
  and inferring them from names is unreliable and itself introduces bias.
- Concentration is not always a bad thing. If the JD explicitly asks for
  candidates from Pune or Noida, high concentration in those cities is aligned,
  not biased.
- The audit compares the top 100 to the full pool, not to the qualified
  applicant pool. A fairer comparison in production would be against all
  candidates who meet the minimum posted requirements.

## Compliance framing

A production version of this system for real hiring in the US, EU, or India
would require:

- Documented fairness metrics tracked over time (Airbnb pattern).
- Human-in-the-loop review of every model-generated shortlist.
- Compliance with NYC Local Law 144 (bias audits for automated hiring tools).
- Compliance with EU AI Act (high-risk system classification for hiring tools).
- Compliance with EEOC guidance on algorithmic bias in hiring.

We frame this submission as a hackathon prototype, not a production hiring
system. Deployment would require the additional work above.
