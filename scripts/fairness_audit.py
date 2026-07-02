"""Fairness and concentration audit of the frozen top-100 submission.

We do NOT audit protected attributes (caste, religion, gender, marital status,
disability, ethnicity). We DO audit structural properties that could indicate
over-selection on one dimension: company concentration, geographic
concentration, and experience distribution.
"""
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.ingestion.loader import iter_candidates

ROOT = Path(__file__).resolve().parents[1]
FINAL_SUB = ROOT / "outputs" / "final_submission.csv"

CONSULTING_FIRMS = {
    "infosys",
    "tcs",
    "tata consultancy",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mphasis",
    "ibm",
    "ey",
    "pwc",
    "kpmg",
    "deloitte",
    "l&t infotech",
    "lti",
    "mindtree",
    "ltimindtree",
    "persistent",
    "hexaware",
    "cyient",
    "coforge",
}


def _is_consulting(company: str) -> bool:
    if not company:
        return False
    c = company.lower()
    return any(firm in c for firm in CONSULTING_FIRMS)


def _collect_pool_stats() -> dict:
    """Collect summary stats over the full candidate pool for context."""
    companies = []
    locations = []
    countries = []
    yoes = []
    consulting_current = 0
    total = 0

    for c in iter_candidates():
        total += 1
        p = c["profile"]
        companies.append(p.get("current_company", "unknown"))
        locations.append(p.get("location", "unknown"))
        countries.append(p.get("country", "unknown"))
        try:
            yoes.append(float(p.get("years_of_experience", 0)))
        except (TypeError, ValueError):
            yoes.append(0.0)
        if _is_consulting(p.get("current_company", "")):
            consulting_current += 1

    return {
        "total": total,
        "company_counts": Counter(companies),
        "location_counts": Counter(locations),
        "country_counts": Counter(countries),
        "yoe_mean": statistics.mean(yoes) if yoes else 0.0,
        "yoe_std": statistics.stdev(yoes) if len(yoes) > 1 else 0.0,
        "consulting_current_pct": 100.0 * consulting_current / total if total else 0.0,
    }


def main() -> None:
    sub = pd.read_csv(FINAL_SUB)
    top100_ids = set(sub["candidate_id"].tolist())

    print("Loading top-100 candidates from JSONL...")
    top100 = []
    for c in iter_candidates():
        if c["candidate_id"] in top100_ids:
            top100.append(c)
            if len(top100) == 100:
                break
    assert len(top100) == 100, f"Expected 100, got {len(top100)}"

    print("Collecting full-pool context stats...")
    pool = _collect_pool_stats()

    profiles = [c["profile"] for c in top100]

    # 1. Company concentration
    companies = [p.get("current_company", "unknown") for p in profiles]
    company_counts = Counter(companies)
    top5_companies = company_counts.most_common(5)
    top5_share = sum(count for _, count in top5_companies)

    # 2. City / location concentration
    locations = [p.get("location", "unknown") for p in profiles]
    city_counts = Counter(locations)
    top3_cities = city_counts.most_common(3)
    top3_city_share = sum(count for _, count in top3_cities)

    # 3. Country
    countries = [p.get("country", "unknown") for p in profiles]
    country_counts = Counter(countries)
    india_count = sum(1 for c in countries if c and c.lower() == "india")

    # 4. YoE distribution
    yoes = [float(p.get("years_of_experience", 0)) for p in profiles]
    yoe_mean = sum(yoes) / len(yoes)
    yoe_min, yoe_max = min(yoes), max(yoes)
    yoe_in_band = sum(1 for y in yoes if 5.0 <= y <= 9.0)
    yoe_std = statistics.stdev(yoes)

    # 5. Notice period distribution
    notice_periods = [
        c.get("redrob_signals", {}).get("notice_period_days", 0) for c in top100
    ]
    notice_bins = Counter()
    for n in notice_periods:
        if n <= 30:
            notice_bins["<= 30 days"] += 1
        elif n <= 60:
            notice_bins["31-60 days"] += 1
        elif n <= 90:
            notice_bins["61-90 days"] += 1
        else:
            notice_bins["> 90 days"] += 1

    # 6. Product vs consulting
    consulting_count = sum(
        1 for p in profiles if _is_consulting(p.get("current_company", ""))
    )
    consulting_history = sum(
        1
        for c in top100
        if any(
            _is_consulting(j.get("company", "")) for j in c.get("career_history", [])
        )
    )

    # 7. Response rate and interview completion
    resp_rates = [
        c.get("redrob_signals", {}).get("recruiter_response_rate", 0.0)
        for c in top100
    ]
    resp_mean = sum(resp_rates) / len(resp_rates)
    int_rates = [
        c.get("redrob_signals", {}).get("interview_completion_rate", 0.0)
        for c in top100
    ]
    int_mean = sum(int_rates) / len(int_rates)

    # Print
    print("\n" + "=" * 72)
    print("FAIRNESS AND CONCENTRATION AUDIT — TOP 100")
    print("=" * 72)

    print(f"\n1. Company concentration")
    print(f"   Unique companies in top 100: {len(company_counts)}")
    print(f"   Top-5 companies share: {top5_share}/100 = {top5_share}%")
    for company, count in top5_companies:
        print(f"     - {company:<40} {count:>3}")
    print(f"   Pool context: {len(pool['company_counts'])} unique companies across {pool['total']} candidates")

    print(f"\n2. Location concentration")
    print(f"   Unique locations in top 100: {len(city_counts)}")
    print(f"   Top-3 locations share: {top3_city_share}/100 = {top3_city_share}%")
    for city, count in top3_cities:
        print(f"     - {city:<40} {count:>3}")
    print(f"   Pool context: {len(pool['location_counts'])} unique locations across {pool['total']} candidates")

    print(f"\n3. Country distribution")
    print(f"   India: {india_count}/100 = {india_count}%")
    for country, count in country_counts.most_common(5):
        print(f"     - {country:<40} {count:>3}")
    india_pool_pct = 100.0 * pool["country_counts"].get("India", 0) / pool["total"]
    print(f"   Pool context: India = {india_pool_pct:.1f}% of full pool")

    print(f"\n4. Years of experience")
    print(f"   Mean: {yoe_mean:.2f}    Std: {yoe_std:.2f}    Min: {yoe_min:.1f}    Max: {yoe_max:.1f}")
    print(f"   Inside JD ideal band [5,9]: {yoe_in_band}/100 = {yoe_in_band}%")
    print(f"   Pool context: mean={pool['yoe_mean']:.2f}, std={pool['yoe_std']:.2f}")

    print(f"\n5. Notice period distribution")
    for band in ["<= 30 days", "31-60 days", "61-90 days", "> 90 days"]:
        print(f"   {band:<15} {notice_bins.get(band, 0):>3}")

    print(f"\n6. Product vs consulting employment")
    print(f"   Currently at consulting firm : {consulting_count}/100")
    print(f"   Has consulting history       : {consulting_history}/100")
    print(f"   Pool context: currently consulting = {pool['consulting_current_pct']:.1f}%")

    print(f"\n7. Behavioral signal averages in top 100")
    print(f"   Mean recruiter_response_rate : {resp_mean:.3f}")
    print(f"   Mean interview_completion    : {int_mean:.3f}")

    print("\n" + "=" * 72)
    print("PROTECTED ATTRIBUTES NOT AUDITED")
    print("=" * 72)
    print("The following were deliberately NOT used in ranking or in this audit:")
    print("  - candidate name (any gender/ethnicity/caste inference)")
    print("  - religion / marital status / disability")
    print("  - age beyond the JD's stated experience range (5-9 years)")
    print("\nThis is a hackathon synthetic-data prototype, not a production hiring")
    print("system. Any production deployment would require additional bias auditing,")
    print("human-in-the-loop review, and legal compliance (NYC Local Law 144,")
    print("EU AI Act, EEOC guidance).")


if __name__ == "__main__":
    main()
