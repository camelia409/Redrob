"""CLI inspector: counts, samples, schema drift, JD length, integrity.

Run from repo root:
    python -m scripts.inspect_data
"""
import time

from src.utils.integrity import verify_data_integrity
from src.ingestion.loader import iter_candidates
from src.ingestion.schema import audit_schema_drift
from src.ingestion.jd import get_jd_text


def main() -> None:
    t0 = time.time()

    print("[1/4] Verifying data integrity...")
    verify_data_integrity()
    print("      OK — all checksums match.")

    print("[2/4] Streaming candidate count + first 5 titles...")
    total = 0
    first5_titles = []
    for i, candidate in enumerate(iter_candidates()):
        total += 1
        if i < 5:
            first5_titles.append(candidate.get("profile", {}).get("current_title", "?"))
    print(f"      total candidates = {total}")
    print(f"      first 5 titles   = {first5_titles}")

    print("[3/4] Schema drift audit (100 sampled)...")
    report = audit_schema_drift(sample_size=100)
    print(f"      sampled: {report['n_sampled']}")
    print(f"      schema top-level keys: {report['schema_top_keys']}")
    if report["keys_missing_in_records"]:
        print(f"      keys missing in records: {report['keys_missing_in_records']}")
    else:
        print("      no missing top-level keys.")
    if report["keys_extra_in_records"]:
        print(f"      keys extra in records:   {report['keys_extra_in_records']}")
    else:
        print("      no extra top-level keys.")

    print("[4/4] JD extraction...")
    jd = get_jd_text()
    print(f"      JD length = {len(jd)} chars, ~{len(jd.split())} words")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
