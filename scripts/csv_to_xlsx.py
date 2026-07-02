"""Convert outputs/final_submission.csv to outputs/final_submission.xlsx.

Portal requires XLSX format. Challenge validator uses CSV. Both are kept.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "final_submission.csv"
XLSX = ROOT / "outputs" / "final_submission.xlsx"


def main() -> None:
    df = pd.read_csv(CSV)
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert list(df.columns) == [
        "candidate_id",
        "rank",
        "score",
        "reasoning",
    ], f"Unexpected columns: {list(df.columns)}"

    df.to_excel(XLSX, index=False, engine="openpyxl")
    print(f"Wrote {XLSX} ({len(df)} rows)")

    # Round-trip verify
    back = pd.read_excel(XLSX, engine="openpyxl")
    assert (
        back["candidate_id"].tolist() == df["candidate_id"].tolist()
    ), "candidate_id mismatch"
    assert back["rank"].tolist() == df["rank"].tolist(), "rank mismatch"
    assert len(back) == 100, "row count mismatch on round-trip"
    print("Round-trip verified: 100 rows, all candidate_ids and ranks match")

    # Byte size sanity
    size_kb = XLSX.stat().st_size / 1024
    print(f"File size: {size_kb:.1f} KB (portal limit 5 MB)")


if __name__ == "__main__":
    main()
