import shutil
import types
from pathlib import Path

import pytest

from src.ingestion.loader import iter_candidates, load_candidates_sample
from src.ingestion.schema import audit_schema_drift
from src.ingestion.jd import get_jd_text, extract_jd_text
from src.utils.integrity import verify_data_integrity
from src.utils.paths import CHALLENGE, JD_TXT


def test_iter_candidates_streams_lazily():
    gen = iter_candidates()
    assert isinstance(gen, types.GeneratorType)
    first_10 = [next(gen) for _ in range(10)]
    assert len(first_10) == 10
    assert "candidate_id" in first_10[0]


def test_load_candidates_sample_default():
    sample = load_candidates_sample()
    assert isinstance(sample, list)
    assert len(sample) == 50
    assert all("candidate_id" in c for c in sample)


def test_schema_no_drift_on_sample():
    report = audit_schema_drift(sample_size=100)
    assert report["n_sampled"] == 100
    assert not report["keys_missing_in_records"], report


def test_jd_extraction_nonempty(tmp_path):
    if JD_TXT.exists():
        JD_TXT.unlink()
    text = extract_jd_text()
    assert len(text) > 500
    assert JD_TXT.exists()


def test_integrity_guard_catches_tamper(tmp_path):
    victim = CHALLENGE / "sample_submission.csv"
    backup = tmp_path / "backup.csv"
    shutil.copy2(victim, backup)
    try:
        with open(victim, "ab") as f:
            f.write(b"X")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            verify_data_integrity()
    finally:
        shutil.copy2(backup, victim)
    # After restoration, integrity should pass again.
    verify_data_integrity()
