"""Central path registry. Import from here everywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # /e/redrob-hackathon/final
DATA = ROOT / "data"
CHALLENGE = DATA / "challenge"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
SILVER = DATA / "silver"
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"

CANDIDATES_JSONL = CHALLENGE / "candidates.jsonl"
CANDIDATE_SCHEMA_JSON = CHALLENGE / "candidate_schema.json"
JOB_DESCRIPTION_DOCX = CHALLENGE / "job_description.docx"
SIGNALS_DOC_DOCX = CHALLENGE / "redrob_signals_doc.docx"
SUBMISSION_SPEC_DOCX = CHALLENGE / "submission_spec.docx"
SAMPLE_CANDIDATES_JSON = CHALLENGE / "sample_candidates.json"
SAMPLE_SUBMISSION_CSV = CHALLENGE / "sample_submission.csv"
CHECKSUMS = CHALLENGE / "CHECKSUMS.sha256"

JD_TXT = CONFIGS / "jd.txt"
