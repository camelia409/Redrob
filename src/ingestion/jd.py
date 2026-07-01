"""Extract plain text of the JD .docx to configs/jd.txt."""
from docx import Document

from src.utils.paths import JOB_DESCRIPTION_DOCX, JD_TXT


def extract_jd_text() -> str:
    """Read job_description.docx and cache the plain text to configs/jd.txt."""
    doc = Document(JOB_DESCRIPTION_DOCX)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(lines)
    JD_TXT.parent.mkdir(parents=True, exist_ok=True)
    JD_TXT.write_text(text, encoding="utf-8")
    return text


def get_jd_text() -> str:
    """Return cached JD text, extracting it first if necessary."""
    if not JD_TXT.exists():
        return extract_jd_text()
    return JD_TXT.read_text(encoding="utf-8")
