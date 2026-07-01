"""Batch duplicate-candidate detector using deterministic fingerprints."""
import hashlib
from collections import Counter
from typing import Dict, Iterator, List, Set, Tuple


def _fingerprint(candidate: Dict) -> str:
    """Deterministic fingerprint of title, company, YoE bucket, and sorted skills."""
    prof = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    title = str(prof.get("current_title", "")).strip().lower()
    company = str(prof.get("current_company", "")).strip().lower()
    yoe = prof.get("years_of_experience")
    # Bucket YoE to avoid over-granularity: integer years.
    yoe_bucket = int(yoe) if isinstance(yoe, (int, float)) and yoe >= 0 else "unknown"

    skills = candidate.get("skills", []) if isinstance(candidate, dict) else []
    skill_names = sorted(
        {str(s.get("name", "")).strip().lower() for s in skills if s.get("name")}
    )

    raw = f"{title}|{company}|{yoe_bucket}|{','.join(skill_names)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def find_duplicate_fingerprints(candidates_iter: Iterator[Dict]) -> Set[str]:
    """Return candidate_ids whose deterministic fingerprint appears more than once.

    Only (candidate_id, fingerprint) tuples are materialized, not full candidates.
    """
    pairs: List[Tuple[str, str]] = []
    for c in candidates_iter:
        cid = c.get("candidate_id")
        if cid is None:
            continue
        pairs.append((cid, _fingerprint(c)))

    counts = Counter(fp for _, fp in pairs)
    duplicate_fingerprints = {fp for fp, cnt in counts.items() if cnt > 1}
    return {cid for cid, fp in pairs if fp in duplicate_fingerprints}
