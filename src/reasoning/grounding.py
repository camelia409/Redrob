"""Zero-hallucination guard for generated reasoning.

Every concrete fact cited in the reasoning text must be present in the
candidate JSON. The guard raises ``HallucinationError`` on any mismatch.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from src.utils.paths import CONFIGS


class HallucinationError(AssertionError):
    """Raised when generated reasoning cites a fact not found in candidate JSON."""


_RUBRIC_PATH = CONFIGS / "rubric_v1.yaml"
_FEATURES_PATH = CONFIGS / "features.yaml"


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_rubric = _load_yaml(_RUBRIC_PATH)
_features = _load_yaml(_FEATURES_PATH)

_KNOWN_SKILLS: Set[str] = set()
for key in ("ml_skills_strict", "ml_skills_broad"):
    _KNOWN_SKILLS.update({str(s).lower() for s in _rubric.get(key, [])})

_KNOWN_COMPANIES: Set[str] = set()
_KNOWN_COMPANIES.update({str(c).lower() for c in _features.get("product_companies", [])})
_KNOWN_COMPANIES.update({str(c).lower() for c in _features.get("consulting_firms", [])})

_KNOWN_CITIES: Set[str] = set()
_KNOWN_CITIES.update({str(c).lower() for c in _features.get("preferred_cities", [])})
_KNOWN_CITIES.update({str(c).lower() for c in _features.get("tier1_indian_cities", [])})


# ---------------------------------------------------------------------------
# Candidate fact extractors
# ---------------------------------------------------------------------------


def _all_skill_names(candidate: Dict) -> Set[str]:
    return {
        str(skill.get("name", "")).strip().lower()
        for skill in candidate.get("skills", [])
        if skill.get("name")
    }


def _all_company_names(candidate: Dict) -> Set[str]:
    profile = candidate.get("profile", {})
    companies = {str(profile.get("current_company", "")).strip().lower()}
    for job in candidate.get("career_history", []):
        company = str(job.get("company", "")).strip().lower()
        if company:
            companies.add(company)
    return companies


def _all_locations(candidate: Dict) -> Set[str]:
    profile = candidate.get("profile", {})
    raw = {
        str(profile.get("location", "")).strip().lower(),
        str(profile.get("country", "")).strip().lower(),
    }
    # Also include individual city tokens (e.g. "chennai" from "chennai, tamil nadu").
    locations: Set[str] = set()
    for item in raw:
        if item:
            locations.add(item)
            for token in re.split(r"[,/\-]+", item):
                token = token.strip()
                if token:
                    locations.add(token)
    return locations


def _all_numeric_values(candidate: Dict) -> Set[float]:
    """Collect every numeric value appearing anywhere in the candidate JSON."""
    values: Set[float] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, bool):
            pass
        elif isinstance(obj, (int, float)):
            values.add(float(obj))

    walk(candidate)
    return values


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _has_word(text_lower: str, phrase_lower: str) -> bool:
    """Case-insensitive whole-word (or whole-phrase) substring check."""
    pattern = r"(?<!\w)" + re.escape(phrase_lower) + r"(?!\w)"
    return re.search(pattern, text_lower) is not None


def _check_skills(text_lower: str, candidate: Dict) -> None:
    allowed = _all_skill_names(candidate)

    # Titles are legitimate places for skill-like phrases (e.g. "Machine Learning
    # Engineer"), so a known skill appearing in a title should not be flagged.
    titles = {
        str(candidate.get("profile", {}).get("current_title", "")).strip().lower()
    }
    for job in candidate.get("career_history", []):
        title = str(job.get("title", "")).strip().lower()
        if title:
            titles.add(title)

    def _is_grounded_skill(skill: str) -> bool:
        """A skill is grounded if the candidate claims it explicitly, as part
        of a longer skill name, or in a job/current title."""
        if skill in allowed:
            return True
        if any(skill in allowed_skill for allowed_skill in allowed):
            return True
        return any(skill in title for title in titles)

    for skill in _KNOWN_SKILLS:
        if skill in text_lower and not _is_grounded_skill(skill):
            # Use whole-word check to avoid substring false positives.
            if _has_word(text_lower, skill):
                raise HallucinationError(
                    f"Hallucinated skill: '{skill}' not found in candidate skills or titles"
                )


def _check_companies(text_lower: str, candidate: Dict) -> None:
    allowed = _all_company_names(candidate)
    for company in _KNOWN_COMPANIES:
        if company in text_lower and company not in allowed:
            if _has_word(text_lower, company):
                raise HallucinationError(
                    f"Hallucinated company: '{company}' not found in candidate profile"
                )


def _check_locations(text_lower: str, candidate: Dict) -> None:
    allowed = _all_locations(candidate)
    for city in _KNOWN_CITIES:
        if city in text_lower and city not in allowed:
            if _has_word(text_lower, city):
                raise HallucinationError(
                    f"Hallucinated location: '{city}' not found in candidate profile"
                )


def _check_numbers(text_lower: str, candidate: Dict, tolerance: float = 0.11) -> None:
    allowed = _all_numeric_values(candidate)
    # Find standalone numbers, not substrings of tokens like "bm25".
    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", text_lower):
        value = float(match.group(1))
        # Zero is common and often benign (e.g., "0 endorsements"); still must be
        # grounded. It is almost always present somewhere in numeric fields.
        if not any(abs(value - allowed_val) <= tolerance for allowed_val in allowed):
            raise HallucinationError(
                f"Hallucinated number: {value} not found in candidate numeric fields"
            )


def assert_grounded(candidate: Dict, reasoning: str) -> None:
    """Raise HallucinationError if ``reasoning`` cites any fact not in candidate JSON."""
    if not reasoning or not isinstance(reasoning, str):
        raise HallucinationError("Reasoning is empty or not a string")

    text_lower = reasoning.lower()

    _check_skills(text_lower, candidate)
    _check_companies(text_lower, candidate)
    _check_locations(text_lower, candidate)
    _check_numbers(text_lower, candidate)
