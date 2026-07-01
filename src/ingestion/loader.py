"""Candidate loading API — streaming preferred."""
from typing import Iterator, List, Dict, Any

import orjson

from src.utils.paths import CANDIDATES_JSONL, SAMPLE_CANDIDATES_JSON
from src.utils.io import iter_jsonl


def iter_candidates() -> Iterator[Dict[str, Any]]:
    """Stream all candidates lazily from candidates.jsonl. Use this by default."""
    yield from iter_jsonl(CANDIDATES_JSONL)


def load_candidates_sample(n: int = 50) -> List[Dict[str, Any]]:
    """Load the small, committed sample_candidates.json file into memory."""
    data = orjson.loads(SAMPLE_CANDIDATES_JSON.read_bytes())
    if isinstance(data, list):
        return data[:n]
    return [data]
