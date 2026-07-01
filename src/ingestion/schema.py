"""Compare declared schema vs actual candidate records."""
import random
from typing import Dict, Any, List, Set

import orjson

from src.utils.paths import CANDIDATE_SCHEMA_JSON
from src.ingestion.loader import iter_candidates


def load_schema() -> Dict[str, Any]:
    """Load candidate_schema.json."""
    return orjson.loads(CANDIDATE_SCHEMA_JSON.read_bytes())


def _schema_top_keys(schema: Dict[str, Any]) -> Set[str]:
    """Return the set of top-level property names declared in the schema."""
    props = schema.get("properties", {})
    return set(props.keys())


def audit_schema_drift(sample_size: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Stream candidates, reservoir-sample `sample_size`, and compare keys to schema."""
    schema_keys = _schema_top_keys(load_schema())
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []

    for i, candidate in enumerate(iter_candidates()):
        if i < sample_size:
            reservoir.append(candidate)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = candidate

    missing: Dict[str, int] = {k: 0 for k in schema_keys}
    extras: Dict[str, int] = {}

    for candidate in reservoir:
        candidate_keys = set(candidate.keys())
        for k in schema_keys - candidate_keys:
            missing[k] += 1
        for k in candidate_keys - schema_keys:
            extras[k] = extras.get(k, 0) + 1

    return {
        "n_sampled": len(reservoir),
        "schema_top_keys": sorted(schema_keys),
        "keys_missing_in_records": {k: v for k, v in missing.items() if v > 0},
        "keys_extra_in_records": extras,
    }
