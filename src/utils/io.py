"""Streaming I/O helpers. Never load full JSONL into memory."""
import csv
from pathlib import Path
from typing import Iterator, Dict, Any, List

import orjson


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield decoded JSON objects one at a time from a JSONL file."""
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)


def read_csv_if_exists(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV file into a list of dicts; return empty list if missing."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
