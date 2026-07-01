"""Streaming I/O helpers. Never load full JSONL into memory."""
from pathlib import Path
from typing import Iterator, Dict, Any

import orjson


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield decoded JSON objects one at a time from a JSONL file."""
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)
