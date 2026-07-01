"""Verify every staged challenge file matches its SHA-256 fingerprint."""
import hashlib
from pathlib import Path
from typing import Dict

from src.utils.paths import CHALLENGE, CHECKSUMS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expected() -> Dict[str, str]:
    """Parse CHECKSUMS.sha256 into {filename: hex_digest}."""
    expected: Dict[str, str] = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("*").strip()] = digest
    return expected


def verify_data_integrity() -> Dict[str, bool]:
    """Check every file listed in CHECKSUMS.sha256; raise on mismatch or missing file."""
    expected = load_expected()
    results: Dict[str, bool] = {}
    for name, digest in expected.items():
        path = CHALLENGE / name
        if not path.exists():
            raise FileNotFoundError(f"Expected staged file missing: {name}")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"Checksum mismatch for {name}\n"
                f"  expected {digest}\n"
                f"  got      {actual}"
            )
        results[name] = True
    return results
