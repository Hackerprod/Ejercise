"""Regenerate final-closure self-hash from the exact bytes written to disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PLACEHOLDER = "__SELF_HASH__"
ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "campaign"
    / "t2_nobypass2_final_closure"
    / "results.json"
)


def replace_hash_value(data: bytes, value: str) -> bytes:
    needle = f'"artifact_self_hash": "{value}"'.encode("utf-8")
    replacement = f'"artifact_self_hash": "{PLACEHOLDER}"'.encode("utf-8")
    if data.count(needle) != 1:
        raise ValueError("expected exactly one artifact_self_hash field")
    return data.replace(needle, replacement, 1)


def main() -> None:
    data = ARTIFACT.read_bytes()
    current = json.loads(data.decode("utf-8"))["artifact_self_hash"]
    unsigned = replace_hash_value(data, current)
    digest = hashlib.sha256(unsigned).hexdigest()
    final = data.replace(
        f'"artifact_self_hash": "{current}"'.encode("utf-8"),
        f'"artifact_self_hash": "{digest}"'.encode("utf-8"),
        1,
    )
    ARTIFACT.write_bytes(final)

    written = ARTIFACT.read_bytes()
    written_hash = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    verified = hashlib.sha256(replace_hash_value(written, written_hash)).hexdigest()
    if verified != written_hash:
        raise RuntimeError(f"self-hash verification failed: {verified} != {written_hash}")
    print(json.dumps({"artifact": str(ARTIFACT), "artifact_self_hash": written_hash, "verified": True}))


if __name__ == "__main__":
    main()
