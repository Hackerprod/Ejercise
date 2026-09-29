"""Local/container-independent verification of frozen OMEGA R1 source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_COMMIT = "2901e58834bc0d8d8225ccaf51ca136901468698"
R1_IDENTITY = "269a4d79b5a1e6df8c230962e2b8c9e237095f18"
FROZEN_DOCKERFILE = (
    "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_execution_readiness/"
    "Dockerfile.frozen.6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432"
)
EXPECTED_HASHES = {
    "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py": "b4e93b231f8c85e77a4f3a4c1ba726b54c9325cfac145d856d862ed8b71d65cd",
    "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/requirements.lock": "38de8910f95cf6894d79e934ebb739a884392c37bc3af5930d59df8b0132d501",
    "t1_trainability_lab_v0.1.0/t1_trainability/model.py": "bc0250593dd7db03cc185f140c9603a3e63b70afb565bc548ceba21a9eb075a6",
    FROZEN_DOCKERFILE: "6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_expected_hashes(root: Path, expected: dict[str, str] = EXPECTED_HASHES) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        actual_hash = sha256(path) if path.is_file() else None
        results[relative] = {
            "expected": expected_hash,
            "actual": actual_hash,
            "matches": actual_hash == expected_hash,
        }
    return results


def provenance_report(root: Path) -> dict[str, Any]:
    hashes = verify_expected_hashes(root)
    return {
        "source_commit": SOURCE_COMMIT,
        "R1_identity": R1_IDENTITY,
        "hashes": hashes,
        "verified": all(item["matches"] for item in hashes.values()),
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = provenance_report(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
