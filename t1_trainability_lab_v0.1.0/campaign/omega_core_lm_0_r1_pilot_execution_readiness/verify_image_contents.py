"""Verify frozen OMEGA R1 artifacts from inside a built image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SOURCE_COMMIT = "2901e58834bc0d8d8225ccaf51ca136901468698"
R1_IDENTITY = "269a4d79b5a1e6df8c230962e2b8c9e237095f18"
IMAGE_METADATA_PATH = "/opt/omega/readiness_report.json"
IMAGE_EXPECTED_HASHES = {
    "/opt/omega/omega_nominal_microbatch_runner.py": "b4e93b231f8c85e77a4f3a4c1ba726b54c9325cfac145d856d862ed8b71d65cd",
    "/opt/omega/requirements.lock": "38de8910f95cf6894d79e934ebb739a884392c37bc3af5930d59df8b0132d501",
    "/opt/omega/t1_trainability/model.py": "bc0250593dd7db03cc185f140c9603a3e63b70afb565bc548ceba21a9eb075a6",
    "/opt/omega/Dockerfile.frozen.6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432": "6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rooted(root: Path, image_path: str) -> Path:
    return root / image_path.lstrip("/")


def verify_image_root(root: Path = Path("/")) -> dict[str, Any]:
    """Verify paths that exist in the image, without inspecting the host checkout."""
    hashes: dict[str, dict[str, Any]] = {}
    for image_path, expected_hash in IMAGE_EXPECTED_HASHES.items():
        path = _rooted(root, image_path)
        actual_hash = sha256(path) if path.is_file() else None
        hashes[image_path] = {
            "expected": expected_hash,
            "actual": actual_hash,
            "matches": actual_hash == expected_hash,
        }
    metadata_path = _rooted(root, IMAGE_METADATA_PATH)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    source_commit = metadata.get("source_commit")
    r1_identity = metadata.get("R1_identity")
    identity_matches = source_commit == SOURCE_COMMIT and r1_identity == R1_IDENTITY
    return {
        "source_commit": source_commit,
        "R1_identity": r1_identity,
        "expected_source_commit": SOURCE_COMMIT,
        "expected_R1_identity": R1_IDENTITY,
        "identity_matches": identity_matches,
        "hashes": hashes,
        "verified": identity_matches and all(item["matches"] for item in hashes.values()),
    }


def docker_command(image: str) -> list[str]:
    """Run verifier with image entrypoint bypassed; image boot entrypoint starts sshd."""
    return [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        image,
        "/opt/omega/verify_image_contents.py",
        "--in-image",
    ]


def verify_image(image: str) -> int:
    try:
        completed = subprocess.run(
            docker_command(image),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        print(json.dumps({"verified": False, "error": str(error)}, sort_keys=True))
        return 2

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode
    try:
        return 0 if json.loads(completed.stdout)["verified"] else 1
    except (json.JSONDecodeError, KeyError):
        return 1


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--in-image", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/"))
    args = parser.parse_args()
    if args.in_image:
        report = verify_image_root(args.root.resolve())
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["verified"] else 1
    if not args.image:
        parser.error("image name is required unless --in-image is set")
    return verify_image(args.image)


if __name__ == "__main__":
    raise SystemExit(_main())
