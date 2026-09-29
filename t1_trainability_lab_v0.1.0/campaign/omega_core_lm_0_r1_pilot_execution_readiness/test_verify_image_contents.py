import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import EXPECTED_HASHES, FROZEN_DOCKERFILE
from verify_image_contents import (
    IMAGE_EXPECTED_HASHES,
    R1_IDENTITY,
    SOURCE_COMMIT,
    docker_command,
    verify_image_root,
)


ROOT = Path(__file__).resolve().parents[3]


def test_image_verifier_checks_expected_paths_and_identity(tmp_path: Path) -> None:
    source_paths = {
        "/opt/omega/omega_nominal_microbatch_runner.py": ROOT / next(
            path for path in EXPECTED_HASHES if path.endswith("omega_nominal_microbatch_runner.py")
        ),
        "/opt/omega/requirements.lock": ROOT / next(
            path for path in EXPECTED_HASHES if path.endswith("requirements.lock")
        ),
        "/opt/omega/t1_trainability/model.py": ROOT / next(
            path for path in EXPECTED_HASHES if path.endswith("t1_trainability/model.py")
        ),
        "/opt/omega/" + Path(FROZEN_DOCKERFILE).name: ROOT / FROZEN_DOCKERFILE,
    }
    for image_path, source_path in source_paths.items():
        target = tmp_path / image_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    metadata_target = tmp_path / "opt" / "omega" / "readiness_report.json"
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_execution_readiness/readiness_report.json",
        metadata_target,
    )

    report = verify_image_root(tmp_path)
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["R1_identity"] == R1_IDENTITY
    assert report["identity_matches"] is True
    assert report["verified"] is True


def test_docker_command_bypasses_ssh_entrypoint() -> None:
    assert docker_command("example/image@sha256:abc") == [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "example/image@sha256:abc",
        "/opt/omega/verify_image_contents.py",
        "--in-image",
    ]


def test_image_hashes_are_frozen_values() -> None:
    assert IMAGE_EXPECTED_HASHES["/opt/omega/Dockerfile.frozen." + "6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432"] == "6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432"
