import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import EXPECTED_HASHES, FROZEN_DOCKERFILE, provenance_report, verify_expected_hashes


ROOT = Path(__file__).resolve().parents[3]


def test_frozen_source_hashes_and_identity_verify() -> None:
    report = provenance_report(ROOT)
    assert report["verified"] is True
    assert report["source_commit"] == "2901e58834bc0d8d8225ccaf51ca136901468698"
    assert report["R1_identity"] == "269a4d79b5a1e6df8c230962e2b8c9e237095f18"
    assert set(EXPECTED_HASHES) == {
        "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py",
        "t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/requirements.lock",
        "t1_trainability_lab_v0.1.0/t1_trainability/model.py",
        FROZEN_DOCKERFILE,
    }
    assert all(item["matches"] for item in report["hashes"].values())


def test_hash_mismatch_is_reported_without_mutating_files() -> None:
    expected = dict(EXPECTED_HASHES)
    first = next(iter(expected))
    expected[first] = "0" * 64
    results = verify_expected_hashes(ROOT, expected)
    assert results[first]["matches"] is False
    assert results[first]["actual"]
