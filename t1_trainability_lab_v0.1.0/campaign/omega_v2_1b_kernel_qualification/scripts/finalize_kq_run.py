"""Analysis-only finalizer for an already completed immutable candidate KQ run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
CAMPAIGN_ROOT = UNIT_ROOT.parent
V2_1_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
V2_0_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
RESULTS_ROOT = UNIT_ROOT / "results" / "omega_v2_1b_kernel_qualification" / "candidate_01" / "run_02"
ATTEMPT02_ROOT = V2_1_ROOT / "results" / "omega_v2_1_physical" / "attempt_02"
V2_0_SEAL = V2_0_ROOT / "V2_0_RESULT_SEAL.json"
sys.path.insert(0, str(UNIT_ROOT))
from scripts import run_kq  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def sealed_source_paths(build_manifest: dict[str, Any]) -> list[Path]:
    return [UNIT_ROOT / name for name in build_manifest["source_sha256"]]


def reconstruct_v2_0_weight_manifest(expected_state_hash: str) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(V2_0_ROOT))
    from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed = 20260929
    block = ContractualCoreBlock(512, dtype=torch.float32, seed=seed)
    value_digest = hashlib.sha256()
    raw_parts = []
    matrix_shapes = []
    for name in MATRIX_FAMILIES:
        tensor = getattr(block, name).detach().contiguous().cpu()
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        value_digest.update(name.encode("utf-8")); value_digest.update(b"\0"); value_digest.update(raw)
        raw_parts.append(raw)
        matrix_shapes.append({"name": name, "shape": list(tensor.shape)})
    value_hash = value_digest.hexdigest()
    if value_hash != expected_state_hash:
        raise RuntimeError("analysis-only V2-0 weight reconstruction does not match measured KQ source-weight hash")
    return {
        "source": "V2-0 omega_v2.core.ContractualCoreBlock FP32 source tensors, pre-Q4",
        "source_path_abs": str((V2_0_ROOT / "omega_v2" / "core.py").resolve()),
        "seed": seed,
        "d": 512,
        "weight_state_sha256": value_hash,
        "weight_stream_sha256": hashlib.sha256(b"".join(raw_parts)).hexdigest(),
        "weight_stream_bytes": sum(len(part) for part in raw_parts),
        "matrix_families": matrix_shapes,
        "reconstructed_after_timing_for_provenance_only": True,
    }


def main() -> int:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"completed candidate_01 KQ data is missing: {RESULTS_ROOT}")
    immutable_native = RESULTS_ROOT / "native_kq_candidate_01.json"
    immutable_pytorch = RESULTS_ROOT / "pytorch_control.json"
    if not immutable_native.is_file() or not immutable_pytorch.is_file():
        raise RuntimeError("KQ result cannot be finalized: native or PyTorch measurement artifact is missing")
    if (RESULTS_ROOT / "OMEGA_V2_1B_KQ_REPORT.md").exists():
        raise FileExistsError("KQ result report already exists and is immutable")

    build_manifest = json.loads((RESULTS_ROOT / "build_manifest.json").read_text(encoding="utf-8"))
    native = json.loads(immutable_native.read_text(encoding="utf-8"))
    pytorch = json.loads(immutable_pytorch.read_text(encoding="utf-8"))
    provenance = build_manifest["git_provenance"]
    measured_source_hashes = build_manifest["source_sha256"]
    changed_measured_sources = [
        name for name, expected in measured_source_hashes.items()
        if sha256_file(UNIT_ROOT / name) != expected
    ]
    if changed_measured_sources:
        raise RuntimeError(f"KQ_STOP: measured source files changed after the candidate run: {changed_measured_sources}")
    executable = Path(build_manifest["executable_absolute_path"])
    if sha256_file(executable) != build_manifest["executable_sha256"]:
        raise RuntimeError("KQ_STOP: measured KQ executable hash changed after candidate run")

    attempt02_before = run_kq.validate_attempt02_preservation()
    if attempt02_before["artifact_manifest_sha256"] != native["attempt02_manifest_sha256"]:
        raise RuntimeError("KQ_STOP: attempt_02 preservation manifest differs from native run provenance")
    attempt02_after = {**attempt02_before, "artifact_hashes_verified_after_kq": True}
    source_weight_info = reconstruct_v2_0_weight_manifest(native["source_weight_sha256"])

    e_q4 = float(native["efficiencies"]["E_Q4"])
    e_full4 = float(native["efficiencies"]["E_FULL_4"])
    e_full16 = float(native["efficiencies"]["E_FULL_16"])
    native_k4 = float(native["full_resident"]["m8_k4_for_s_native"]["median_seconds"])
    pytorch_k4 = float(pytorch["S_native_cells"]["d512_m8_K4"]["pytorch_fp32_original_median_seconds"])
    s_native = pytorch_k4 / native_k4
    correctness = (
        native["correctness"]["toy_d32_m4_k2"].get("pass") is True
        and native["correctness"]["full_d512_outputs_finite"] is True
        and native["correctness"]["full_checksum_stable"] is True
    )
    decision = run_kq.classify_candidate1(correctness=correctness, e_q4=e_q4, e_full4=e_full4, e_full16=e_full16, s_native=s_native)
    pytorch["S_native_cells"]["d512_m8_K4"].update({
        "native_median_seconds": native_k4,
        "S_native": s_native,
        "S_native_minimum": 1.20,
    })
    attempt02_m1 = run_kq.attempt02_m1_diagnostics()
    tests, conformance = run_kq.make_test_rows(
        native, pytorch, e_q4 >= .25, e_full4 >= .60, e_full16 >= .60, s_native >= 1.20,
        attempt02_before, attempt02_after, attempt02_m1, source_weight_info, "candidate_01",
    )
    if [row["name"] for row in tests] != run_kq.KQ_TEST_NAMES:
        raise RuntimeError("analysis finalizer test report does not match the frozen 17-test inventory")
    test_report = {
        "schema": "omega-v2-1b-kq-test-report-v1",
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "test_count": len(tests),
        "pass_count": sum(row["status"] == "PASS" for row in tests),
        "fail_count": sum(row["status"] == "FAIL" for row in tests),
        "skip_count": sum(row["status"] == "SKIP" for row in tests),
        "tests": tests,
    }

    metrics = {
        "correctness": correctness,
        "E_Q4": {"value": e_q4, "minimum": .25, "pass": e_q4 >= .25},
        "E_FULL_4": {"value": e_full4, "minimum": .60, "pass": e_full4 >= .60},
        "E_FULL_16": {"value": e_full16, "minimum": .60, "pass": e_full16 >= .60},
        "S_native": {"value": s_native, "minimum": 1.20, "pass": s_native >= 1.20},
        "scratch_per_worker_bytes": {"value": native["scratch_per_worker_bytes"], "maximum": 65536, "pass": native["scratch_per_worker_bytes"] <= 65536},
    }
    summary = {
        "schema": "omega-v2-1b-kq-summary-v1",
        "terminal_status": decision["terminal_status"],
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "source_provenance": provenance,
        "measurement_implementation_commit": build_manifest["implementation_commit"],
        "postmeasurement_analysis_commit": run_kq.git("rev-parse", "HEAD"),
        "measurement_source_manifest_sha256": build_manifest["source_sha256"],
        "source_weight_manifest": source_weight_info,
        "attempt02_preservation": attempt02_after,
        "native": native,
        "pytorch": pytorch,
        "m1_residency_diagnostic_no_gate": attempt02_m1,
        "gates": metrics,
        "candidate_state": {
            "max_candidates": 2,
            "candidate01_measured": True,
            "candidate02_measured": False,
            "candidate02_allowed": decision["candidate2_allowed"],
            "attempt03_executed": False,
            "attempt03_allowed_after_freeze": decision["all_five_gates_pass"],
        },
        "machine_balance_diagnostic": native["machine_balance_diagnostic"],
        "test_report": test_report,
        "global_status": "CONFORMANCE_HOLD",
        "gpu": "HOLD",
        "T3": "HOLD",
    }
    write_json(RESULTS_ROOT / "source_weight_manifest.json", source_weight_info)
    write_json(RESULTS_ROOT / "attempt02_preservation.json", attempt02_after)
    write_json(RESULTS_ROOT / "test_report.json", test_report)
    write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml", {"OMEGA_CONFORMANCE_BLOCK": conformance})
    write_json(RESULTS_ROOT / "summary_metrics.json", summary)
    if decision["all_five_gates_pass"]:
        write_json(RESULTS_ROOT / "candidate_freeze_manifest.json", {
            "schema": "omega-v2-1b-candidate-freeze-v1",
            "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
            "frozen": True,
            "attempt03_allowed_exactly_once": True,
            "implementation_commit": build_manifest["implementation_commit"],
            "executable_sha256": build_manifest["executable_sha256"],
            "source_sha256": build_manifest["source_file_sha256"],
            "compiler_flags": build_manifest["compile_flags"],
            "selected_worker_cpu_set_ids": attempt02_before["frozen_worker_cpu_set_ids"],
            "candidate_metrics": metrics,
        })

    postanalysis_file = Path(__file__).resolve()
    pre_report_paths = [
        *sealed_source_paths(build_manifest), *run_kq.dependency_files(), executable, V2_0_SEAL,
        Path(attempt02_before["artifact_manifest_path_abs"]), RESULTS_ROOT / "build_manifest.json",
        immutable_native, immutable_pytorch, RESULTS_ROOT / "source_weight_manifest.json",
        RESULTS_ROOT / "attempt02_preservation.json", RESULTS_ROOT / "test_report.json",
        RESULTS_ROOT / "summary_metrics.json", RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml",
        RESULTS_ROOT / "native_kq_stdout.log", RESULTS_ROOT / "native_kq_stderr.log",
        RESULTS_ROOT / "pytorch_control_stdout.log", RESULTS_ROOT / "pytorch_control_stderr.log",
        *(RESULTS_ROOT / "native_correctness_outputs" / name for name in (
            "candidate1_full_m4_k1.bin", "candidate1_full_m16_k1.bin", "candidate1_full_m8_k4.bin",
        )),
        postanalysis_file,
    ]
    freeze_path = RESULTS_ROOT / "candidate_freeze_manifest.json"
    if freeze_path.is_file():
        pre_report_paths.append(freeze_path)
    report_hashes = {str(path.resolve()): sha256_file(path) for path in pre_report_paths if path.is_file()}
    report = run_kq.format_kq_report(summary, RESULTS_ROOT, executable, build_manifest["executable_sha256"], build_manifest, report_hashes)
    report_path = RESULTS_ROOT / "OMEGA_V2_1B_KQ_REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    report_sidecar = RESULTS_ROOT / "OMEGA_V2_1B_KQ_REPORT.md.sha256"
    report_digest = sha256_file(report_path)
    report_sidecar.write_text(report_digest + "\n", encoding="ascii", newline="\n")
    result_paths = [*pre_report_paths, report_path, report_sidecar]
    artifact_hashes = {
        "schema": "omega-v2-1b-kq-artifact-hashes-v1",
        "measurement_implementation_commit": build_manifest["implementation_commit"],
        "postmeasurement_analysis_commit": summary["postmeasurement_analysis_commit"],
        "report_self_sha256": report_digest,
        "artifacts": {
            str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in result_paths if path.is_file()
        },
    }
    write_json(RESULTS_ROOT / "artifact_hashes.json", artifact_hashes)
    verified = all(record["sha256"] == sha256_file(Path(path)) for path, record in artifact_hashes["artifacts"].items())
    verified = verified and sha256_file(report_path) == report_sidecar.read_text(encoding="ascii").strip()
    if not verified:
        raise RuntimeError("KQ_RESULT_HASH_FAILURE after analysis-only finalization")
    print(json.dumps({
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "terminal_status": decision["terminal_status"],
        "source_implementation_commit": build_manifest["implementation_commit"],
        "analysis_commit": summary["postmeasurement_analysis_commit"],
        "results_root_abs": str(RESULTS_ROOT.resolve()),
        "report_abs": str(report_path.resolve()),
        "report_sha256": report_digest,
        "artifact_hashes_abs": str((RESULTS_ROOT / "artifact_hashes.json").resolve()),
        "artifact_hashes_verified": verified,
        "tests_passed": test_report["pass_count"],
        "tests_failed": test_report["fail_count"],
        "tests_skipped": test_report["skip_count"],
        **{key: metrics[key]["value"] for key in ("E_Q4", "E_FULL_4", "E_FULL_16", "S_native")},
    }, indent=2, sort_keys=True))
    return 0 if decision["all_five_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
