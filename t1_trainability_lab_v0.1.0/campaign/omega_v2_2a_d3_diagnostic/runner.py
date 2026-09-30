"""D3 diagnostic source seal, report persistence, and GO-gated CUDA runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from . import ATTEMPT00_ROOT, PACKAGE_ROOT, V20_ROOT, V22A_ROOT, V22A_R1_RESULT
from .core import configure_v2_2a_diagnostic_execution
from .diagnostic import create_cpu_fp64_oracle, create_cuda_d3_run, run_diagnostic


SPEC_PATH = PACKAGE_ROOT / "OMEGA_V2_2A_D3_DIAGNOSTIC_SPEC.md"
SOURCE_SEAL_PATH = PACKAGE_ROOT / "SOURCE_SEAL.json"
V22A_SOURCE_SEAL = V22A_ROOT / "SOURCE_SEAL_R1.json"
ATTEMPT00_MANIFEST = ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json"
ATTEMPT00_INCIDENT = ATTEMPT00_ROOT / "INCIDENT.json"
V22A_RESULTS_MANIFEST = V22A_R1_RESULT.parent / "artifact_hashes.json"
RESULTS_ROOT = PACKAGE_ROOT / "results" / "omega_v2_2a_d3_diagnostic_attempt01"
HELD_OUT_D3Q_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def source_hashes() -> dict[str, str]:
    paths = [SPEC_PATH, PACKAGE_ROOT / ".gitattributes", *sorted(PACKAGE_ROOT.rglob("*.py"))]
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in paths if path.is_file()}


def v2_0_reference_hashes() -> dict[str, str]:
    paths = [
        V20_ROOT / "omega_v2" / "core.py",
        V20_ROOT / "omega_v2" / "variants.py",
        V20_ROOT / "omega_v2" / "ledger.py",
        V20_ROOT / "results" / "omega_v2_0_conformance" / "omega_v2_flop_ledger.json",
        V20_ROOT / "V2_0_RESULT_SEAL.json",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def v2_2a_reference_hashes() -> dict[str, str]:
    paths = [
        V22A_R1_RESULT,
        V22A_RESULTS_MANIFEST,
        V22A_SOURCE_SEAL,
        V22A_ROOT / "core.py",
        V22A_ROOT / "variants.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def create_source_seal() -> dict[str, Any]:
    if SOURCE_SEAL_PATH.exists():
        raise FileExistsError(f"D3 diagnostic source seal is immutable: {SOURCE_SEAL_PATH}")
    v22a_seal = json.loads(V22A_SOURCE_SEAL.read_text(encoding="utf-8"))
    attempt00 = json.loads(ATTEMPT00_INCIDENT.read_text(encoding="utf-8"))
    seal = {
        "schema": "omega-v2-2a-d3-diagnostic-source-seal-v1",
        "spec_sha256": sha256_file(SPEC_PATH),
        "python_source_sha256": source_hashes(),
        "v2_0_reference_sha256": v2_0_reference_hashes(),
        "v2_2a_r1_reference_sha256": v2_2a_reference_hashes(),
        "attempt_00_classification": attempt00["classification"],
        "attempt_00_incident_json_sha256": sha256_file(ATTEMPT00_INCIDENT),
        "attempt_00_artifact_manifest_sha256": sha256_file(ATTEMPT00_MANIFEST),
        "attempt_00_manifest_sidecar_sha256": (ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json.sha256").read_text(encoding="ascii").strip(),
        "environment": v22a_seal["environment"],
        "source_seal_process_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_device_query_performed": False,
            "cuda_kernel_launches": 0,
        },
        "fixed_seeds": {
            "MASTER_SEED": 20260930,
            "WEIGHT_SEED": 20260930,
            "INPUT_SEED": 20261938,
            "LOSS_W_SEED": 20262938,
        },
        "held_out_D3Q_seeds_forbidden": list(HELD_OUT_D3Q_SEEDS),
        "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
        "may_rescue_V2_2A": False,
        "architectural_verdict": None,
        "official_diagnostic_started": False,
        "cuda_kernel_launches_during_source_seal": 0,
    }
    write_json(SOURCE_SEAL_PATH, seal)
    return seal


def _validate_runtime_environment(seal: dict[str, Any]) -> None:
    expected = seal.get("environment", {})
    if expected.get("torch_version") and str(torch.__version__) != expected["torch_version"]:
        raise RuntimeError("D3_DIAGNOSTIC_ENVIRONMENT_MISMATCH: PyTorch version differs from the sealed V2-2A-r1 environment")
    if expected.get("torch_cuda_runtime") and torch.version.cuda != expected["torch_cuda_runtime"]:
        raise RuntimeError("D3_DIAGNOSTIC_ENVIRONMENT_MISMATCH: CUDA runtime differs from the sealed V2-2A-r1 environment")


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# OMEGA-V2-2A-D3-DIAGNOSTIC", "",
        f"- classification: `{result['classification']}`",
        f"- may_rescue_V2_2A: `{result['may_rescue_V2_2A']}`",
        f"- architectural_verdict: `{result['architectural_verdict']}`",
        f"- V2-2A-r1 terminal: `{result['v2_2a_r1_terminal']}`",
        f"- V2-2A-r1 modified: `{result['v2_2a_r1_modified']}`",
        f"- held-out D3Q seeds touched: `{result['held_out_D3Q_seeds_touched']}`", "",
        "## CUDA D3 run reproducibility", "",
        f"- run 01/02 bitwise reproducible: `{result['reproducibility']['all_bitwise_equal']}`", "",
        "## Run 1 and run 2 family metrics", "",
        "| Run | Family | Sum variant | E_L2 | E_inf | max_abs | max_rel_old |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for run_name, metrics_key in (("run_01", "run_01_metrics"), ("run_02", "run_02_metrics")):
        for family, family_row in result[metrics_key].items():
            for variant, metrics in family_row["summation_metrics"].items():
                lines.append(f"| {run_name} | {family} | {variant} | {metrics['E_L2']:.9g} | {metrics['E_inf']:.9g} | {metrics['max_abs']:.9g} | {metrics['max_rel_old']:.9g} |")
    lines.extend([
        "", "## Reproducibility and persisted tensor hashes", "", "```json",
        json.dumps({"reproducibility": result["reproducibility"], "tensor_bundles": result["tensor_bundles"]}, sort_keys=True, indent=2), "```",
        "", "## Per-family old-comparator indices and ULP details", "", "```json",
        json.dumps({"run_01": result["run_01_metrics"], "run_02": result["run_02_metrics"]}, sort_keys=True, indent=2), "```",
        "", "## V2-2A-r1 read-only cross-check", "", "```json",
        json.dumps(result["V2_2A_r1_read_only_crosscheck"], sort_keys=True, indent=2), "```",
        "", "## CPU FP64 oracle diagnostic", "", "```json",
        json.dumps({"copy_checks": result["cpu_fp64_copy_checks"], "metrics": result["cpu_fp64_arithmetic_oracle"], "tensor_bundle": result["tensor_bundles"]["cpu_fp64"]}, sort_keys=True, indent=2), "```",
        "", "## Environment and source hashes", "", "```json",
        json.dumps(result["source_seal"], sort_keys=True, indent=2), "```",
        "", "No thresholds or PASS/FAIL scientific verdicts are applied in this diagnostic.", "",
    ])
    return "\n".join(lines)


def _seal_diagnostic_result(result: dict[str, Any], result_root: Path) -> dict[str, Any]:
    report_path = result_root / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md"
    report_path.write_text(_report_markdown(result), encoding="utf-8", newline="\n")
    report_sha = sha256_file(report_path)
    sidecar = result_root / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md.sha256"
    sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")
    metrics_path = result_root / "d3_diagnostic_metrics.json"
    write_json(metrics_path, result)
    evidence_paths = [report_path, sidecar, metrics_path, SOURCE_SEAL_PATH, SPEC_PATH]
    evidence_paths.extend(sorted(PACKAGE_ROOT.rglob("*.py")))
    evidence_paths.extend(Path(path) for path in v2_0_reference_hashes())
    evidence_paths.extend(Path(path) for path in v2_2a_reference_hashes())
    evidence_paths.extend((ATTEMPT00_INCIDENT, ATTEMPT00_MANIFEST, ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json.sha256"))
    evidence_paths.extend(result_root.glob("*.pt"))
    manifest = {
        "schema": "omega-v2-2a-d3-diagnostic-artifact-hashes-v1",
        "classification": result["classification"],
        "report_sha256": report_sha,
        "artifacts": {str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence_paths if path.is_file()},
    }
    manifest_path = result_root / "artifact_hashes.json"
    write_json(manifest_path, manifest)
    verified = all(Path(name).is_file() and sha256_file(Path(name)) == entry["sha256"] and Path(name).stat().st_size == entry["size_bytes"] for name, entry in manifest["artifacts"].items())
    verified = verified and sha256_file(report_path) == sidecar.read_text(encoding="ascii").strip()
    write_json(result_root / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(manifest["artifacts"])})
    return {"report_sha256": report_sha, "artifact_manifest_sha256": sha256_file(manifest_path), "artifact_count": len(manifest["artifacts"]), "verified": verified}


def _seal_failure_result(result_root: Path, failure: dict[str, Any]) -> dict[str, Any]:
    write_json(result_root / "execution_failure.json", failure)
    report = result_root / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md"
    report.write_text("# OMEGA-V2-2A-D3-DIAGNOSTIC\n\n- terminal_classification: `EXECUTION_FAILURE`\n- error_type: `" + failure["error_type"] + "`\n- error: `" + failure["error"] + "`\n\n## Environment and source hashes\n\n```json\n" + json.dumps(failure.get("source_seal", {}), sort_keys=True, indent=2) + "\n```\n", encoding="utf-8", newline="\n")
    report_sha = sha256_file(report)
    sidecar = result_root / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md.sha256"
    sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")
    files = [*result_root.glob("*.pt"), result_root / "execution_failure.json", report, sidecar, SOURCE_SEAL_PATH, SPEC_PATH]
    files.extend(sorted(PACKAGE_ROOT.rglob("*.py")))
    files.extend(Path(path) for path in v2_0_reference_hashes())
    files.extend(Path(path) for path in v2_2a_reference_hashes())
    files.extend((ATTEMPT00_INCIDENT, ATTEMPT00_MANIFEST, ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json.sha256"))
    manifest = {"schema": "omega-v2-2a-d3-diagnostic-artifact-hashes-v1", "classification": "CALIBRATION_DIAGNOSTIC_ONLY", "report_sha256": report_sha, "artifacts": {str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files if path.is_file()}}
    manifest_path = result_root / "artifact_hashes.json"
    write_json(manifest_path, manifest)
    verified = all(Path(name).is_file() and sha256_file(Path(name)) == item["sha256"] and Path(name).stat().st_size == item["size_bytes"] for name, item in manifest["artifacts"].items())
    verified = verified and sha256_file(report) == sidecar.read_text(encoding="ascii").strip()
    write_json(result_root / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(manifest["artifacts"])})
    return {"report_sha256": report_sha, "artifact_manifest_sha256": sha256_file(manifest_path), "verified": verified}


def execute_diagnostic(*, explicit_go: bool = False, result_root: Path = RESULTS_ROOT, cuda_run_fn: Callable = create_cuda_d3_run, fp64_fn: Callable = create_cpu_fp64_oracle) -> dict[str, Any]:
    if not explicit_go:
        raise RuntimeError("D3_DIAGNOSTIC_STOP: execution requires explicit diagnostic GO")
    if result_root.exists():
        raise FileExistsError(f"D3 diagnostic result slot is immutable: {result_root}")
    source_seal = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
    if source_seal["spec_sha256"] != sha256_file(SPEC_PATH) or source_seal["python_source_sha256"] != source_hashes():
        raise RuntimeError("D3_DIAGNOSTIC_SOURCE_SEAL_MISMATCH")
    _validate_runtime_environment(source_seal)
    try:
        result = run_diagnostic(result_root=result_root, run_cuda_fn=cuda_run_fn, run_cpu_fp64_fn=fp64_fn)
        result["source_seal"] = source_seal
        if result.get("terminal_classification") == "DIAGNOSTIC_COMPLETE":
            result["artifact_seal"] = _seal_diagnostic_result(result, result_root)
        else:
            result["artifact_seal"] = _seal_failure_result(result_root, result)
        return result
    except Exception as error:
        if not result_root.exists():
            result_root.mkdir(parents=True, exist_ok=False)
        failure = {
            "schema": "omega-v2-2a-d3-diagnostic-execution-failure-v1",
            "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
            "may_rescue_V2_2A": False,
            "architectural_verdict": None,
            "terminal_classification": "EXECUTION_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "held_out_D3Q_seeds_touched": [],
            "source_seal": source_seal,
        }
        write_json(result_root / "execution_failure.json", failure)
        failure["artifact_seal"] = _seal_failure_result(result_root, failure)
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--seal-source", action="store_true", help="seal diagnostic spec/source/environment; does not execute D3")
    modes.add_argument("--run-diagnostic", action="store_true", help="execute the separate D3 calibration diagnostic")
    parser.add_argument("--go-d3-diagnostic", action="store_true", help="acknowledge the judge's explicit CUDA GO for this diagnostic")
    args = parser.parse_args(argv)
    if args.seal_source:
        seal = create_source_seal()
        print(json.dumps({"phase": "D3_DIAGNOSTIC_SOURCE_SEAL", "spec_sha256": seal["spec_sha256"], "source_hashes": seal["python_source_sha256"], "attempt00_artifact_manifest_sha256": seal["attempt_00_artifact_manifest_sha256"], "official_diagnostic_started": False}, indent=2, sort_keys=True))
        return 0
    if not args.go_d3_diagnostic:
        raise RuntimeError("D3_DIAGNOSTIC_STOP: CUDA diagnostic needs explicit diagnostic GO")
    seal = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
    if seal["spec_sha256"] != sha256_file(SPEC_PATH) or seal["python_source_sha256"] != source_hashes():
        raise RuntimeError("D3_DIAGNOSTIC_SOURCE_SEAL_MISMATCH")
    _validate_runtime_environment(seal)
    configure_v2_2a_diagnostic_execution()
    if not torch.cuda.is_available():
        raise RuntimeError("D3_DIAGNOSTIC_EXECUTION_FAILURE: CUDA unavailable")
    report = execute_diagnostic(explicit_go=True)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report.get("terminal_classification") == "DIAGNOSTIC_COMPLETE" else 1
