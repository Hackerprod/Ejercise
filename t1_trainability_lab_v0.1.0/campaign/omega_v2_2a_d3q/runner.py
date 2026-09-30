"""Seed-gated D3Q runner, evidence persistence, and source seal creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor

from omega_v2.core import MATRIX_FAMILIES

from . import D3_DIAGNOSTIC_ROOT, PACKAGE_ROOT, V20_ROOT, V22A_ROOT
from .config import (
    CALIBRATION_SEED,
    DryRunSeedSlot,
    OFFICIAL_MASTER_SEEDS,
    SeedPlan,
    make_seed_plan,
    official_seed_plans,
)
from .core import configure_d3q_execution, run_calibration_seed_cpu_qa, run_seed_cuda
from .metrics import (
    evaluate_oracle_gate,
    evaluate_primary_gates,
    file_sha256,
    old_max_rel_diagnostic,
    sum_u4_diagnostics,
    tensor_raw_sha256,
)


SPEC_PATH = PACKAGE_ROOT / "OMEGA_V2_2A_D3Q_SPEC.md"
SOURCE_SEAL_PATH = PACKAGE_ROOT / "SOURCE_SEAL.json"
OFFICIAL_RESULTS_ROOT = PACKAGE_ROOT / "results" / "omega_v2_2a_d3q_official_attempt01"
CALIBRATION_SMOKE_ROOT = PACKAGE_ROOT / "results" / "qa" / "omega_v2_2a_d3q_calibration_smoke_01"
OFFICIAL_LAUNCH_LOG_ROOT = PACKAGE_ROOT / "results" / "omega_v2_2a_d3q_launch_logs_01"

D3_SOURCE_COMMIT = "8e5a755903ead3c5732c05b22b79ff0a089c07e6"
D3_RESULT_COMMIT = "ceaaf329f0e5d8690a303b596c8342311ca073d6"
D3_SOURCE_SEAL = D3_DIAGNOSTIC_ROOT / "SOURCE_SEAL.json"
D3_RESULT_ROOT = D3_DIAGNOSTIC_ROOT / "results" / "omega_v2_2a_d3_diagnostic_attempt01"
D3_RESULT_HASH_MANIFEST = D3_RESULT_ROOT / "artifact_hashes.json"
D3_CALIBRATION_GRADIENT_BUNDLE = D3_RESULT_ROOT / "cuda_gradients_run_01.pt"
D3_CALIBRATION_GRADIENT_BUNDLE_SHA256 = "1D9EFE93B6D02AD58768682D6B1C7E4B52C5516D3254DF214121310932E43629"
D3_OPERATOR_ABORT = D3_DIAGNOSTIC_ROOT / "results" / "omega_v2_2a_d3_diagnostic_operator_abort_00" / "OPERATOR_ABORT_00.json"
V22A_SOURCE_SEAL = V22A_ROOT / "SOURCE_SEAL_R1.json"
V22A_R1_RESULT = V22A_ROOT / "results" / "omega_v2_2a_local_preflight" / "preflight_result.json"
V22A_R1_MANIFEST = V22A_R1_RESULT.parent / "artifact_hashes.json"
V22A_ATTEMPT00_ROOT = V22A_ROOT / "results" / "attempt_00"
V22A_ATTEMPT00_INCIDENT = V22A_ATTEMPT00_ROOT / "INCIDENT.json"
V22A_ATTEMPT00_MANIFEST = V22A_ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json"
V22A_ATTEMPT00_SIDECAR = V22A_ATTEMPT00_ROOT / "INCIDENT_ARTIFACT_HASHES.json.sha256"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def source_hashes() -> dict[str, str]:
    paths = [SPEC_PATH, PACKAGE_ROOT / ".gitattributes", *sorted(PACKAGE_ROOT.rglob("*.py"))]
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in paths if path.is_file()}


def reference_hashes() -> dict[str, str]:
    paths = [
        V20_ROOT / "omega_v2" / "core.py",
        V20_ROOT / "omega_v2" / "variants.py",
        V20_ROOT / "omega_v2" / "ledger.py",
        V20_ROOT / "V2_0_RESULT_SEAL.json",
        V22A_ROOT / "core.py",
        V22A_ROOT / "variants.py",
        V22A_SOURCE_SEAL,
        V22A_R1_RESULT,
        V22A_R1_MANIFEST,
        D3_SOURCE_SEAL,
        D3_RESULT_ROOT / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md",
        D3_RESULT_HASH_MANIFEST,
        D3_OPERATOR_ABORT,
        V22A_ATTEMPT00_INCIDENT,
        V22A_ATTEMPT00_MANIFEST,
        V22A_ATTEMPT00_SIDECAR,
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def create_source_seal() -> dict[str, Any]:
    if SOURCE_SEAL_PATH.exists():
        raise FileExistsError(f"D3Q source seal is immutable: {SOURCE_SEAL_PATH}")
    smoke_manifest_path = CALIBRATION_SMOKE_ROOT / "artifact_hashes.json"
    smoke_verified_path = CALIBRATION_SMOKE_ROOT / "artifact_hashes_verified.json"
    if not smoke_manifest_path.is_file() or not smoke_verified_path.is_file():
        raise RuntimeError("D3Q_SOURCE_SEAL_STOP: authorized calibration-seed harness smoke is required first")
    smoke_verified = json.loads(smoke_verified_path.read_text(encoding="utf-8"))
    if smoke_verified.get("verified") is not True:
        raise RuntimeError("D3Q_SOURCE_SEAL_STOP: calibration-seed harness smoke artifacts are not verified")
    smoke_metrics = json.loads((CALIBRATION_SMOKE_ROOT / "calibration_smoke.json").read_text(encoding="utf-8"))
    if smoke_metrics.get("spec_sha256") != sha256_file(SPEC_PATH) or smoke_metrics.get("source_sha256") != source_hashes():
        raise RuntimeError("D3Q_SOURCE_SEAL_STOP: calibration smoke spec/source snapshot is stale")
    if smoke_metrics.get("terminal_status") != "CALIBRATION_SMOKE_COMPLETE" or smoke_metrics.get("D3Q_verdict") is not None:
        raise RuntimeError("D3Q_SOURCE_SEAL_STOP: calibration smoke must complete without a D3Q verdict")
    d3_seal = json.loads(D3_SOURCE_SEAL.read_text(encoding="utf-8"))
    d3_artifact_verification = json.loads((D3_RESULT_ROOT / "artifact_hashes_verified.json").read_text(encoding="utf-8"))
    v22a_seal = json.loads(V22A_SOURCE_SEAL.read_text(encoding="utf-8"))
    attempt00 = json.loads(V22A_ATTEMPT00_INCIDENT.read_text(encoding="utf-8"))
    seal = {
        "schema": "omega-v2-2a-d3q-source-seal-v1",
        "spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": source_hashes(),
        "reference_sha256": reference_hashes(),
        "source_commits": {
            "d3_diagnostic_source": D3_SOURCE_COMMIT,
            "d3_diagnostic_result": D3_RESULT_COMMIT,
        },
        "d3_diagnostic_terminal": "DIAGNOSTIC_COMPLETE",
        "d3_diagnostic_artifacts_verified": d3_artifact_verification["verified"],
        "d3_diagnostic_report_sha256": sha256_file(D3_RESULT_ROOT / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md"),
        "d3_operator_abort_00": {
            "classification": "D3_DIAG_ATTEMPT_00_OPERATOR_ABORT_PRE_CUDA",
            "record_sha256": sha256_file(D3_OPERATOR_ABORT),
            "cuda_cells_started": 0,
            "protocol_changed": False,
        },
        "v2_2a_attempt_00": {
            "classification": attempt00["classification"],
            "incident_sha256": sha256_file(V22A_ATTEMPT00_INCIDENT),
            "artifact_manifest_sha256": sha256_file(V22A_ATTEMPT00_MANIFEST),
            "artifact_manifest_sidecar_sha256": V22A_ATTEMPT00_SIDECAR.read_text(encoding="ascii").strip(),
        },
        "v2_2a_r1_environment_reference": v22a_seal["environment"],
        "d3_diagnostic_calibration_environment_reference": d3_seal["environment"],
        "source_seal_process_environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_device_query_performed_during_seal": False,
            "cuda_kernel_launches_during_seal": 0,
        },
        "frozen_configuration": {"d": 256, "m": 8, "B": 8, "K": 4, "dtype": "torch.float32", "optimizer": "NONE"},
        "calibration_seed_forbidden_from_verdict": CALIBRATION_SEED,
        "official_master_seeds": list(OFFICIAL_MASTER_SEEDS),
        "seed_derivation": {"WEIGHT_SEED": "S", "INPUT_SEED": "S+1008", "LOSS_W_SEED": "S+2008"},
        "calibration_smoke_artifact_manifest_sha256": sha256_file(smoke_manifest_path),
        "calibration_smoke_artifacts_verified": smoke_verified["verified"],
        "official_d3q_started": False,
        "cuda_kernel_launches_during_source_seal": 0,
    }
    write_json(SOURCE_SEAL_PATH, seal)
    return seal


def _verify_source_seal() -> dict[str, Any]:
    seal = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
    if seal["spec_sha256"] != sha256_file(SPEC_PATH) or seal["source_sha256"] != source_hashes():
        raise RuntimeError("D3Q_SOURCE_SEAL_MISMATCH")
    if seal["reference_sha256"] != reference_hashes():
        raise RuntimeError("D3Q_REFERENCE_SEAL_MISMATCH")
    return seal


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError("empty D3Q case id")
    return cleaned


def prepare_official_launch(result_root: str | Path = OFFICIAL_RESULTS_ROOT) -> dict[str, Any]:
    """Describe the official invocation without creating the results slot."""
    root = Path(result_root)
    if root.exists():
        raise FileExistsError(f"D3Q official result slot is immutable: {root}")
    return {
        "module": "omega_v2_2a_d3q",
        "arguments": ["--run-official", "--go-d3q-official"],
        "result_root": str(root.resolve()),
        "launch_log_root": str(OFFICIAL_LAUNCH_LOG_ROOT.resolve()),
        "slot_created": False,
    }


def _walk_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _walk_tensors(item, next_prefix)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{prefix}/{index}")


def _persist_seed_result(result_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    case_id = _safe_component(result["case_id"])
    bundle_path = result_root / f"{case_id}_raw_gradients.pt"
    tensor_bundle = result["tensor_bundle"]
    torch.save({"schema": "omega-v2-2a-d3q-gradient-bundle-v1", "case_id": case_id, "families": tensor_bundle}, bundle_path)
    hashes: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for logical_name, tensor in _walk_tensors(tensor_bundle):
        name = f"{case_id}/{logical_name}"
        raw_hash = tensor_raw_sha256(name, tensor)
        hashes[name] = raw_hash
        metadata[name] = {"shape": [int(x) for x in tensor.shape], "dtype": str(tensor.dtype), "raw_sha256": raw_hash}
    return {
        "bundle_path": bundle_path.name,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256_file(bundle_path),
        "tensor_raw_sha256": hashes,
        "tensor_metadata": metadata,
    }


def _seed_summary(result: dict[str, Any], bundle_record: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in result.items() if key != "tensor_bundle"}
    summary["tensor_bundle"] = bundle_record
    return summary


def _required_seed_gates_pass(result: dict[str, Any]) -> bool:
    families = result.get("families", {})
    copy_checks = result.get("cpu_fp64_copy_checks", {})
    return bool(
        result.get("seed_pass") is True
        and result.get("structural_pass") is True
        and result.get("finite") is True
        and result.get("cpu_fp64_oracle_all_pass") is True
        and len(families) == len(MATRIX_FAMILIES)
        and set(families) == set(MATRIX_FAMILIES)
        and all(item.get("seed_family_pass") is True for item in families.values())
        and len(copy_checks) == 3
        and all(value is True for value in copy_checks.values())
    )


def _report_markdown(result: dict[str, Any]) -> str:
    if result.get("classification") == "CALIBRATION_ONLY":
        smoke = result.get("calibration_smoke", {})
        seed_execution = smoke.get("seed_execution", {})
        return "\n".join([
            "# OMEGA-V2-2A-D3Q Calibration Seed Harness Smoke", "",
            "- classification: `CALIBRATION_ONLY`",
            "- D3Q_verdict: `NONE`",
            f"- seed plan: `{json.dumps(smoke.get('seed_plan', {}), sort_keys=True)}`",
            "- held-out seed used: `False`",
            "- scientific gate decisions voting: `False`",
            f"- structural_pass: `{seed_execution.get('structural_pass')}`",
            f"- finite: `{seed_execution.get('finite')}`",
            f"- terminal_status: `{result['terminal_status']}`",
            f"- duration_seconds: `{seed_execution.get('duration_seconds')}`",
            f"- vram_peak_allocated_bytes (NON-GATE): `{seed_execution.get('vram_peak_allocated_bytes')}`",
            f"- vram_peak_reserved_bytes (NON-GATE): `{seed_execution.get('vram_peak_reserved_bytes')}`", "",
            "This harness smoke is calibration-only and contributes no scientific D3Q verdict.", "",
        ])
    lines = [
        "# OMEGA-V2-2A-D3Q", "",
        f"- terminal_status: `{result['terminal_status']}`",
        "- classification: `INDEPENDENT_HELDOUT_VALIDATION`",
        "- calibration seed 20260930 included in official verdict: `False`",
        f"- official seed slots completed: `{len(result['seed_results'])}` / 5",
        f"- hard_stop: `{result['hard_stop']}`", "",
        "## Per-seed decisions", "",
        "| Seed slot | Structural | Finite | Oracle | Seed result | Duration (s) | Peak allocated (bytes) | Peak reserved (bytes) |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for seed in result["seed_results"]:
        structural = seed.get("structural_pass")
        finite = seed.get("finite")
        oracle = seed.get("cpu_fp64_oracle_all_pass")
        lines.append(
            f"| {seed['case_id']} | {structural} | {finite} | {oracle} | {seed['terminal_status']} | "
            f"{seed.get('duration_seconds', 0.0):.9g} | {seed.get('vram_peak_allocated_bytes')} | {seed.get('vram_peak_reserved_bytes')} |"
        )
    for seed in result["seed_results"]:
        lines.extend(["", f"## Seed {seed['case_id']}", ""])
        if seed.get("structural") is not None:
            lines.extend(["### Structural trace checks", "", "```json", json.dumps(seed["structural"], sort_keys=True, indent=2), "```"])
        lines.extend(["", "### Per-family scientific and oracle decisions", "", "```json", json.dumps(seed.get("families", {}), sort_keys=True, indent=2), "```"])
        lines.extend(["", "### Persisted gradient bundle", "", "```json", json.dumps(seed.get("tensor_bundle", {}), sort_keys=True, indent=2), "```"])
    lines.extend([
        "", "## Aggregation rule", "",
        "All five held-out seeds and all seven families must pass. No majority, averaging, outlier exclusion, or per-seed retry is used.",
        "S_reverse equality and old max_rel remain NON-GATE diagnostics.", "",
    ])
    return "\n".join(lines)


def _finalize_result(result: dict[str, Any], result_root: Path, *, source_seal: dict[str, Any] | None) -> dict[str, Any]:
    calibration_smoke = result.get("classification") == "CALIBRATION_ONLY"
    metrics_path = result_root / ("calibration_smoke.json" if calibration_smoke else "d3q_metrics.json")
    if source_seal is not None:
        result["source_seal_summary"] = {
            "spec_sha256": source_seal["spec_sha256"],
            "source_sha256": source_seal["source_sha256"],
            "source_commits": source_seal["source_commits"],
        }
    write_json(metrics_path, result)
    report_path = result_root / ("CALIBRATION_SMOKE_REPORT.md" if calibration_smoke else "OMEGA_V2_2A_D3Q_REPORT.md")
    report_path.write_text(_report_markdown(result), encoding="utf-8", newline="\n")
    report_sha = sha256_file(report_path)
    report_sidecar = result_root / ("CALIBRATION_SMOKE_REPORT.md.sha256" if calibration_smoke else "OMEGA_V2_2A_D3Q_REPORT.md.sha256")
    report_sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")

    evidence_paths: list[Path] = [metrics_path, report_path, report_sidecar, *sorted(result_root.glob("*_raw_gradients.pt"))]
    evidence_paths.extend(path for path in (result_root / "hard_stop.json", result_root / "calibration_smoke.json") if path.is_file())
    evidence_paths.extend([SPEC_PATH, PACKAGE_ROOT / ".gitattributes", *sorted(PACKAGE_ROOT.rglob("*.py"))])
    if SOURCE_SEAL_PATH.is_file():
        evidence_paths.append(SOURCE_SEAL_PATH)
    manifest = {
        "schema": "omega-v2-2a-d3q-calibration-smoke-hashes-v1" if calibration_smoke else "omega-v2-2a-d3q-artifact-hashes-v1",
        "terminal_status": result["terminal_status"],
        "classification": "INDEPENDENT_HELDOUT_VALIDATION",
        "report_sha256": report_sha,
        "artifacts": {
            str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in evidence_paths
            if path.is_file()
        },
    }
    manifest_path = result_root / "artifact_hashes.json"
    write_json(manifest_path, manifest)
    verified = all(
        Path(name).is_file()
        and Path(name).stat().st_size == item["size_bytes"]
        and sha256_file(Path(name)) == item["sha256"]
        for name, item in manifest["artifacts"].items()
    )
    verified = verified and sha256_file(report_path) == report_sidecar.read_text(encoding="ascii").strip()
    write_json(result_root / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(manifest["artifacts"])})
    result["artifact_seal"] = {
        "verified": bool(verified),
        "artifact_count": len(manifest["artifacts"]),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": report_sha,
    }
    return result


def _execute_seed_sequence(
    seed_cases: list[Any],
    seed_executor: Callable[[Any, Callable[[], None]], dict[str, Any]],
    result_root: str | Path,
    *,
    source_seal: dict[str, Any] | None = None,
    expected_count: int = 5,
) -> dict[str, Any]:
    root = Path(result_root)
    if root.exists():
        raise FileExistsError(f"D3Q result slot is immutable: {root}")
    if len(seed_cases) != expected_count:
        raise ValueError(f"D3Q control flow requires exactly {expected_count} seed slots")
    started = time.perf_counter()
    boundary = {"crossed": False}
    seed_results: list[dict[str, Any]] = []
    hard_stop: dict[str, Any] | None = None

    def mark_boundary() -> None:
        boundary["crossed"] = True

    for index, case in enumerate(seed_cases):
        case_id = str(case.case_id) if isinstance(case, DryRunSeedSlot) else f"seed_{case.master_seed}"
        try:
            seed_result = seed_executor(case, mark_boundary)
            seed_result["case_id"] = case_id
            seed_result["seed_pass"] = _required_seed_gates_pass(seed_result)
            if not seed_result["seed_pass"] and seed_result.get("structural_pass") is False:
                seed_result["terminal_status"] = "SEED_FAIL_STRUCTURAL"
            elif not seed_result["seed_pass"]:
                seed_result["terminal_status"] = "SEED_FAIL"
            # Persisting a held-out seed result itself crosses the consumption boundary.
            mark_boundary()
            if not root.exists():
                root.mkdir(parents=True, exist_ok=False)
            bundle_record = _persist_seed_result(root, seed_result)
            seed_results.append(_seed_summary(seed_result, bundle_record))
        except Exception as error:
            if not boundary["crossed"]:
                return {
                    "schema": "omega-v2-2a-d3q-pre-cuda-abort-v1",
                    "terminal_status": "PRE_CUDA_ABORT_UNCONSUMED",
                    "consumed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seed_results": [],
                    "hard_stop": None,
                }
            hard_stop = {"case_id": case_id, "error_type": type(error).__name__, "error": str(error), "seed_index": index}
            if not root.exists():
                root.mkdir(parents=True, exist_ok=False)
            write_json(root / "hard_stop.json", hard_stop)
            break

    if hard_stop is not None:
        not_run = [
            str(case.case_id) if isinstance(case, DryRunSeedSlot) else f"seed_{case.master_seed}"
            for case in seed_cases[len(seed_results) + 1 :]
        ]
        for case_id in not_run:
            seed_results.append({"case_id": case_id, "terminal_status": "NOT_RUN_AFTER_TECHNICAL_HARD_STOP", "seed_pass": False})
    all_pass = len(seed_results) == expected_count and all(_required_seed_gates_pass(item) for item in seed_results) and hard_stop is None
    summary = {
        "schema": "omega-v2-2a-d3q-result-v1",
        "classification": "INDEPENDENT_HELDOUT_VALIDATION",
        "terminal_status": "OMEGA_V2_2A_D3Q_PASS" if all_pass else "OMEGA_V2_2A_D3Q_FAIL",
        "planned_seed_slots": expected_count,
        "completed_seed_slots": sum(item.get("terminal_status") not in ("NOT_RUN_AFTER_TECHNICAL_HARD_STOP",) for item in seed_results),
        "seed_results": seed_results,
        "hard_stop": hard_stop,
        "calibration_seed_in_official_verdict": False,
        "held_out_seed_majority_or_average_used": False,
        "per_seed_retries": 0,
        "duration_seconds": time.perf_counter() - started,
        "official_d3q_started": not all(isinstance(case, DryRunSeedSlot) for case in seed_cases),
    }
    if not root.exists():
        # A pre-CUDA abort has no scientific result and does not consume the slot.
        if hard_stop is None and not boundary["crossed"]:
            summary["terminal_status"] = "PRE_CUDA_ABORT_UNCONSUMED"
            return summary
        root.mkdir(parents=True, exist_ok=False)
    return _finalize_result(summary, root, source_seal=source_seal)


def _validate_runtime_environment(target: dict[str, Any]) -> None:
    if target.get("torch_version") and str(torch.__version__) != target["torch_version"]:
        raise RuntimeError("D3Q_ENVIRONMENT_MISMATCH: PyTorch version differs from sealed V2-2A-r1")
    if target.get("torch_cuda_runtime") and torch.version.cuda != target["torch_cuda_runtime"]:
        raise RuntimeError("D3Q_ENVIRONMENT_MISMATCH: CUDA runtime differs from sealed V2-2A-r1")


def _validate_reference_environment(source_seal: dict[str, Any]) -> None:
    _validate_runtime_environment(source_seal["v2_2a_r1_environment_reference"])


def run_official_d3q(
    *,
    go_d3q_official: bool,
    result_root: str | Path = OFFICIAL_RESULTS_ROOT,
    seed_executor: Callable[[Any, Callable[[], None]], dict[str, Any]] = run_seed_cuda,
) -> dict[str, Any]:
    """Official entry point; held-out plans are materialized only after explicit GO and seal checks."""
    if not go_d3q_official:
        raise RuntimeError("D3Q_STOP: explicit --go-d3q-official is required")
    root = Path(result_root)
    if root.exists():
        raise FileExistsError(f"D3Q official result slot is immutable: {root}")
    seal = _verify_source_seal()
    _validate_reference_environment(seal)
    configure_d3q_execution()
    if not torch.cuda.is_available():
        raise RuntimeError("D3Q_PRE_CUDA_ABORT: CUDA unavailable")
    # Held-out plans are first materialized after all pre-CUDA authorization checks.
    plans = official_seed_plans()
    return _execute_seed_sequence(list(plans), seed_executor, root, source_seal=seal, expected_count=5)


def load_calibration_diagnostic_reference(
    reference_path: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the sealed D3 calibration bundle hash before deserializing it."""
    if reference_path is None:
        reference_path = D3_CALIBRATION_GRADIENT_BUNDLE
    if expected_sha256 is None:
        expected_sha256 = D3_CALIBRATION_GRADIENT_BUNDLE_SHA256
    actual_sha256 = sha256_file(reference_path)
    if actual_sha256.upper() != expected_sha256.upper():
        raise RuntimeError("D3Q_CALIBRATION_REFERENCE_SHA256_MISMATCH")
    bundle = torch.load(reference_path, map_location="cpu", weights_only=True)
    if bundle.get("schema") != "omega-v2-2a-d3-gradient-bundle-v1":
        raise RuntimeError("D3Q_CALIBRATION_REFERENCE_SCHEMA_MISMATCH")
    return {"path": str(Path(reference_path).resolve()), "sha256": actual_sha256, "bundle": bundle}


def calibration_seed_gradient_crosscheck(
    smoke_seed_result: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Report exact gradient equality only; the comparison never affects smoke status."""
    current = smoke_seed_result["tensor_bundle"]["cuda_fp32"]
    frozen = reference["bundle"]["families"]
    rows = []
    for family in MATRIX_FAMILIES:
        current_family = current[family]
        frozen_family = frozen[family]
        for logical_name, smoke_key in (
            ("gR", "gR_cuda_fp32"),
            ("gU0", "gU0_cuda_fp32"),
            ("gU1", "gU1_cuda_fp32"),
            ("gU2", "gU2_cuda_fp32"),
            ("gU3", "gU3_cuda_fp32"),
        ):
            smoke_tensor = current_family[smoke_key].detach().to(device="cpu")
            frozen_tensor = frozen_family[logical_name].detach().to(device="cpu")
            rows.append({
                "family": family,
                "tensor": logical_name,
                "torch_equal": bool(torch.equal(smoke_tensor, frozen_tensor)),
                "smoke_dtype": str(smoke_tensor.dtype),
                "reference_dtype": str(frozen_tensor.dtype),
                "smoke_shape": [int(size) for size in smoke_tensor.shape],
                "reference_shape": [int(size) for size in frozen_tensor.shape],
            })
    return {
        "classification": "NON_GATE_REPORT_ONLY",
        "reference_sha256": reference["sha256"],
        "comparison": "torch.equal for gR and gU0..gU3 across seven families",
        "rows": rows,
        "all_bitwise_equal": all(row["torch_equal"] for row in rows),
        "affects_smoke_terminal_status": False,
        "affects_D3Q_verdict": False,
    }


def run_calibration_seed_smoke(
    *,
    go_calibration_smoke: bool,
    result_root: str | Path = CALIBRATION_SMOKE_ROOT,
    seed_executor: Callable[[Any, Callable[[], None]], dict[str, Any]] = run_seed_cuda,
) -> dict[str, Any]:
    """Separate non-verdict GPU smoke using only calibration seed 20260930."""
    if not go_calibration_smoke:
        raise RuntimeError("D3Q_SMOKE_STOP: explicit calibration-seed smoke GO is required")
    root = Path(result_root)
    if root.exists():
        raise FileExistsError(f"D3Q calibration smoke slot is immutable: {root}")
    if root.resolve() == OFFICIAL_RESULTS_ROOT.resolve():
        raise ValueError("calibration-seed smoke output must be outside the official result slot")
    plan = make_seed_plan(CALIBRATION_SEED, mode="calibration_smoke")
    reference = load_calibration_diagnostic_reference()
    v22a_environment = json.loads(V22A_SOURCE_SEAL.read_text(encoding="utf-8"))["environment"]
    _validate_runtime_environment(v22a_environment)
    configure_d3q_execution()
    if not torch.cuda.is_available():
        raise RuntimeError("D3Q_SMOKE_PRE_CUDA_ABORT: CUDA unavailable")
    boundary = {"crossed": False}
    try:
        seed_result = seed_executor(plan, lambda: boundary.__setitem__("crossed", True))
    except Exception as error:
        if not boundary["crossed"]:
            return {
                "schema": "omega-v2-2a-d3q-calibration-smoke-abort-v1",
                "classification": "CALIBRATION_ONLY",
                "D3Q_verdict": None,
                "terminal_status": "PRE_CUDA_ABORT_UNCONSUMED",
                "consumed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        root.mkdir(parents=True, exist_ok=False)
        failure = {
            "schema": "omega-v2-2a-d3q-calibration-smoke-failure-v1",
            "classification": "CALIBRATION_ONLY",
            "D3Q_verdict": None,
            "terminal_status": "CALIBRATION_SMOKE_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_heldout_seeds_touched": False,
            "scientific_gates_voting": False,
        }
        write_json(root / "calibration_smoke_failure.json", failure)
        return _finalize_result(
            {"schema": failure["schema"], "classification": failure["classification"], "terminal_status": failure["terminal_status"], "hard_stop": failure, "seed_results": []},
            root,
            source_seal=None,
        )
    seed_result["case_id"] = "calibration_seed_20260930"
    crosscheck = calibration_seed_gradient_crosscheck(seed_result, reference)
    seed_result["D3_diagnostic_crosscheck_NON_GATE"] = crosscheck
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    bundle_record = _persist_seed_result(root, seed_result)
    smoke_result = {
        "schema": "omega-v2-2a-d3q-calibration-smoke-v1",
        "classification": "CALIBRATION_ONLY",
        "D3Q_verdict": None,
        "seed_plan": plan.as_dict(),
        "seed_execution": _seed_summary(seed_result, bundle_record),
        "D3_diagnostic_crosscheck_NON_GATE": crosscheck,
        "scientific_gates_voting": False,
        "official_heldout_seeds_touched": False,
        "source_sha256": source_hashes(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "terminal_status": "CALIBRATION_SMOKE_COMPLETE" if (
            seed_result.get("structural_pass")
            and seed_result.get("finite")
            and seed_result.get("cpu_fp64_oracle_all_pass")
            and all(seed_result.get("cpu_fp64_copy_checks", {}).values())
        ) else "CALIBRATION_SMOKE_FAILURE",
    }
    return _finalize_result(
        {
            "schema": smoke_result["schema"],
            "classification": smoke_result["classification"],
            "terminal_status": smoke_result["terminal_status"],
            "hard_stop": None,
            "seed_results": [smoke_result["seed_execution"]],
            "D3Q_verdict": None,
            "source_sha256": smoke_result["source_sha256"],
            "spec_sha256": smoke_result["spec_sha256"],
            "D3_diagnostic_crosscheck_NON_GATE": crosscheck,
            "calibration_smoke": smoke_result,
        },
        root,
        source_seal=None,
    )


def _stub_seed_executor(case: DryRunSeedSlot, mark_boundary: Callable[[], None]) -> dict[str, Any]:
    """CPU tensors only; case is symbolic and carries no numeric seed."""
    mark_boundary()
    raw_by_family: dict[str, dict[str, Tensor]] = {}
    oracle_by_family: dict[str, dict[str, Tensor]] = {}
    family_decisions: dict[str, Any] = {}
    structural = {
        "initial_clone_report": {
            "initial_values_bitwise_equal": True,
            "r4_reuses_one_block_object": True,
            "u4_has_four_block_objects": True,
            "u4_block_storages_pairwise_disjoint": True,
            "u4_storage_disjoint_from_r4": True,
        },
        "round_trace_equalities": [{"round": index, "torch_equal": True} for index in range(1, 5)],
        "final_output_equal": True,
        "pass": True,
    }
    for family_index, family in enumerate(MATRIX_FAMILIES, start=1):
        g_r32 = torch.tensor([float(family_index), float(family_index * 2)], dtype=torch.float32)
        g_u32 = [torch.tensor([family_index / 4, family_index / 2], dtype=torch.float32) for _ in range(4)]
        sums32 = sum_u4_diagnostics(g_u32)
        s64 = (((g_u32[0].double() + g_u32[1].double()) + g_u32[2].double()) + g_u32[3].double())
        g_r64 = g_r32.double()
        primary = evaluate_primary_gates(g_r64, s64)
        old = old_max_rel_diagnostic(g_r32, sums32["S_stack"])
        old["S_reverse_equal_gR_NON_GATE"] = torch.equal(sums32["S_reverse"], g_r32)
        cpu_oracle = {"gR64_cpu": g_r64.clone(), **{f"gU{i}_64_cpu": g_u32[i].double().clone() for i in range(4)}}
        cpu_sum = (((cpu_oracle["gU0_64_cpu"] + cpu_oracle["gU1_64_cpu"]) + cpu_oracle["gU2_64_cpu"]) + cpu_oracle["gU3_64_cpu"])
        oracle_gate = evaluate_oracle_gate(cpu_oracle["gR64_cpu"], cpu_sum)
        raw_by_family[family] = {
            "gR_cuda_fp32": g_r32,
            **{f"gU{i}_cuda_fp32": g_u32[i] for i in range(4)},
            **{f"{name}_cuda": value for name, value in sums32.items()},
            "gR64_primary_cuda": g_r64,
            "S64_primary_cuda": s64,
        }
        oracle_by_family[family] = {**cpu_oracle, "S64_cpu": cpu_sum}
        family_decisions[family] = {
            "primary_S64": primary,
            "diagnostics_NON_GATE": {"old_max_rel": old, "S_reverse_equal_gR_NON_GATE": old["S_reverse_equal_gR_NON_GATE"]},
            "cpu_fp64_oracle": oracle_gate,
            "operational": {"all_cuda_family_tensors_finite": True, "all_cpu_oracle_tensors_finite": True},
            "seed_family_pass": bool(primary["scientific_gates_pass"] and oracle_gate["pass"]),
        }
    return {
        "case_id": case.case_id,
        "seed_plan": None,
        "structural": structural,
        "structural_pass": True,
        "cpu_fp64_copy_checks": {
            "initial_FP32_weights_to_FP64_exact_value_preserved": True,
            "x_FP32_to_FP64_exact_value_preserved": True,
            "w_FP32_to_FP64_exact_value_preserved": True,
        },
        "families": family_decisions,
        "tensor_bundle": {
            "cuda_fp32": raw_by_family,
            "cpu_fp64_oracle": oracle_by_family,
            "structural_traces": {"round_equal": torch.ones(4, dtype=torch.bool)},
        },
        "finite": True,
        "cpu_fp64_oracle_all_pass": True,
        "duration_seconds": 0.001,
        "vram_peak_allocated_bytes": 0,
        "vram_peak_reserved_bytes": 0,
        "terminal_status": "SEED_PASS",
        "seed_pass": all(row["seed_family_pass"] for row in family_decisions.values()),
    }


def full_official_control_flow_dry_run(
    result_root: str | Path,
    *,
    seed_cases: list[DryRunSeedSlot] | None = None,
    seed_executor: Callable[[Any, Callable[[], None]], dict[str, Any]] = _stub_seed_executor,
) -> dict[str, Any]:
    """Run the complete five-slot orchestrator with opaque symbolic stubs only."""
    cases = seed_cases or [DryRunSeedSlot(f"dryrun_slot_{index:02d}") for index in range(1, 6)]
    if any(not isinstance(case, DryRunSeedSlot) for case in cases):
        raise ValueError("D3Q full-control-flow QA accepts symbolic dry-run slots only")
    return _execute_seed_sequence(cases, seed_executor, result_root, expected_count=5)


def pre_cuda_official_path_dry_run(result_root: str | Path = OFFICIAL_RESULTS_ROOT) -> dict[str, Any]:
    """Exercise launch planning only; it never derives seeds or creates the result slot."""
    launch = prepare_official_launch(result_root)
    return {
        "phase": "PRE_CUDA_OFFICIAL_PATH_DRY_RUN",
        "launch_plan": launch,
        "official_seed_values_materialized": False,
        "result_slot_created": Path(result_root).exists(),
        "cuda_kernels_launched": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qa-calibration", action="store_true", help="CPU-only QA with calibration seed only")
    modes.add_argument("--pre-cuda-official-dry-run", action="store_true", help="official launch guards only; no seed generation")
    modes.add_argument("--full-official-control-flow-dry-run", action="store_true", help="five symbolic stub slots; no CUDA or held-out seed data")
    modes.add_argument("--calibration-seed-smoke", action="store_true", help="one separate calibration-only CUDA smoke")
    modes.add_argument("--run-official", action="store_true", help="five-seed official D3Q run")
    modes.add_argument("--seal-source", action="store_true", help="create SOURCE_SEAL.json after authorized smoke")
    parser.add_argument("--go-calibration-seed-smoke", action="store_true", help="explicit GO for the isolated calibration smoke")
    parser.add_argument("--go-d3q-official", action="store_true", help="explicit GO for the five held-out seeds")
    args = parser.parse_args(argv)

    if args.qa_calibration:
        result = run_calibration_seed_cpu_qa(CALIBRATION_SEED)
    elif args.pre_cuda_official_dry_run:
        result = pre_cuda_official_path_dry_run()
    elif args.full_official_control_flow_dry_run:
        with tempfile.TemporaryDirectory(prefix="omega_d3q_full_dry_run_") as temp_dir:
            result = full_official_control_flow_dry_run(Path(temp_dir) / "results")
    elif args.calibration_seed_smoke:
        result = run_calibration_seed_smoke(go_calibration_smoke=args.go_calibration_seed_smoke)
    elif args.run_official:
        result = run_official_d3q(go_d3q_official=args.go_d3q_official)
    else:
        result = create_source_seal()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0
