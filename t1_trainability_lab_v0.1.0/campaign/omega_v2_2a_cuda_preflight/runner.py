"""V2-2A source sealing and authorized CUDA preflight runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from omega_v2.core import ContractualCoreBlock, configure_reference_execution
from omega_v2.ledger import count_one_round_macs, state_dict_sha256

from .checks import (
    cuda_correctness_gate,
    gradient_sharing_gate,
    initialization_and_trace_gate,
    iso_flop_gate,
    k_flex_backward_cell,
    k_flex_forward_cell,
    optimizer_smoke,
    parameter_count,
    parameter_storage_gate,
    schema_sha256,
    state_dict_values_equal,
    tensor_finite,
)
from .core import configure_v2_2a_execution
from .variants import (
    BATCH,
    D,
    INPUT_SEED,
    LOSS_W_SEED,
    M,
    MASTER_SEED,
    TARGET_SEED,
    WEIGHT_SEED,
    build_initial_variants,
    copy_fixed_tensor_to_cuda,
    make_fixed_inputs,
    make_cpu_normal,
    weight_copy_report,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
CAMPAIGN_ROOT = PACKAGE_ROOT.parent
V20_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
V20_LEDGER = V20_ROOT / "results" / "omega_v2_0_conformance" / "omega_v2_flop_ledger.json"
V20_SEAL = V20_ROOT / "V2_0_RESULT_SEAL.json"
SOURCE_SEAL_PATH = PACKAGE_ROOT / "SOURCE_SEAL_R1.json"
RESULTS_ROOT = PACKAGE_ROOT / "results" / "omega_v2_2a_local_preflight"
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_GPU_NAME_FRAGMENT = "GTX 1650 SUPER"
EXPECTED_COMPUTE_CAPABILITY = (7, 5)
MAX_WALL_SECONDS = 30 * 60
VRAM_BUDGET_BYTES = 3 * 1024**3
REQUIRED_FLOPS = 538_968_064


class HardStop(RuntimeError):
    """A runtime condition that prevents the next registered gate."""


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
    source_paths = [PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md", *sorted(PACKAGE_ROOT.rglob("*.py"))]
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in source_paths if path.is_file()}


def v20_hashes() -> dict[str, str]:
    paths = [
        V20_ROOT / "omega_v2" / "core.py",
        V20_ROOT / "omega_v2" / "variants.py",
        V20_ROOT / "omega_v2" / "ledger.py",
        V20_LEDGER,
        V20_SEAL,
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _nvidia_smi_metadata() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "path": None, "query_returncode": None, "query": None}
    result = subprocess.run(
        [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    return {
        "available": result.returncode == 0,
        "path": executable,
        "query_returncode": result.returncode,
        "query": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def environment_record() -> dict[str, Any]:
    configure_reference_execution()
    configure_v2_2a_execution()
    available = bool(torch.cuda.is_available())
    device = None
    capability = None
    total_memory = None
    if available:
        properties = torch.cuda.get_device_properties(0)
        device = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
        total_memory = int(properties.total_memory)
    nvidia_smi = _nvidia_smi_metadata()
    runtime = torch.version.cuda
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": runtime,
        "cuda_available": available,
        "device_name": device,
        "compute_capability": capability,
        "device_total_memory_bytes": total_memory,
        "nvidia_smi": nvidia_smi,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "dtype": "torch.float32",
        "amp": False,
        "dropout": 0,
    }


def environment_matches(record: dict[str, Any]) -> bool:
    return (
        record["torch_version"] == EXPECTED_TORCH
        and record["torch_cuda_runtime"] == EXPECTED_CUDA
        and record["cuda_available"] is True
        and EXPECTED_GPU_NAME_FRAGMENT in str(record["device_name"])
        and tuple(record["compute_capability"] or ()) == EXPECTED_COMPUTE_CAPABILITY
        and record["nvidia_smi"].get("available") is True
        and EXPECTED_GPU_NAME_FRAGMENT in str(record["nvidia_smi"].get("query"))
        and record["cublas_workspace_config"] == ":4096:8"
        and record["deterministic_algorithms"] is True
        and record["allow_tf32_matmul"] is False
        and record["allow_tf32_cudnn"] is False
        and record["cudnn_benchmark"] is False
        and record["cudnn_deterministic"] is True
        and record["torch_num_threads"] == 1
        and record["cuda_autocast_enabled"] is False
    )


def create_source_seal() -> dict[str, Any]:
    previous_seal_sha256 = None
    if SOURCE_SEAL_PATH.exists():
        previous_bytes_sha256 = sha256_file(SOURCE_SEAL_PATH)
        previous_seal_sha256 = previous_bytes_sha256
        previous = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        current_spec_sha256 = sha256_file(PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md")
        current_source_hashes = source_hashes()
        if previous.get("spec_sha256") == current_spec_sha256 and previous.get("python_source_sha256") == current_source_hashes:
            raise FileExistsError(f"matching source seal already exists: {SOURCE_SEAL_PATH}")
        archive_path = PACKAGE_ROOT / "SOURCE_SEAL_PREIMPLEMENTATION.json"
        if archive_path.exists():
            if sha256_file(archive_path) != previous_bytes_sha256:
                raise FileExistsError(f"pre-implementation source seal archive already differs: {archive_path}")
            SOURCE_SEAL_PATH.unlink()
        else:
            SOURCE_SEAL_PATH.replace(archive_path)
    env = environment_record()
    source_map = source_hashes()
    frozen_v20 = v20_hashes()
    attempt00_root = PACKAGE_ROOT / "results" / "attempt_00"
    attempt00_incident = attempt00_root / "INCIDENT.json"
    attempt00_artifacts = attempt00_root / "INCIDENT_ARTIFACT_HASHES.json"
    if not attempt00_incident.is_file() or not attempt00_artifacts.is_file():
        raise FileNotFoundError("ATTEMPT_00 incident record and hash manifest are required for SOURCE_SEAL_R1")
    attempt00_record = json.loads(attempt00_incident.read_text(encoding="utf-8"))
    seal = {
        "schema": "omega-v2-2a-source-environment-seal-v1",
        "supersedes_preimplementation_source_seal_sha256": previous_seal_sha256,
        "attempt_00_classification": attempt00_record["classification"],
        "attempt_00_incident_sha256": sha256_file(attempt00_incident),
        "attempt_00_artifact_manifest_sha256": sha256_file(attempt00_artifacts),
        "spec_sha256": sha256_file(PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md"),
        "python_source_sha256": source_map,
        "v2_0_source_and_ledger_sha256": frozen_v20,
        "environment": env,
        "environment_matches_md317": environment_matches(env),
        "fixed_seeds": {
            "MASTER_SEED": MASTER_SEED,
            "WEIGHT_SEED": WEIGHT_SEED,
            "INPUT_SEED": INPUT_SEED,
            "LOSS_W_SEED": LOSS_W_SEED,
            "TARGET_SEED": TARGET_SEED,
        },
        "official_preflight_started": False,
        "cuda_kernel_launches_during_source_seal": 0,
    }
    write_json(SOURCE_SEAL_PATH, seal)
    return seal


def _measure_cell(name: str, action: Callable[[], Any], started: float) -> tuple[Any | None, dict[str, Any]]:
    if time.perf_counter() - started > MAX_WALL_SECONDS:
        raise HardStop("V2-2A total wall-time limit exceeded before next cell")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        value = action()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        within_budget = peak_allocated <= VRAM_BUDGET_BYTES
        return value, {
            "cell": name,
            "status": "OK" if within_budget else "OOM_CAPACITY_DIAGNOSTIC",
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "within_3_gib_allocated_budget": within_budget,
            "wall_seconds": elapsed,
        }
    except torch.cuda.OutOfMemoryError as error:
        try:
            torch.cuda.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            peak_allocated = None
            peak_reserved = None
        torch.cuda.empty_cache()
        return None, {
            "cell": name,
            "status": "OOM_CAPACITY_DIAGNOSTIC",
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "within_3_gib_allocated_budget": False,
            "wall_seconds": time.perf_counter() - t0,
            "error": str(error),
        }


def _gate_row(name: str, report: dict[str, Any] | None, memory: dict[str, Any]) -> dict[str, Any]:
    if report is None:
        return {"gate": name, "status": memory["status"], "pass": None, "architectural_fail": False, "report": None, "memory": memory}
    return {"gate": name, "status": "PASS" if report.get("pass") is True else "FAIL", "pass": report.get("pass") is True, "report": report, "memory": memory}


def _new_cuda_copy(module):
    import copy
    return copy.deepcopy(module).to(device="cuda")


def _official_run() -> dict[str, Any]:
    if not SOURCE_SEAL_PATH.is_file():
        raise RuntimeError("V2_2A_SOURCE_SEAL_REQUIRED: run --seal-source and commit before official preflight")
    source_seal = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
    if source_seal["spec_sha256"] != sha256_file(PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md") or source_seal["python_source_sha256"] != source_hashes():
        raise RuntimeError("V2_2A_SOURCE_SEAL_MISMATCH: sources changed after source seal")
    if not source_seal["environment_matches_md317"]:
        raise RuntimeError("V2_2A_ENVIRONMENT_INVALID: source-sealed environment does not match MD/317")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"official V2-2A result slot already exists: {RESULTS_ROOT}")

    current_environment = environment_record()
    if current_environment != source_seal["environment"]:
        raise RuntimeError("V2_2A_ENVIRONMENT_SEAL_MISMATCH: current torch/CUDA/driver/GPU settings differ from source seal")
    run_started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "omega-v2-2a-local-preflight-v1",
        "phase": "OMEGA-V2-2A-LOCAL-GPU-PREFLIGHT",
        "status": "RUNNING",
        "source_seal_sha256": sha256_file(SOURCE_SEAL_PATH),
        "environment": source_seal["environment"],
        "gates": {},
        "cuda_cells": [],
        "execution_order": ["environment/config", "parameter/storage", "CPU-CUDA correctness", "R4-U4 parity", "gradient identity", "iso-FLOP", "K-flex", "R4 smoke", "U4 smoke", "VRAM/runtime", "conformance"],
        "official_preflight_started": True,
        "runpod": "HOLD",
        "T3": "HOLD",
    }
    gate_results: list[dict[str, Any]] = []
    hard_stop_reason: str | None = None
    try:
        # 1. Environment/config seal is checked before the first tensor/model CUDA copy.
        result["gates"]["environment_config"] = {"status": "PASS", "pass": True, "source_seal_sha256": result["source_seal_sha256"]}

        # 2. Parameter/storage ledger is introspected on CPU before device copies.
        pair = build_initial_variants(device="cpu")
        parameter_cpu = parameter_storage_gate(pair["r4_cpu"], pair["u4_cpu"])
        result["gates"]["parameter_storage"] = parameter_cpu
        gate_results.append({"gate": "D4_parameter_storage_cpu", "status": "PASS" if parameter_cpu["pass"] else "FAIL", "pass": parameter_cpu["pass"], "report": parameter_cpu})

        inputs = make_fixed_inputs()
        if not all(inputs["cpu_to_cuda_copy_bitwise"].values()):
            result["gates"]["fixed_tensor_copy"] = {"status": "FAIL", "pass": False, **inputs["cpu_to_cuda_copy_bitwise"]}
        else:
            result["gates"]["fixed_tensor_copy"] = {"status": "PASS", "pass": True, **inputs["cpu_to_cuda_copy_bitwise"]}
        gate_results.append({"gate": "fixed_tensor_copy", "status": result["gates"]["fixed_tensor_copy"]["status"], "pass": result["gates"]["fixed_tensor_copy"]["pass"], "report": result["gates"]["fixed_tensor_copy"]})

        r4_cuda = _new_cuda_copy(pair["r4_cpu"])
        u4_cuda = _new_cuda_copy(pair["u4_cpu"])
        x_cuda = inputs["x_cuda"]
        weights_copy = weight_copy_report(pair["r4_cpu"], pair["u4_cpu"], r4_cuda, u4_cuda)
        weight_copy_pass = weights_copy["all_bitwise_equal"]
        result["gates"]["cpu_to_cuda_weight_copy"] = {**weights_copy, "status": "PASS" if weight_copy_pass else "FAIL", "pass": weight_copy_pass}
        gate_results.append({"gate": "cpu_to_cuda_weight_copy", "status": "PASS" if weight_copy_pass else "FAIL", "pass": weight_copy_pass, "report": weights_copy})

        # D4 device storage identity is introspection-only and precedes numerical gates.
        parameter_gpu = parameter_storage_gate(r4_cuda, u4_cuda)
        result["gates"]["parameter_storage_cuda"] = parameter_gpu
        gate_results.append({"gate": "D4_parameter_storage_cuda", "status": "PASS" if parameter_gpu["pass"] else "FAIL", "pass": parameter_gpu["pass"], "report": parameter_gpu})

        # 3. CPU↔CUDA correctness: K1 and K4 are separate VRAM-accounted cells.
        correctness_cells: list[dict[str, Any]] = []
        result["gates"]["cuda_correctness"] = {"schema": "omega-v2-2a-cuda-correctness-gate-v1", "cells": correctness_cells, "pass": None}
        for k in (1, 4):
            def action(k=k):
                report = cuda_correctness_gate(pair["r4_cpu"].recurrent.block, r4_cuda.recurrent.block, inputs["x_cpu"], x_cuda, k_values=(k,))
                return report
            report, memory = _measure_cell(f"D1_R4_K{k}", action, run_started)
            result["cuda_cells"].append(memory)
            if report is not None:
                correctness_cells.extend(report["cells"])
            gate_results.append(_gate_row(f"D1_R4_K{k}", report, memory))
            result["gates"]["cuda_correctness"]["pass"] = all(row.get("pass") is True for row in gate_results if row["gate"].startswith("D1_"))

        # 4. R4/U4 initial parity and K4 traces.
        parity_report, memory = _measure_cell(
            "D2_R4_U4_K4_PARITY",
            lambda: initialization_and_trace_gate(r4_cuda, u4_cuda, x_cuda, cpu_to_cuda_weights_bitwise_equal=weight_copy_pass),
            run_started,
        )
        result["cuda_cells"].append(memory)
        gate_results.append(_gate_row("D2_init_trace_parity", parity_report, memory))
        result["gates"]["init_trace_parity"] = parity_report or {"status": "OOM_CAPACITY_DIAGNOSTIC", "pass": False}

        # 5. Gradient-sharing identity at the initial graph.
        gradient_report, memory = _measure_cell("D3_GRADIENT_IDENTITY_K4", lambda: gradient_sharing_gate(r4_cuda, u4_cuda, x_cuda, inputs["loss_weights_cuda"]), run_started)
        result["cuda_cells"].append(memory)
        gate_results.append(_gate_row("D3_gradient_sharing", gradient_report, memory))
        result["gates"]["gradient_sharing"] = gradient_report or {"status": "OOM_CAPACITY_DIAGNOSTIC", "pass": False}

        # 6. Integer iso-FLOP ledger from V2-0 functions; no timing.
        flop_report = iso_flop_gate(pair["r4_cpu"], pair["u4_cpu"])
        result["gates"]["iso_flop"] = flop_report
        gate_results.append({"gate": "D5_iso_flop", "status": "PASS" if flop_report["pass"] else "FAIL", "pass": flop_report["pass"], "report": flop_report})

        # 7. K-flex; every forward and backward K is its own memory cell.
        r4_block_cuda = r4_cuda.recurrent.block
        k_flex_rows: list[dict[str, Any]] = []
        result["gates"]["k_flex"] = {"schema": "omega-v2-2a-k-flex-v1", "rows": k_flex_rows, "pass": None}
        for k in (1, 2, 4, 8, 16):
            cell, memory = _measure_cell(f"D6_KFLEX_FWD_K{k}", lambda k=k: k_flex_forward_cell(r4_block_cuda, x_cuda, k), run_started)
            result["cuda_cells"].append(memory)
            if cell is None:
                cell = {"phase": "forward", "K": k, "status": memory["status"], "pass": None}
            else:
                cell_pass = cell.get("output_finite") is True and cell.get("schema_unchanged") is True and cell.get("value_unchanged") is True and cell.get("parameter_count_unchanged") is True
                cell["pass"] = cell_pass
            k_flex_rows.append(cell)
            result["gates"]["k_flex"]["rows"] = k_flex_rows
        for k in (1, 4, 8, 16):
            cell, memory = _measure_cell(f"D6_KFLEX_BWD_K{k}", lambda k=k: k_flex_backward_cell(r4_block_cuda, x_cuda, inputs["loss_weights_cuda"], k), run_started)
            result["cuda_cells"].append(memory)
            if cell is None:
                cell = {"phase": "backward", "K": k, "status": memory["status"], "pass": None}
            else:
                cell["pass"] = cell.get("finite") is True and cell.get("schema_unchanged") is True and cell.get("value_unchanged") is True and cell.get("parameter_count_unchanged") is True
            k_flex_rows.append(cell)
            result["gates"]["k_flex"]["rows"] = k_flex_rows
        k_flex_pass = all(row.get("pass") is True for row in k_flex_rows)
        k_flex_capacity_diag = any(row.get("pass") is None for row in k_flex_rows)
        kflex_report = {"schema": "omega-v2-2a-k-flex-v1", "rows": k_flex_rows, "SCHEMA_SHA256_final": schema_sha256(r4_block_cuda), "VALUE_SHA256_final": state_dict_sha256(r4_block_cuda), "parameter_count_final": parameter_count(r4_block_cuda), "pass": k_flex_pass}
        result["gates"]["k_flex"] = kflex_report
        gate_results.append({"gate": "D6_k_flex", "status": "PASS" if k_flex_pass else ("OOM_CAPACITY_DIAGNOSTIC" if k_flex_capacity_diag else "FAIL"), "pass": k_flex_pass if not k_flex_capacity_diag else None, "architectural_fail": not k_flex_capacity_diag and not k_flex_pass, "report": kflex_report})

        # 8. R4 optimizer smoke; fresh model from the same CPU initialization.
        r4_smoke = _new_cuda_copy(pair["r4_cpu"])
        r4_smoke_report, memory = _measure_cell("D7_R4_SMOKE_20_UPDATES", lambda: optimizer_smoke(r4_smoke, x_cuda, inputs["target_cuda"], label="R4"), run_started)
        result["cuda_cells"].append(memory)
        gate_results.append(_gate_row("D7_R4_optimizer_smoke", r4_smoke_report, memory))
        result["gates"]["R4_smoke"] = r4_smoke_report or {"status": "OOM_CAPACITY_DIAGNOSTIC", "pass": False}

        # 9. U4 optimizer smoke; fresh model, independent optimizer, no R4/U4 loss comparison.
        u4_smoke = _new_cuda_copy(pair["u4_cpu"])
        u4_smoke_report, memory = _measure_cell("D7_U4_SMOKE_20_UPDATES", lambda: optimizer_smoke(u4_smoke, x_cuda, inputs["target_cuda"], label="U4"), run_started)
        result["cuda_cells"].append(memory)
        gate_results.append(_gate_row("D7_U4_optimizer_smoke", u4_smoke_report, memory))
        result["gates"]["U4_smoke"] = u4_smoke_report or {"status": "OOM_CAPACITY_DIAGNOSTIC", "pass": False}

    except HardStop as error:
        hard_stop_reason = str(error)
    except torch.cuda.OutOfMemoryError as error:
        hard_stop_reason = f"unrecoverable OOM outside a recoverable cell: {error}"
    except RuntimeError as error:
        hard_stop_reason = f"CUDA/runtime hard-stop: {error}"

    if hard_stop_reason:
        required_rows = (
            "D1_R4_K1", "D1_R4_K4", "D2_init_trace_parity", "D3_gradient_sharing",
            "D4_parameter_storage_cpu", "fixed_tensor_copy", "cpu_to_cuda_weight_copy", "D4_parameter_storage_cuda",
            "D5_iso_flop", "D6_k_flex", "D7_R4_optimizer_smoke", "D7_U4_optimizer_smoke", "D8_vram_wall_contract",
        )
        existing_rows = {row["gate"] for row in gate_results}
        for name in required_rows:
            if name not in existing_rows:
                row = {"gate": name, "status": "NOT_RUN_DUE_TO_HARD_STOP", "pass": None, "architectural_fail": False}
                gate_results.append(row)
                result["gates"].setdefault(name, row)

    # 10. VRAM/runtime summary. Capacity-only diagnostic rows are not marked as architectural FAIL.
    capacity_diagnostics = [row for row in result["cuda_cells"] if row.get("status") == "OOM_CAPACITY_DIAGNOSTIC"]
    result["gates"]["vram_runtime"] = {
        "status": "OOM_CAPACITY_DIAGNOSTIC" if capacity_diagnostics else ("PASS" if not hard_stop_reason else "NOT_RUN_DUE_TO_HARD_STOP"),
        "allocated_budget_bytes": VRAM_BUDGET_BYTES,
        "cells": result["cuda_cells"],
        "capacity_diagnostic_count": len(capacity_diagnostics),
        "total_wall_seconds": time.perf_counter() - run_started,
        "total_wall_limit_seconds": MAX_WALL_SECONDS,
        "within_wall_limit": time.perf_counter() - run_started <= MAX_WALL_SECONDS,
    }
    vram_pass = not capacity_diagnostics and result["gates"]["vram_runtime"]["within_wall_limit"] and hard_stop_reason is None
    gate_results.append({
        "gate": "D8_vram_wall_contract",
        "status": "PASS" if vram_pass else ("OOM_CAPACITY_DIAGNOSTIC" if capacity_diagnostics else ("NOT_RUN_DUE_TO_HARD_STOP" if hard_stop_reason else "FAIL")),
        "pass": vram_pass,
        "architectural_fail": False if capacity_diagnostics else not vram_pass,
        "report": result["gates"]["vram_runtime"],
    })
    result["gate_results"] = gate_results
    result["hard_stop_reason"] = hard_stop_reason
    result["terminal_status"] = (
        "OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL"
        if hard_stop_reason or any(row.get("pass") is False for row in gate_results) or result["gates"]["vram_runtime"]["within_wall_limit"] is False
        else "OMEGA_V2_2A_LOCAL_PREFLIGHT_PASS"
    )
    result["status"] = result["terminal_status"]
    result["residency_verdict"] = "NOT_ESTABLISHED"
    return result


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# OMEGA-V2-2A Local CUDA Preflight", "",
        f"- terminal_status: `{result['terminal_status']}`",
        f"- source_seal_sha256: `{result['source_seal_sha256']}`",
        f"- wall_seconds: `{result['gates']['vram_runtime']['total_wall_seconds']:.6f}`",
        f"- residency_verdict: `{result['residency_verdict']}`", "", "## Ordered gate results", "",
        "| Gate | Status | Pass |", "|---|---|---:|",
    ]
    for row in result.get("gate_results", []):
        lines.append(f"| {row['gate']} | {row['status']} | {row['pass']} |")
    lines.extend(["", "## Gate reports", "", "```json", json.dumps(result["gates"], sort_keys=True, indent=2), "```", ""])
    if result.get("hard_stop_reason"):
        lines.extend(["## Hard stop", "", result["hard_stop_reason"], ""])
    return "\n".join(lines)


def _seal_official_result(result: dict[str, Any]) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    report_path = RESULTS_ROOT / "OMEGA_V2_2A_REPORT.md"
    report_path.write_text(_report_markdown(result), encoding="utf-8", newline="\n")
    report_sha = sha256_file(report_path)
    sidecar = RESULTS_ROOT / "OMEGA_V2_2A_REPORT.md.sha256"
    sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")
    result_path = RESULTS_ROOT / "preflight_result.json"
    write_json(result_path, result)
    artifact_paths = [PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md", SOURCE_SEAL_PATH, report_path, sidecar, result_path]
    artifact_paths.extend(sorted(PACKAGE_ROOT.rglob("*.py")))
    artifact_paths.extend(Path(path) for path in v20_hashes())
    artifact_hashes = {
        str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in artifact_paths if path.is_file()
    }
    manifest_path = RESULTS_ROOT / "artifact_hashes.json"
    write_json(manifest_path, {"schema": "omega-v2-2a-artifact-hashes-v1", "terminal_status": result["terminal_status"], "report_sha256": report_sha, "artifacts": artifact_hashes})
    verified = all(Path(path).is_file() and sha256_file(Path(path)) == row["sha256"] for path, row in artifact_hashes.items())
    write_json(RESULTS_ROOT / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(artifact_hashes)})
    return {"report_sha256": report_sha, "artifact_manifest_sha256": sha256_file(manifest_path), "artifact_count": len(artifact_hashes), "verified": verified}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--seal-source", action="store_true", help="minimal environment query plus source/spec hash seal; no conformance gates")
    modes.add_argument("--official-preflight", action="store_true", help="execute V2-2A only with explicit --go-v2-2a")
    parser.add_argument("--go-v2-2a", action="store_true", help="acknowledge the judge's explicit V2-2A GO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seal_source:
        seal = create_source_seal()
        print(json.dumps({"phase": "V2_2A_SOURCE_ENVIRONMENT_SEAL", "spec_sha256": seal["spec_sha256"], "source_sha256": seal["python_source_sha256"], "environment": seal["environment"], "environment_matches_md317": seal["environment_matches_md317"], "official_preflight_started": False, "cuda_kernel_launches": 0}, indent=2, sort_keys=True))
        return 0 if seal["environment_matches_md317"] else 1
    if not args.go_v2_2a:
        raise RuntimeError("V2_2A_STOP: official GPU preflight requires the judge's explicit GO after source seal review")
    result = _official_run()
    seal = _seal_official_result(result)
    print(json.dumps({"terminal_status": result["terminal_status"], "report_sha256": seal["report_sha256"], "artifact_manifest_sha256": seal["artifact_manifest_sha256"], "artifact_count": seal["artifact_count"], "artifacts_verified": seal["verified"], "official_preflight_started": True}, indent=2, sort_keys=True))
    return 0 if seal["verified"] and result["terminal_status"] == "OMEGA_V2_2A_LOCAL_PREFLIGHT_PASS" else 1
