"""D3 calibration run analysis, raw gradient persistence, and read-only R1 cross-check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from omega_v2.core import MATRIX_FAMILIES

from . import ATTEMPT00_ROOT, V22A_R1_RESULT
from .core import (
    BATCH,
    D,
    INPUT_SEED,
    LOSS_W_SEED,
    M,
    MASTER_SEED,
    WEIGHT_SEED,
    create_cpu_fp64_oracle,
    create_cuda_d3_run,
)
from .metrics import diagnostic_metrics, file_sha256, old_comparator_argmax_details, tensor_raw_sha256


SUM_NAMES = ("S_forward", "S_reverse", "S_pairwise", "S_stack", "S_fp64")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def _raw_tensor_hashes(bundle: dict[str, Any], run_prefix: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for family, tensors in bundle.items():
        for tensor_name, tensor in tensors.items():
            logical_name = f"{run_prefix}/{family}/{tensor_name}"
            hashes[logical_name] = tensor_raw_sha256(logical_name, tensor)
    return hashes


def _tensor_metadata(bundle: dict[str, Any], run_prefix: str) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for family, tensors in bundle.items():
        for tensor_name, tensor in tensors.items():
            logical_name = f"{run_prefix}/{family}/{tensor_name}"
            metadata[logical_name] = {
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype),
                "raw_sha256": tensor_raw_sha256(logical_name, tensor),
            }
    return metadata


def _run_reproducibility(run1: dict[str, Any], run2: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for family in MATRIX_FAMILIES:
        for tensor_name in ("gR", "gU0", "gU1", "gU2", "gU3"):
            equal = torch.equal(run1["families_cuda"][family][tensor_name], run2["families_cuda"][family][tensor_name])
            rows.append({"family": family, "tensor": tensor_name, "torch_equal": equal})
    rows.append({"family": "initial_weights", "tensor": "V2_0_state_dict_sha256", "torch_equal": run1["initial_weight_state_dict_sha256"] == run2["initial_weight_state_dict_sha256"]})
    rows.append({"family": "initial_weights", "tensor": "R4_U4_initial_clone", "torch_equal": run1["initial_clone_report"]["initial_values_bitwise_equal"] and run2["initial_clone_report"]["initial_values_bitwise_equal"]})
    rows.append({"family": "initial_state", "tensor": "run_02_compared_to_run_01_before_forward", "torch_equal": bool(run2["initial_comparison_to_run_01"]["all_bitwise_equal"])})
    rows.append({"family": "fixed_inputs", "tensor": "x", "torch_equal": torch.equal(run1["x_cpu"], run2["x_cpu"])})
    rows.append({"family": "fixed_inputs", "tensor": "w", "torch_equal": torch.equal(run1["w_cpu"], run2["w_cpu"])})
    rows.append({"family": "initial_copy", "tensor": "run_01", "torch_equal": run1["input_copy_bitwise_equal"] and run1["loss_weight_copy_bitwise_equal"] and run1["weight_copy_report"]["all_bitwise_equal"]})
    rows.append({"family": "initial_copy", "tensor": "run_02", "torch_equal": run2["input_copy_bitwise_equal"] and run2["loss_weight_copy_bitwise_equal"] and run2["weight_copy_report"]["all_bitwise_equal"]})
    return {"schema": "omega-v2-2a-d3-reproducibility-v1", "rows": rows, "all_bitwise_equal": all(row["torch_equal"] for row in rows)}


def _metric_matrix(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_family: dict[str, Any] = {}
    stack_metrics: dict[str, dict[str, Any]] = {}
    for family in MATRIX_FAMILIES:
        tensors = run["families_cuda"][family]
        g_r = tensors["gR"]
        by_sum: dict[str, Any] = {}
        for sum_name in SUM_NAMES:
            sum_tensor = tensors[sum_name]
            metrics = diagnostic_metrics(g_r, sum_tensor)
            by_sum[sum_name] = metrics
            if sum_name == "S_stack":
                stack_metrics[family] = metrics
        by_family[family] = {
            "shape": [int(size) for size in g_r.shape],
            "dtype": str(g_r.dtype),
            "raw_tensor_sha256": _raw_tensor_hashes({family: tensors}, "run"),
            "tensor_metadata": _tensor_metadata({family: tensors}, "run"),
            "summation_metrics": by_sum,
            "old_comparator_argmax_details": old_comparator_argmax_details(g_r, tensors["S_stack"], stack_metrics[family]),
        }
    return by_family, stack_metrics


def _v2_2a_r1_crosscheck(run1_stack_metrics: dict[str, dict[str, Any]], result_path: Path = V22A_R1_RESULT) -> dict[str, Any]:
    frozen = json.loads(result_path.read_text(encoding="utf-8"))
    frozen_families = frozen["gates"]["gradient_sharing"]["families"]
    by_name = {row["family"]: row for row in frozen_families}
    rows = []
    for family in MATRIX_FAMILIES:
        current = run1_stack_metrics[family]
        sealed = by_name[family]
        rows.append({
            "family": family,
            "max_abs_equal_exact_float": current["max_abs"] == sealed["max_abs"],
            "max_rel_equal_exact_float": current["max_rel_old"] == sealed["max_rel"],
            "L2_equal_exact_float": current["E_L2"] == sealed["L2_relative_error"],
            "diagnostic_max_abs": current["max_abs"], "r1_max_abs": sealed["max_abs"],
            "diagnostic_max_rel_old": current["max_rel_old"], "r1_max_rel": sealed["max_rel"],
            "diagnostic_E_L2": current["E_L2"], "r1_E_L2": sealed["L2_relative_error"],
        })
    return {
        "schema": "omega-v2-2a-d3-r1-metric-crosscheck-v1",
        "source_result_sha256": file_sha256(result_path),
        "terminal_status": frozen["terminal_status"],
        "read_only": True,
        "metrics": ["max_abs", "max_rel", "L2_relative_error"],
        "comparison": "exact Python float equality, informational only; no PASS/FAIL",
        "rows": rows,
    }


def _fp64_oracle_metrics(cuda_run1: dict[str, Any], cpu64: dict[str, Any]) -> dict[str, Any]:
    rows = []
    cuda = cuda_run1["families_cuda"]
    for family in MATRIX_FAMILIES:
        oracle = cpu64["families"][family]
        g_r64 = oracle["gR64"]
        u64 = [oracle[f"gU{index}_64"] for index in range(4)]
        sum_u64 = (((u64[0] + u64[1]) + u64[2]) + u64[3])
        comparisons = [{"comparison": "CPU_FP64_gR64_vs_sum_gU64", **diagnostic_metrics(g_r64, sum_u64)}]
        comparisons.append({"comparison": "CUDA_gR_vs_CPU_FP64_gR64", **diagnostic_metrics(cuda[family]["gR"].cpu(), g_r64)})
        for index in range(4):
            comparisons.append({"comparison": f"CUDA_gU{index}_vs_CPU_FP64_gU{index}", **diagnostic_metrics(cuda[family][f"gU{index}"].cpu(), u64[index])})
        comparisons.append({"comparison": "CUDA_S_stack_vs_CPU_FP64_sum", **diagnostic_metrics(cuda[family]["S_stack"].cpu(), sum_u64)})
        comparisons.append({"comparison": "CUDA_S_fp64_vs_CPU_FP64_sum", **diagnostic_metrics(cuda[family]["S_fp64"].cpu(), sum_u64)})
        rows.append({"family": family, "comparisons": comparisons})
    return {"schema": "omega-v2-2a-d3-cpu-fp64-arithmetic-oracle-v1", "families": rows, "thresholds": None}


def _persist_tensor_bundle(path: Path, bundle: dict[str, Any], prefix: str) -> dict[str, Any]:
    cpu_bundle = {
        family: {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()}
        for family, tensors in bundle.items()
    }
    torch.save({"schema": "omega-v2-2a-d3-gradient-bundle-v1", "run": prefix, "families": cpu_bundle}, path)
    tensor_hashes = _raw_tensor_hashes(cpu_bundle, prefix)
    return {"path": path.name, "size_bytes": path.stat().st_size, "file_sha256": file_sha256(path), "tensor_raw_sha256": tensor_hashes, "tensor_metadata": _tensor_metadata(cpu_bundle, prefix)}


def run_diagnostic(*, result_root: Path, run_cuda_fn=create_cuda_d3_run, run_cpu_fp64_fn=create_cpu_fp64_oracle) -> dict[str, Any]:
    """Run exactly two D3 CUDA diagnostic executions and persist a non-gating report."""
    if result_root.exists():
        raise FileExistsError(f"D3 diagnostic result slot is immutable: {result_root}")
    result_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bundle_records: dict[str, Any] = {}
    completed_cuda_runs = 0
    try:
        run1 = run_cuda_fn()
        bundle_records["cuda_run_01"] = _persist_tensor_bundle(result_root / "cuda_gradients_run_01.pt", run1["tensor_bundle_cpu"], "run_01")
        completed_cuda_runs = 1
        run2 = run_cuda_fn(reference_run=run1)
        bundle_records["cuda_run_02"] = _persist_tensor_bundle(result_root / "cuda_gradients_run_02.pt", run2["tensor_bundle_cpu"], "run_02")
        completed_cuda_runs = 2
        reproducibility = _run_reproducibility(run1, run2)
        run1_metrics, run1_stack_metrics = _metric_matrix(run1)
        run2_metrics, _ = _metric_matrix(run2)
        r1_crosscheck = _v2_2a_r1_crosscheck(run1_stack_metrics)
        cpu64 = run_cpu_fp64_fn(run1["x_cpu"], run1["w_cpu"])
        cpu64_metrics = _fp64_oracle_metrics(run1, cpu64)
        bundle_records["cpu_fp64"] = _persist_tensor_bundle(result_root / "cpu_fp64_oracle.pt", cpu64["families"], "cpu_fp64")
        metrics = {
            "schema": "omega-v2-2a-d3-diagnostic-metrics-v1",
            "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
            "may_rescue_V2_2A": False,
            "architectural_verdict": None,
            "parameters": {"d": D, "m": M, "B": BATCH, "K": 4, "MASTER_SEED": MASTER_SEED, "WEIGHT_SEED": WEIGHT_SEED, "INPUT_SEED": INPUT_SEED, "LOSS_W_SEED": LOSS_W_SEED},
            "held_out_D3Q_seeds_touched": [],
            "run_01_metrics": run1_metrics,
            "run_02_metrics": run2_metrics,
            "run_01": {"input_copy_bitwise_equal": run1["input_copy_bitwise_equal"], "loss_weight_copy_bitwise_equal": run1["loss_weight_copy_bitwise_equal"], "weight_copy_report": run1["weight_copy_report"], "initial_clone_report": run1["initial_clone_report"], "initial_weight_state_dict_sha256": run1["initial_weight_state_dict_sha256"], "fixed_seeds": run1["fixed_seeds"]},
            "run_02": {"input_copy_bitwise_equal": run2["input_copy_bitwise_equal"], "loss_weight_copy_bitwise_equal": run2["loss_weight_copy_bitwise_equal"], "weight_copy_report": run2["weight_copy_report"], "initial_clone_report": run2["initial_clone_report"], "initial_comparison_to_run_01": run2["initial_comparison_to_run_01"], "initial_weight_state_dict_sha256": run2["initial_weight_state_dict_sha256"], "fixed_seeds": run2["fixed_seeds"]},
            "reproducibility": reproducibility,
            "V2_2A_r1_read_only_crosscheck": r1_crosscheck,
            "cpu_fp64_arithmetic_oracle": cpu64_metrics,
            "cpu_fp64_copy_checks": {key: cpu64[key] for key in ("fp32_to_fp64_initial_weights_bitwise_value_preserved", "fp32_to_fp64_x_exact_value_preserved", "fp32_to_fp64_w_exact_value_preserved")},
            "tensor_bundles": bundle_records,
            "wall_seconds": time.perf_counter() - started,
            "terminal_classification": "DIAGNOSTIC_COMPLETE",
            "official_V2_2A_run_repeated": False,
            "v2_2a_r1_terminal": r1_crosscheck["terminal_status"],
            "v2_2a_r1_modified": False,
        }
        _write_json(result_root / "d3_diagnostic_metrics.json", metrics)
        return metrics
    except Exception as error:
        failure = {
            "schema": "omega-v2-2a-d3-diagnostic-execution-failure-v1",
            "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
            "may_rescue_V2_2A": False,
            "architectural_verdict": None,
            "terminal_classification": "EXECUTION_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_cuda_diagnostic_runs": completed_cuda_runs,
            "tensor_bundles_written": bundle_records,
            "held_out_D3Q_seeds_touched": [],
            "wall_seconds": time.perf_counter() - started,
            "thresholds_applied": False,
        }
        _write_json(result_root / "execution_failure.json", failure)
        return failure
