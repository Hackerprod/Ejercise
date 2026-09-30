"""V2-0 R4/U4 implementations, CUDA D3Q execution, and CPU FP64 oracle."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES
from omega_v2.variants import R4Shared, U4Untied, clone_value_and_storage_report
from omega_v2_2a_cuda_preflight.core import configure_v2_2a_execution, cuda_rounds, shared_cuda_rounds
from omega_v2_2a_cuda_preflight.variants import BATCH, D, M, make_cpu_normal, weight_copy_report

from .config import K, SeedPlan
from .metrics import (
    all_finite,
    evaluate_oracle_gate,
    evaluate_primary_gates,
    old_max_rel_diagnostic,
    sum_u4_diagnostics,
)


def configure_d3q_execution() -> None:
    """Apply the deterministic V2-2A CUDA environment without changing equations."""
    configure_v2_2a_execution()
    torch.set_num_threads(1)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("D3Q deterministic algorithms must be ON")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("D3Q TF32 must be OFF")
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("D3Q CUDA AMP/autocast must be OFF")


def make_cpu_normal_for_plan(shape: tuple[int, ...], seed: int) -> Tensor:
    return make_cpu_normal(shape, seed)


def create_cpu_initial_variants(weight_seed: int) -> dict[str, Any]:
    """Create R4 on CPU, then make four independent bitwise U4 clones."""
    r4_cpu = R4Shared.seeded(D, weight_seed, dtype=torch.float32)
    u4_cpu = U4Untied.from_shared(r4_cpu)
    clone_report = clone_value_and_storage_report(r4_cpu, u4_cpu)
    return {"r4_cpu": r4_cpu, "u4_cpu": u4_cpu, "clone_report": clone_report}


def _state_copy_checks(initial: dict[str, Any], r4_cuda: R4Shared, u4_cuda: U4Untied) -> dict[str, Any]:
    return weight_copy_report(initial["r4_cpu"], initial["u4_cpu"], r4_cuda, u4_cuda)


def _cpu_fp64_oracle(
    *,
    r4_cpu: R4Shared,
    u4_cpu: U4Untied,
    x_fp32_cpu: Tensor,
    w_fp32_cpu: Tensor,
) -> dict[str, Any]:
    """Promote exact FP32 initialized weights/data, then run V2-0 on CPU FP64."""
    r4_64 = copy.deepcopy(r4_cpu).to(device="cpu", dtype=torch.float64)
    u4_64 = copy.deepcopy(u4_cpu).to(device="cpu", dtype=torch.float64)
    x64 = x_fp32_cpu.to(device="cpu", dtype=torch.float64)
    w64 = w_fp32_cpu.to(device="cpu", dtype=torch.float64)
    initial_weights_exact = all(
        torch.equal(getattr(r4_cpu.recurrent.block, family).double(), getattr(r4_64.recurrent.block, family))
        for family in MATRIX_FAMILIES
    ) and all(
        torch.equal(
            getattr(u4_cpu.blocks[round_index], family).double(),
            getattr(u4_64.blocks[round_index], family),
        )
        for round_index in range(K)
        for family in MATRIX_FAMILIES
    )
    x_exact = torch.equal(x_fp32_cpu.double(), x64)
    w_exact = torch.equal(w_fp32_cpu.double(), w64)

    r4_state = x64
    for _ in range(K):
        r4_state = r4_64.recurrent.block.step(r4_state)
    r4_loss = torch.sum(r4_state * w64)
    r4_grads = torch.autograd.grad(
        r4_loss,
        tuple(getattr(r4_64.recurrent.block, family) for family in MATRIX_FAMILIES),
    )

    u4_state = x64
    for block in u4_64.blocks:
        u4_state = block.step(u4_state)
    u4_loss = torch.sum(u4_state * w64)
    u4_parameters = [
        getattr(u4_64.blocks[round_index], family)
        for round_index in range(K)
        for family in MATRIX_FAMILIES
    ]
    u4_grads = torch.autograd.grad(u4_loss, tuple(u4_parameters))

    oracle_families: dict[str, dict[str, Tensor]] = {}
    oracle_metrics: dict[str, Any] = {}
    for family_index, family in enumerate(MATRIX_FAMILIES):
        g_r64 = r4_grads[family_index].detach().clone()
        g_u64 = [
            u4_grads[round_index * len(MATRIX_FAMILIES) + family_index].detach().clone()
            for round_index in range(K)
        ]
        sum_u64 = (((g_u64[0] + g_u64[1]) + g_u64[2]) + g_u64[3])
        oracle_families[family] = {
            "gR64_cpu": g_r64.detach().cpu().contiguous(),
            **{f"gU{round_index}_64_cpu": grad.detach().cpu().contiguous() for round_index, grad in enumerate(g_u64)},
            "S64_cpu": sum_u64.detach().cpu().contiguous(),
        }
        oracle_metrics[family] = evaluate_oracle_gate(g_r64, sum_u64)

    return {
        "families": oracle_families,
        "metrics": oracle_metrics,
        "copy_checks": {
            "initial_FP32_weights_to_FP64_exact_value_preserved": bool(initial_weights_exact),
            "x_FP32_to_FP64_exact_value_preserved": bool(x_exact),
            "w_FP32_to_FP64_exact_value_preserved": bool(w_exact),
        },
        "x_fp64_cpu": x64.detach().contiguous(),
        "w_fp64_cpu": w64.detach().contiguous(),
    }


def _cpu_tensor_bundle(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_tensor_bundle(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_bundle(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensor_bundle(item) for item in value)
    return value


def run_seed_cuda(plan: SeedPlan, mark_cuda_cell_started: Callable[[], None]) -> dict[str, Any]:
    """Execute one D3Q seed; this is reachable only through GO-gated smoke/official runners."""
    if plan.mode not in ("official", "calibration_smoke"):
        raise ValueError("D3Q CUDA seed executor accepts only official or calibration_smoke plans")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    initial = create_cpu_initial_variants(plan.weight_seed)
    x_cpu = make_cpu_normal_for_plan((BATCH, M, D), plan.input_seed)
    w_cpu = make_cpu_normal_for_plan((BATCH, M, D), plan.loss_w_seed)

    r4_cuda = copy.deepcopy(initial["r4_cpu"]).to(device="cuda")
    u4_cuda = copy.deepcopy(initial["u4_cpu"]).to(device="cuda")
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    w_cuda = w_cpu.to(device="cuda", dtype=torch.float32)
    weight_copies = _state_copy_checks(initial, r4_cuda, u4_cuda)
    input_copy_exact = torch.equal(x_cpu, x_cuda.cpu())
    loss_weight_copy_exact = torch.equal(w_cpu, w_cuda.cpu())

    # Mark the irreversible boundary immediately before the first held-out CUDA numerical cell.
    mark_cuda_cell_started()
    r4_output, r4_trace = shared_cuda_rounds(
        r4_cuda.recurrent.block,
        x_cuda,
        K,
        return_trace=True,
    )
    u4_output, u4_trace = cuda_rounds(
        u4_cuda.block_for_round,
        x_cuda,
        K,
        return_trace=True,
    )
    trace_rows = [
        {"round": round_index + 1, "torch_equal": bool(torch.equal(r4_trace[round_index], u4_trace[round_index]))}
        for round_index in range(K)
    ]
    final_equal = bool(torch.equal(r4_output, u4_output))
    clone_report = initial["clone_report"]
    structural_pass = bool(
        clone_report["initial_values_bitwise_equal"]
        and clone_report["r4_reuses_one_block_object"]
        and clone_report["u4_has_four_block_objects"]
        and clone_report["u4_block_storages_pairwise_disjoint"]
        and clone_report["u4_storage_disjoint_from_r4"]
        and weight_copies["all_bitwise_equal"]
        and input_copy_exact
        and loss_weight_copy_exact
        and all(row["torch_equal"] for row in trace_rows)
        and final_equal
    )
    structural = {
        "initial_clone_report": clone_report,
        "weight_copy_report": weight_copies,
        "input_copy_bitwise_equal": bool(input_copy_exact),
        "loss_weight_copy_bitwise_equal": bool(loss_weight_copy_exact),
        "round_trace_equalities": trace_rows,
        "final_output_equal": final_equal,
        "pass": structural_pass,
    }

    tensor_bundle: dict[str, Any] = {
        "cuda_fp32": {},
        "cpu_fp64_oracle": {},
        "structural_traces": {
            "R4": {f"round_{i + 1}": value.detach().cpu().contiguous() for i, value in enumerate(r4_trace)},
            "U4": {f"round_{i + 1}": value.detach().cpu().contiguous() for i, value in enumerate(u4_trace)},
            "R4_final": r4_output.detach().cpu().contiguous(),
            "U4_final": u4_output.detach().cpu().contiguous(),
        },
        "inputs": {"x_fp32_cpu": x_cpu.contiguous(), "w_fp32_cpu": w_cpu.contiguous()},
    }
    if not structural_pass:
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        return {
            "case_id": str(plan.master_seed),
            "seed_plan": plan.as_dict(),
            "structural": structural,
            "structural_pass": False,
            "finite": False,
            "families": {},
            "tensor_bundle": _cpu_tensor_bundle(tensor_bundle),
            "duration_seconds": time.perf_counter() - started,
            "vram_peak_allocated_bytes": peak_allocated,
            "vram_peak_reserved_bytes": peak_reserved,
            "terminal_status": "SEED_FAIL_STRUCTURAL",
            "seed_pass": False,
        }

    r4_loss = torch.sum(r4_output * w_cuda)
    r4_grads = torch.autograd.grad(
        r4_loss,
        tuple(getattr(r4_cuda.recurrent.block, family) for family in MATRIX_FAMILIES),
    )
    u4_loss = torch.sum(u4_output * w_cuda)
    u4_parameters = [
        getattr(u4_cuda.blocks[round_index], family)
        for round_index in range(K)
        for family in MATRIX_FAMILIES
    ]
    u4_grads = torch.autograd.grad(u4_loss, tuple(u4_parameters))

    oracle = _cpu_fp64_oracle(
        r4_cpu=initial["r4_cpu"],
        u4_cpu=initial["u4_cpu"],
        x_fp32_cpu=x_cpu,
        w_fp32_cpu=w_cpu,
    )
    families: dict[str, Any] = {}
    cuda_raw_bundle: dict[str, dict[str, Tensor]] = {}
    oracle_raw_bundle = oracle["families"]
    seeds_and_copy_finite = all_finite(x_cpu, w_cpu)
    for family_index, family in enumerate(MATRIX_FAMILIES):
        g_r32 = r4_grads[family_index].detach().clone()
        g_u32 = [
            u4_grads[round_index * len(MATRIX_FAMILIES) + family_index].detach().clone()
            for round_index in range(K)
        ]
        sums32 = sum_u4_diagnostics(g_u32)
        g_r64 = g_r32.to(dtype=torch.float64)
        g_u64 = [grad.to(dtype=torch.float64) for grad in g_u32]
        s64 = (((g_u64[0] + g_u64[1]) + g_u64[2]) + g_u64[3])
        primary = evaluate_primary_gates(g_r64, s64)
        old_relative = old_max_rel_diagnostic(g_r32, sums32["S_stack"])
        old_relative["S_reverse_equal_gR_NON_GATE"] = bool(torch.equal(sums32["S_reverse"], g_r32))
        raw_grads = {"gR_cuda_fp32": g_r32, **{f"gU{index}_cuda_fp32": value for index, value in enumerate(g_u32)}}
        family_bundle = {
            **raw_grads,
            **{f"{name}_cuda": value.detach().clone() for name, value in sums32.items()},
            "gR64_primary_cuda": g_r64.detach().clone(),
            "S64_primary_cuda": s64.detach().clone(),
        }
        cuda_raw_bundle[family] = family_bundle
        oracle_family = oracle_raw_bundle[family]
        oracle_raw_bundle[family] = {
            **oracle_family,
        }
        oracle_finite = all_finite(*oracle_family.values())
        cuda_finite = all_finite(*family_bundle.values())
        oracle_gate = oracle["metrics"][family]
        family_pass = bool(primary["finite"] and primary["scientific_gates_pass"] and oracle_finite and oracle_gate["pass"])
        families[family] = {
            "primary_S64": primary,
            "diagnostics_NON_GATE": {
                "old_max_rel": old_relative,
                "fp32_sum_names": list(sums32),
                "S_reverse_equal_gR_NON_GATE": old_relative["S_reverse_equal_gR_NON_GATE"],
            },
            "cpu_fp64_oracle": oracle_gate,
            "operational": {"all_cuda_family_tensors_finite": bool(cuda_finite), "all_cpu_oracle_tensors_finite": bool(oracle_finite)},
            "seed_family_pass": family_pass,
        }
        seeds_and_copy_finite = seeds_and_copy_finite and cuda_finite and oracle_finite and primary["finite"]

    tensor_bundle["cuda_fp32"] = cuda_raw_bundle
    tensor_bundle["cpu_fp64_oracle"] = oracle_raw_bundle
    copy_checks = oracle["copy_checks"]
    oracle_copy_pass = all(copy_checks.values())
    oracle_pass = all(item["cpu_fp64_oracle"]["pass"] for item in families.values())
    seed_pass = bool(structural_pass and seeds_and_copy_finite and oracle_copy_pass and oracle_pass and all(row["seed_family_pass"] for row in families.values()))
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "case_id": str(plan.master_seed),
        "seed_plan": plan.as_dict(),
        "structural": structural,
        "structural_pass": structural_pass,
        "cpu_fp64_copy_checks": copy_checks,
        "families": families,
        "tensor_bundle": _cpu_tensor_bundle(tensor_bundle),
        "finite": bool(seeds_and_copy_finite),
        "cpu_fp64_oracle_all_pass": bool(oracle_pass),
        "duration_seconds": time.perf_counter() - started,
        "vram_peak_allocated_bytes": peak_allocated,
        "vram_peak_reserved_bytes": peak_reserved,
        "terminal_status": "SEED_PASS" if seed_pass else "SEED_FAIL",
        "seed_pass": seed_pass,
    }


def run_calibration_seed_cpu_qa(master_seed: int) -> dict[str, Any]:
    """CPU-only harness QA; the QA guard admits only calibration seed 20260930."""
    from .config import make_seed_plan

    plan = make_seed_plan(master_seed, mode="qa")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = create_cpu_initial_variants(plan.weight_seed)
    x_cpu = make_cpu_normal_for_plan((BATCH, M, D), plan.input_seed)
    w_cpu = make_cpu_normal_for_plan((BATCH, M, D), plan.loss_w_seed)
    r4_output, r4_trace = initial["r4_cpu"](x_cpu, return_trace=True)
    u4_output, u4_trace = initial["u4_cpu"](x_cpu, return_trace=True)
    trace_equal = [bool(torch.equal(r4_trace[i], u4_trace[i])) for i in range(K)]
    final_equal = bool(torch.equal(r4_output, u4_output))
    r4_loss = torch.sum(r4_output * w_cpu)
    u4_loss = torch.sum(u4_output * w_cpu)
    r4_grads = torch.autograd.grad(r4_loss, tuple(initial["r4_cpu"].parameters()))
    u4_grads = torch.autograd.grad(u4_loss, tuple(initial["u4_cpu"].parameters()))
    return {
        "mode": "QA_CALIBRATION_ONLY",
        "seed_plan": plan.as_dict(),
        "structural": initial["clone_report"],
        "round_trace_equalities": trace_equal,
        "final_output_equal": final_equal,
        "all_fp32_gradients_finite": all_finite(*r4_grads, *u4_grads),
        "cuda_used": False,
        "D3Q_verdict": None,
        "pass": bool(
            initial["clone_report"]["initial_values_bitwise_equal"]
            and all(trace_equal)
            and final_equal
            and all_finite(*r4_grads, *u4_grads)
        ),
    }
