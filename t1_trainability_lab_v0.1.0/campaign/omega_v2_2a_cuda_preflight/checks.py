"""Deterministic V2-2A gate computations and metadata checks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

import torch
from torch import Tensor

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES, state_dict_values_equal
from omega_v2.ledger import count_one_round_macs, state_dict_sha256
from omega_v2.variants import R4Shared, U4Untied

from .core import GPU_OPERATION_ORDER, cpu_reference_rounds, cuda_rounds, shared_cuda_rounds
from .variants import family_parameters


def tensor_finite(value: Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all().item())


def schema_sha256(module: torch.nn.Module) -> str:
    records = [
        {"name": name, "shape": [int(size) for size in value.shape], "dtype": str(value.dtype)}
        for name, value in sorted(module.state_dict().items())
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_value_sha256(module: torch.nn.Module) -> str:
    return state_dict_sha256(module)


def parameter_count(module: torch.nn.Module) -> int:
    unique: dict[tuple[int, int], torch.nn.Parameter] = {}
    for parameter in module.parameters():
        storage = parameter.untyped_storage()
        unique.setdefault((int(storage.data_ptr()), int(storage.nbytes())), parameter)
    return sum(int(parameter.numel()) for parameter in unique.values())


def compare_cuda_to_cpu(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    ref = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    cand = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate shapes differ: {tuple(ref.shape)} != {tuple(cand.shape)}")
    absolute = (cand - ref).abs()
    allowed = 1e-5 + 1e-4 * ref.abs().clamp_min(1e-6)
    scaled = absolute / allowed
    l2_error = torch.linalg.vector_norm(absolute)
    l2_reference = torch.linalg.vector_norm(ref)
    l2_relative = float((l2_error / torch.maximum(l2_reference, torch.tensor(1e-6))).item())
    max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
    max_scaled = float(scaled.max().item()) if scaled.numel() else 0.0
    return {
        "max_abs": max_abs,
        "max_scaled_error": max_scaled,
        "L2_relative_error": l2_relative,
        "elementwise_pass": max_scaled <= 1.0,
        "L2_pass": l2_relative <= 1e-5,
        "pass": max_scaled <= 1.0 and l2_relative <= 1e-5,
    }


def cuda_correctness_gate(
    cpu_block: ContractualCoreBlock,
    cuda_block: ContractualCoreBlock,
    x_cpu: Tensor,
    x_cuda: Tensor,
    *,
    k_values: tuple[int, ...] = (1, 4),
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for k in k_values:
        if k not in (1, 4):
            raise ValueError("CUDA correctness is preregistered for K={1,4}")
        if k == 1:
            cpu_final = cpu_reference_rounds(cpu_block, x_cpu, 1)
            cuda_final = shared_cuda_rounds(cuda_block, x_cuda, 1)
            cells.append({"K": 1, "item": "final_output", **compare_cuda_to_cpu(cpu_final, cuda_final)})
        else:
            cpu_final, cpu_trace = cpu_reference_rounds(cpu_block, x_cpu, 4, return_trace=True)
            cuda_final, cuda_trace = shared_cuda_rounds(cuda_block, x_cuda, 4, return_trace=True)
            for round_index, (cpu_round, cuda_round) in enumerate(zip(cpu_trace, cuda_trace), start=1):
                cells.append({"K": 4, "item": f"round_trace_{round_index}", **compare_cuda_to_cpu(cpu_round, cuda_round)})
            cells.append({"K": 4, "item": "final_output", **compare_cuda_to_cpu(cpu_final, cuda_final)})
    return {
        "schema": "omega-v2-2a-cuda-correctness-v1",
        "reference": "V2-0 ContractualCoreBlock CPU FP32 exact equations",
        "d": 256,
        "m": 8,
        "B": 8,
        "cells": cells,
        "pass": all(row["pass"] for row in cells),
    }


def _fallback_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float | bool]:
    ref = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    cand = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    absolute = (cand - ref).abs()
    relative = absolute / ref.abs().clamp_min(1e-6)
    max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    return {"max_abs": max_abs, "max_rel": max_rel, "pass": max_abs <= 1e-6 and max_rel <= 1e-5}


def initialization_and_trace_gate(r4: R4Shared, u4: U4Untied, x_cuda: Tensor, *, cpu_to_cuda_weights_bitwise_equal: bool) -> dict[str, Any]:
    initial_values_equal = all(
        torch.equal(getattr(r4.recurrent.block, family), getattr(u4.blocks[0], family))
        for family in MATRIX_FAMILIES
    )
    for block in u4.blocks[1:]:
        initial_values_equal = initial_values_equal and all(
            torch.equal(getattr(u4.blocks[0], family), getattr(block, family))
            for family in MATRIX_FAMILIES
        )

    with torch.no_grad():
        r4_final, r4_trace = shared_cuda_rounds(r4.recurrent.block, x_cuda, 4, return_trace=True)
        u4_final, u4_trace = cuda_rounds(u4.block_for_round, x_cuda, 4, return_trace=True)
    trace_equal = [torch.equal(left, right) for left, right in zip(r4_trace, u4_trace)]
    final_equal = torch.equal(r4_final, u4_final)
    bitwise_pass = cpu_to_cuda_weights_bitwise_equal and initial_values_equal and all(trace_equal) and final_equal
    result: dict[str, Any] = {
        "schema": "omega-v2-2a-r4-u4-init-parity-v1",
        "initial_values_bitwise_equal": initial_values_equal,
        "cpu_to_cuda_weights_bitwise_equal": cpu_to_cuda_weights_bitwise_equal,
        "round_trace_torch_equal": trace_equal,
        "final_output_torch_equal": final_equal,
        "bitwise_pass": bitwise_pass,
        "fallback_used": False,
    }
    if bitwise_pass:
        result["pass"] = True
        return result

    with torch.no_grad():
        r4_final_repeat, r4_trace_repeat = shared_cuda_rounds(r4.recurrent.block, x_cuda, 4, return_trace=True)
        u4_final_repeat, u4_trace_repeat = cuda_rounds(u4.block_for_round, x_cuda, 4, return_trace=True)
        _, _, r4_ops = shared_cuda_rounds(r4.recurrent.block, x_cuda, 4, return_operation_traces=True)
        _, _, u4_ops = cuda_rounds(u4.block_for_round, x_cuda, 4, return_operation_traces=True)
    r4_reproducible = all(torch.equal(a, b) for a, b in zip(r4_trace, r4_trace_repeat)) and torch.equal(r4_final, r4_final_repeat)
    u4_reproducible = all(torch.equal(a, b) for a, b in zip(u4_trace, u4_trace_repeat)) and torch.equal(u4_final, u4_final_repeat)
    first_divergence = None
    for round_index, (left_round, right_round) in enumerate(zip(r4_ops, u4_ops), start=1):
        for operation in GPU_OPERATION_ORDER:
            if not torch.equal(left_round[operation], right_round[operation]):
                first_divergence = {"round": round_index, "operation": operation}
                break
        if first_divergence is not None:
            break
    fallback_rows = [
        {"item": f"round_trace_{index}", **_fallback_metrics(left, right)}
        for index, (left, right) in enumerate(zip(r4_trace, u4_trace), start=1)
    ]
    fallback_rows.append({"item": "final_output", **_fallback_metrics(r4_final, u4_final)})
    fallback_pass = r4_reproducible and u4_reproducible and first_divergence is not None and all(row["pass"] for row in fallback_rows)
    result.update({
        "fallback_used": True,
        "r4_repeat_bitwise": r4_reproducible,
        "u4_repeat_bitwise": u4_reproducible,
        "first_divergence": first_divergence,
        "fallback_metrics": fallback_rows,
        "pass": fallback_pass,
    })
    return result


def gradient_sharing_gate(r4: R4Shared, u4: U4Untied, x_cuda: Tensor, loss_weights_cuda: Tensor) -> dict[str, Any]:
    r4_block = r4.recurrent.block
    r4.zero_grad(set_to_none=True)
    u4.zero_grad(set_to_none=True)
    r4_output = shared_cuda_rounds(r4_block, x_cuda, 4)
    r4_loss = torch.sum(r4_output * loss_weights_cuda)
    r4_grads = torch.autograd.grad(r4_loss, tuple(getattr(r4_block, name) for name in MATRIX_FAMILIES))

    u4_output = cuda_rounds(u4.block_for_round, x_cuda, 4)
    u4_loss = torch.sum(u4_output * loss_weights_cuda)
    u4_blocks_grads = [
        torch.autograd.grad(u4_loss, tuple(getattr(block, name) for name in MATRIX_FAMILIES), retain_graph=index < 3)
        for index, block in enumerate(u4.blocks)
    ]

    families: list[dict[str, Any]] = []
    for family_index, family in enumerate(MATRIX_FAMILIES):
        g_r = r4_grads[family_index]
        g_u = torch.stack([round_grads[family_index] for round_grads in u4_blocks_grads], dim=0).sum(dim=0)
        delta = (g_r - g_u).abs()
        elementwise_denominator = torch.maximum(torch.maximum(g_r.abs(), g_u.abs()), torch.tensor(1e-6, device=g_r.device))
        max_abs = float(delta.max().item())
        max_rel = float((delta / elementwise_denominator).max().item())
        norm_r = torch.linalg.vector_norm(g_r)
        norm_u = torch.linalg.vector_norm(g_u)
        l2_relative = float((torch.linalg.vector_norm(g_r - g_u) / torch.maximum(torch.maximum(norm_r, norm_u), torch.tensor(1e-6, device=g_r.device))).item())
        finite = tensor_finite(g_r) and tensor_finite(g_u)
        family_pass = finite and l2_relative <= 1e-5 and max_abs <= 2e-5 and max_rel <= 2e-4
        families.append({"family": family, "max_abs": max_abs, "max_rel": max_rel, "L2_relative_error": l2_relative, "finite": finite, "pass": family_pass})
    return {"schema": "omega-v2-2a-gradient-sharing-v1", "loss": "sum(y * loss_weights)", "families": families, "pass": all(row["pass"] for row in families)}


def parameter_storage_gate(r4: R4Shared, u4: U4Untied) -> dict[str, Any]:
    r4_block = r4.recurrent.block
    r4_parameters = [getattr(r4_block, family) for family in MATRIX_FAMILIES]
    u4_parameters = [[getattr(block, family) for family in MATRIX_FAMILIES] for block in u4.blocks]
    r4_unique_count = parameter_count(r4_block)
    u4_unique_count = parameter_count(u4)
    r4_storage_ids = [int(parameter.untyped_storage().data_ptr()) for parameter in r4_parameters]
    r4_one_storage_per_family = len(set(r4_storage_ids)) == len(MATRIX_FAMILIES)
    u4_family_storage_ids = [[int(parameter.untyped_storage().data_ptr()) for parameter in block] for block in u4_parameters]
    u4_four_storages_per_family = all(len(set(ids)) == 4 for ids in zip(*u4_family_storage_ids))
    u4_pairwise_disjoint = all(
        not (set(u4_family_storage_ids[i]) & set(u4_family_storage_ids[j]))
        for i in range(4) for j in range(i + 1, 4)
    )
    u4_disjoint_from_r4 = all(not (set(r4_storage_ids) & set(ids)) for ids in u4_family_storage_ids)
    passed = (
        r4_unique_count == 1_048_576
        and u4_unique_count == 4_194_304
        and r4_one_storage_per_family
        and u4_four_storages_per_family
        and u4_pairwise_disjoint
        and u4_disjoint_from_r4
    )
    return {
        "schema": "omega-v2-2a-parameter-storage-v1",
        "R4_unique_parameters": r4_unique_count,
        "U4_unique_parameters": u4_unique_count,
        "R4_one_storage_per_family": r4_one_storage_per_family,
        "U4_four_storages_per_family": u4_four_storages_per_family,
        "U4_storages_pairwise_disjoint": u4_pairwise_disjoint,
        "U4_storage_disjoint_from_R4": u4_disjoint_from_r4,
        "pass": passed,
    }


def iso_flop_gate(r4_cpu: R4Shared, u4_cpu: U4Untied) -> dict[str, Any]:
    block = r4_cpu.recurrent.block
    per_round_b1 = count_one_round_macs(block, m=8, batch_size=1)
    per_round_b8 = count_one_round_macs(block, m=8, batch_size=8)
    r4_flops = 4 * per_round_b8["forward_flops_per_round"]
    u4_rows = [count_one_round_macs(block_variant, m=8, batch_size=8) for block_variant in u4_cpu.blocks]
    u4_flops = sum(row["forward_flops_per_round"] for row in u4_rows)
    r4_non_gemm = {name: 4 * value for name, value in per_round_b8["non_gemm_per_round"].items()}
    u4_non_gemm = {name: sum(row["non_gemm_per_round"][name] for row in u4_rows) for name in r4_non_gemm}
    return {
        "schema": "omega-v2-2a-flop-ledger-v1",
        "mac_convention": "1 MAC = 2 FLOPs",
        "per_round_b1_flops": per_round_b1["forward_flops_per_round"],
        "k4_b1_flops": 4 * per_round_b1["forward_flops_per_round"],
        "k4_b8_R4_forward_flops": r4_flops,
        "k4_b8_U4_forward_flops": u4_flops,
        "R4_non_gemm_counts_total": r4_non_gemm,
        "U4_non_gemm_counts_total": u4_non_gemm,
        "exact_integer_flop_equality": r4_flops == u4_flops == 538_968_064,
        "non_gemm_counts_identical": r4_non_gemm == u4_non_gemm,
        "pass": r4_flops == u4_flops == 538_968_064 and r4_non_gemm == u4_non_gemm,
    }


def k_flex_gate(block: ContractualCoreBlock, x_cuda: Tensor, loss_weights_cuda: Tensor) -> dict[str, Any]:
    schema_before = schema_sha256(block)
    value_before = state_dict_sha256(block)
    count_before = parameter_count(block)
    rows: list[dict[str, Any]] = []
    for k in (1, 2, 4, 8, 16):
        with torch.no_grad():
            output = shared_cuda_rounds(block, x_cuda, k)
        forward_finite = tensor_finite(output)
        rows.append({"phase": "forward", "K": k, "finite": forward_finite,
                     "schema_unchanged": schema_sha256(block) == schema_before,
                     "value_unchanged": state_dict_sha256(block) == value_before,
                     "parameter_count_unchanged": parameter_count(block) == count_before})
    for k in (1, 4, 8, 16):
        block.zero_grad(set_to_none=True)
        input_leaf = x_cuda.detach().clone().requires_grad_(True)
        output = shared_cuda_rounds(block, input_leaf, k)
        loss = torch.sum(output * loss_weights_cuda)
        loss.backward()
        grads_finite = all(tensor_finite(getattr(block, family).grad) for family in MATRIX_FAMILIES)
        row = {
            "phase": "backward", "K": k,
            "output_finite": tensor_finite(output), "loss_finite": tensor_finite(loss),
            "input_gradient_finite": tensor_finite(input_leaf.grad), "parameter_gradients_finite": grads_finite,
            "schema_unchanged": schema_sha256(block) == schema_before,
            "value_unchanged": state_dict_sha256(block) == value_before,
            "parameter_count_unchanged": parameter_count(block) == count_before,
        }
        row["finite"] = all(row[key] for key in ("output_finite", "loss_finite", "input_gradient_finite", "parameter_gradients_finite"))
        rows.append(row)
        block.zero_grad(set_to_none=True)
    passed = all(row["finite"] and row["schema_unchanged"] and row["value_unchanged"] and row["parameter_count_unchanged"] for row in rows)
    return {
        "schema": "omega-v2-2a-k-flex-v1",
        "SCHEMA_SHA256_before": schema_before,
        "VALUE_SHA256_before": value_before,
        "parameter_count_before": count_before,
        "rows": rows,
        "SCHEMA_SHA256_after": schema_sha256(block),
        "VALUE_SHA256_after": state_dict_sha256(block),
        "parameter_count_after": parameter_count(block),
        "pass": passed and schema_sha256(block) == schema_before and state_dict_sha256(block) == value_before and parameter_count(block) == count_before,
    }


def k_flex_forward_cell(block: ContractualCoreBlock, x_cuda: Tensor, k: int) -> dict[str, Any]:
    schema_before = schema_sha256(block)
    value_before = state_dict_sha256(block)
    count_before = parameter_count(block)
    with torch.no_grad():
        output = shared_cuda_rounds(block, x_cuda, k)
    return {
        "phase": "forward", "K": k,
        "output_finite": tensor_finite(output),
        "schema_unchanged": schema_sha256(block) == schema_before,
        "value_unchanged": state_dict_sha256(block) == value_before,
        "parameter_count_unchanged": parameter_count(block) == count_before,
    }


def k_flex_backward_cell(block: ContractualCoreBlock, x_cuda: Tensor, loss_weights_cuda: Tensor, k: int) -> dict[str, Any]:
    schema_before = schema_sha256(block)
    value_before = state_dict_sha256(block)
    count_before = parameter_count(block)
    block.zero_grad(set_to_none=True)
    input_leaf = x_cuda.detach().clone().requires_grad_(True)
    output = shared_cuda_rounds(block, input_leaf, k)
    loss = torch.sum(output * loss_weights_cuda)
    loss.backward()
    result = {
        "phase": "backward", "K": k,
        "output_finite": tensor_finite(output),
        "loss_finite": tensor_finite(loss),
        "input_gradient_finite": tensor_finite(input_leaf.grad),
        "parameter_gradients_finite": all(tensor_finite(getattr(block, family).grad) for family in MATRIX_FAMILIES),
        "schema_unchanged": schema_sha256(block) == schema_before,
        "value_unchanged": state_dict_sha256(block) == value_before,
        "parameter_count_unchanged": parameter_count(block) == count_before,
    }
    result["finite"] = all(result[key] for key in ("output_finite", "loss_finite", "input_gradient_finite", "parameter_gradients_finite"))
    block.zero_grad(set_to_none=True)
    return result


def optimizer_smoke(variant: R4Shared | U4Untied, x_cuda: Tensor, target_cuda: Tensor, *, label: str) -> dict[str, Any]:
    parameters = list(variant.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    block_for_round: Callable[[int], ContractualCoreBlock]
    if isinstance(variant, R4Shared):
        block_for_round = lambda _round_index: variant.recurrent.block
    else:
        block_for_round = variant.block_for_round

    with torch.no_grad():
        initial_output = cuda_rounds(block_for_round, x_cuda, 4)
        loss0 = torch.mean((initial_output - target_cuda) ** 2)
    l0 = float(loss0.item())
    initial_finite = tensor_finite(initial_output) and tensor_finite(loss0)
    step_rows: list[dict[str, Any]] = []
    for step in range(1, 21):
        optimizer.zero_grad(set_to_none=True)
        output = cuda_rounds(block_for_round, x_cuda, 4)
        loss = torch.mean((output - target_cuda) ** 2)
        loss.backward()
        gradients_finite = all(tensor_finite(parameter.grad) for parameter in parameters)
        optimizer.step()
        parameters_finite = all(tensor_finite(parameter) for parameter in parameters)
        optimizer_state_finite = all(
            tensor_finite(value)
            for state in optimizer.state.values()
            for key, value in state.items()
            if key in ("exp_avg", "exp_avg_sq")
        )
        step_pass = tensor_finite(output) and tensor_finite(loss) and gradients_finite and parameters_finite and optimizer_state_finite
        step_rows.append({
            "step": step, "output_finite": tensor_finite(output), "loss_finite": tensor_finite(loss),
            "gradients_finite": gradients_finite, "parameters_finite": parameters_finite,
            "adam_exp_avg_exp_avg_sq_finite": optimizer_state_finite, "pass": step_pass,
        })
    with torch.no_grad():
        final_output = cuda_rounds(block_for_round, x_cuda, 4)
        loss20 = torch.mean((final_output - target_cuda) ** 2)
    l20 = float(loss20.item())
    loss_gate = l20 <= 0.99 * l0
    return {
        "schema": "omega-v2-2a-optimizer-smoke-v1", "variant": label,
        "updates": len(step_rows), "optimizer": "AdamW", "lr": 3e-4, "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0, "scheduler": None, "gradient_clipping": False,
        "L0": l0, "L20": l20, "L0_finite": initial_finite, "loss_decrease_pass": loss_gate,
        "steps": step_rows, "pass": len(step_rows) == 20 and initial_finite and loss_gate and all(row["pass"] for row in step_rows),
    }
