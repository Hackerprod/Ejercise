"""D1/D2/D4/D6/D7 records; D3 gates and ULP are reused unchanged from D3Q."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor

from omega_v2.ledger import state_dict_sha256
from omega_v2_2a_d3q.metrics import (
    evaluate_primary_gates,
    local_ulp_scalar,
    old_max_rel_diagnostic,
    tensor_raw_sha256,
)

from .config import (
    D1_ELEMENTWISE_ABS_TOL,
    D1_ELEMENTWISE_FLOOR,
    D1_ELEMENTWISE_REL_TOL,
    D1_NORMWISE_E_L2_LIMIT,
    D1_NORMWISE_FLOOR,
)


def all_finite(*tensors: Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def d1_correctness_metrics(y_cuda: Tensor, y_cpu: Tensor) -> dict[str, Any]:
    if tuple(y_cuda.shape) != tuple(y_cpu.shape):
        raise ValueError("D1 CPU/CUDA outputs must have the same shape")
    cuda64 = y_cuda.detach().to(device="cpu", dtype=torch.float64)
    cpu64 = y_cpu.detach().to(device="cpu", dtype=torch.float64)
    abs_delta = (cuda64 - cpu64).abs()
    allowed = D1_ELEMENTWISE_ABS_TOL + D1_ELEMENTWISE_REL_TOL * torch.maximum(
        cpu64.abs(), torch.full_like(cpu64, D1_ELEMENTWISE_FLOOR)
    )
    elementwise_pass = bool(torch.all(abs_delta <= allowed).item())
    l2_error = torch.linalg.vector_norm(cuda64 - cpu64)
    l2_reference = torch.linalg.vector_norm(cpu64)
    e_l2 = float((l2_error / torch.maximum(l2_reference, torch.tensor(D1_NORMWISE_FLOOR, dtype=torch.float64))).item())
    finite = all_finite(cuda64, cpu64, abs_delta, allowed) and math.isfinite(e_l2)
    return {
        "finite": bool(finite),
        "elementwise_gate": {
            "abs_tolerance": D1_ELEMENTWISE_ABS_TOL,
            "rel_tolerance": D1_ELEMENTWISE_REL_TOL,
            "cpu_floor": D1_ELEMENTWISE_FLOOR,
            "max_abs": float(abs_delta.amax().item()),
            "max_scaled_error": float((abs_delta / allowed).amax().item()),
            "pass": bool(finite and elementwise_pass),
        },
        "normwise_gate": {
            "E_L2": e_l2,
            "limit": D1_NORMWISE_E_L2_LIMIT,
            "denominator_floor": D1_NORMWISE_FLOOR,
            "pass": bool(finite and e_l2 <= D1_NORMWISE_E_L2_LIMIT),
        },
        "shape": [int(dim) for dim in y_cuda.shape],
        "dtype_cuda": str(y_cuda.dtype),
        "dtype_cpu": str(y_cpu.dtype),
    }


def d2_structural_parity(
    *,
    clone_report: dict[str, Any],
    round_traces_r4: list[Tensor],
    round_traces_u4: list[Tensor],
    final_r4: Tensor,
    final_u4: Tensor,
) -> dict[str, Any]:
    if len(round_traces_r4) != 4 or len(round_traces_u4) != 4:
        raise ValueError("D2 K4 requires exactly four traces per variant")
    clone_equal = bool(clone_report["initial_values_bitwise_equal"])
    trace_rows = [
        {"round": index + 1, "torch_equal": bool(torch.equal(round_traces_r4[index], round_traces_u4[index]))}
        for index in range(4)
    ]
    final_equal = bool(torch.equal(final_r4, final_u4))
    return {
        "initial_clone_report": clone_report,
        "initial_weights_equal": clone_equal,
        "round_trace_equalities": trace_rows,
        "final_equal": final_equal,
        "pass": bool(clone_equal and all(row["torch_equal"] for row in trace_rows) and final_equal),
        "non_gate_fallback_used": False,
    }


def unique_parameter_count(module: torch.nn.Module) -> int:
    unique: dict[tuple[int, int], torch.nn.Parameter] = {}
    for parameter in module.parameters():
        storage = parameter.untyped_storage()
        unique.setdefault((int(storage.data_ptr()), int(storage.nbytes())), parameter)
    return sum(int(parameter.numel()) for parameter in unique.values())


def storage_conformance(shared: torch.nn.Module, untied: torch.nn.Module, families: tuple[str, ...]) -> dict[str, Any]:
    shared_block = shared.recurrent.block
    u4_blocks = list(untied.blocks)
    rows = []
    initial_equal = True
    r4_storage_ok = True
    u4_storage_ok = True
    no_alias_r4 = True
    for family in families:
        shared_value = getattr(shared_block, family)
        u4_values = [getattr(block, family) for block in u4_blocks]
        equal = all(torch.equal(shared_value.detach().cpu(), value.detach().cpu()) for value in u4_values)
        initial_equal = initial_equal and equal
        r4_storage_ok = r4_storage_ok and len({(shared_value.untyped_storage().data_ptr(), shared_value.untyped_storage().nbytes())}) == 1
        identities = [(value.untyped_storage().data_ptr(), value.untyped_storage().nbytes()) for value in u4_values]
        u4_distinct = len(set(identities)) == 4
        u4_storage_ok = u4_storage_ok and u4_distinct
        no_alias = all(identity != (shared_value.untyped_storage().data_ptr(), shared_value.untyped_storage().nbytes()) for identity in identities)
        no_alias_r4 = no_alias_r4 and no_alias
        rows.append({"family": family, "initial_bitwise_equal": equal, "u4_four_disjoint_storages": u4_distinct, "u4_disjoint_from_r4": no_alias})
    r4_count = unique_parameter_count(shared)
    u4_count = unique_parameter_count(untied)
    return {
        "families": rows,
        "R4_unique_parameters": r4_count,
        "U4_unique_parameters": u4_count,
        "R4_parameter_count_pass": r4_count == 4_194_304,
        "U4_parameter_count_pass": u4_count == 16_777_216,
        "initial_bitwise_equal": initial_equal,
        "R4_one_storage_per_family": r4_storage_ok,
        "U4_four_pairwise_disjoint_storages_per_family": u4_storage_ok,
        "U4_disjoint_from_R4": no_alias_r4,
        "pass": bool(initial_equal and r4_storage_ok and u4_storage_ok and no_alias_r4 and r4_count == 4_194_304 and u4_count == 16_777_216),
    }


def schema_sha256(module: torch.nn.Module) -> str:
    schema = [
        {"name": name, "dtype": str(tensor.dtype), "shape": [int(dim) for dim in tensor.shape]}
        for name, tensor in sorted(module.state_dict().items())
    ]
    payload = (json.dumps(schema, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def value_state_sha256(module: torch.nn.Module) -> str:
    """Use the exact V2-0 `state_dict_sha256` tensor-name/dtype/shape/raw-byte recipe."""
    return state_dict_sha256(module)


def d6_invariance_snapshot(module: torch.nn.Module) -> dict[str, Any]:
    """Capture one module's D6 schema/value/count at a specific point in a cell."""
    return {
        "SCHEMA_SHA256": schema_sha256(module),
        "VALUE_SHA256": value_state_sha256(module),
        "parameter_count": unique_parameter_count(module),
    }


def d6_invariance_record(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Persist CUDA-model before/after snapshots and their exact invariance decision."""
    record = {
        "SCHEMA_SHA256_before": before["SCHEMA_SHA256"],
        "SCHEMA_SHA256_after": after["SCHEMA_SHA256"],
        "VALUE_SHA256_before": before["VALUE_SHA256"],
        "VALUE_SHA256_after": after["VALUE_SHA256"],
        "parameter_count_before": before["parameter_count"],
        "parameter_count_after": after["parameter_count"],
    }
    record["pass"] = bool(
        record["SCHEMA_SHA256_before"] == record["SCHEMA_SHA256_after"]
        and record["VALUE_SHA256_before"] == record["VALUE_SHA256_after"]
        and record["parameter_count_before"] == record["parameter_count_after"]
    )
    return record


def optimizer_state_canonical_sha256(optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> str:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    entries = []
    for parameter, state in optimizer.state.items():
        state_row: dict[str, Any] = {}
        for key, value in sorted(state.items()):
            if isinstance(value, Tensor):
                tensor = value.detach().to(device="cpu").contiguous()
                state_row[key] = {
                    "dtype": str(tensor.dtype),
                    "shape": [int(dim) for dim in tensor.shape],
                    "raw_sha256": tensor_raw_sha256(f"optimizer/{names.get(id(parameter), 'unknown')}/{key}", tensor),
                }
            else:
                state_row[key] = value
        entries.append({"parameter_name": names.get(id(parameter), "unknown"), "parameter_shape": [int(dim) for dim in parameter.shape], "state": state_row})
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            key: [names.get(id(value), "unknown") for value in values] if key == "params" else value
            for key, values in group.items()
        })
    payload = (json.dumps({"param_groups": groups, "state": sorted(entries, key=lambda item: item["parameter_name"])}, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_parameter_state_sha256(module: torch.nn.Module) -> str:
    return state_dict_sha256(module)


def finite_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, bool]:
    exp_avg_values = []
    exp_avg_sq_values = []
    all_state_tensors = []
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                all_state_tensors.append(value)
                if key == "exp_avg":
                    exp_avg_values.append(value)
                elif key == "exp_avg_sq":
                    exp_avg_sq_values.append(value)
    return {
        "all_optimizer_state_finite": all_finite(*all_state_tensors),
        "exp_avg_finite": all_finite(*exp_avg_values),
        "exp_avg_sq_finite": all_finite(*exp_avg_sq_values),
    }


def d1_elementwise_and_normwise(y_cuda: Tensor, y_cpu: Tensor) -> dict[str, Any]:
    return d1_correctness_metrics(y_cuda, y_cpu)


def d3_d512_gates(g_r32: Tensor, g_u32: list[Tensor]) -> dict[str, Any]:
    if len(g_u32) != 4:
        raise ValueError("D3 requires gU0..gU3")
    d3q_sums = (((g_u32[0].double() + g_u32[1].double()) + g_u32[2].double()) + g_u32[3].double())
    g_r64 = g_r32.double()
    primary = evaluate_primary_gates(g_r64, d3q_sums)
    from omega_v2_2a_d3q.metrics import sum_u4_variants

    diagnostics = sum_u4_variants(g_u32)
    old_max_rel = old_max_rel_diagnostic(g_r32, diagnostics["S_stack"])
    old_max_rel["S_reverse_equal_gR_NON_GATE"] = bool(torch.equal(diagnostics["S_reverse"], g_r32))
    return {
        "primary_S64": primary,
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {"old_max_rel": old_max_rel, "sum_variant_names": list(diagnostics)},
        "pass": bool(primary["finite"] and primary["scientific_gates_pass"]),
        "sums_fp32": diagnostics,
        "gR64": g_r64,
        "S64": d3q_sums,
    }
