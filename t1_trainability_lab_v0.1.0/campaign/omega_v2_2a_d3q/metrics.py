"""Frozen D3Q FP64 gates and diagnostic-only FP32 comparator metrics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


GATE_A_LIMIT = 2e-7
GATE_B_LIMIT = 5e-7
GATE_C_ULPS = 8.0
ORACLE_L2_LIMIT = 1e-15
ORACLE_INF_LIMIT = 1e-14


def inclusive_threshold_pass(value: float, limit: float) -> bool:
    return math.isfinite(float(value)) and float(value) <= float(limit)


def scale_aware_gate_c_pass(max_abs: float, m32: float | Tensor, *, finite: bool = True) -> tuple[bool, float | None]:
    ulp = local_ulp_scalar(m32)
    passed = bool(finite and ulp is not None and inclusive_threshold_pass(max_abs, GATE_C_ULPS * ulp))
    return passed, ulp


def local_ulp_scalar(value: Tensor | float) -> float | None:
    """D3-DIAGNOSTIC ULP: nextafter in FP32 applied to the magnitude."""
    scalar = float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
    if not math.isfinite(scalar):
        return None
    magnitude = torch.tensor(abs(scalar), dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(magnitude)):
        return None
    zero = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    infinity = torch.tensor(float("inf"), dtype=torch.float32, device="cpu")
    if float(magnitude.item()) == 0.0:
        return float(torch.nextafter(zero, infinity).item())
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32, device="cpu")
    if bool(magnitude == largest_finite):
        return float((magnitude - torch.nextafter(magnitude, zero)).item())
    next_value = torch.nextafter(magnitude, infinity)
    if not bool(torch.isfinite(next_value)):
        return float((magnitude - torch.nextafter(magnitude, zero)).item())
    return float((next_value - magnitude).item())


def to_fp32_round_nearest_even(value: float | Tensor) -> Tensor:
    """Use torch's ordinary FP64-to-FP32 conversion (round-to-nearest-even)."""
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).to(dtype=torch.float32)
    return torch.as_tensor(float(value), dtype=torch.float64, device="cpu").to(dtype=torch.float32)


def all_finite(*tensors: Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def fp64_comparison_metrics(g_r64: Tensor, sum_u64: Tensor) -> dict[str, Any]:
    if g_r64.dtype != torch.float64 or sum_u64.dtype != torch.float64:
        raise TypeError("D3Q primary metrics require FP64 tensors")
    if tuple(g_r64.shape) != tuple(sum_u64.shape):
        raise ValueError("D3Q compared gradients must have identical shapes")
    delta = g_r64 - sum_u64
    absolute = delta.abs()
    norm_r = torch.linalg.vector_norm(g_r64)
    norm_s = torch.linalg.vector_norm(sum_u64)
    norm_delta = torch.linalg.vector_norm(delta)
    norm_r_inf = g_r64.abs().amax()
    norm_s_inf = sum_u64.abs().amax()
    norm_delta_inf = absolute.amax()
    l2_denominator = torch.maximum(
        torch.maximum(norm_r, norm_s),
        torch.tensor(1e-6, dtype=torch.float64, device=g_r64.device),
    )
    inf_denominator = torch.maximum(
        torch.maximum(norm_r_inf, norm_s_inf),
        torch.tensor(1e-12, dtype=torch.float64, device=g_r64.device),
    )
    old_relative = absolute / torch.maximum(
        torch.maximum(g_r64.abs(), sum_u64.abs()),
        torch.tensor(1e-6, dtype=torch.float64, device=g_r64.device),
    )
    return {
        "E_L2": float((norm_delta / l2_denominator).item()),
        "E_inf": float((norm_delta_inf / inf_denominator).item()),
        "max_abs": float(absolute.amax().item()),
        "max_rel_old": float(old_relative.amax().item()),
        "norm_gR64_l2": float(norm_r.item()),
        "norm_S64_l2": float(norm_s.item()),
        "norm_gR64_inf": float(norm_r_inf.item()),
        "norm_S64_inf": float(norm_s_inf.item()),
        "shape": [int(size) for size in g_r64.shape],
    }


def evaluate_primary_gates(g_r64: Tensor, sum_u64: Tensor) -> dict[str, Any]:
    metrics = fp64_comparison_metrics(g_r64, sum_u64)
    finite = all_finite(g_r64, sum_u64)
    magnitude = max(metrics["norm_gR64_inf"], metrics["norm_S64_inf"])
    m32_tensor = to_fp32_round_nearest_even(magnitude)
    m32 = float(m32_tensor.item())
    ulp_m32 = local_ulp_scalar(m32_tensor)
    metric_finite = all(
        math.isfinite(metrics[key])
        for key in ("E_L2", "E_inf", "max_abs", "max_rel_old", "norm_gR64_l2", "norm_S64_l2", "norm_gR64_inf", "norm_S64_inf")
    )
    gate_finite = finite and metric_finite
    gate_a = gate_finite and inclusive_threshold_pass(metrics["E_L2"], GATE_A_LIMIT)
    gate_b = gate_finite and inclusive_threshold_pass(metrics["E_inf"], GATE_B_LIMIT)
    gate_c, _ = scale_aware_gate_c_pass(metrics["max_abs"], m32_tensor, finite=gate_finite)
    return {
        "finite": gate_finite and math.isfinite(m32) and ulp_m32 is not None,
        "metrics": metrics,
        "scale_context": {"M64": magnitude, "M32": m32, "ULP_M32": ulp_m32},
        "gates": {
            "A": {"name": "normwise_L2", "limit": GATE_A_LIMIT, "value": metrics["E_L2"], "pass": bool(gate_a)},
            "B": {"name": "normwise_Linf", "limit": GATE_B_LIMIT, "value": metrics["E_inf"], "pass": bool(gate_b)},
            "C": {
                "name": "max_abs_scale_aware",
                "limit_ulps": GATE_C_ULPS,
                "value": metrics["max_abs"],
                "limit_value": None if ulp_m32 is None else GATE_C_ULPS * ulp_m32,
                "pass": bool(gate_c),
            },
        },
        "scientific_gates_pass": bool(gate_a and gate_b and gate_c),
    }


def evaluate_oracle_gate(g_r64_cpu: Tensor, sum_u64_cpu: Tensor) -> dict[str, Any]:
    metrics = fp64_comparison_metrics(g_r64_cpu, sum_u64_cpu)
    finite = all_finite(g_r64_cpu, sum_u64_cpu) and all(
        math.isfinite(metrics[key]) for key in ("E_L2", "E_inf", "max_abs", "max_rel_old")
    )
    l2_pass = finite and inclusive_threshold_pass(metrics["E_L2"], ORACLE_L2_LIMIT)
    inf_pass = finite and inclusive_threshold_pass(metrics["E_inf"], ORACLE_INF_LIMIT)
    return {
        "finite": bool(finite),
        "metrics": metrics,
        "gates": {
            "E_L2_FP64": {"limit": ORACLE_L2_LIMIT, "value": metrics["E_L2"], "pass": bool(l2_pass)},
            "E_inf_FP64": {"limit": ORACLE_INF_LIMIT, "value": metrics["E_inf"], "pass": bool(inf_pass)},
        },
        "pass": bool(l2_pass and inf_pass),
        "max_abs_FP64_threshold": None,
    }


def sum_u4_diagnostics(grads_u32: list[Tensor]) -> dict[str, Tensor]:
    if len(grads_u32) != 4:
        raise ValueError("D3Q U4 diagnostics require exactly four CUDA FP32 gradients")
    g0, g1, g2, g3 = grads_u32
    if any(g.dtype != torch.float32 for g in grads_u32):
        raise TypeError("captured CUDA gradients must be FP32 before diagnostic sums")
    return {
        "S_forward": ((g0 + g1) + g2) + g3,
        "S_reverse": ((g3 + g2) + g1) + g0,
        "S_pairwise": (g0 + g1) + (g2 + g3),
        "S_stack": torch.stack([g0, g1, g2, g3]).sum(0),
    }


def old_max_rel_diagnostic(g_r32: Tensor, s_stack32: Tensor) -> dict[str, Any]:
    if g_r32.dtype != torch.float32 or s_stack32.dtype != torch.float32:
        raise TypeError("old max_rel diagnostic is defined on CUDA FP32 tensors")
    delta = (g_r32 - s_stack32).abs()
    denominator = torch.maximum(
        torch.maximum(g_r32.abs(), s_stack32.abs()),
        torch.tensor(1e-6, dtype=torch.float32, device=g_r32.device),
    )
    relative = delta / denominator
    flat = relative.contiguous().reshape(-1)
    index = int(torch.argmax(flat).item())
    old_floor_active = bool(
        torch.maximum(g_r32.abs(), s_stack32.abs()).contiguous().reshape(-1)[index].item() <= 1e-6
    )
    scalar_g = g_r32.contiguous().reshape(-1)[index]
    scalar_s = s_stack32.contiguous().reshape(-1)[index]
    abs_error = float(delta.contiguous().reshape(-1)[index].item())
    ulp_g = local_ulp_scalar(scalar_g)
    ulp_s = local_ulp_scalar(scalar_s)
    remainder = index
    coordinates = [0] * g_r32.ndim
    for axis in range(g_r32.ndim - 1, -1, -1):
        size = int(g_r32.shape[axis])
        coordinates[axis] = remainder % size
        remainder //= size
    return {
        "max_rel_old": float(flat[index].item()),
        "argmax_flat_index": index,
        "argmax_multi_index_row_major": coordinates,
        "gR_value": float(scalar_g.item()),
        "sum_gU_old_value": float(scalar_s.item()),
        "abs_error": abs_error,
        "old_denominator": float(denominator.contiguous().reshape(-1)[index].item()),
        "floor_1e_6_active": old_floor_active,
        "ulp_gR": ulp_g,
        "ulp_sum_gU_old": ulp_s,
        "abs_error_over_ulp_gR": abs_error / ulp_g if ulp_g else None,
        "abs_error_over_ulp_sum_gU_old": abs_error / ulp_s if ulp_s else None,
    }


def tensor_raw_sha256(name: str, tensor: Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
