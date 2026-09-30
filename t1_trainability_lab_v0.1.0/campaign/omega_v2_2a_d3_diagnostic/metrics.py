"""Post-hoc D3 error, ULP, argmax, and raw-tensor hash diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor


def sum_u4_variants(grads_u: list[Tensor]) -> dict[str, Tensor]:
    if len(grads_u) != 4:
        raise ValueError("U4 summation diagnostics require exactly four round gradients")
    g0, g1, g2, g3 = grads_u
    return {
        "S_forward": ((g0 + g1) + g2) + g3,
        "S_reverse": ((g3 + g2) + g1) + g0,
        "S_pairwise": (g0 + g1) + (g2 + g3),
        "S_stack": torch.stack([g0, g1, g2, g3]).sum(0),
        "S_fp64": (((g0.double() + g1.double()) + g2.double()) + g3.double()),
    }


def local_ulp_scalar(value: Tensor | float) -> float | None:
    scalar = float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
    if not math.isfinite(scalar):
        return None
    magnitude = torch.tensor(abs(scalar), dtype=torch.float32, device="cpu")
    zero = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    infinity = torch.tensor(float("inf"), dtype=torch.float32, device="cpu")
    if float(magnitude.item()) == 0.0:
        return float(torch.nextafter(zero, infinity).item())
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32, device="cpu")
    if bool(magnitude == largest_finite):
        return float((magnitude - torch.nextafter(magnitude, zero)).item())
    return float((torch.nextafter(magnitude, infinity) - magnitude).item())


def _common_float_tensors(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    dtype = torch.promote_types(left.dtype, right.dtype)
    return left.to(dtype=dtype), right.to(dtype=dtype)


def diagnostic_metrics(g_r: Tensor, sum_g_u: Tensor) -> dict[str, Any]:
    if tuple(g_r.shape) != tuple(sum_g_u.shape):
        raise ValueError("D3 gradient tensors must have identical shapes")
    left, right = _common_float_tensors(g_r, sum_g_u)
    difference = left - right
    absolute = difference.abs()
    one = torch.ones((), dtype=left.dtype, device=left.device)
    old_floor = torch.tensor(1e-6, dtype=left.dtype, device=left.device)
    old_denominator = torch.maximum(torch.maximum(left.abs(), right.abs()), old_floor)
    relative_old = absolute / old_denominator
    norm_left = torch.linalg.vector_norm(left)
    norm_right = torch.linalg.vector_norm(right)
    norm_diff = torch.linalg.vector_norm(difference)
    e_l2 = norm_diff / torch.maximum(torch.maximum(norm_left, norm_right), torch.tensor(1e-6, dtype=left.dtype, device=left.device))
    inf_left = left.abs().amax()
    inf_right = right.abs().amax()
    inf_diff = absolute.amax()
    e_inf = inf_diff / torch.maximum(torch.maximum(inf_left, inf_right), torch.tensor(1e-12, dtype=left.dtype, device=left.device))
    flat_rel = relative_old.contiguous().reshape(-1)
    flat_abs = absolute.contiguous().reshape(-1)
    rel_index = int(torch.argmax(flat_rel).item())
    abs_index = int(torch.argmax(flat_abs).item())
    return {
        "E_L2": float(e_l2.item()),
        "E_inf": float(e_inf.item()),
        "max_abs": float(flat_abs.max().item()),
        "max_rel_old": float(flat_rel.max().item()),
        "norm_gR_l2": float(norm_left.item()),
        "norm_gR_inf": float(inf_left.item()),
        "norm_sum_gU_inf": float(inf_right.item()),
        "max_abs_flat_index": abs_index,
        "max_rel_old_flat_index": rel_index,
        "shape": [int(size) for size in g_r.shape],
        "dtype_gR": str(g_r.dtype),
        "dtype_sum_gU": str(sum_g_u.dtype),
        "floor_1e_6_active_at_max_rel_old": bool(old_denominator.contiguous().reshape(-1)[rel_index].item() == 1e-6),
    }


def _unravel_c_index(index: int, shape: tuple[int, ...]) -> list[int]:
    coordinates = [0] * len(shape)
    remainder = int(index)
    for axis in range(len(shape) - 1, -1, -1):
        size = int(shape[axis])
        coordinates[axis] = remainder % size
        remainder //= size
    return coordinates


def old_comparator_argmax_details(g_r: Tensor, s_stack: Tensor, metrics: dict[str, Any]) -> dict[str, Any]:
    left, right = _common_float_tensors(g_r, s_stack)
    diff = (left - right).abs()
    raw_denominator = torch.maximum(left.abs(), right.abs())
    denominator = torch.maximum(raw_denominator, torch.tensor(1e-6, dtype=left.dtype, device=left.device))
    old_relative_tensor = diff / denominator
    shape = tuple(int(size) for size in g_r.shape)
    result: dict[str, Any] = {}
    for label, flat_index_key in (("argmax_max_rel_old", "max_rel_old_flat_index"), ("argmax_max_abs", "max_abs_flat_index")):
        flat_index = int(metrics[flat_index_key])
        multi_index = _unravel_c_index(flat_index, shape)
        index = tuple(multi_index)
        g_r_value = float(g_r[index].detach().cpu().item())
        sum_value = float(s_stack[index].detach().cpu().item())
        error_value = float(diff[index].detach().cpu().item())
        denom_value = float(denominator[index].detach().cpu().item())
        ulp_g_r = local_ulp_scalar(g_r[index])
        ulp_sum = local_ulp_scalar(s_stack[index])
        result[label] = {
            "flat_index": flat_index,
            "multi_index_row_major": list(multi_index),
            "gR_value": g_r_value,
            "sum_gU_old_value": sum_value,
            "abs_error": error_value,
            "old_denominator": denom_value,
            "old_relative_error": float(old_relative_tensor[index].detach().cpu().item()),
            "floor_1e_6_active": float(raw_denominator[index].detach().cpu().item()) <= 1e-6,
            "ulp_gR": ulp_g_r,
            "ulp_sum_gU_old": ulp_sum,
            "error_over_ulp_gR": error_value / ulp_g_r if ulp_g_r else None,
            "error_over_ulp_sum_gU_old": error_value / ulp_sum if ulp_sum else None,
        }
    norm_inf_stack = float(s_stack.detach().abs().amax().cpu().item())
    norm_inf_ulp = local_ulp_scalar(norm_inf_stack)
    result["tensor_magnitude_context"] = {
        "norm_gR_l2": metrics["norm_gR_l2"],
        "norm_gR_inf": metrics["norm_gR_inf"],
        "norm_S_stack_inf": norm_inf_stack,
        "ulp_norm_S_stack_inf": norm_inf_ulp,
        "max_abs_over_ulp_norm_S_stack_inf": metrics["max_abs"] / norm_inf_ulp if norm_inf_ulp else None,
    }
    return result


def tensor_raw_sha256(name: str, tensor: Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
