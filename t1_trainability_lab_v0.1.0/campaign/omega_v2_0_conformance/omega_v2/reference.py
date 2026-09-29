"""Independent toy reference and numerical conformance checks."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .core import ContractualCoreBlock, MATRIX_FAMILIES, configure_reference_execution, make_seeded_core
from .variants import fixed_input


def _dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def _matvec(weight: Tensor, vector: list[float]) -> list[float]:
    rows = weight.detach().cpu().tolist()
    return [_dot(row, vector) for row in rows]


def _rms_vector(vector: list[float], eps: float) -> list[float]:
    denom = math.sqrt(sum(value * value for value in vector) / len(vector) + eps)
    return [value / denom for value in vector]


def _vectorized_forward(block: ContractualCoreBlock, state: Tensor, K: int) -> Tensor:
    current = state
    for _ in range(K):
        current = block.step(current)
    return current


def naive_step(block: ContractualCoreBlock, state: Tensor) -> Tensor:
    """Deliberately scalar/unfused FP64 equation reference for small toy inputs."""
    if state.dtype != torch.float64 or state.ndim != 3 or state.device.type != "cpu":
        raise ValueError("naive reference accepts CPU FP64 [B,m,d] only")
    batch_size, m, d = (int(v) for v in state.shape)
    if d != block.d:
        raise ValueError("toy state width does not match core width")
    output: list[list[list[float]]] = []
    sqrt_d = math.sqrt(d)
    for batch in range(batch_size):
        slots = state[batch].detach().tolist()
        normalized = [_rms_vector(slot, 1e-6) for slot in slots]
        queries = [_matvec(block.W_Q, slot) for slot in normalized]
        keys = [_matvec(block.W_K, slot) for slot in normalized]
        values = [_matvec(block.W_V, slot) for slot in normalized]
        hidden: list[list[float]] = []
        for query_index in range(m):
            scores = [_dot(queries[query_index], keys[key_index]) / sqrt_d for key_index in range(m)]
            peak = max(scores)
            exps = [math.exp(score - peak) for score in scores]
            denom = sum(exps)
            probs = [value / denom for value in exps]
            attended = [sum(probs[j] * values[j][feature] for j in range(m)) for feature in range(d)]
            projected = _matvec(block.W_O, attended)
            hidden.append([slots[query_index][feature] + projected[feature] for feature in range(d)])

        out_slots: list[list[float]] = []
        for slot in hidden:
            z = _rms_vector(slot, 1e-6)
            gate = _matvec(block.W_gate, z)
            up = _matvec(block.W_up, z)
            product = [(g / (1.0 + math.exp(-g))) * u for g, u in zip(gate, up)]
            mlp = _matvec(block.W_down, product)
            out_slots.append([slot[index] + mlp[index] for index in range(d)])
        output.append(out_slots)
    return torch.tensor(output, dtype=torch.float64, device="cpu")


def naive_forward(block: ContractualCoreBlock, state: Tensor, K: int) -> Tensor:
    current = state
    for _ in range(K):
        current = naive_step(block, current)
    return current


def toy_reference_report() -> dict[str, Any]:
    configure_reference_execution()
    block = make_seeded_core(8, 20260929, dtype=torch.float64)
    state = fixed_input(8, 2, 20260930, dtype=torch.float64)
    reference = naive_forward(block, state, K=2)
    vectorized = block_result = state
    for _ in range(2):
        vectorized = block.step(vectorized)
    abs_error = (vectorized - reference).abs()
    rel_error = abs_error / torch.clamp(reference.abs(), min=1e-30)
    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())
    return {
        "naive_reference": {
            "dtype": "FP64",
            "d": 8,
            "m": 2,
            "K": 2,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "abs_tolerance": 1e-12,
            "rel_tolerance": 1e-10,
            "pass": max_abs <= 1e-12 and max_rel <= 1e-10,
        }
    }


def toy_gradcheck_report() -> dict[str, Any]:
    configure_reference_execution()
    block = make_seeded_core(8, 20260931, dtype=torch.float64)
    state = fixed_input(8, 2, 20260932, dtype=torch.float64)
    state.requires_grad_(True)

    def scalar_loss() -> Tensor:
        output = _vectorized_forward(block, state, K=2)
        return output.square().mean() + 0.13 * output.sum()

    parameters = [getattr(block, family) for family in MATRIX_FAMILIES]
    gradients = torch.autograd.grad(scalar_loss(), parameters)
    probes = {
        "W_Q": (0, 0),
        "W_K": (1, 2),
        "W_V": (2, 3),
        "W_O": (3, 4),
        "W_gate": (4, 5),
        "W_up": (5, 6),
        "W_down": (6, 7),
    }
    eps = 1e-6
    rows = []
    for family, analytic_gradient in zip(MATRIX_FAMILIES, gradients):
        parameter = getattr(block, family)
        index = probes[family]
        original = float(parameter[index].item())
        with torch.no_grad():
            parameter[index] = original + eps
        plus = float(scalar_loss().item())
        with torch.no_grad():
            parameter[index] = original - eps
        minus = float(scalar_loss().item())
        with torch.no_grad():
            parameter[index] = original
        numeric = (plus - minus) / (2.0 * eps)
        analytic = float(analytic_gradient[index].item())
        relative = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12)
        rows.append(
            {
                "family": family,
                "index": list(index),
                "autograd": analytic,
                "central_difference": numeric,
                "relative_error": relative,
                "epsilon": eps,
                "pass": relative <= 1e-5,
            }
        )
    return {
        "gradcheck": {
            "dtype": "FP64",
            "d": 8,
            "m": 2,
            "K": 2,
            "finite_difference": "central",
            "max_relative_error": max(row["relative_error"] for row in rows),
            "tolerance": 1e-5,
            "families": rows,
            "pass": all(row["pass"] for row in rows),
        }
    }


def zero_weight_residual_report() -> dict[str, Any]:
    block = make_seeded_core(8, 20260933, dtype=torch.float32)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    state = fixed_input(8, 2, 20260934, dtype=torch.float32)
    one = block.step(state)
    multi = _vectorized_forward(block, state, K=2)
    return {
        "zero_weight_residual": {
            "step_bitwise_identity": torch.equal(one, state),
            "K2_bitwise_identity": torch.equal(multi, state),
            "pass": torch.equal(one, state) and torch.equal(multi, state),
        }
    }


def slot_permutation_report() -> dict[str, Any]:
    block = make_seeded_core(8, 20260935, dtype=torch.float64)
    state = fixed_input(8, 4, 20260936, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    left = _vectorized_forward(block, state[:, permutation, :], K=2)
    right = _vectorized_forward(block, state, K=2)[:, permutation, :]
    difference = (left - right).abs()
    max_abs = float(difference.max().item())
    passed = torch.allclose(left, right[:, :, :], atol=1e-12, rtol=1e-10)
    return {
        "slot_permutation_equivariance": {
            "dtype": "FP64",
            "d": 8,
            "m": 4,
            "K": 2,
            "permutation": permutation.tolist(),
            "max_abs_error": max_abs,
            "abs_tolerance": 1e-12,
            "rel_tolerance": 1e-10,
            "pass": bool(passed),
        }
    }


def toy_acceptance_report() -> dict[str, Any]:
    reports = {}
    reports.update(toy_reference_report())
    reports.update(toy_gradcheck_report())
    reports.update(zero_weight_residual_report())
    reports.update(slot_permutation_report())
    reports["status"] = "PASS" if all(row["pass"] for key, row in reports.items() if key != "status") else "FAIL"
    return reports
