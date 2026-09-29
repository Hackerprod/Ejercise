"""Dependency-free paired Student-t confidence interval for five deltas."""

from __future__ import annotations

import math
from typing import Sequence

import torch


T_CRITICAL_DF4_975 = 2.7764451051977987


def paired_deltas(nll_k1: Sequence[float], nll_k4: Sequence[float]) -> torch.Tensor:
    """Return paired depth deltas d_i = NLL_K1,i - NLL_K4,i."""

    left = torch.as_tensor(nll_k1, dtype=torch.float64)
    right = torch.as_tensor(nll_k4, dtype=torch.float64)
    if left.ndim != 1 or right.ndim != 1 or left.numel() != right.numel():
        raise ValueError("paired NLL inputs must be one-dimensional and equal length")
    if left.numel() < 2:
        raise ValueError("at least two paired deltas are required")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("NLL inputs must be finite")
    return left - right


def paired_student_t_ci(
    nll_k1: Sequence[float], nll_k4: Sequence[float], *, t_critical: float = T_CRITICAL_DF4_975
) -> dict[str, float | int | list[float]]:
    """Compute mean and two-sided 95% CI from paired seed-level deltas."""

    deltas = paired_deltas(nll_k1, nll_k4)
    count = int(deltas.numel())
    if count != 5:
        raise ValueError("OMEGA pilot CI requires exactly five paired deltas")
    mean_delta = float(deltas.mean())
    sample_std = float(deltas.std(unbiased=True))
    standard_error = sample_std / math.sqrt(count)
    margin = float(t_critical) * standard_error
    return {
        "n": count,
        "deltas": [float(value) for value in deltas.tolist()],
        "mean_delta": mean_delta,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "t_critical": float(t_critical),
        "lower": mean_delta - margin,
        "upper": mean_delta + margin,
    }


def depth_gate_passes(mean_delta: float, threshold: float = 0.05) -> bool:
    """Evaluate unchanged scientific gate separately from CI calculation."""

    return float(mean_delta) >= float(threshold)
