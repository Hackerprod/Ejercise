"""Canonical shared R1 distillation objective for backend quality runs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


LOSS_CONTRACT = "R1_MASKED_TOKEN_MEAN_V1"
TEMPERATURE = 2.0
CE_WEIGHT = 0.5
KL_WEIGHT = 0.5


def _validate_inputs(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> int:
    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must have shape [B,L,V], got {tuple(student_logits.shape)}")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have identical [B,L,V] shape")
    expected_tokens = tuple(student_logits.shape[:2])
    if tuple(targets.shape) != expected_tokens or tuple(valid_mask.shape) != expected_tokens:
        raise ValueError("targets and valid_mask must both have shape [B,L]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be torch.bool")
    if student_logits.shape[-1] <= 1:
        raise ValueError("vocabulary dimension must exceed one")
    if student_logits.device != teacher_logits.device or student_logits.device != targets.device or student_logits.device != valid_mask.device:
        raise ValueError("student, teacher, targets, and mask must share device")
    valid_tokens = int(valid_mask.sum().item())
    if valid_tokens <= 0:
        raise ValueError("R1 distillation update requires valid_tokens > 0")
    return valid_tokens


def r1_masked_token_loss_numerators(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> dict[str, Any]:
    """Return CE/KL token sums and mask count, suitable for exact chunk aggregation."""
    valid_tokens = _validate_inputs(student_logits, teacher_logits, targets, valid_mask)
    batch, length, vocabulary = student_logits.shape
    ce_per_token = F.cross_entropy(
        student_logits.reshape(batch * length, vocabulary),
        targets.reshape(batch * length),
        reduction="none",
    ).reshape(batch, length)

    # Teacher is a fixed target distribution; no gradient may flow into it.
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits.detach() / TEMPERATURE, dim=-1)
    student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1) * (TEMPERATURE**2)

    weights = valid_mask.to(dtype=student_logits.dtype)
    return {
        "ce_numerator": (ce_per_token * weights).sum(),
        "kl_numerator": (kl_per_token * weights).sum(),
        "valid_tokens": valid_tokens,
    }


def r1_masked_token_mean_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> dict[str, Any]:
    """Compute R1 CE + teacher||student KL, each averaged over valid tokens."""
    numerators = r1_masked_token_loss_numerators(student_logits, teacher_logits, targets, valid_mask)
    denominator = numerators["valid_tokens"]
    ce = numerators["ce_numerator"] / denominator
    kl = numerators["kl_numerator"] / denominator
    return {
        "ce": ce,
        "kl": kl,
        "total": CE_WEIGHT * ce + KL_WEIGHT * kl,
        "valid_tokens": denominator,
    }
