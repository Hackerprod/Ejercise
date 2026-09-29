from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LAB_ROOT = REPO_ROOT / "t1_trainability_lab_v0.1.0"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LAB_ROOT / "scripts"))
sys.path.insert(0, str(LAB_ROOT / "campaign" / "omega_teacher_logit_cache"))

from r1_masked_token_mean_loss import (  # noqa: E402
    TEMPERATURE,
    r1_masked_token_loss_numerators,
    r1_masked_token_mean_loss,
)
from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    distillation_loss as r1_reference_loss,
)
from run_omega_teacher_logit_cache import (  # noqa: E402
    distillation_loss as historical_helper_loss,
)


def _sample(batch: int = 2, length: int = 5, vocab: int = 11) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(9173 + batch * 100 + length * 10 + vocab)
    student = torch.randn(batch, length, vocab, dtype=torch.float64)
    teacher = torch.randn(batch, length, vocab, dtype=torch.float64)
    targets = torch.randint(vocab, (batch, length), dtype=torch.long)
    mask = torch.rand(batch, length) > 0.35
    if not bool(mask.any()):
        mask[0, 0] = True
    return student, teacher, targets, mask


def test_r1_oracle_components_and_logits_gradient_match() -> None:
    student, teacher, targets, mask = _sample()
    student_ref = student.clone().requires_grad_(True)
    student_new = student.clone().requires_grad_(True)
    reference = r1_reference_loss(student_ref, teacher, targets, mask)
    actual = r1_masked_token_mean_loss(student_new, teacher, targets, mask)
    for key in ("ce", "kl", "total"):
        torch.testing.assert_close(actual[key], reference[key], rtol=1e-12, atol=1e-12)
    reference_grad = torch.autograd.grad(reference["total"], student_ref)[0]
    actual_grad = torch.autograd.grad(actual["total"], student_new)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)
    assert actual["valid_tokens"] == int(mask.sum())


def test_repeated_positions_are_length_invariant_and_old_helper_scales_kl() -> None:
    student, teacher, targets, _ = _sample(batch=2, length=1, vocab=13)
    canonical_kls: list[torch.Tensor] = []
    for length in (1, 3, 256):
        student_l = student.expand(-1, length, -1).contiguous()
        teacher_l = teacher.expand(-1, length, -1).contiguous()
        targets_l = targets.expand(-1, length).contiguous()
        mask = torch.ones_like(targets_l, dtype=torch.bool)
        actual = r1_masked_token_mean_loss(student_l, teacher_l, targets_l, mask)
        historical = historical_helper_loss(student_l, teacher_l, targets_l)
        canonical_kls.append(actual["kl"])
        torch.testing.assert_close(historical["kl"] / actual["kl"], torch.tensor(float(length), dtype=actual["kl"].dtype), rtol=1e-12, atol=1e-12)
    for value in canonical_kls[1:]:
        torch.testing.assert_close(value, canonical_kls[0], rtol=1e-12, atol=1e-12)


def test_nonuniform_mask_excludes_invalid_positions_from_numerator_and_denominator() -> None:
    student, teacher, targets, _ = _sample(batch=2, length=6, vocab=9)
    mask = torch.tensor([[True, False, True, False, False, True], [False, True, False, True, False, False]])
    baseline_student = student.clone().requires_grad_(True)
    baseline = r1_masked_token_mean_loss(baseline_student, teacher, targets, mask)

    changed_student = student.clone()
    changed_teacher = teacher.clone()
    changed_targets = targets.clone()
    changed_student[~mask] = torch.linspace(-1000.0, 1000.0, int((~mask).sum()) * student.shape[-1], dtype=student.dtype).reshape(-1, student.shape[-1])
    changed_teacher[~mask] = -changed_student[~mask]
    changed_targets[~mask] = (changed_targets[~mask] + 1) % student.shape[-1]
    changed_student.requires_grad_(True)
    changed = r1_masked_token_mean_loss(changed_student, changed_teacher, changed_targets, mask)
    for key in ("ce", "kl", "total"):
        torch.testing.assert_close(changed[key], baseline[key], rtol=0, atol=0)
    assert changed["valid_tokens"] == int(mask.sum()) == 5
    gradient = torch.autograd.grad(changed["total"], changed_student)[0]
    assert torch.count_nonzero(gradient[~mask]).item() == 0


def test_three_dimensional_and_valid_token_flattened_forms_match_objective_and_gradient() -> None:
    student, teacher, targets, mask = _sample(batch=3, length=7, vocab=17)
    student_3d = student.clone().requires_grad_(True)
    actual_3d = r1_masked_token_mean_loss(student_3d, teacher, targets, mask)
    grad_3d = torch.autograd.grad(actual_3d["total"], student_3d)[0]

    selected_student = student[mask].reshape(1, -1, student.shape[-1]).clone().requires_grad_(True)
    selected_teacher = teacher[mask].reshape(1, -1, teacher.shape[-1])
    selected_targets = targets[mask].reshape(1, -1)
    flat_mask = torch.ones_like(selected_targets, dtype=torch.bool)
    actual_flat = r1_masked_token_mean_loss(selected_student, selected_teacher, selected_targets, flat_mask)
    grad_flat = torch.autograd.grad(actual_flat["total"], selected_student)[0]

    for key in ("ce", "kl", "total"):
        torch.testing.assert_close(actual_3d[key], actual_flat[key], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_3d[mask], grad_flat.reshape(-1, student.shape[-1]), rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(grad_3d[~mask]).item() == 0


def test_unequal_chunk_partitions_combine_numerators_not_means() -> None:
    student, teacher, targets, _ = _sample(batch=3, length=5, vocab=13)
    mask = torch.tensor([
        [True, True, True, True, True],
        [True, False, True, False, False],
        [False, False, True, False, False],
    ])
    whole = r1_masked_token_mean_loss(student, teacher, targets, mask)
    chunks = [
        r1_masked_token_loss_numerators(student[:1], teacher[:1], targets[:1], mask[:1]),
        r1_masked_token_loss_numerators(student[1:], teacher[1:], targets[1:], mask[1:]),
    ]
    total_valid = sum(item["valid_tokens"] for item in chunks)
    ce = sum((item["ce_numerator"] for item in chunks), torch.zeros((), dtype=student.dtype)) / total_valid
    kl = sum((item["kl_numerator"] for item in chunks), torch.zeros((), dtype=student.dtype)) / total_valid
    torch.testing.assert_close(ce, whole["ce"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kl, whole["kl"], rtol=1e-12, atol=1e-12)
    unweighted_mean_of_chunk_means = (chunks[0]["ce_numerator"] / chunks[0]["valid_tokens"] + chunks[1]["ce_numerator"] / chunks[1]["valid_tokens"]) / 2
    assert not torch.isclose(unweighted_mean_of_chunk_means, whole["ce"], rtol=1e-8, atol=1e-8)


def test_teacher_direction_temperature_scaling_and_no_teacher_gradient() -> None:
    student, teacher, targets, mask = _sample(batch=2, length=4, vocab=9)
    student = student.requires_grad_(True)
    teacher = teacher.requires_grad_(True)
    actual = r1_masked_token_mean_loss(student, teacher, targets, mask)

    student_log_probs = F.log_softmax(student / TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher.detach() / TEMPERATURE, dim=-1)
    expected_kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (TEMPERATURE**2)
    expected_kl = expected_kl_per_token[mask].mean()
    reverse_kl_per_token = F.kl_div(
        F.log_softmax(teacher.detach() / TEMPERATURE, dim=-1),
        F.softmax(student.detach() / TEMPERATURE, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (TEMPERATURE**2)
    reverse_kl = reverse_kl_per_token[mask].mean()
    torch.testing.assert_close(actual["kl"], expected_kl, rtol=1e-12, atol=1e-12)
    assert not torch.isclose(actual["kl"], reverse_kl, rtol=1e-6, atol=1e-6)

    actual["total"].backward()
    assert teacher.grad is None
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


def test_all_invalid_batch_fails_closed() -> None:
    student, teacher, targets, _ = _sample(batch=2, length=3, vocab=7)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_tokens > 0"):
        r1_masked_token_mean_loss(student, teacher, targets, mask)
