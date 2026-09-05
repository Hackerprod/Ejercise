from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_u0c_c1_joint import delta_loss_per_coordinate


def old_delta_loss(predicted: torch.Tensor, target_deltas: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return ((predicted - target_deltas) * active).square().sum() / active.sum().clamp_min(1)


def test_delta_loss_matches_mse_over_active_transitions() -> None:
    torch.manual_seed(7)
    predicted = torch.randn(2, 4, 8)
    target = torch.randn_like(predicted)
    active = torch.tensor([[[True], [False], [True], [False]], [[False], [True], [True], [False]]])
    expected = ((predicted - target)[active.expand_as(predicted)] ** 2).mean()
    actual = delta_loss_per_coordinate(predicted, target, active)
    torch.testing.assert_close(actual, expected)


def test_old_delta_loss_is_64_times_new_delta_loss() -> None:
    torch.manual_seed(11)
    predicted = torch.randn(3, 5, 64)
    target = torch.randn_like(predicted)
    active = torch.tensor([[[True], [True], [False], [True], [False]]] * 3)
    old = old_delta_loss(predicted, target, active)
    new = delta_loss_per_coordinate(predicted, target, active)
    torch.testing.assert_close(old, 64 * new)


def test_old_delta_gradient_is_64_times_new_delta_gradient() -> None:
    torch.manual_seed(13)
    predicted_old = torch.randn(2, 4, 64, requires_grad=True)
    target = torch.randn_like(predicted_old)
    active = torch.tensor([[[True], [False], [True], [True]], [[False], [True], [False], [True]]])
    old_gradient = torch.autograd.grad(old_delta_loss(predicted_old, target, active), predicted_old)[0]
    predicted_new = predicted_old.detach().clone().requires_grad_()
    new_gradient = torch.autograd.grad(delta_loss_per_coordinate(predicted_new, target, active), predicted_new)[0]
    torch.testing.assert_close(old_gradient, 64 * new_gradient)


def test_unit_error_per_active_coordinate_has_loss_one() -> None:
    predicted = torch.ones(2, 3, 64)
    target = torch.zeros_like(predicted)
    active = torch.ones(2, 3, 1, dtype=torch.bool)
    assert delta_loss_per_coordinate(predicted, target, active).item() == 1.0


def test_padding_does_not_change_delta_loss() -> None:
    torch.manual_seed(17)
    predicted = torch.randn(2, 3, 8)
    target = torch.randn_like(predicted)
    active = torch.tensor([[[True], [False], [True]], [[False], [True], [False]]])
    padded_predicted = torch.cat((predicted, torch.zeros(2, 2, 8)), dim=1)
    padded_target = torch.cat((target, torch.zeros(2, 2, 8)), dim=1)
    padded_active = torch.cat((active, torch.zeros(2, 2, 1, dtype=torch.bool)), dim=1)
    torch.testing.assert_close(
        delta_loss_per_coordinate(predicted, target, active),
        delta_loss_per_coordinate(padded_predicted, padded_target, padded_active),
    )


def test_duplicating_coordinates_does_not_change_delta_loss() -> None:
    torch.manual_seed(19)
    predicted = torch.randn(2, 3, 8)
    target = torch.randn_like(predicted)
    active = torch.tensor([[[True], [True], [False]], [[False], [True], [True]]])
    duplicated_predicted = predicted.repeat_interleave(2, dim=-1)
    duplicated_target = target.repeat_interleave(2, dim=-1)
    torch.testing.assert_close(
        delta_loss_per_coordinate(predicted, target, active),
        delta_loss_per_coordinate(duplicated_predicted, duplicated_target, active),
    )


def test_all_inactive_delta_loss_and_gradient_are_zero() -> None:
    predicted = torch.randn(2, 4, 16, requires_grad=True)
    target = torch.randn_like(predicted)
    active = torch.zeros(2, 4, 1, dtype=torch.bool)
    loss = delta_loss_per_coordinate(predicted, target, active)
    gradient = torch.autograd.grad(loss, predicted)[0]
    assert loss.item() == 0.0
    assert torch.count_nonzero(gradient) == 0
