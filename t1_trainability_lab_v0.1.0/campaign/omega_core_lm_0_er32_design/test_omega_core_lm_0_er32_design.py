"""Synthetic ER32 correctness gate; no campaign imports or real data."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omega_core_lm_0_er32_design import (  # noqa: E402
    ATOL,
    RTOL,
    LOSS_TEMPERATURE,
    OmegaCoreLM0ER32,
    ce_kl_loss,
)


VOCAB = 19
RANK = 5
DIMENSION = 7
SLOTS = 3


def make_pair(rounds: int) -> tuple[OmegaCoreLM0ER32, OmegaCoreLM0ER32]:
    explicit = OmegaCoreLM0ER32.fresh(
        seed=20260917,
        vocab_size=VOCAB,
        rank=RANK,
        dimension=DIMENSION,
        slots=SLOTS,
        rounds=rounds,
        implementation="explicit",
    )
    efficient = OmegaCoreLM0ER32(
        vocab_size=VOCAB,
        rank=RANK,
        dimension=DIMENSION,
        slots=SLOTS,
        rounds=rounds,
        implementation="efficient",
    )
    efficient.load_state_dict(explicit.state_dict())
    return explicit, efficient


def batch_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, False], [True, False, True]])
    teacher = torch.linspace(-0.7, 0.8, steps=tokens.numel() * VOCAB).view(2, 3, VOCAB)
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
    previous = torch.linspace(-0.2, 0.3, steps=2 * SLOTS * DIMENSION).view(2, SLOTS, DIMENSION)
    return tokens, valid_mask, teacher, targets, previous


def assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("rounds", [1, 4])
def test_explicit_and_efficient_logits_states_and_loss(rounds: int) -> None:
    explicit, efficient = make_pair(rounds)
    tokens, valid_mask, teacher, targets, previous = batch_inputs()
    result_explicit = explicit(tokens, previous, valid_mask)
    result_efficient = efficient(tokens, previous, valid_mask)

    for left, right in zip(result_explicit, result_efficient):
        assert_close(left, right)
    loss_explicit = ce_kl_loss(result_explicit[1], teacher, targets, valid_mask)
    loss_efficient = ce_kl_loss(result_efficient[1], teacher, targets, valid_mask)
    assert LOSS_TEMPERATURE == 2.0
    for key in ("ce", "kl", "total"):
        assert_close(loss_explicit[key], loss_efficient[key])


@pytest.mark.parametrize("rounds", [1, 4])
def test_explicit_and_efficient_gradients_and_clipping(rounds: int) -> None:
    explicit, efficient = make_pair(rounds)
    tokens, valid_mask, teacher, targets, previous = batch_inputs()
    loss_explicit = ce_kl_loss(explicit(tokens, previous, valid_mask)[1], teacher, targets, valid_mask)["total"]
    loss_efficient = ce_kl_loss(efficient(tokens, previous, valid_mask)[1], teacher, targets, valid_mask)["total"]
    loss_explicit.backward()
    loss_efficient.backward()

    explicit_parameters = dict(explicit.named_parameters())
    efficient_parameters = dict(efficient.named_parameters())
    assert {"C", "U"}.issubset(explicit_parameters)
    assert {"C", "U"}.issubset(efficient_parameters)
    assert explicit_parameters["C"].grad is not None
    assert explicit_parameters["U"].grad is not None
    assert explicit_parameters["token_projection.weight"].grad is not None
    for name, parameter in explicit_parameters.items():
        assert parameter.grad is not None, name
        assert_close(parameter.grad, efficient_parameters[name].grad)

    norm_explicit = clip_grad_norm_(explicit.parameters(), max_norm=0.15)
    norm_efficient = clip_grad_norm_(efficient.parameters(), max_norm=0.15)
    assert_close(norm_explicit, norm_efficient)
    for name, parameter in explicit_parameters.items():
        assert_close(parameter.grad, efficient_parameters[name].grad)


@pytest.mark.parametrize("rounds", [1, 4])
def test_one_adamw_update_matches_parameters_and_both_moments(rounds: int) -> None:
    explicit, efficient = make_pair(rounds)
    tokens, valid_mask, teacher, targets, previous = batch_inputs()
    optimizer_explicit = torch.optim.AdamW(explicit.parameters(), lr=0.003, weight_decay=0.01)
    optimizer_efficient = torch.optim.AdamW(efficient.parameters(), lr=0.003, weight_decay=0.01)
    loss_explicit = ce_kl_loss(explicit(tokens, previous, valid_mask)[1], teacher, targets, valid_mask)["total"]
    loss_efficient = ce_kl_loss(efficient(tokens, previous, valid_mask)[1], teacher, targets, valid_mask)["total"]
    loss_explicit.backward()
    loss_efficient.backward()
    clip_grad_norm_(explicit.parameters(), max_norm=0.15)
    clip_grad_norm_(efficient.parameters(), max_norm=0.15)
    optimizer_explicit.step()
    optimizer_efficient.step()

    efficient_parameters = dict(efficient.named_parameters())
    for name, parameter in explicit.named_parameters():
        other = efficient_parameters[name]
        assert_close(parameter, other)
        state = optimizer_explicit.state[parameter]
        other_state = optimizer_efficient.state[other]
        assert_close(state["exp_avg"], other_state["exp_avg"])
        assert_close(state["exp_avg_sq"], other_state["exp_avg_sq"])


def test_factors_are_only_tied_input_output_parameters() -> None:
    model = OmegaCoreLM0ER32(vocab_size=VOCAB, rank=RANK, dimension=DIMENSION, slots=SLOTS, rounds=1)
    names = dict(model.named_parameters())
    assert set(("C", "U")).issubset(names)
    assert not any(parameter.shape == (VOCAB, DIMENSION) for parameter in names.values())
    assert not any("embedding" in name.lower() or "output_projection" in name.lower() for name in names)
    assert model.C.shape == (VOCAB, RANK)
    assert model.U.shape == (RANK, DIMENSION)


def test_window_zero_mask_and_window_one_state_transfer() -> None:
    explicit, efficient = make_pair(rounds=4)
    initial = explicit.initial_state(2)
    window_zero = torch.tensor([[1, 8], [2, 3]], dtype=torch.long)
    mask_zero = torch.tensor([[True, False], [True, True]])
    window_one = torch.tensor([[4, 5], [6, 7]], dtype=torch.long)
    mask_one = torch.ones_like(window_one, dtype=torch.bool)

    state_zero, _, states_zero = explicit(window_zero, initial, mask_zero)
    state_one, logits_one, states_one = explicit(window_one, state_zero, mask_one)
    efficient_zero = efficient(window_zero, initial, mask_zero)
    efficient_one = efficient(window_one, efficient_zero[0], mask_one)
    for left, right in zip((state_zero, states_zero, state_one, logits_one), (efficient_zero[0], efficient_zero[2], efficient_one[0], efficient_one[1])):
        assert_close(left, right)

    single_zero = explicit(window_zero[:, :1], initial, torch.ones((2, 1), dtype=torch.bool))[0]
    assert_close(state_zero[0:1], single_zero[0:1])
    assert states_one.shape == (2, 2, SLOTS, DIMENSION)


def test_fresh_seed_is_deterministic_and_preserves_global_rng_stream() -> None:
    torch.manual_seed(31415)
    expected = torch.rand(4)
    torch.manual_seed(31415)
    rng_before = torch.get_rng_state()
    first = OmegaCoreLM0ER32.fresh(seed=9, vocab_size=VOCAB, rank=RANK, dimension=DIMENSION, slots=SLOTS, rounds=1)
    rng_after = torch.get_rng_state()
    actual = torch.rand(4)
    second = OmegaCoreLM0ER32.fresh(seed=9, vocab_size=VOCAB, rank=RANK, dimension=DIMENSION, slots=SLOTS, rounds=1)
    assert torch.equal(rng_before, rng_after)
    assert torch.equal(actual, expected)
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, dict(second.named_parameters())[name])


def test_factor_initialization_matches_embedding_marginal_variance() -> None:
    model = OmegaCoreLM0ER32.fresh(
        seed=20260917,
        vocab_size=4096,
        rank=32,
        dimension=64,
        slots=SLOTS,
        rounds=1,
    )
    effective_embedding = model.C @ model.U
    marginal_variance = effective_embedding.var(dim=0, unbiased=False).mean()
    torch.testing.assert_close(marginal_variance, torch.ones_like(marginal_variance), atol=0.1, rtol=0.1)


def _run_nominal_two_window_update(model: OmegaCoreLM0ER32) -> dict[str, object]:
    tokens_zero = torch.tensor([[1, 8], [2, 3]], dtype=torch.long)
    mask_zero = torch.tensor([[True, False], [True, True]])
    tokens_one = torch.tensor([[4, 5], [6, 7]], dtype=torch.long)
    mask_one = torch.ones_like(tokens_one, dtype=torch.bool)
    teacher_zero = torch.linspace(-0.7, 0.8, steps=tokens_zero.numel() * VOCAB).view(2, 2, VOCAB)
    teacher_one = torch.linspace(0.8, -0.7, steps=tokens_one.numel() * VOCAB).view(2, 2, VOCAB)
    targets_zero = torch.tensor([[2, 3], [5, 6]], dtype=torch.long)
    targets_one = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    state_zero, logits_zero, _ = model(tokens_zero, model.initial_state(2), mask_zero)
    loss_zero = ce_kl_loss(logits_zero, teacher_zero, targets_zero, mask_zero)["total"]
    loss_zero.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    state_zero = state_zero.detach()

    optimizer.zero_grad()
    state_one, logits_one, _ = model(tokens_one, state_zero, mask_one)
    loss_one = ce_kl_loss(logits_one, teacher_one, targets_one, mask_one)["total"]
    loss_one.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()

    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_state = {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state[parameter].items()
        }
        for name, parameter in model.named_parameters()
    }
    return {
        "losses": (loss_zero.detach(), loss_one.detach()),
        "states": (state_zero, state_one.detach()),
        "gradients": gradients,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
    }


@pytest.mark.parametrize("rounds", [1, 4])
def test_nominal_two_window_causal_training_matches_explicit_and_efficient(rounds: int) -> None:
    explicit, efficient = make_pair(rounds)
    expected = _run_nominal_two_window_update(explicit)
    actual = _run_nominal_two_window_update(efficient)

    for expected_loss, actual_loss in zip(expected["losses"], actual["losses"]):
        assert_close(expected_loss, actual_loss)
    for expected_state, actual_state in zip(expected["states"], actual["states"]):
        assert_close(expected_state, actual_state)
    for name, expected_gradient in expected["gradients"].items():
        assert_close(expected_gradient, actual["gradients"][name])
    for name, expected_parameter in expected["parameters"].items():
        assert_close(expected_parameter, actual["parameters"][name])
    for name, expected_state in expected["optimizer_state"].items():
        actual_state = actual["optimizer_state"][name]
        for key in ("exp_avg", "exp_avg_sq"):
            assert_close(expected_state[key], actual_state[key])
        assert expected_state["step"] == actual_state["step"]
        assert expected_state["step"] == 2.0


def test_two_window_without_detach_reaches_window_zero_graph() -> None:
    model = OmegaCoreLM0ER32.fresh(
        seed=20260917,
        vocab_size=VOCAB,
        rank=RANK,
        dimension=DIMENSION,
        slots=SLOTS,
        rounds=4,
    )
    tokens_zero = torch.tensor([[1, 8], [2, 3]], dtype=torch.long)
    tokens_one = torch.tensor([[4, 5], [6, 7]], dtype=torch.long)
    state_zero = model(tokens_zero, model.initial_state(2))[0]
    state_zero.retain_grad()
    state_one = model(tokens_one, state_zero)[0]
    state_one.sum().backward()

    assert state_zero.requires_grad
    assert state_zero.grad_fn is not None
    assert state_one.requires_grad
    assert state_zero.grad is not None
