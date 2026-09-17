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
    first = OmegaCoreLM0ER32.fresh(seed=9, vocab_size=VOCAB, rank=RANK, dimension=DIMENSION, slots=SLOTS, rounds=1)
    actual = torch.rand(4)
    second = OmegaCoreLM0ER32.fresh(seed=9, vocab_size=VOCAB, rank=RANK, dimension=DIMENSION, slots=SLOTS, rounds=1)
    assert_close(actual, expected)
    for name, parameter in first.named_parameters():
        assert_close(parameter, dict(second.named_parameters())[name])
