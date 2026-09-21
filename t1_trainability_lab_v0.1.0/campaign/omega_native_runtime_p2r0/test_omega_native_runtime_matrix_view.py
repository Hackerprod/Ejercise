from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import omega_recurrent_production_bridge as bridge  # noqa: E402


def test_production_bridge_preserves_fused_prelude_gradients() -> None:
    torch.manual_seed(241)
    batch, sequence, slots, dimension, rounds = 1, 2, 2, 3, 1
    state_dimension = slots * dimension
    model = nn.Module()
    model.prelude = nn.Linear(2 * dimension, state_dimension)
    token_features = torch.randn(batch, sequence, dimension)
    token_part = F.linear(token_features, model.prelude.weight[:, :dimension], model.prelude.bias)
    state_part_weight = model.prelude.weight[:, dimension:]
    assert tuple(state_part_weight.shape) == (state_dimension, dimension)
    assert tuple(state_part_weight.stride()) == (2 * dimension, 1)
    assert state_part_weight.data_ptr() == model.prelude.weight.data_ptr() + dimension * model.prelude.weight.element_size()

    previous_state = torch.randn(batch, slots, dimension)
    prelude_norm_weight = torch.randn(state_dimension)
    block_qkv_weight = torch.randn(3 * dimension, dimension) * 0.1
    block_qkv_bias = torch.randn(3 * dimension) * 0.1
    block_out_weight = torch.randn(dimension, dimension) * 0.1
    block_out_bias = torch.randn(dimension) * 0.1
    block_fc1_weight = torch.randn(4 * dimension, dimension) * 0.1
    block_fc1_bias = torch.randn(4 * dimension) * 0.1
    block_fc2_weight = torch.randn(dimension, 4 * dimension) * 0.1
    block_fc2_bias = torch.randn(dimension) * 0.1
    block_norm_weight = torch.randn(dimension)
    depth_embedding = torch.randn(rounds, dimension) * 0.1
    gate_logits = torch.randn(rounds, dimension) * 0.1

    next_state, readout_states = bridge.apply(
        token_part,
        previous_state,
        state_part_weight,
        prelude_norm_weight,
        block_qkv_weight,
        block_qkv_bias,
        block_out_weight,
        block_out_bias,
        block_fc1_weight,
        block_fc1_bias,
        block_fc2_weight,
        block_fc2_bias,
        block_norm_weight,
        depth_embedding,
        gate_logits,
        rounds,
    )
    loss = token_part.square().sum() + next_state.square().sum() + readout_states.square().sum()
    loss.backward()

    assert model.prelude.weight.grad is not None
    token_gradient = model.prelude.weight.grad[:, :dimension]
    state_gradient = model.prelude.weight.grad[:, dimension:]
    assert float(token_gradient.abs().sum()) > 0.0
    assert float(state_gradient.abs().sum()) > 0.0
