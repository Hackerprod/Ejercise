"""Minimal recurrent core specified by T1-A.

The module operates on states shaped ``[batch, slots, dimension]``. A single
state shaped ``[slots, dimension]`` is also accepted and returned unbatched.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


Baseline = Literal["single", "shared", "untied", "vector-state"]


class RMSNorm(nn.Module):
    """Root-mean-square normalization without a bias term."""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return value * scale * self.weight


class SlotMix(nn.Module):
    """Single-head scaled dot-product attention over slots.

    Q/K/V retain dimension ``D``. The output projection is shared across all
    slots and rounds. No residual is applied here; the recurrent residual is
    applied by :class:`RecurrentCore` exactly as specified by T1-A.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.scale = dimension**-0.5

    def forward(self, state: Tensor) -> Tensor:
        query = self.query(state)
        key = self.key(state)
        value = self.value(state)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        return self.output(torch.matmul(attention, value))


class CoreMLP(nn.Module):
    """Per-slot shared F_theta: Linear(D, 4D) -> GELU -> Linear(4D, D)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)


class RecurrentCore(nn.Module):
    """T1-A recurrent core and its four required baseline configurations.

    ``variant`` controls only the baseline comparison:

    - ``single``: one core and one round.
    - ``shared``: one core reused for all configured rounds.
    - ``untied``: one independently parameterized core per round.
    - ``vector-state``: one shared core with ``slots=1``.

    The learned gate is a per-dimension, per-round vector. Its sigmoid logits
    are initialized so each gate is approximately 0.1: this permits gradient
    flow through F_theta from the first update while limiting the initial
    residual magnitude. The depth embedding has dimension D so it can be added
    directly to U without an extra projection.
    """

    VALID_DIMENSIONS = (64, 128)
    VALID_SLOTS = (1, 4, 8)
    VALID_ROUNDS = (1, 2, 4, 6, 8)
    VALID_BASELINES = ("single", "shared", "untied", "vector-state")

    def __init__(
        self,
        dimension: int = 64,
        slots: int = 1,
        rounds: int = 1,
        variant: Baseline = "shared",
    ) -> None:
        super().__init__()
        self._validate_configuration(dimension, slots, rounds, variant)
        self.dimension = dimension
        self.slots = slots
        self.rounds = rounds
        self.variant = variant

        self.slot_mix = SlotMix(dimension)
        self.depth_embedding = nn.Embedding(rounds, dimension)

        core_count = rounds if variant == "untied" else 1
        self.cores = nn.ModuleList(CoreMLP(dimension) for _ in range(core_count))

        gate_probability = 0.1
        gate_logit = math.log(gate_probability / (1.0 - gate_probability))
        self.gate_logits = nn.Parameter(torch.full((rounds, dimension), gate_logit))
        self.rms_norm = RMSNorm(dimension)

    @classmethod
    def _validate_configuration(
        cls,
        dimension: int,
        slots: int,
        rounds: int,
        variant: str,
    ) -> None:
        if dimension not in cls.VALID_DIMENSIONS:
            raise ValueError(f"dimension must be one of {cls.VALID_DIMENSIONS}")
        if slots not in cls.VALID_SLOTS:
            raise ValueError(f"slots must be one of {cls.VALID_SLOTS}")
        if rounds not in cls.VALID_ROUNDS:
            raise ValueError(f"rounds must be one of {cls.VALID_ROUNDS}")
        if variant not in cls.VALID_BASELINES:
            raise ValueError(f"variant must be one of {cls.VALID_BASELINES}")
        if variant == "single" and rounds != 1:
            raise ValueError("single baseline requires rounds=1")
        if variant == "vector-state" and slots != 1:
            raise ValueError("vector-state baseline requires slots=1")

    def _check_state(self, state: Tensor) -> tuple[Tensor, bool]:
        if state.ndim not in (2, 3):
            raise ValueError("state must have shape [S, D] or [B, S, D]")
        unbatched = state.ndim == 2
        batched = state.unsqueeze(0) if unbatched else state
        if batched.shape[1:] != (self.slots, self.dimension):
            raise ValueError(
                f"state must have trailing shape {(self.slots, self.dimension)}, "
                f"got {tuple(batched.shape[1:])}"
            )
        return batched, unbatched

    def forward(self, state: Tensor, *, return_states: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Run configured recurrent rounds and optionally return all states."""

        state, unbatched = self._check_state(state)
        states = [state]
        for round_index in range(self.rounds):
            mixed = self.slot_mix(state)
            depth = self.depth_embedding.weight[round_index].view(1, 1, self.dimension)
            core_index = round_index if self.variant == "untied" else 0
            update = self.cores[core_index](mixed + depth)
            gate = torch.sigmoid(self.gate_logits[round_index]).view(1, 1, self.dimension)
            state = self.rms_norm(state + gate * update)
            states.append(state)

        if unbatched:
            state = state.squeeze(0)
            states = [item.squeeze(0) for item in states]
        if return_states:
            return state, tuple(states)
        return state
