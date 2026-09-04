"""Pointer-chasing memory reader and fixed-alpha pre-norm recurrent core."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .model import CoreMLP, RMSNorm


class PointerCore(nn.Module):
    """S=1 differentiable key-addressed recurrent core using fixed alpha."""

    def __init__(self, dimension: int = 64, key_count: int = 256, rounds: int = 4, transition: str = "residual_pre_norm") -> None:
        super().__init__()
        if transition not in {"residual_pre_norm", "pointer_replacement"}:
            raise ValueError("transition must be residual_pre_norm or pointer_replacement")
        self.dimension = dimension
        self.key_count = key_count
        self.rounds = rounds
        self.transition = transition
        self.key_embedding = nn.Embedding(key_count, dimension)
        self.query = nn.Linear(dimension, dimension)
        self.core = CoreMLP(dimension)
        self.input_norm = RMSNorm(dimension)
        self.output_norm = RMSNorm(dimension)
        self.alpha = rounds**-0.5
        self.scale = dimension**-0.5

    def initial_state(self, start_keys: Tensor) -> Tensor:
        return self.key_embedding(start_keys).unsqueeze(1)

    def _read_one_hop(
        self,
        normalized_state: Tensor,
        memory_sources: Tensor,
        memory_destinations: Tensor,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        query = self.query(normalized_state[:, 0, :])
        current_logits = torch.matmul(query, self.key_embedding.weight.transpose(0, 1)) * self.scale
        # Differentiable key-addressed mask: only rows whose source key matches
        # the state-produced current distribution receive substantial mass.
        source_log_mask = torch.log_softmax(current_logits, dim=-1).gather(1, memory_sources)
        if memory_mask is not None:
            source_log_mask = source_log_mask.masked_fill(~memory_mask, torch.finfo(source_log_mask.dtype).min)
        attention = torch.softmax(source_log_mask, dim=-1)
        destinations = self.key_embedding(memory_destinations)
        return torch.matmul(attention.unsqueeze(1), destinations).squeeze(1)

    def forward(
        self,
        start_keys: Tensor,
        memory_sources: Tensor,
        memory_destinations: Tensor,
        memory_mask: Tensor | None = None,
        *,
        rounds: int | None = None,
        required_hops: Tensor | None = None,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        if start_keys.ndim != 1:
            raise ValueError("start_keys must have shape [B]")
        if memory_sources.ndim != 2 or memory_destinations.shape != memory_sources.shape:
            raise ValueError("memory tensors must have shape [B, memory_size]")
        run_rounds = self.rounds if rounds is None else rounds
        if not 1 <= run_rounds <= self.rounds:
            raise ValueError(f"rounds must be between 1 and {self.rounds}")
        state = self.initial_state(start_keys)
        states = [state]
        for round_index in range(run_rounds):
            normalized = self.input_norm(state)
            retrieved = self._read_one_hop(normalized, memory_sources, memory_destinations, memory_mask).unsqueeze(1)
            if self.transition == "pointer_replacement":
                next_state = retrieved
            else:
                delta = self.core(normalized + retrieved)
                next_state = state + self.alpha * delta
            if required_hops is not None:
                active = (required_hops > round_index).view(-1, 1, 1)
                state = torch.where(active, next_state, state)
            else:
                state = next_state
            states.append(state)
        if return_states:
            return state, tuple(states)
        return state


class PointerHead(nn.Module):
    """Final normalized state to global opaque-key logits."""

    def __init__(self, dimension: int, key_embedding: nn.Embedding) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.norm = RMSNorm(dimension)
        self.key_embedding = key_embedding
        self.scale = dimension**-0.5

    def forward(self, state: Tensor) -> Tensor:
        normalized = self.norm(state[:, 0, :])
        query = self.query(normalized)
        return torch.matmul(query, self.key_embedding.weight.transpose(0, 1)) * self.scale
