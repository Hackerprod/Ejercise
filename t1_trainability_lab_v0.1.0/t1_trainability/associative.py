"""Typed associative-recall core: stable query slot plus replacement value slot."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import RMSNorm


class AssociativeCore(nn.Module):
    """Key-addressed reader with P=1 stable query and V=1 replacement value."""

    def __init__(self, dimension: int = 64, key_count: int = 32, value_count: int = 32, rounds: int = 4) -> None:
        super().__init__()
        self.dimension = dimension
        self.key_count = key_count
        self.value_count = value_count
        self.rounds = rounds
        self.key_embedding = nn.Embedding(key_count, dimension)
        self.value_embedding = nn.Embedding(value_count, dimension)
        self.query = nn.Linear(dimension, dimension)
        self.query_norm = RMSNorm(dimension)
        self.alpha = None
        self.scale = dimension**-0.5

    def read_value(self, pointer: Tensor, pair_keys: Tensor, pair_values: Tensor, pair_mask: Tensor) -> Tensor:
        query = self.query(self.query_norm(pointer[:, 0, :]))
        key_logits = torch.matmul(query, self.key_embedding.weight.transpose(0, 1)) * self.scale
        source_logits = key_logits.gather(1, pair_keys)
        source_logits = source_logits.masked_fill(~pair_mask, torch.finfo(source_logits.dtype).min)
        attention = torch.softmax(source_logits, dim=-1)
        values = self.value_embedding(pair_values)
        return torch.matmul(attention.unsqueeze(1), values).squeeze(1).unsqueeze(1)

    def forward(self, query_keys: Tensor, pair_keys: Tensor, pair_values: Tensor, pair_mask: Tensor, *, rounds: int = 4, return_states: bool = False):
        if rounds < 1 or rounds > self.rounds:
            raise ValueError(f"rounds must be between 1 and {self.rounds}")
        pointer = self.key_embedding(query_keys).unsqueeze(1)
        value = torch.zeros_like(pointer)
        pointer_states = [pointer]
        value_states = [value]
        for _ in range(rounds):
            value = self.read_value(pointer, pair_keys, pair_values, pair_mask)
            pointer_states.append(pointer)
            value_states.append(value)
        if return_states:
            return pointer, value, tuple(pointer_states), tuple(value_states)
        return pointer, value


class AssociativeHead(nn.Module):
    """Task head reading only normalized retrieved-value slot V."""

    def __init__(self, dimension: int, value_embedding: nn.Embedding) -> None:
        super().__init__()
        self.norm = RMSNorm(dimension)
        self.query = nn.Linear(dimension, dimension)
        self.value_embedding = value_embedding
        self.scale = dimension**-0.5

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.norm(value[:, 0, :])
        query = self.query(normalized)
        return torch.matmul(query, self.value_embedding.weight.transpose(0, 1)) * self.scale
