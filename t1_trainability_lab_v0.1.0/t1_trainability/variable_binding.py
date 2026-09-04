"""Typed variable-binding core with reference replacement and value overwrite."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import RMSNorm


class VariableBindingCore(nn.Module):
    """Shared key-addressed reader over ASSIGN and ATTR rows.

    Reference slot ``R`` is initialized with VAR:X and replaced by OBJECT:N
    on round one. Value slot ``V`` is written only by ATTR rows from round two
    onward. No workspace slot or cross-slot mixing is used.
    """

    ASSIGN = 0
    ATTR = 1

    def __init__(self, dimension: int = 64, reference_count: int = 33, value_count: int = 8, rounds: int = 4) -> None:
        super().__init__()
        self.dimension = dimension
        self.reference_count = reference_count
        self.value_count = value_count
        self.rounds = rounds
        self.reference_embedding = nn.Embedding(reference_count, dimension)
        self.value_embedding = nn.Embedding(value_count, dimension)
        self.query = nn.Linear(dimension, dimension)
        self.query_norm = RMSNorm(dimension)
        self.scale = dimension**-0.5

    def _read(
        self,
        reference: Tensor,
        sources: Tensor,
        destinations: Tensor,
        row_types: Tensor,
        row_mask: Tensor,
        row_type: int,
        destination_embedding: nn.Embedding,
    ) -> Tensor:
        query = self.query(self.query_norm(reference[:, 0, :]))
        key_logits = torch.matmul(query, self.reference_embedding.weight.transpose(0, 1)) * self.scale
        source_logits = key_logits.gather(1, sources)
        allowed = row_mask & (row_types == row_type)
        source_logits = source_logits.masked_fill(~allowed, torch.finfo(source_logits.dtype).min)
        attention = torch.softmax(source_logits, dim=-1)
        # Destination id domains differ by row type (OBJECT vs COLOR). Masked
        # rows still pass through embedding lookup, so replace their indices
        # with a valid zero before selecting typed destinations.
        safe_destinations = destinations.masked_fill(row_types != row_type, 0)
        values = destination_embedding(safe_destinations)
        return torch.matmul(attention.unsqueeze(1), values)

    def forward(
        self,
        query_references: Tensor,
        sources: Tensor,
        destinations: Tensor,
        row_types: Tensor,
        row_mask: Tensor,
        *,
        rounds: int = 4,
        return_states: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        if rounds < 1 or rounds > self.rounds:
            raise ValueError(f"rounds must be between 1 and {self.rounds}")
        if sources.shape != destinations.shape or sources.shape != row_types.shape or sources.shape != row_mask.shape:
            raise ValueError("memory tensors must have identical shape [B, memory_size]")
        reference = self.reference_embedding(query_references).unsqueeze(1)
        value = torch.zeros_like(reference)
        reference_states = [reference]
        value_states = [value]
        for round_index in range(rounds):
            if round_index == 0:
                reference = self._read(
                    reference,
                    sources,
                    destinations,
                    row_types,
                    row_mask,
                    self.ASSIGN,
                    self.reference_embedding,
                )
            else:
                value = self._read(
                    reference,
                    sources,
                    destinations,
                    row_types,
                    row_mask,
                    self.ATTR,
                    self.value_embedding,
                )
            reference_states.append(reference)
            value_states.append(value)
        if return_states:
            return reference, value, tuple(reference_states), tuple(value_states)
        return reference, value


class VariableBindingHead(nn.Module):
    """Classify colors from normalized value slot only."""

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
