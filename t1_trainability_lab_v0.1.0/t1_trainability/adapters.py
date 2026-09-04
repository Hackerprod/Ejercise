"""Input and output adapters for synthetic T1 task examples."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .data import OUTPUT_CARDINALITIES, TaskName
from .model import RMSNorm


class InputAdapter(nn.Module):
    """Encode typed token sequences into initial ``[B, S, D]`` states.

    Learned slot queries perform one cross-attention read over token and
    positional embeddings. This adapter adds no recurrent computation.
    """

    def __init__(self, vocabulary_size: int, dimension: int, slots: int, max_length: int = 64) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.slots = slots
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocabulary_size, dimension)
        self.position_embedding = nn.Embedding(max_length, dimension)
        self.slot_queries = nn.Parameter(torch.randn(slots, dimension))
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.scale = dimension**-0.5

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        batch_size, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length={self.max_length}")
        positions = torch.arange(length, device=input_ids.device).view(1, length)
        tokens = self.token_embedding(input_ids) + self.position_embedding(positions)
        queries = self.query(self.slot_queries).unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.key(tokens)
        values = self.value(tokens)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            if not attention_mask.any(dim=-1).all():
                raise ValueError("each sequence must contain at least one non-padding token")
            scores = scores.masked_fill(~attention_mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return self.output(torch.matmul(weights, values))


class OutputReader(nn.Module):
    """Query-conditioned slot pooling with one linear head per task."""

    def __init__(self, vocabulary_size: int, dimension: int) -> None:
        super().__init__()
        self.query_embedding = nn.Embedding(vocabulary_size, dimension)
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.final_norm = RMSNorm(dimension)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(dimension, cardinality) for task, cardinality in OUTPUT_CARDINALITIES.items()}
        )
        self.scale = math.sqrt(dimension) ** -1

    def forward(self, state: Tensor, query_ids: Tensor, task: TaskName) -> Tensor:
        if state.ndim != 3:
            raise ValueError("state must have shape [B, S, D]")
        if query_ids.ndim != 1 or query_ids.shape[0] != state.shape[0]:
            raise ValueError("query_ids must have shape [B]")
        query = self.query(self.query_embedding(query_ids)).unsqueeze(1)
        keys = self.key(state)
        values = self.value(state)
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        pooled = self.output(torch.matmul(weights, values)).squeeze(1)
        pooled = self.final_norm(pooled)
        try:
            head = self.heads[task]
        except KeyError as error:
            raise ValueError(f"Unknown task: {task}") from error
        return head(pooled)
