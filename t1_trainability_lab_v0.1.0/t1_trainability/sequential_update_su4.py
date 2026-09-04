"""SU-4 shared operator trunk with operation-specific output heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import RMSNorm


class SequentialUpdateSU4Core(nn.Module):
    """One-step typed operator with shared trunk and three small heads."""

    OP_ADD = 0
    OP_SUB = 1
    OP_MUL = 2

    def __init__(self, dimension: int = 64, value_count: int = 32, operation_count: int = 3) -> None:
        super().__init__()
        self.dimension = dimension
        self.value_count = value_count
        self.operation_count = operation_count
        self.value_embedding = nn.Embedding(value_count, dimension)
        self.operand_embedding = nn.Embedding(value_count, dimension)
        self.operation_embedding = nn.Embedding(operation_count, dimension)
        self.input_norm = RMSNorm(dimension)
        self.left_projection = nn.Linear(dimension, dimension)
        self.right_projection = nn.Linear(dimension, dimension)
        self.trunk = nn.Sequential(
            nn.Linear(4 * dimension, 4 * dimension),
            nn.SiLU(),
            nn.Linear(4 * dimension, dimension),
        )
        self.operation_heads = nn.ModuleList(nn.Linear(dimension, value_count) for _ in range(operation_count))

    def operator_input(self, register: Tensor, operand: Tensor, operation: Tensor) -> Tensor:
        left = self.left_projection(self.input_norm(self.value_embedding(register)))
        right = self.right_projection(self.operand_embedding(operand))
        op_state = self.operation_embedding(operation)
        return torch.cat((left, right, op_state, left - right), dim=-1)

    def forward(self, register: Tensor, operand: Tensor, operation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.operator_input(register, operand, operation)
        hidden = self.trunk(features)
        all_logits = torch.stack([head(hidden) for head in self.operation_heads], dim=1)
        selected_logits = all_logits[torch.arange(register.shape[0], device=register.device), operation]
        probabilities = torch.softmax(selected_logits, dim=-1)
        next_register = probabilities @ self.value_embedding.weight
        return next_register.unsqueeze(1), selected_logits, hidden
