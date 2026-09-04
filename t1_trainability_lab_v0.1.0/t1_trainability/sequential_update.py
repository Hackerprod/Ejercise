"""Typed sequential-update core with replacement-only register semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import RMSNorm


class SequentialUpdateCore(nn.Module):
    """Apply one indexed operation per round to one register slot.

    The step reader is intentionally deterministic: round ``r`` selects the
    row whose ``step_index == r``. The shared operator then emits a complete
    candidate register state, which replaces the previous value.
    """

    OP_ADD = 0
    OP_SUB = 1
    OP_MUL = 2

    def __init__(self, dimension: int = 64, value_count: int = 32, operation_count: int = 3, max_steps: int = 6) -> None:
        super().__init__()
        self.dimension = dimension
        self.value_count = value_count
        self.max_steps = max_steps
        self.value_embedding = nn.Embedding(value_count, dimension)
        self.operation_embedding = nn.Embedding(operation_count, dimension)
        self.operand_embedding = nn.Embedding(value_count, dimension)
        self.input_norm = RMSNorm(dimension)
        self.operator = nn.Sequential(
            nn.Linear(3 * dimension, 4 * dimension),
            nn.GELU(),
            nn.Linear(4 * dimension, value_count),
        )

    def read_operation(self, operation_types: Tensor, operands: Tensor, step_index: int) -> tuple[Tensor, Tensor]:
        """Read exactly indexed operation row for current round."""

        if not 0 <= step_index < operation_types.shape[1]:
            raise ValueError(f"step_index {step_index} outside memory width {operation_types.shape[1]}")
        return operation_types[:, step_index], operands[:, step_index]

    def forward(
        self,
        initial_values: Tensor,
        operation_types: Tensor,
        operands: Tensor,
        step_mask: Tensor,
        *,
        rounds: int = 6,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        if rounds < 1 or rounds > self.max_steps:
            raise ValueError(f"rounds must be between 1 and {self.max_steps}")
        if operation_types.shape != operands.shape or operation_types.shape != step_mask.shape:
            raise ValueError("operation tensors must have identical shape [B, max_steps]")
        register = self.value_embedding(initial_values).unsqueeze(1)
        states = [register]
        for round_index in range(rounds):
            operation_type, operand = self.read_operation(operation_types, operands, round_index)
            operation_state = self.operation_embedding(operation_type)
            operand_state = self.operand_embedding(operand)
            normalized = self.input_norm(register[:, 0, :])
            logits = self.operator(torch.cat((normalized, operation_state, operand_state), dim=-1))
            candidate = torch.softmax(logits, dim=-1) @ self.value_embedding.weight
            active = step_mask[:, round_index].view(-1, 1, 1)
            register = torch.where(active, candidate.unsqueeze(1), register)
            states.append(register)
        if return_states:
            return register, tuple(states)
        return register


class SequentialUpdateHead(nn.Module):
    """Decode current register slot into 32 value classes."""

    def __init__(self, dimension: int, value_embedding: nn.Embedding) -> None:
        super().__init__()
        self.norm = RMSNorm(dimension)
        self.query = nn.Linear(dimension, dimension)
        self.value_embedding = value_embedding
        self.scale = dimension**-0.5

    def forward(self, register: Tensor) -> Tensor:
        normalized = self.norm(register[:, 0, :])
        query = self.query(normalized)
        return torch.matmul(query, self.value_embedding.weight.transpose(0, 1)) * self.scale
