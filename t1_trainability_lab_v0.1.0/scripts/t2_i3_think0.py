"""T2-I3 THINK-0: shared recurrent residual workspace integration cell."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

WORKSPACE_SLOTS = 2
WORKSPACE_DIMENSION = 32
ROLE_DIMENSION = 4
STEP_DIMENSION = 4
INPUT_DIMENSION = 104
HIDDEN_DIMENSION = 32
THINK_ROUNDS = 2


class Think0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(WORKSPACE_DIMENSION, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(WORKSPACE_DIMENSION, elementwise_affine=False)
        self.anchor_norm = nn.LayerNorm(WORKSPACE_DIMENSION, elementwise_affine=False)
        self.role_embedding = nn.Embedding(WORKSPACE_SLOTS, ROLE_DIMENSION)
        self.step_embedding = nn.Embedding(THINK_ROUNDS, STEP_DIMENSION)
        self.w1 = nn.Linear(INPUT_DIMENSION, HIDDEN_DIMENSION)
        self.w2 = nn.Linear(HIDDEN_DIMENSION, WORKSPACE_DIMENSION)
        self.gate = nn.Linear(INPUT_DIMENSION, 1)

    def _validate(self, workspace: Tensor) -> None:
        if workspace.ndim != 3 or tuple(workspace.shape[1:]) != (WORKSPACE_SLOTS, WORKSPACE_DIMENSION): raise ValueError("THINK expects S0 with shape [batch, 2, 32]")

    def _step(self, state: Tensor, anchor: Tensor, round_index: int) -> Tensor:
        context = state.mean(dim=1, keepdim=True).expand(-1, WORKSPACE_SLOTS, -1); role = self.role_embedding.weight.unsqueeze(0).expand(state.shape[0], -1, -1); step = self.step_embedding.weight[round_index].view(1, 1, -1).expand(state.shape[0], WORKSPACE_SLOTS, -1); features = torch.cat((self.slot_norm(state), self.context_norm(context), self.anchor_norm(anchor), role, step), dim=-1); delta = self.w2(F.silu(self.w1(features))); gate = torch.sigmoid(self.gate(features)); return state + gate * delta

    def run_rounds(self, workspace: Tensor, rounds: int) -> Tensor:
        self._validate(workspace)
        if rounds < 0 or rounds > THINK_ROUNDS: raise ValueError(f"rounds must be in [0, {THINK_ROUNDS}]")
        state = workspace; anchor = workspace.detach()
        for round_index in range(rounds): state = self._step(state, anchor, round_index)
        return state

    def forward(self, workspace: Tensor) -> Tensor:
        self._validate(workspace); return self.run_rounds(workspace, THINK_ROUNDS)


def parameter_count(model: nn.Module) -> int: return sum(parameter.numel() for parameter in model.parameters())


def architecture_report(model: Think0 | None = None) -> dict[str, Any]:
    think = Think0() if model is None else model
    return {"module": "Think0", "task": "T2-I3", "input": "[batch,2,32]", "rounds": THINK_ROUNDS, "shared_weights": True, "attention": False, "layer_norm_affine": False, "input_dimension": INPUT_DIMENSION, "hidden_dimension": HIDDEN_DIMENSION, "role_embedding_shape": [2, ROLE_DIMENSION], "step_embedding_shape": [THINK_ROUNDS, STEP_DIMENSION], "w1_shape": [HIDDEN_DIMENSION, INPUT_DIMENSION], "w2_shape": [WORKSPACE_DIMENSION, HIDDEN_DIMENSION], "gate_shape": [1, INPUT_DIMENSION], "trainable_parameters": parameter_count(think), "pooling": "recurrent residual workspace integration"}
