"""Continuous residual workspace accumulator for T1-W."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import RMSNorm


class WorkspaceCore(nn.Module):
    """Shared residual accumulator with inference-only transition ablations."""

    MODES = ("residual", "frozen", "replaced")

    def __init__(self, dimension: int = 64, *, identity_bypass: bool = False) -> None:
        super().__init__()
        self.dimension = dimension
        self.identity_bypass = identity_bypass
        self.input_norm = RMSNorm(dimension)
        if identity_bypass:
            self.correction_mlp = nn.Sequential(
                nn.Linear(2 * dimension, 4 * dimension),
                nn.SiLU(),
                nn.Linear(4 * dimension, dimension),
            )
            nn.init.zeros_(self.correction_mlp[-1].weight)
            nn.init.zeros_(self.correction_mlp[-1].bias)
        else:
            self.operator = nn.Sequential(
                nn.Linear(2 * dimension, 4 * dimension),
                nn.SiLU(),
                nn.Linear(4 * dimension, dimension),
            )

    def forward(self, vectors: Tensor, lengths: Tensor, *, rounds: int = 6, mode: str = "residual", return_states: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if vectors.ndim != 3 or vectors.shape[-1] != self.dimension:
            raise ValueError("vectors must have shape [B, max_h, dimension]")
        if lengths.ndim != 1 or lengths.shape[0] != vectors.shape[0]:
            raise ValueError("lengths must have shape [B]")
        if not 1 <= rounds <= vectors.shape[1]:
            raise ValueError(f"rounds must be between 1 and {vectors.shape[1]}")
        workspace = torch.zeros_like(vectors[:, 0, :])
        states = [workspace]
        for round_index in range(rounds):
            evidence = vectors[:, round_index, :]
            normalized_workspace = self.input_norm(workspace)
            operator_input = torch.cat((evidence, normalized_workspace), dim=-1)
            if self.identity_bypass:
                delta = evidence + self.correction_mlp(operator_input)
            else:
                delta = self.operator(operator_input)
            active = (lengths > round_index).view(-1, 1)
            if mode == "residual":
                candidate = workspace + delta
            elif mode == "replaced":
                candidate = delta
            else:
                candidate = workspace
            workspace = torch.where(active, candidate, workspace)
            states.append(workspace)
        if return_states:
            return workspace, tuple(states)
        return workspace
