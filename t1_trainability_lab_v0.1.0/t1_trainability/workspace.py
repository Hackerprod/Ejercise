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


class TransformCorrectionMLP(nn.Module):
    """Shared full-width workspace correction conditioned by transform opcode."""

    def __init__(self, dimension: int = 64, *, transform_count: int = 4, embedding_width: int = 16) -> None:
        super().__init__()
        if dimension < 1 or transform_count < 1 or embedding_width < 1:
            raise ValueError("dimension, transform_count, and embedding_width must be positive")
        self.dimension = dimension
        self.transform_count = transform_count
        self.embedding_width = embedding_width
        self.transform_embedding = nn.Embedding(transform_count, embedding_width)
        with torch.no_grad():
            self.transform_embedding.weight.zero_()
            self.transform_embedding.weight[:, :transform_count].copy_(torch.eye(transform_count))
        # Multiplicative opcode gating gives one shared MLP direct access to a
        # transform-conditioned payload basis without introducing per-transform
        # heads.  Workspace context remains a separate input and stays in-graph.
        self.network = nn.Sequential(
            nn.Linear((transform_count + 1) * dimension + embedding_width, 4 * dimension),
            nn.SiLU(),
            nn.Linear(4 * dimension, dimension),
        )
        self.payload_basis = nn.Linear(transform_count * dimension, dimension)
        # Preserve exact identity transport at initialization.
        nn.init.zeros_(self.payload_basis.weight)
        nn.init.zeros_(self.payload_basis.bias)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, payload: Tensor, workspace: Tensor, transform_id: Tensor) -> Tensor:
        if payload.ndim != 2 or payload.shape[-1] != self.dimension:
            raise ValueError("payload must have shape [B, D]")
        if workspace.shape != payload.shape:
            raise ValueError("workspace must have shape [B, D]")
        if transform_id.ndim != 1 or transform_id.shape[0] != payload.shape[0]:
            raise ValueError("transform_id must have shape [B]")
        normalized_workspace = workspace * torch.rsqrt(workspace.square().mean(dim=-1, keepdim=True) + 1e-6)
        instruction = self.transform_embedding(transform_id.to(dtype=torch.long))
        gated_payload = (payload.unsqueeze(-1) * instruction[:, : self.transform_count].unsqueeze(1)).flatten(start_dim=1)
        return self.payload_basis(gated_payload) + self.network(torch.cat((gated_payload, normalized_workspace, instruction), dim=-1))
