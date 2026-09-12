"""T2-I3 COMP-0: zero-preserving symmetric composition correction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Composition0(nn.Module):
    """Permutation-invariant correction over two frozen Writer slots."""

    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Linear(32, 16)
        self.w_out = nn.Linear(16, 32)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, slots: Tensor) -> Tensor:
        if slots.ndim != 3 or tuple(slots.shape[1:]) != (2, 32):
            raise ValueError("COMP-0 expects slots with shape [batch, 2, 32]")
        h = F.silu(self.phi(slots[:, 0])) + F.silu(self.phi(slots[:, 1]))
        correction = self.w_out(F.silu(h))
        return slots.sum(dim=1) + correction


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
