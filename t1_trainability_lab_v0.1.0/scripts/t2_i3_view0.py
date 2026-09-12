"""T2-I3 VIEW-0 independent role views over frozen Writer slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class View0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(68, 15), nn.SiLU(), nn.Linear(15, 1))

    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        if slots.ndim != 3 or tuple(slots.shape[1:]) != (2, 32):
            raise ValueError("VIEW-0 expects slots with shape [batch, 2, 32]")
        batch = slots.shape[0]
        baseline = slots.sum(dim=1)
        outputs = []
        for role in range(2):
            view = torch.zeros((batch, 32), dtype=slots.dtype, device=slots.device)
            role_code = F.one_hot(torch.tensor(role, device=slots.device), 2).to(slots.dtype).expand(batch, -1)
            for slot in range(2):
                slot_code = F.one_hot(torch.tensor(slot, device=slots.device), 2).to(slots.dtype).expand(batch, -1)
                features = torch.cat((slots[:, slot], baseline, role_code, slot_code), dim=-1)
                gate = torch.sigmoid(self.gate(features))
                view = view + gate * slots[:, slot]
            outputs.append(view)
        return outputs[0], outputs[1]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
