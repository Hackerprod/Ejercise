"""T2-I3 THINK-1: anchored recurrent proposal workspace integration cell."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from t2_i3_think0 import WORKSPACE_SLOTS, Think0


class Think1(Think0):
    """THINK-0 with anchored proposal interpolation and identical parameters."""

    def _step(self, state: Tensor, anchor: Tensor, round_index: int) -> Tensor:
        context = state.mean(dim=1, keepdim=True).expand(-1, WORKSPACE_SLOTS, -1)
        role = self.role_embedding.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
        step = self.step_embedding.weight[round_index].view(1, 1, -1).expand(state.shape[0], WORKSPACE_SLOTS, -1)
        features = torch.cat((self.slot_norm(state), self.context_norm(context), self.anchor_norm(anchor), role, step), dim=-1)
        delta = self.w2(F.silu(self.w1(features)))
        gate = torch.sigmoid(self.gate(features))
        return state + gate * (anchor + delta - state)
