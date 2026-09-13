"""Synthetic-parameter integration against the real T6 GenericNRoleBinder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
ARGUMENTS = tuple(f"ARG_{index:02d}" for index in range(32))
ACTIVE_OPERATORS = ("OP_V", "OP_W", "OP_X", "OP_Y")
NOOP_OPERATOR = "OP_Z"
LINK = "LINK"


def integration_manifest() -> dict[str, object]:
    """Build only an in-memory manifest; no campaign file is written."""
    old_tokens = [*ARGUMENTS, *ACTIVE_OPERATORS, LINK]
    return {"token_ids": {name: index for index, name in enumerate([*old_tokens, NOOP_OPERATOR])}, "permutation": list(range(32))}


class AppendedNoopEmbedding(nn.Module):
    """Frozen old table plus one appended trainable row."""

    def __init__(self, old_weights: Tensor, noop_weights: Tensor) -> None:
        super().__init__()
        self.old_embeddings = nn.Parameter(old_weights.detach().clone(), requires_grad=False)
        self.noop_embedding = nn.Parameter(noop_weights.detach().clone())

    def forward(self, token_ids: Tensor) -> Tensor:
        table = torch.cat((self.old_embeddings, self.noop_embedding.unsqueeze(0)), dim=0)
        return table[token_ids]


class T7ProductionCoreNoopIntegration(nn.Module):
    """Real GenericNRoleBinder forward with an append-only NOOP embedding."""

    def __init__(self, *, seed: int = 7701) -> None:
        super().__init__()
        self.core = GenericNRoleBinder(integration_manifest(), seed, list(ROLES))
        old_weights = self.core.embedding.weight[:-1].detach()
        noop_weights = self.core.embedding.weight[-1].detach()
        self.core.embedding = AppendedNoopEmbedding(old_weights, noop_weights)
        self.noop_token_id = len(ARGUMENTS) + len(ACTIVE_OPERATORS) + 1
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_(name == "embedding.noop_embedding")

    def forward(self, token_ids: Tensor, lengths: Tensor) -> dict[str, object]:
        return self.core(token_ids, lengths)

    def noop_candidate_scores(self) -> Tensor:
        rows: list[list[int]] = []
        positions: list[int] = []
        for index in range(32):
            rows.append([self.noop_token_id, index])
            positions.append(1)
        rows.append([self.noop_token_id, 0])
        positions.append(0)
        rows.append([len(ARGUMENTS) + len(ACTIVE_OPERATORS), self.noop_token_id])
        positions.append(1)
        ids = torch.tensor(rows, dtype=torch.long)
        lengths = torch.tensor([2] * 32 + [1, 2], dtype=torch.long)
        scores = self.forward(ids, lengths)["scores"]
        stacked = torch.stack(tuple(scores), dim=0)
        positions_tensor = torch.tensor(positions, dtype=torch.long).view(1, -1, 1)
        return stacked.gather(2, positions_tensor.expand(4, 34, 1)).squeeze(2)


def noop_difference_matrix(core: T7ProductionCoreNoopIntegration, thresholds: Tensor) -> Tensor:
    differences = core.noop_candidate_scores() - thresholds.view(4, 1)
    if differences.shape != (4, 34):
        raise AssertionError("production-core NOOP matrix must be [4,34]")
    return differences


def noop_none_loss(core: T7ProductionCoreNoopIntegration, thresholds: Tensor) -> Tensor:
    return F.softplus(noop_difference_matrix(core, thresholds)).mean()
