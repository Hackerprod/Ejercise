"""Isolated synthetic preflight for T7 NOOP-to-NONE supervision.

This module deliberately owns no campaign paths, checkpoints, manifests, or
optimizer.  It models the ordinary lookup/scorer path with a synthetic frozen
four-role binder and exposes only the new NOOP embedding as trainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
ARGUMENT_COUNT = 32
NOOP_TOKEN = "OP_NOOP"
LINK_TOKEN = "LINK"
DIMENSION = 16
NOOP_TARGETS: tuple[None, None, None, None] = (None, None, None, None)


@dataclass(frozen=True)
class NoopContext:
    """One structural context and the position whose score is supervised."""

    name: str
    tokens: tuple[str, str]
    score_position: int


def noop_contexts() -> tuple[NoopContext, ...]:
    """Return exactly U: 32 operator-to-argument plus two structural contexts."""
    contexts = [
        NoopContext(f"{NOOP_TOKEN}->ARG_{index:02d}", (NOOP_TOKEN, f"ARG_{index:02d}"), 1)
        for index in range(ARGUMENT_COUNT)
    ]
    contexts.extend(
        (
            NoopContext("START->OP_NOOP", (NOOP_TOKEN,), 0),
            NoopContext("LINK->OP_NOOP", (LINK_TOKEN, NOOP_TOKEN), 1),
        )
    )
    if len(contexts) != 34:
        raise AssertionError("T7 NOOP context cardinality must be 34")
    return tuple(contexts)


class SyntheticNoopBinder(nn.Module):
    """Deterministic four-query scorer with only appended NOOP row trainable."""

    def __init__(self, *, seed: int = 1701, dimension: int = DIMENSION) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        old_vocab_size = ARGUMENT_COUNT + 5
        old_embeddings = torch.randn(old_vocab_size, dimension, generator=generator)
        self.old_vocab_size = old_vocab_size
        self.noop_token_id = old_vocab_size
        self.old_embeddings = nn.Parameter(old_embeddings, requires_grad=False)
        self.noop_embedding = nn.Parameter(torch.randn(dimension, generator=generator))
        self.query_bank = nn.Parameter(torch.randn(4, dimension, generator=generator), requires_grad=False)
        self.key_network = nn.Linear(2 * dimension, dimension)
        self.local_norm = nn.LayerNorm(dimension)
        self.w_v = nn.Linear(dimension, dimension)
        self.thresholds = nn.Parameter(torch.tensor([0.25, 0.5, 0.75, 1.0]), requires_grad=False)
        for module in (self.key_network, self.local_norm, self.w_v):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        with torch.no_grad():
            for parameter in (self.key_network.weight, self.local_norm.weight, self.w_v.weight):
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
            for parameter in (self.key_network.bias, self.local_norm.bias, self.w_v.bias):
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    @property
    def vocab_size(self) -> int:
        return self.old_vocab_size + 1

    def lookup(self, token_ids: Tensor) -> Tensor:
        """Ordinary embedding lookup over old rows plus appended NOOP row."""
        embedding_table = torch.cat((self.old_embeddings, self.noop_embedding.unsqueeze(0)), dim=0)
        return embedding_table[token_ids]

    def score(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        """Run ordinary previous-token/key/query scoring; no NOOP shortcut."""
        if token_ids.ndim != 2 or lengths.shape != (token_ids.shape[0],):
            raise ValueError("token_ids must be [B,T] and lengths must be [B]")
        embeddings = self.lookup(token_ids)
        zero = torch.zeros((token_ids.shape[0], 1, embeddings.shape[-1]), dtype=embeddings.dtype)
        previous = torch.cat((zero, embeddings[:, :-1]), dim=1)
        keys = self.local_norm(self.key_network(torch.cat((previous, embeddings), dim=-1)))
        values = self.w_v(embeddings)
        scores = torch.einsum("btd,rd->brt", keys, self.query_bank) / 4.0
        valid = torch.arange(token_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        return scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    def noop_candidate_scores(self) -> Tensor:
        """Return [4,34] scores at OP_NOOP positions through ordinary scorer."""
        token_ids: list[list[int]] = []
        positions: list[int] = []
        for context in noop_contexts():
            row = [self.noop_token_id if token == NOOP_TOKEN else self.old_vocab_size - 1 if token == LINK_TOKEN else int(token[4:]) for token in context.tokens]
            token_ids.append(row)
            positions.append(context.score_position)
        width = max(len(row) for row in token_ids)
        ids = torch.full((len(token_ids), width), self.old_vocab_size - 1, dtype=torch.long)
        lengths = torch.tensor([len(row) for row in token_ids], dtype=torch.long)
        for index, row in enumerate(token_ids):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        scores = self.score(ids, lengths).transpose(0, 1)
        positions_tensor = torch.tensor(positions, dtype=torch.long)
        return scores.gather(2, positions_tensor.view(1, -1, 1).expand(4, -1, 1)).squeeze(2)


def noop_difference_matrix(binder: SyntheticNoopBinder) -> Tensor:
    """Compute score-b differences with shape [4,34]."""
    return binder.noop_candidate_scores() - binder.thresholds.unsqueeze(1)


def noop_none_loss(binder: SyntheticNoopBinder) -> Tensor:
    """Mean softplus(score-b) over exactly 4*34 comparisons."""
    differences = noop_difference_matrix(binder)
    if differences.shape != (4, 34):
        raise AssertionError("NOOP difference matrix must have shape [4,34]")
    return F.softplus(differences).mean()
