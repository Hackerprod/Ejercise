"""Self-contained synthetic ER32 model and explicit/direct equivalence helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ATOL = 1e-4
RTOL = 1e-4
LOSS_TEMPERATURE = 2.0


@dataclass(frozen=True)
class ER32Config:
    """Production-shape defaults; tests intentionally use smaller values."""

    vocab_size: int = 50257
    rank: int = 32
    dimension: int = 128
    slots: int = 8
    rounds: int = 4


DEFAULT_CONFIG = ER32Config()


class _SharedRoundBlock(nn.Module):
    """Compact stand-in for the unchanged shared recurrent update block."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.update = nn.Linear(dimension, dimension)

    def forward(self, state: Tensor, anchor: Tensor) -> Tensor:
        return state + torch.tanh(self.update(state + anchor))


class OmegaCoreLM0ER32(nn.Module):
    """Synthetic causal recurrent core with linked ER32 input/output factors.

    ``implementation`` changes only vocabulary projection order. All other
    parameters and recurrent operations are shared in meaning between views.
    """

    def __init__(
        self,
        *,
        vocab_size: int = DEFAULT_CONFIG.vocab_size,
        rank: int = DEFAULT_CONFIG.rank,
        dimension: int = DEFAULT_CONFIG.dimension,
        slots: int = DEFAULT_CONFIG.slots,
        rounds: int = DEFAULT_CONFIG.rounds,
        implementation: str = "efficient",
    ) -> None:
        super().__init__()
        if implementation not in {"explicit", "efficient"}:
            raise ValueError(f"unknown implementation: {implementation}")
        if rounds not in {1, 4}:
            raise ValueError("rounds must be 1 or 4")
        self.vocab_size = vocab_size
        self.rank = rank
        self.dimension = dimension
        self.slots = slots
        self.rounds = rounds
        self.implementation = implementation

        # One pair serves both input and output. No dense [V, D] parameter.
        self.C = nn.Parameter(torch.empty(vocab_size, rank))
        self.U = nn.Parameter(torch.empty(rank, dimension))

        # Compact recurrent fixture. These parameters are deliberately
        # independent of vocabulary projection implementation.
        self.token_projection = nn.Linear(dimension, slots * dimension)
        self.state_projection = nn.Linear(dimension, slots * dimension)
        self.shared_block = _SharedRoundBlock(dimension)
        self.readout = nn.Linear(slots * dimension, dimension, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use fresh factor parameters and standard reference-style rest init."""

        nn.init.normal_(self.C, mean=0.0, std=1.0)
        nn.init.normal_(self.U, mean=0.0, std=1.0 / math.sqrt(self.rank))
        self.token_projection.reset_parameters()
        self.state_projection.reset_parameters()
        self.shared_block.update.reset_parameters()
        self.readout.reset_parameters()

    @classmethod
    def fresh(cls, *, seed: int, **kwargs: object) -> "OmegaCoreLM0ER32":
        """Build deterministic fresh parameters without consuming global RNG."""

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return cls(**kwargs)

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension, device=device)

    def _input_embeddings(self, tokens: Tensor) -> Tensor:
        if self.implementation == "explicit":
            E = self.C @ self.U
            return F.embedding(tokens, E)
        return self.C[tokens] @ self.U

    def _output_logits(self, hidden: Tensor) -> Tensor:
        scale = self.dimension ** -0.5
        if self.implementation == "explicit":
            E = self.C @ self.U
            return F.linear(hidden, E) * scale
        return (hidden @ self.U.transpose(-2, -1)) @ self.C.transpose(-2, -1) * scale

    def recur_states(
        self,
        tokens: Tensor,
        previous_state: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return continuation state and candidate states for every position."""

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T]")
        if previous_state.shape != (tokens.shape[0], self.slots, self.dimension):
            raise ValueError("previous_state has incompatible shape")
        if valid_mask is not None and valid_mask.shape != tokens.shape:
            raise ValueError("valid_mask must match tokens shape")

        token_features = self._input_embeddings(tokens)
        state = previous_state
        candidates: list[Tensor] = []
        for position in range(tokens.shape[1]):
            write = self.token_projection(token_features[:, position])
            write = write + self.state_projection(state.mean(dim=1))
            anchor = write.view(tokens.shape[0], self.slots, self.dimension)
            candidate = anchor
            for _ in range(self.rounds):
                candidate = self.shared_block(candidate, anchor)
            candidates.append(candidate)
            if valid_mask is None:
                state = candidate
            else:
                active = valid_mask[:, position].view(-1, 1, 1)
                state = torch.where(active, candidate, state)
        return state, torch.stack(candidates, dim=1)

    def forward_window(
        self,
        tokens: Tensor,
        previous_state: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return final state, dense logits, and candidate states."""

        final_state, states = self.recur_states(tokens, previous_state, valid_mask)
        hidden = self.readout(states.flatten(start_dim=2))
        return final_state, self._output_logits(hidden), states

    def forward(
        self,
        tokens: Tensor,
        previous_state: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if previous_state is None:
            previous_state = self.initial_state(tokens.shape[0], device=tokens.device)
        return self.forward_window(tokens, previous_state, valid_mask)


def ce_kl_loss(
    logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    temperature: float = LOSS_TEMPERATURE,
) -> dict[str, Tensor]:
    """Return masked CE, temperature-scaled KL, and ``0.5 CE + 0.5 KL``."""

    if logits.shape != teacher_logits.shape:
        raise ValueError("logits and teacher_logits must have the same shape")
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or valid_mask.shape != targets.shape:
        raise ValueError("expected logits [B, T, V], targets/mask [B, T]")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_teacher = teacher_logits.reshape_as(flat_logits)
    flat_targets = targets.reshape(-1)
    weights = valid_mask.reshape(-1).to(dtype=logits.dtype)
    denom = weights.sum().clamp_min(1.0)

    ce_values = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    student_log_probs = F.log_softmax(flat_logits / temperature, dim=-1)
    teacher_probs = F.softmax(flat_teacher / temperature, dim=-1)
    kl_values = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl_values = kl_values * (temperature**2)
    ce = (ce_values * weights).sum() / denom
    kl = (kl_values * weights).sum() / denom
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}
