"""Corrected CPU-only OMEGA R1 fast candidate for Step 1 equivalence.

This module is an isolated candidate. The proposal module is read-only reference
material; this file does not import or modify it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


EPS = 1e-6
GATE_ATOL = 1e-5


class FastWorkspaceUpdateBlock(nn.Module):
    """Reference WorkspaceUpdateBlock with explicit fused parameter mapping.

    Mapping:
    - qkv rows [0:D], [D:2D], [2D:3D] <- query, key, value rows.
    - out <- slot_mix.output.
    - fc1/fc2 <- core.network[0]/core.network[2].
    - norm_weight <- rms_norm.weight.

    SDPA is called with dropout_p=0.0 and is_causal=False. Its input has one
    attention head and the slots are the sequence dimension.
    """

    def __init__(self, dimension: int, slots: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.slots = slots
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.out = nn.Linear(dimension, dimension)
        self.fc1 = nn.Linear(dimension, 4 * dimension)
        self.fc2 = nn.Linear(4 * dimension, dimension)
        self.norm_weight = nn.Parameter(torch.ones(dimension))

    @torch.no_grad()
    def load_from_reference(self, reference_block: nn.Module) -> None:
        slot_mix = reference_block.slot_mix
        core = reference_block.core.network
        self.qkv.weight.copy_(torch.cat((slot_mix.query.weight, slot_mix.key.weight, slot_mix.value.weight), dim=0))
        self.qkv.bias.copy_(torch.cat((slot_mix.query.bias, slot_mix.key.bias, slot_mix.value.bias), dim=0))
        self.out.weight.copy_(slot_mix.output.weight)
        self.out.bias.copy_(slot_mix.output.bias)
        self.fc1.weight.copy_(core[0].weight)
        self.fc1.bias.copy_(core[0].bias)
        self.fc2.weight.copy_(core[2].weight)
        self.fc2.bias.copy_(core[2].bias)
        self.norm_weight.copy_(reference_block.rms_norm.weight)

    def forward(self, state: Tensor, anchor: Tensor, depth_bias: Tensor, gate: Tensor) -> Tensor:
        dimension = self.dimension
        query, key, value = self.qkv(state + anchor).split(dimension, dim=-1)
        mixed = F.scaled_dot_product_attention(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(1)
        mixed = self.out(mixed)
        update = self.fc2(F.gelu(F.linear(mixed, self.fc1.weight, depth_bias)))
        return F.rms_norm(torch.addcmul(state, gate, update), (dimension,), self.norm_weight, EPS)


class OmegaCoreLMFast(nn.Module):
    """CPU/eager/FP32 candidate with corrected masked recurrence semantics."""

    def __init__(self, *, vocab_size: int, dimension: int = 128, slots: int = 8, rounds: int = 4, variant: str = "shared") -> None:
        super().__init__()
        if variant not in {"shared", "untied"}:
            raise ValueError(variant)
        self.vocab_size = vocab_size
        self.dimension = dimension
        self.slots = slots
        self.rounds = rounds
        self.variant = variant
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.prelude = nn.Linear(2 * dimension, slots * dimension)
        self.prelude_norm_weight = nn.Parameter(torch.ones(slots * dimension))
        block_count = 1 if variant == "shared" else rounds
        self.blocks = nn.ModuleList(FastWorkspaceUpdateBlock(dimension, slots) for _ in range(block_count))
        self.depth_embedding = nn.Embedding(rounds, dimension)
        self.gate_logits = nn.Parameter(torch.full((rounds, dimension), -2.1972245773362196))
        self.readout_norm_weight = nn.Parameter(torch.ones(slots * dimension))
        self.output_projection = nn.Linear(slots * dimension, dimension, bias=False)

    @classmethod
    def from_reference(cls, reference: nn.Module) -> "OmegaCoreLMFast":
        """Create candidate from one already initialized reference instance."""
        fast = cls(
            vocab_size=reference.vocab_size,
            dimension=reference.dimension,
            slots=reference.slots,
            rounds=reference.rounds,
            variant=reference.variant,
        ).to(next(reference.parameters()).device)
        with torch.no_grad():
            fast.embedding.weight.copy_(reference.embedding.weight)
            fast.prelude.weight.copy_(reference.prelude.weight)
            fast.prelude.bias.copy_(reference.prelude.bias)
            fast.prelude_norm_weight.copy_(reference.prelude_norm.weight)
            for fast_block, reference_block in zip(fast.blocks, reference.blocks):
                fast_block.load_from_reference(reference_block)
            fast.depth_embedding.weight.copy_(reference.depth_embedding.weight)
            fast.gate_logits.copy_(reference.gate_logits)
            fast.readout_norm_weight.copy_(reference.readout_norm.weight)
            fast.output_projection.weight.copy_(reference.output_projection.weight)
        return fast

    def initial_state(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension, device=device, dtype=torch.float32)

    def _round_constants(self) -> tuple[list[Tensor], Tensor]:
        """Recompute differentiable per-round values on every call/window."""
        gates = torch.sigmoid(self.gate_logits)
        biases: list[Tensor] = []
        for round_index in range(self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            depth = self.depth_embedding.weight[round_index]
            biases.append(block.fc1.bias + F.linear(depth, block.fc1.weight))
        return biases, gates

    def recur_states(
        self,
        tokens: Tensor,
        previous_state: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return final, continuation, candidate, and readout states.

        Candidate next state is computed first. Readout state/logits are based
        on that candidate. Only then does valid_mask select the continuation
        state. This prevents masked positions from reading out reverted state.
        """
        batch, time = tokens.shape
        state_dim = self.slots * self.dimension
        prelude_weight = self.prelude.weight
        token_part = F.linear(self.embedding(tokens), prelude_weight[:, : self.dimension], self.prelude.bias)
        state_part_weight = prelude_weight[:, self.dimension :]
        depth_biases, gates = self._round_constants()
        blocks = [self.blocks[0] if self.variant == "shared" else self.blocks[index] for index in range(self.rounds)]

        state = previous_state
        continuation_states: list[Tensor] = []
        candidate_states: list[Tensor] = []
        readout_states: list[Tensor] = []
        for position in range(time):
            write = token_part[:, position] + F.linear(state.mean(dim=1), state_part_weight)
            anchor = F.rms_norm(state.flatten(start_dim=1) + write, (state_dim,), self.prelude_norm_weight, EPS).view(batch, self.slots, self.dimension)
            candidate = anchor
            for round_index, block in enumerate(blocks):
                candidate = block(candidate, anchor, depth_biases[round_index], gates[round_index])
            readout_state = candidate
            if valid_mask is None:
                continuation = candidate
            else:
                active = valid_mask[:, position].view(-1, 1, 1)
                continuation = torch.where(active, candidate, state)
            candidate_states.append(candidate)
            readout_states.append(readout_state)
            continuation_states.append(continuation)
            state = continuation
        return state, torch.stack(continuation_states, dim=1), torch.stack(candidate_states, dim=1), torch.stack(readout_states, dim=1)

    def project(self, states: Tensor) -> Tensor:
        flat = states.flatten(start_dim=2)
        return self.output_projection(F.rms_norm(flat, (flat.shape[-1],), self.readout_norm_weight, EPS))

    def logits_from_projected(self, projected: Tensor) -> Tensor:
        return F.linear(projected, self.embedding.weight) * (self.dimension ** -0.5)

    def logits_from_states(self, states: Tensor) -> Tensor:
        return self.logits_from_projected(self.project(states))

    def forward_window(self, tokens: Tensor, previous_state: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        final_state, continuation_states, candidate_states, readout_states = self.recur_states(tokens, previous_state, valid_mask)
        logits = self.logits_from_states(readout_states)
        return final_state, logits, {
            "continuation_states": continuation_states,
            "candidate_states": candidate_states,
            "readout_states": readout_states,
        }


def parameter_mapping(reference: nn.Module, fast: OmegaCoreLMFast) -> dict[str, tuple[str, ...]]:
    """Return complete candidate -> reference parameter mapping.

    Fused Q/K/V rows map to original rows in exact order. Shared blocks map to
    reference block 0; untied blocks map by round. Every source parameter must
    appear exactly once across this mapping.
    """
    mapping: dict[str, tuple[str, ...]] = {
        "embedding.weight": ("embedding.weight",),
        "prelude.weight": ("prelude.weight",),
        "prelude.bias": ("prelude.bias",),
        "prelude_norm_weight": ("prelude_norm.weight",),
        "depth_embedding.weight": ("depth_embedding.weight",),
        "gate_logits": ("gate_logits",),
        "readout_norm_weight": ("readout_norm.weight",),
        "output_projection.weight": ("output_projection.weight",),
    }
    for index in range(fast.rounds if fast.variant == "untied" else 1):
        reference_index = index if reference.variant == "untied" else 0
        prefix = f"blocks.{index}"
        reference_prefix = f"blocks.{reference_index}"
        mapping.update(
            {
                f"{prefix}.qkv.weight": (
                    f"{reference_prefix}.slot_mix.query.weight",
                    f"{reference_prefix}.slot_mix.key.weight",
                    f"{reference_prefix}.slot_mix.value.weight",
                ),
                f"{prefix}.qkv.bias": (
                    f"{reference_prefix}.slot_mix.query.bias",
                    f"{reference_prefix}.slot_mix.key.bias",
                    f"{reference_prefix}.slot_mix.value.bias",
                ),
                f"{prefix}.out.weight": (f"{reference_prefix}.slot_mix.output.weight",),
                f"{prefix}.out.bias": (f"{reference_prefix}.slot_mix.output.bias",),
                f"{prefix}.fc1.weight": (f"{reference_prefix}.core.network.0.weight",),
                f"{prefix}.fc1.bias": (f"{reference_prefix}.core.network.0.bias",),
                f"{prefix}.fc2.weight": (f"{reference_prefix}.core.network.2.weight",),
                f"{prefix}.fc2.bias": (f"{reference_prefix}.core.network.2.bias",),
                f"{prefix}.norm_weight": (f"{reference_prefix}.rms_norm.weight",),
            }
        )
    return mapping


def teacher_targets(teacher_logits: Tensor, temperature: float = 2.0) -> tuple[Tensor, Tensor]:
    """Return FP32 teacher probabilities and exact negative entropy constant."""
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        return teacher_probs, (teacher_probs * teacher_log_probs).sum(dim=-1)


def distillation_loss_lse(
    model: OmegaCoreLMFast,
    states: Tensor,
    teacher_probs: Tensor,
    teacher_neg_entropy: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    temperature: float = 2.0,
) -> dict[str, Tensor]:
    """Compute CE, KL, and total with logsumexp while retaining teacher entropy."""
    logits = model.logits_from_states(states)
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    weights = valid_mask.reshape(-1).to(dtype=logits.dtype)
    flat_teacher_probs = teacher_probs.reshape(-1, vocab_size)
    ce_per_token = torch.logsumexp(flat_logits, dim=-1) - flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    scaled_logits = flat_logits / temperature
    kl_per_token = (
        teacher_neg_entropy.reshape(-1)
        - (flat_teacher_probs * flat_logits).sum(dim=-1) / temperature
        + torch.logsumexp(scaled_logits, dim=-1)
    ) * temperature**2
    denominator = weights.sum().clamp_min(1.0)
    ce = (ce_per_token * weights).sum() / denominator
    kl = (kl_per_token * weights).sum() / denominator
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}
