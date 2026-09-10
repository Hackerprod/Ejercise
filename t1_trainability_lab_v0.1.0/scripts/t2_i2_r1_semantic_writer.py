"""T2-I2-R1 local competitive semantic writer with a NULL sink."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from t2_i1_instruction import parse_instruction_i1

VOCABULARY_SIZE = 39
TOKEN_DIMENSION = 16
SLOT_DIMENSION = 32
SEMANTIC_SLOTS = 2
ROUTING_SLOTS = 3  # two semantic slots plus NULL
MAX_TOKENS = 5
MAX_POSITIONS = 6


class CompetitiveSemanticWriter(nn.Module):
    """Assign each local token state competitively to slot 0, slot 1, or NULL."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY_SIZE, TOKEN_DIMENSION)
        self.position = nn.Embedding(MAX_POSITIONS, TOKEN_DIMENSION)
        self.local_binding = nn.Linear(3 * TOKEN_DIMENSION, TOKEN_DIMENSION)
        self.local_norm = nn.LayerNorm(TOKEN_DIMENSION)
        self.slot_queries = nn.Parameter(torch.empty(ROUTING_SLOTS, TOKEN_DIMENSION))
        self.slot_value = nn.Linear(TOKEN_DIMENSION, SLOT_DIMENSION)
        self.slot_output = nn.Linear(SLOT_DIMENSION, SLOT_DIMENSION)
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details: bool = False) -> Tensor | dict[str, Tensor]:
        if token_ids.ndim != 2 or lengths.ndim != 1 or token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("token_ids must be [batch, tokens] and lengths must be [batch]")
        if bool((lengths < 1).any()) or bool((lengths > MAX_TOKENS).any()):
            raise ValueError(f"sequence lengths must be in [1, {MAX_TOKENS}]")
        batch, token_count = token_ids.shape
        if token_count > MAX_TOKENS:
            raise ValueError(f"token_ids exceeds max token count {MAX_TOKENS}")
        positions = torch.arange(1, token_count + 1, device=token_ids.device).unsqueeze(0)
        tokens = self.embedding(token_ids) + self.position(positions)
        valid = torch.arange(token_count, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        zeros = torch.zeros((batch, 1, TOKEN_DIMENSION), dtype=tokens.dtype, device=tokens.device)
        left = torch.cat((zeros, tokens[:, :-1]), dim=1)
        right = torch.cat((tokens[:, 1:], zeros), dim=1)
        local = self.local_norm(F.silu(self.local_binding(torch.cat((left, tokens, right), dim=-1))))
        scores = torch.einsum("sd,btd->bst", self.slot_queries, local) / (TOKEN_DIMENSION**0.5)
        probabilities = torch.softmax(scores, dim=1).masked_fill(~valid.unsqueeze(1), 0.0)
        values = self.slot_value(local)
        mass = probabilities[:, :SEMANTIC_SLOTS].sum(dim=-1, keepdim=True)
        pooled = torch.einsum("bst,btd->bsd", probabilities[:, :SEMANTIC_SLOTS], values) / (1e-8 + mass)
        occupancy = mass / (1e-8 + mass)
        slots = self.slot_output(pooled) * occupancy
        if not return_details:
            return slots
        return {"slots": slots, "local_states": local, "routing_probabilities": probabilities, "semantic_mass": mass, "values": values}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tensorize(instructions: list[str]) -> tuple[Tensor, Tensor]:
    parsed = [parse_instruction_i1(text) for text in instructions]
    lengths = torch.tensor([len(item.token_ids) for item in parsed], dtype=torch.long)
    token_ids = torch.zeros((len(parsed), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(parsed):
        token_ids[index, : len(item.token_ids)] = torch.tensor(item.token_ids, dtype=torch.long)
    return token_ids, lengths


def architecture_report(model: CompetitiveSemanticWriter | None = None) -> dict[str, Any]:
    writer = CompetitiveSemanticWriter() if model is None else model
    return {
        "module": "CompetitiveSemanticWriter",
        "vocabulary_size": VOCABULARY_SIZE,
        "token_dimension": TOKEN_DIMENSION,
        "slot_dimension": SLOT_DIMENSION,
        "semantic_slot_count": SEMANTIC_SLOTS,
        "routing_slot_count": ROUTING_SLOTS,
        "max_tokens": MAX_TOKENS,
        "max_positions": MAX_POSITIONS,
        "local_window": "-1,0,+1",
        "global_self_attention": False,
        "conditioning": "slot_0 + slot_1",
        "null_sink": True,
        "null_value_projection": False,
        "trainable_parameters": parameter_count(writer),
        "supervisor_trainable_parameters": 0,
        "parameter_breakdown": {
            "token_embedding": 624,
            "position_embedding": 96,
            "local_binding": 784,
            "local_layer_norm": 32,
            "slot_queries_including_null": 48,
            "semantic_value_projection": 544,
            "shared_slot_output": 1056,
        },
        "routing": "softmax over slots per token, then semantic-slot mass normalization",
        "slot_score_shape": "[batch, 3, tokens]",
        "slot_gate_input": None,
    }
