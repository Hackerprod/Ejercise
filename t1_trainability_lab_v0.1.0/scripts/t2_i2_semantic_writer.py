"""T2-I2 learned semantic writer over complete instruction sequences."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from t2_i1_instruction import parse_instruction_i1


VOCABULARY_SIZE = 39
TOKEN_DIMENSION = 16
SLOT_DIMENSION = 32
SLOT_COUNT = 2
MAX_TOKENS = 5
MAX_POSITIONS = MAX_TOKENS + 1  # fixed zero CLS + up to five grammar tokens


class SemanticWriter(nn.Module):
    """Map one complete token sequence to two content-addressed semantic slots."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY_SIZE, TOKEN_DIMENSION)
        self.position = nn.Embedding(MAX_POSITIONS, TOKEN_DIMENSION)
        self.self_attention = nn.MultiheadAttention(TOKEN_DIMENSION, 1, batch_first=True)
        # First norm is non-affine; second carries the sole 32-parameter norm budget.
        self.attention_norm = nn.LayerNorm(TOKEN_DIMENSION, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(TOKEN_DIMENSION, 2 * TOKEN_DIMENSION),
            nn.SiLU(),
            nn.Linear(2 * TOKEN_DIMENSION, TOKEN_DIMENSION),
        )
        self.ffn_norm = nn.LayerNorm(TOKEN_DIMENSION)
        self.slot_queries = nn.Parameter(torch.empty(SLOT_COUNT, TOKEN_DIMENSION))
        self.slot_key = nn.Linear(TOKEN_DIMENSION, TOKEN_DIMENSION)
        self.slot_value = nn.Linear(TOKEN_DIMENSION, SLOT_DIMENSION)
        self.slot_output = nn.Linear(SLOT_DIMENSION, SLOT_DIMENSION)
        self.slot_gates = nn.ModuleList(nn.Linear(2 * TOKEN_DIMENSION, 1) for _ in range(SLOT_COUNT))
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details: bool = False) -> Tensor | dict[str, Tensor]:
        if token_ids.ndim != 2 or lengths.ndim != 1 or token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("token_ids must be [batch, tokens] and lengths must be [batch]")
        if bool((lengths < 1).any()) or bool((lengths > MAX_TOKENS).any()):
            raise ValueError(f"sequence lengths must be in [1, {MAX_TOKENS}]")
        batch, token_count = token_ids.shape
        if token_count > MAX_TOKENS:
            raise ValueError(f"token_ids exceeds max token count {MAX_TOKENS}")

        token = self.embedding(token_ids)
        cls = torch.zeros((batch, 1, TOKEN_DIMENSION), dtype=token.dtype, device=token.device)
        hidden = torch.cat((cls, token), dim=1)
        positions = torch.arange(token_count + 1, device=token_ids.device).unsqueeze(0)
        hidden = hidden + self.position(positions)
        padded = torch.arange(token_count + 1, device=token_ids.device).unsqueeze(0) >= (lengths + 1).unsqueeze(1)
        attended, attention = self.self_attention(hidden, hidden, hidden, key_padding_mask=padded, need_weights=True)
        hidden = self.attention_norm(hidden + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))

        keys = self.slot_key(hidden)
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.einsum("bsd,btd->bst", queries, keys) / (TOKEN_DIMENSION**0.5)
        scores = scores.masked_fill(padded.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bst,btd->bsd", weights, hidden)
        values = self.slot_value(context)
        projected = self.slot_output(values)
        gates = torch.stack([torch.sigmoid(gate(torch.cat((queries[:, index], context[:, index]), dim=-1))) for index, gate in enumerate(self.slot_gates)], dim=1)
        slots = projected * gates
        if not return_details:
            return slots
        return {"slots": slots, "attention": attention, "slot_weights": weights, "context": context, "gates": gates}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tensorize(instructions: list[str]) -> tuple[Tensor, Tensor]:
    parsed = [parse_instruction_i1(text) for text in instructions]
    lengths = torch.tensor([len(item.token_ids) for item in parsed], dtype=torch.long)
    token_ids = torch.zeros((len(parsed), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(parsed):
        token_ids[index, : len(item.token_ids)] = torch.tensor(item.token_ids, dtype=torch.long)
    return token_ids, lengths


def architecture_report(model: SemanticWriter | None = None) -> dict[str, Any]:
    writer = SemanticWriter() if model is None else model
    return {
        "module": "SemanticWriter",
        "vocabulary_size": VOCABULARY_SIZE,
        "token_dimension": TOKEN_DIMENSION,
        "slot_dimension": SLOT_DIMENSION,
        "slot_count": SLOT_COUNT,
        "max_tokens": MAX_TOKENS,
        "max_positions": MAX_POSITIONS,
        "cls": "fixed_zero_vector",
        "conditioning": "slot_0 + slot_1",
        "trainable_parameters": parameter_count(writer),
        "supervisor_trainable_parameters": 0,
        "parameter_breakdown": {
            "token_embedding": 624,
            "position_embedding": 96,
            "self_attention_qkv": 816,
            "self_attention_output": 272,
            "ffn_in": 544,
            "ffn_out": 528,
            "layer_norms": 32,
            "slot_queries": 32,
            "slot_key": 272,
            "slot_value": 544,
            "slot_output": 1056,
            "slot_gates": 66,
        },
        "slot_gate_input": "query(16) + pooled_context(16) = 32",
        "slot_gate_output": "sigmoid scalar per slot",
    }
