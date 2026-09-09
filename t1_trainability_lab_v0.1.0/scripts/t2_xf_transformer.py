"""Predeclared matched-control Transformer for T2-XF.

XF-SEQ uses ``encode_instruction`` once. XF-CLAUSE calls the same encoder for
each clause and sums the resulting 32-dimensional vectors. No architecture or
pooling variant belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from t2_i1_instruction import TOKEN_IDS, tokenize_instruction_i1


CLS_TOKEN = "CLS"
CLS_ID = len(TOKEN_IDS)
VOCAB_SIZE = CLS_ID + 1
MAX_TOKENS = 6


class MatchedTransformerEncoder(nn.Module):
    """One shared 5,380-parameter Transformer encoder producing 32D vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, 20)
        self.position_embedding = nn.Embedding(MAX_TOKENS, 20)
        layer = nn.TransformerEncoderLayer(
            d_model=20,
            nhead=4,
            dim_feedforward=48,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(20)
        self.projection = nn.Linear(20, 32)

    def forward(self, token_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > MAX_TOKENS - 1:
            raise ValueError(f"token_ids must have shape [batch, <= {MAX_TOKENS - 1}]")
        batch_size, sequence_length = token_ids.shape
        cls = torch.full((batch_size, 1), CLS_ID, dtype=token_ids.dtype, device=token_ids.device)
        tokens = torch.cat((cls, token_ids), dim=1)
        positions = torch.arange(sequence_length + 1, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        if padding_mask is not None:
            if padding_mask.shape != token_ids.shape:
                raise ValueError("padding_mask must match token_ids shape")
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=token_ids.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        return self.projection(self.final_norm(hidden[:, 0]))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def encode_instruction(model: MatchedTransformerEncoder, instructions: list[str]) -> Tensor:
    """Encode complete instructions for XF-SEQ without syntactic splitting."""
    rows = [tokenize_instruction_i1(instruction) for instruction in instructions]
    return _encode_rows(model, rows)


def encode_clauses(model: MatchedTransformerEncoder, instructions: list[str]) -> Tensor:
    """Encode clauses independently with shared weights, then add vectors."""
    first_rows: list[tuple[str, ...]] = []
    second_rows: list[tuple[str, ...]] = []
    for instruction in instructions:
        tokens = tokenize_instruction_i1(instruction)
        split = tokens.index("AND") if "AND" in tokens else len(tokens)
        first_rows.append(tokens[:split])
        second_rows.append(tokens[split + 1 :] if split < len(tokens) else ())
    first = _encode_rows(model, first_rows)
    if not any(second_rows):
        return first
    second = _encode_rows(model, [row if row else ("NOOP", "VALUE_0") for row in second_rows])
    mask = torch.tensor([bool(row) for row in second_rows], dtype=first.dtype, device=first.device).unsqueeze(1)
    return first + second * mask


def _encode_rows(model: MatchedTransformerEncoder, rows: list[tuple[str, ...]]) -> Tensor:
    if not rows:
        return torch.empty((0, 32))
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    if int(lengths.max()) > MAX_TOKENS - 1:
        raise ValueError("instruction exceeds Transformer position capacity")
    token_ids = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    padding_mask = torch.ones_like(token_ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = torch.tensor([TOKEN_IDS[token] for token in row], dtype=torch.long)
        padding_mask[index, : len(row)] = False
    return model(token_ids, padding_mask)


@dataclass(frozen=True)
class ArchitectureReport:
    vocabulary_size: int
    max_tokens_with_cls: int
    trainable_parameters: int
    components: dict[str, int]


def architecture_report() -> ArchitectureReport:
    model = MatchedTransformerEncoder()
    components = {
        "token_embedding": model.token_embedding.weight.numel(),
        "position_embedding": model.position_embedding.weight.numel(),
        "transformer_layer": sum(parameter.numel() for parameter in model.transformer.parameters()),
        "final_layer_norm": sum(parameter.numel() for parameter in model.final_norm.parameters()),
        "projection": sum(parameter.numel() for parameter in model.projection.parameters()),
    }
    return ArchitectureReport(VOCAB_SIZE, MAX_TOKENS, parameter_count(model), components)


if __name__ == "__main__":
    report = architecture_report()
    print({
        "vocabulary_size": report.vocabulary_size,
        "max_tokens_with_cls": report.max_tokens_with_cls,
        "trainable_parameters": report.trainable_parameters,
        "components": report.components,
    })
