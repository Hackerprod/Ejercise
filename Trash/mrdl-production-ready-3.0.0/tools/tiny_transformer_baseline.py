#!/usr/bin/env python3
"""CPU-oriented Transformer baseline using the exact MRDL tokenizer.

This tool is intentionally isolated from the C++ runtime. It exists only for
budget-matched comparison on the same token IDs and optionally the same frozen
Q8 input embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - operational error path
    raise SystemExit(
        "Missing optional baseline dependencies. Run: "
        "python3 -m pip install -r tools/requirements-transformer.txt"
    ) from exc

PAD = 0
BOS = 1
EOS = 2
UNK = 3
BYTE_BASE = 4
BYTE_COUNT = 256
FIRST_LEARNED = BYTE_BASE + BYTE_COUNT


@dataclass(frozen=True)
class TokenizerData:
    pieces: tuple[bytes, ...]
    lowercase: bool

    @property
    def vocabulary_size(self) -> int:
        return len(self.pieces)


@dataclass
class Metrics:
    tokens: int = 0
    negative_log_likelihood: float = 0.0
    correct: int = 0
    elapsed_seconds: float = 0.0

    def to_json(self) -> dict[str, float | int]:
        denominator = max(self.tokens, 1)
        loss = self.negative_log_likelihood / denominator
        return {
            "tokens": self.tokens,
            "loss": loss,
            "perplexity": math.exp(min(loss, 80.0)),
            "accuracy": self.correct / denominator,
            "tokens_per_second": self.tokens / self.elapsed_seconds
            if self.elapsed_seconds > 0.0
            else 0.0,
            "elapsed_seconds": self.elapsed_seconds,
        }


def read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("unexpected end of file")
    return data


def load_tokenizer(path: Path) -> TokenizerData:
    with path.open("rb") as stream:
        magic, version, flags, count, _payload_hash = struct.unpack(
            "<8sIIQQ", read_exact(stream, 32)
        )
        if not magic.startswith(b"MRDLTOK") or version != 1:
            raise ValueError(f"unsupported tokenizer file: {path}")
        pieces: list[bytes] = []
        for _ in range(count):
            (size,) = struct.unpack("<Q", read_exact(stream, 8))
            if size > 1 << 30:
                raise ValueError("corrupt tokenizer piece length")
            pieces.append(read_exact(stream, size))
        if stream.read(1):
            raise ValueError("trailing tokenizer bytes")
    if len(pieces) < FIRST_LEARNED:
        raise ValueError("tokenizer lacks byte fallback rows")
    return TokenizerData(tuple(pieces), bool(flags & 1))


def ascii_lower(data: bytes) -> bytes:
    return bytes((byte + 32 if 65 <= byte <= 90 else byte) for byte in data)


def piece_class(byte: int) -> int:
    if byte in b" \t\n\r\v\f":
        return 0
    if 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122 or byte == 95 or byte >= 128:
        return 1
    return 2


def split_pieces(data: bytes, lowercase: bool) -> Iterable[bytes]:
    if lowercase:
        data = ascii_lower(data)
    start = 0
    while start < len(data):
        category = piece_class(data[start])
        end = start + 1
        while end < len(data) and piece_class(data[end]) == category:
            if category == 2 and data[end] != data[start]:
                break
            end += 1
        yield data[start:end]
        start = end


def make_encoder(tokenizer: TokenizerData):
    lookup = {
        piece: token_id
        for token_id, piece in enumerate(tokenizer.pieces[FIRST_LEARNED:], FIRST_LEARNED)
    }

    def encode(data: bytes, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        result: list[int] = [BOS] if add_bos else []
        for piece in split_pieces(data, tokenizer.lowercase):
            token_id = lookup.get(piece)
            if token_id is not None:
                result.append(token_id)
            else:
                result.extend(BYTE_BASE + byte for byte in piece)
        if add_eos:
            result.append(EOS)
        return result

    return encode


def corpus_tokens(path: Path, tokenizer: TokenizerData, maximum: int = 0) -> torch.Tensor:
    encode = make_encoder(tokenizer)
    tokens: list[int] = []
    with path.open("rb") as stream:
        for raw in stream:
            if raw.endswith(b"\n"):
                raw = raw[:-1]
            raw += b"\n"
            tokens.extend(encode(raw, add_bos=True, add_eos=True))
            if maximum and len(tokens) >= maximum:
                del tokens[maximum:]
                break
    if len(tokens) < 3:
        raise ValueError(f"corpus contains too few tokens: {path}")
    return torch.tensor(tokens, dtype=torch.long)


class WindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, tokens: Tensor, context: int, stride: int) -> None:
        if context < 2 or stride < 1:
            raise ValueError("invalid context/stride")
        self.tokens = tokens
        self.context = context
        self.stride = stride
        available = max(int(tokens.numel()) - context - 1, 0)
        self.length = available // stride + (1 if available >= 0 and tokens.numel() > context else 0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.stride
        end = start + self.context
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        context: int,
        dimension: int,
        heads: int,
        layers: int,
        feed_forward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.context = context
        self.token_embedding = nn.Embedding(vocabulary, dimension)
        self.position_embedding = nn.Embedding(context, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feed_forward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, vocabulary, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if length > self.context:
            raise ValueError("input exceeds configured context")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        hidden = self.encoder(hidden, mask=self.causal_mask[:length, :length])
        return self.output(self.norm(hidden))


def load_mrdl_q8_embeddings(path: Path) -> Tensor:
    header_format = "<8sIIQIIQQ16s"
    header_size = struct.calcsize(header_format)
    with path.open("rb") as stream:
        header = struct.unpack(header_format, read_exact(stream, header_size))
    magic, version, storage_kind, rows, dimension, row_stride, _hash, _seed, _reserved = header
    if not magic.startswith(b"MRDLEMB") or version != 1 or storage_kind != 1:
        raise ValueError("unsupported MRDL embedding file")
    if row_stride != 4 + dimension:
        raise ValueError("invalid MRDL embedding row stride")
    expected = header_size + rows * row_stride
    if path.stat().st_size != expected:
        raise ValueError("MRDL embedding length mismatch")
    raw = np.memmap(path, mode="r", dtype=np.uint8, offset=header_size, shape=(rows, row_stride))
    scales = raw[:, :4].copy().reshape(rows, 4).view("<f4").reshape(rows)
    quantized = raw[:, 4:].view(np.int8).astype(np.float32, copy=True)
    matrix = quantized * scales[:, None]
    return torch.from_numpy(matrix)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    result = Metrics()
    started = time.perf_counter()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)
            logits = model(inputs)
            result.negative_log_likelihood += float(
                criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()
            )
            result.correct += int((logits.argmax(dim=-1) == targets).sum().item())
            result.tokens += int(targets.numel())
    result.elapsed_seconds = time.perf_counter() - started
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--eval-corpus", type=Path, required=True)
    parser.add_argument("--mrdl-embeddings", type=Path)
    parser.add_argument("--freeze-input-embeddings", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("transformer-baseline.pt"))
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feed-forward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-eval-tokens", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    parser.add_argument("--seed", type=int, default=5571588)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if args.context < 2 or args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("context, batch-size and epochs must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tokenizer = load_tokenizer(args.tokenizer)
    train_tokens = corpus_tokens(args.train_corpus, tokenizer, args.max_train_tokens)
    eval_tokens = corpus_tokens(args.eval_corpus, tokenizer, args.max_eval_tokens)
    train_data = WindowDataset(train_tokens, args.context, args.stride)
    eval_data = WindowDataset(eval_tokens, args.context, args.context)
    if not len(train_data) or not len(eval_data):
        raise SystemExit("corpus is shorter than the configured context")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyTransformerLM(
        tokenizer.vocabulary_size,
        args.context,
        args.d_model,
        args.heads,
        args.layers,
        args.feed_forward,
        args.dropout,
    )
    if args.mrdl_embeddings:
        weights = load_mrdl_q8_embeddings(args.mrdl_embeddings)
        if tuple(weights.shape) != tuple(model.token_embedding.weight.shape):
            raise SystemExit(
                f"embedding shape {tuple(weights.shape)} != model input shape "
                f"{tuple(model.token_embedding.weight.shape)}"
            )
        with torch.no_grad():
            model.token_embedding.weight.copy_(weights)
        model.token_embedding.weight.requires_grad_(not args.freeze_input_embeddings)

    device = torch.device(args.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    history: list[dict[str, float | int]] = []
    training_started = time.perf_counter()
    processed_tokens = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_nll = 0.0
        epoch_tokens = 0
        epoch_correct = 0
        epoch_started = time.perf_counter()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            count = int(targets.numel())
            epoch_nll += float(loss.item()) * count
            epoch_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            epoch_tokens += count
            processed_tokens += count
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch + 1,
            "tokens": epoch_tokens,
            "loss": epoch_nll / max(epoch_tokens, 1),
            "perplexity": math.exp(min(epoch_nll / max(epoch_tokens, 1), 80.0)),
            "accuracy": epoch_correct / max(epoch_tokens, 1),
            "tokens_per_second": epoch_tokens / elapsed if elapsed > 0 else 0.0,
        }
        history.append(row)
        print(json.dumps({"event": "epoch", **row}, sort_keys=True), flush=True)

    evaluation = evaluate(model, eval_loader, device)
    result = {
        "model": "tiny_transformer_baseline",
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "vocabulary": tokenizer.vocabulary_size,
        "train_source_tokens": int(train_tokens.numel()),
        "eval_source_tokens": int(eval_tokens.numel()),
        "processed_training_targets": processed_tokens,
        "training_seconds": time.perf_counter() - training_started,
        "history": history,
        "evaluation": evaluation.to_json(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(
        {
            "format": 1,
            "result": result,
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(args.output)
    metrics_path = args.output.with_suffix(args.output.suffix + ".json")
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
