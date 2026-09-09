"""T2-I0 Baseline A: learned instruction encoder and explicit bit heads."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from t2_i0_instruction import parse_instruction, tokenize_instruction
from train_u0c_ctrl2_o import sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_baseline_a_seed4801"


class SharedInstructionEncoder(nn.Module):
    def __init__(self, vocabulary_size: int = 36, embedding_size: int = 16, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.recurrence = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output_size = hidden_size

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        sequence, _ = self.recurrence(self.embedding(token_ids))
        return sequence[torch.arange(token_ids.shape[0]), lengths - 1]


class BaselineAClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedInstructionEncoder()
        self.floor_head = nn.Linear(32, 2)
        self.avoid_head = nn.Linear(32, 2)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        representation = self.encoder(token_ids, lengths)
        return self.floor_head(representation), self.avoid_head(representation)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def examples(include_and: bool) -> list[tuple[str, tuple[int, int]]]:
    rows = [("KEEP", (0, 0))]
    rows.extend((f"AT_LEAST VALUE_{value}", (1, 0)) for value in range(32))
    rows.extend((f"AVOID VALUE_{value}", (0, 1)) for value in range(32))
    if include_and:
        rows.extend((f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}", (1, 1)) for lower in range(32) for forbidden in range(31))
    return rows


def tensorize(rows: list[tuple[str, tuple[int, int]]]) -> dict[str, Tensor]:
    parsed = [parse_instruction(text) for text, _ in rows]
    lengths = torch.tensor([len(item.token_ids) for item in parsed], dtype=torch.long)
    width = int(lengths.max())
    token_ids = torch.zeros((len(parsed), width), dtype=torch.long)
    for index, item in enumerate(parsed):
        token_ids[index, : len(item.token_ids)] = torch.tensor(item.token_ids)
    bits = torch.tensor([constraints for _, constraints in rows], dtype=torch.long)
    return {"token_ids": token_ids, "lengths": lengths, "bits": bits, "instructions": rows}


def classify(model: BaselineAClassifier, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    floor_logits, avoid_logits = model(data["token_ids"], data["lengths"])
    return floor_logits.argmax(-1), avoid_logits.argmax(-1)


def metric(model: BaselineAClassifier, data: dict[str, Tensor]) -> dict[str, Any]:
    with torch.no_grad():
        floor, avoid = classify(model, data)
    expected = data["bits"]
    correct = (floor == expected[:, 0]) & (avoid == expected[:, 1])
    return {"samples": len(expected), "correct": int(correct.sum()), "accuracy": float(correct.float().mean()), "floor_accuracy": float((floor == expected[:, 0]).float().mean()), "avoid_accuracy": float((avoid == expected[:, 1]).float().mean())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=4801); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    train_data = tensorize(examples(False)); val_data = tensorize(examples(False)); test_data = tensorize(examples(False)); and_data = tensorize(examples(True)[65:])
    model = BaselineAClassifier(); torch.save({"model": model.state_dict(), "seed": args.seed, "initialization": "random from explicit seed", "trainable_parameters": parameter_count(model), "and_trained": False}, args.output_root / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(args.seed + 1); metrics: list[dict[str, Any]] = []
    for step in range(1, 5001):
        progress = (step - 1) / 4999; lr = 1e-3 + (1e-5 - 1e-3) * progress; optimizer.param_groups[0]["lr"] = lr; indices = torch.randint(len(train_data["bits"]), (len(train_data["bits"]),), generator=generator); optimizer.zero_grad(set_to_none=True)
        floor_logits, avoid_logits = model(train_data["token_ids"][indices], train_data["lengths"][indices]); loss = (F.cross_entropy(floor_logits, train_data["bits"][indices, 0]) + F.cross_entropy(avoid_logits, train_data["bits"][indices, 1])) / 2; loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            val_metric = metric(model, val_data); metrics.append({"step": step, "train_loss": float(loss.item()), "learning_rate": lr, **{f"val_{key}": value for key, value in val_metric.items()}})
    torch.save({"model": model.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(model), "and_trained": False}, args.output_root / "final.pt")
    (args.output_root / "training_metrics.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics), encoding="utf-8")
    metadata = {"train": {"hash": sha256(args.output_root / "initial.pt"), **metric(model, train_data)}, "val": metric(model, val_data), "test": metric(model, test_data), "heldout_and": metric(model, and_data), "and_trained": False, "forms": {"train": ["KEEP", "AT_LEAST VALUE_L", "AVOID VALUE_F"], "heldout": ["AT_LEAST VALUE_L AND AVOID VALUE_F"]}}
    (args.output_root / "data_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I0", "baseline": "A", "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(model), "and_trained": False, "metrics": metadata, "checkpoint": {"path": str(args.output_root / "final.pt"), "sha256": sha256(args.output_root / "final.pt")}}
    (args.output_root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(args.output_root / "results.json"), "checkpoint_sha256": result["checkpoint"]["sha256"], "status": result["status"], "metrics": metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
