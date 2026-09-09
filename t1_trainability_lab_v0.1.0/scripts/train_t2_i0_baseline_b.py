"""T2-I0 Baseline B: latent instruction conditioning with frozen CTRL-7 core."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from t2_i0_instruction import parse_instruction
from train_t2_i0_baseline_a import SharedInstructionEncoder, examples, tensorize
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from train_u0c_ctrl2_o import sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
CTRL7_CHECKPOINT = SOURCE_ROOT / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_baseline_b_seed4901"


class LatentConditionedSupervisor(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        core = GoalConditionedSupervisor614()
        core.load_state_dict(torch.load(checkpoint, weights_only=False)["supervisor"], strict=True)
        self.observation_projection = core.observation_projection
        self.output_projection = core.output_projection
        for parameter in self.observation_projection.parameters(): parameter.requires_grad = False
        for parameter in self.output_projection.parameters(): parameter.requires_grad = False

    def forward(self, observation: Tensor, condition: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.observation_projection(observation) + condition))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def instruction_text(constraints: tuple[int, int], lower: int, forbidden: int) -> str:
    if constraints == (0, 0): return "KEEP"
    if constraints == (1, 0): return f"AT_LEAST VALUE_{lower}"
    if constraints == (0, 1): return f"AVOID VALUE_{forbidden}"
    raise ValueError("AND must remain held out")


def instruction_table() -> tuple[Tensor, Tensor, dict[str, int]]:
    rows = examples(False); data = tensorize(rows); ids = {text: index for index, (text, _) in enumerate(rows)}
    return data["token_ids"], data["lengths"], ids


def load_data(root: Path, split: str, token_ids: Tensor, lengths: Tensor, ids: dict[str, int]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    observations = torch.load(root / split / "observations.pt", weights_only=False)
    labels = torch.load(root / split / "labels.pt", weights_only=False)
    instruction_ids = []
    for constraint, lower, forbidden in zip(labels["constraints"].tolist(), labels["lower"].tolist(), labels["forbidden"].tolist()):
        instruction_ids.append(ids[instruction_text(tuple(constraint), lower, forbidden)])
    labels = {**labels, "instruction_id": torch.tensor(instruction_ids, dtype=torch.long)}
    return observations, labels


def train(encoder: SharedInstructionEncoder, supervisor: LatentConditionedSupervisor, train_data: tuple[dict[str, Tensor], dict[str, Tensor]], val_data: tuple[dict[str, Tensor], dict[str, Tensor]], token_ids: Tensor, lengths: Tensor, root: Path, seed: int) -> dict[str, Any]:
    train_observations, train_labels = train_data; val_observations, val_labels = val_data
    actions = (0, 1, 2, 3, 5)
    goal_names = {(0, 0): "NONE", (1, 0): "FLOOR", (0, 1): "AVOID"}
    buckets = {name: {action: torch.where((train_labels["constraints"][:, 0] == key[0]) & (train_labels["constraints"][:, 1] == key[1]) & (train_labels["action"] == action))[0] for action in actions} for key, name in goal_names.items()}
    batch_counts = {"NONE": {0: 8, 1: 8, 2: 8, 5: 8}, "FLOOR": {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, "AVOID": {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); metrics = []
    for step in range(1, 5001):
        selected = []
        for goal, counts in batch_counts.items():
            for action, count in counts.items():
                bucket = buckets[goal][action]
                if not len(bucket): raise RuntimeError(f"missing bucket {goal}/{action}")
                selected.append(bucket[torch.randint(len(bucket), (count,), generator=generator)])
        indices = torch.cat(selected); progress = (step - 1) / 4999; lr = 1e-3 + (1e-5 - 1e-3) * progress; optimizer.param_groups[0]["lr"] = lr; optimizer.zero_grad(set_to_none=True)
        condition = encoder(token_ids[train_labels["instruction_id"][indices]], lengths[train_labels["instruction_id"][indices]])
        loss = F.cross_entropy(supervisor(train_observations["features"][indices], condition), train_labels["action"][indices]); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            with torch.no_grad():
                val_condition = encoder(token_ids[val_labels["instruction_id"]], lengths[val_labels["instruction_id"]]); logits = supervisor(val_observations["features"], val_condition); accuracy = float((logits.argmax(-1) == val_labels["action"]).float().mean())
            metrics.append({"step": step, "train_loss": float(loss), "val_loss": float(F.cross_entropy(logits, val_labels["action"])), "val_accuracy": accuracy, "learning_rate": lr})
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": 5000, "trainable_parameters": parameter_count(encoder), "frozen_supervisor_parameters": parameter_count(supervisor), "and_trained": False}, root / "final.pt")
    (root / "training_metrics.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics), encoding="utf-8")
    return {"updates": 5000, "seed": seed, "batch_size": 128, "batch_counts": batch_counts, "final_validation": metrics[-1], "encoder_trainable_parameters": parameter_count(encoder), "supervisor_trainable_parameters": sum(p.numel() for p in supervisor.parameters() if p.requires_grad), "and_trained": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=4901); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); token_ids, lengths, ids = instruction_table(); train_data = load_data(SOURCE_ROOT, "train", token_ids, lengths, ids); val_data = load_data(SOURCE_ROOT, "val", token_ids, lengths, ids)
    encoder = SharedInstructionEncoder(); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); torch.save({"encoder": encoder.state_dict(), "seed": args.seed, "trainable_parameters": parameter_count(encoder), "and_trained": False}, args.output_root / "initial.pt")
    training = train(encoder, supervisor, train_data, val_data, token_ids, lengths, args.output_root, args.seed)
    metadata = {"train_rows": len(train_data[1]["action"]), "val_rows": len(val_data[1]["action"]), "instruction_forms": [text for text, _ in examples(False)], "and_trained": False, "source_observations": sha256(SOURCE_ROOT / "train" / "observations.pt"), "source_labels": sha256(SOURCE_ROOT / "train" / "labels.pt")}
    (args.output_root / "data_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I0", "baseline": "B", "seed": args.seed, "training": training, "data": metadata, "checkpoint": {"path": str(args.output_root / "final.pt"), "sha256": sha256(args.output_root / "final.pt")}, "and_trained": False}
    (args.output_root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(args.output_root / "results.json"), "checkpoint_sha256": result["checkpoint"]["sha256"], "status": result["status"], "training": training}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
