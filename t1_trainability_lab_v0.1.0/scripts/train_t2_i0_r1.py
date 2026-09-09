"""T2-I0-R1 deconfounded Baseline A/B training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_baseline_a import SharedInstructionEncoder
from train_u0c_ctrl2_o import sha256
from t2_i0_instruction_r1 import parse_instruction_r1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
CTRL7_CHECKPOINT = SOURCE_ROOT / "final.pt"
OUTPUT_A = CAMPAIGN_ROOT / "t2_i0_r1_baseline_a_seed5301"
OUTPUT_B = CAMPAIGN_ROOT / "t2_i0_r1_baseline_b_seed5401"


class SharedInstructionEncoderR1(SharedInstructionEncoder):
    def __init__(self) -> None:
        super().__init__(vocabulary_size=37, embedding_size=16, hidden_size=32)


class BaselineAR1Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.encoder = SharedInstructionEncoderR1(); self.floor_head = nn.Linear(32, 2); self.avoid_head = nn.Linear(32, 2)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        representation = self.encoder(token_ids, lengths); return self.floor_head(representation), self.avoid_head(representation)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def r1_examples(include_heldout: bool = False) -> list[tuple[str, tuple[int, int]]]:
    rows = [("NOOP VALUE_0", (0, 0)), ("NOOP VALUE_0 AND NOOP VALUE_0", (0, 0))]
    for value in range(32):
        rows.extend(((f"AT_LEAST VALUE_{value}", (1, 0)), (f"AT_LEAST VALUE_{value} AND NOOP VALUE_0", (1, 0)), (f"NOOP VALUE_0 AND AT_LEAST VALUE_{value}", (1, 0))))
        rows.extend(((f"AVOID VALUE_{value}", (0, 1)), (f"AVOID VALUE_{value} AND NOOP VALUE_0", (0, 1)), (f"NOOP VALUE_0 AND AVOID VALUE_{value}", (0, 1))))
    if include_heldout:
        rows.extend((f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}", (1, 1)) for lower in range(32) for forbidden in range(31))
        rows.extend((f"AVOID VALUE_{forbidden} AND AT_LEAST VALUE_{lower}", (1, 1)) for lower in range(32) for forbidden in range(31))
    return rows


def tensorize(rows: list[tuple[str, tuple[int, int]]]) -> dict[str, Tensor]:
    parsed = [parse_instruction_r1(text) for text, _ in rows]; lengths = torch.tensor([len(item.token_ids) for item in parsed], dtype=torch.long); tokens = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(parsed): tokens[index, : len(item.token_ids)] = torch.tensor(item.token_ids)
    return {"token_ids": tokens, "lengths": lengths, "bits": torch.tensor([bits for _, bits in rows], dtype=torch.long)}


def instruction_for(constraints: tuple[int, int], lower: int, forbidden: int, variant: int, dummy: int = 0) -> str:
    dummy_text = f"NOOP VALUE_{dummy}"
    if constraints == (0, 0): return dummy_text if variant == 0 else f"{dummy_text} AND {dummy_text}"
    operator = f"AT_LEAST VALUE_{lower}" if constraints == (1, 0) else f"AVOID VALUE_{forbidden}"
    return (operator, f"{operator} AND {dummy_text}", f"{dummy_text} AND {operator}")[variant]


def source_data(split: str, token_rows: list[tuple[str, tuple[int, int]]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    observations = torch.load(SOURCE_ROOT / split / "observations.pt", weights_only=False); labels = torch.load(SOURCE_ROOT / split / "labels.pt", weights_only=False); lookup = {text: index for index, (text, _) in enumerate(token_rows)}; instruction_ids = []
    for index, (bits, lower, forbidden) in enumerate(zip(labels["constraints"].tolist(), labels["lower"].tolist(), labels["forbidden"].tolist())):
        key = tuple(bits); variant = index % (2 if key == (0, 0) else 3); instruction_ids.append(lookup[instruction_for(key, lower, forbidden, variant)])
    return observations, {**labels, "instruction_id": torch.tensor(instruction_ids, dtype=torch.long)}


def train_a(model: BaselineAR1Classifier, data: dict[str, Tensor], seed: int) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); rows = data["bits"]
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; indices = torch.randint(len(rows), (128,), generator=generator); optimizer.zero_grad(set_to_none=True); floor, avoid = model(data["token_ids"][indices], data["lengths"][indices]); loss = (F.cross_entropy(floor, rows[indices, 0]) + F.cross_entropy(avoid, rows[indices, 1])) / 2; loss.backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "trainable_parameters": parameter_count(model), "final_loss": float(loss.detach())}


def train_b(encoder: SharedInstructionEncoderR1, supervisor: LatentConditionedSupervisor, data: tuple[dict[str, Tensor], dict[str, Tensor]], seed: int) -> dict[str, float]:
    observations, labels = data; optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); buckets = {key: {action: torch.where((labels["constraints"][:, 0] == key[0]) & (labels["constraints"][:, 1] == key[1]) & (labels["action"] == action))[0] for action in (0, 1, 2, 3, 5)} for key in ((0, 0), (1, 0), (0, 1))}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    for step in range(1, 5001):
        selected = []; progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
        for key, action_counts in counts.items():
            for action, count in action_counts.items():
                bucket = buckets[key][action]; selected.append(bucket[torch.randint(len(bucket), (count,), generator=generator)])
        indices = torch.cat(selected); optimizer.zero_grad(set_to_none=True); condition = encoder(data[0]["_token_ids"][labels["instruction_id"][indices]], data[0]["_lengths"][labels["instruction_id"][indices]]); loss = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices]); loss.backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "trainable_parameters": parameter_count(encoder), "frozen_supervisor_trainable": 0, "final_loss": float(loss.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed-a", type=int, default=5301); parser.add_argument("--seed-b", type=int, default=5401); args = parser.parse_args(); torch.manual_seed(args.seed_a); random.seed(args.seed_a)
    rows = r1_examples(False); token_data = tensorize(rows); a = BaselineAR1Classifier(); a_training = train_a(a, token_data, args.seed_a); OUTPUT_A.mkdir(parents=True, exist_ok=True); torch.save({"model": a.state_dict(), "seed": args.seed_a, "updates": 5000, "and_trained": False, "trainable_parameters": parameter_count(a)}, OUTPUT_A / "final.pt")
    torch.manual_seed(args.seed_b); random.seed(args.seed_b); encoder = SharedInstructionEncoderR1(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); train_observations, train_labels = source_data("train", rows); train_observations = {**train_observations, "_token_ids": token_data["token_ids"], "_lengths": token_data["lengths"]}; b_training = train_b(encoder, core, (train_observations, train_labels), args.seed_b); OUTPUT_B.mkdir(parents=True, exist_ok=True); torch.save({"encoder": encoder.state_dict(), "seed": args.seed_b, "updates": 5000, "and_trained": False, "trainable_parameters": parameter_count(encoder)}, OUTPUT_B / "final.pt")
    result = {"status": "trained", "task": "T2-I0-R1", "baseline_a": a_training, "baseline_b": b_training, "instruction_forms": len(rows), "heldout_forms_excluded": True, "checkpoints": {"a": sha256(OUTPUT_A / "final.pt"), "b": sha256(OUTPUT_B / "final.pt")}}
    for root, payload in ((OUTPUT_A, {"status": "trained", "baseline": "A", **result}), (OUTPUT_B, {"status": "trained", "baseline": "B", **result})): (root / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
