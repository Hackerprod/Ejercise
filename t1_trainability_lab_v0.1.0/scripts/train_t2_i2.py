"""Train T2-I2 SemanticWriter; downstream CTRL-7 remains frozen."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from t2_i1_instruction import parse_instruction_i1
from t2_i2_semantic_writer import SemanticWriter, architecture_report, parameter_count, tensorize
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT, SOURCE_ROOT
from train_u0c_ctrl2_o import sha256

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"


def output_for_seed(seed: int) -> Path:
    return CAMPAIGN_ROOT / f"t2_i2_seed{seed}"


def checkpoint_for_seed(seed: int) -> Path:
    return output_for_seed(seed) / "final.pt"


def dummy_values(value: int) -> tuple[int, ...]:
    return tuple((value + offset) % 32 for offset in (0, 7, 13, 23))


def instruction_for_label(constraints: tuple[int, int], lower: int, forbidden: int, index: int, form: int) -> str:
    if constraints == (0, 0):
        dummy = index % 32
        return f"NOOP VALUE_{dummy}" if form == 1 else f"NOOP VALUE_{dummy} AND NOOP VALUE_{(dummy + 11) % 32}"
    value = lower if constraints == (1, 0) else forbidden
    aliases = ("AT_LEAST", "MINIMUM") if constraints == (1, 0) else ("AVOID", "EXCLUDE")
    alias = aliases[index % 2]
    dummy = dummy_values(value)[index % 4]
    operator = f"{alias} VALUE_{value}"
    if form == 1:
        return operator
    noop = f"NOOP VALUE_{dummy}"
    return f"{operator} AND {noop}" if form == 2 else f"{noop} AND {operator}"


def corpus() -> list[tuple[str, tuple[int, int], int]]:
    rows: list[tuple[str, tuple[int, int], int]] = []
    for dummy in range(32):
        rows.append((f"NOOP VALUE_{dummy}", (0, 0), 1))
        rows.append((f"NOOP VALUE_{dummy} AND NOOP VALUE_{(dummy + 11) % 32}", (0, 0), 2))
    for constraints in ((1, 0), (0, 1)):
        aliases = ("AT_LEAST", "MINIMUM") if constraints == (1, 0) else ("AVOID", "EXCLUDE")
        for alias in aliases:
            for value in range(32):
                for dummy in dummy_values(value):
                    operator = f"{alias} VALUE_{value}"
                    rows.append((operator, constraints, 1))
                    rows.append((f"{operator} AND NOOP VALUE_{dummy}", constraints, 2))
                    rows.append((f"NOOP VALUE_{dummy} AND {operator}", constraints, 2))
    return rows


def load_source(split: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    observations = torch.load(SOURCE_ROOT / split / "observations.pt", weights_only=False)
    labels = torch.load(SOURCE_ROOT / split / "labels.pt", weights_only=False)
    return observations, labels


def train(writer: SemanticWriter, supervisor: LatentConditionedSupervisor, observations: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], seed: int) -> dict[str, object]:
    actions = (0, 1, 2, 3, 5)
    goals = ((0, 0), (1, 0), (0, 1))
    buckets = {goal: {action: torch.where((labels["constraints"][:, 0] == goal[0]) & (labels["constraints"][:, 1] == goal[1]) & (labels["action"] == action))[0] for action in actions} for goal in goals}
    counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 1)
    last_loss = torch.tensor(0.0)
    for step in range(1, 5001):
        selected: list[torch.Tensor] = []
        forms: list[int] = []
        for goal, action_counts in counts.items():
            for action, count in action_counts.items():
                half = count // 2
                bucket = buckets[goal][action]
                if not len(bucket):
                    raise RuntimeError(f"missing bucket {goal}/{action}")
                selected.extend((bucket[torch.randint(len(bucket), (half,), generator=generator)], bucket[torch.randint(len(bucket), (count - half,), generator=generator)]))
                two_clause = count - half
                forms.extend((1,) * half + (2,) * (two_clause // 2) + (3,) * (two_clause - two_clause // 2))
        indices = torch.cat(selected)
        texts = [instruction_for_label(tuple(labels["constraints"][index].tolist()), int(labels["lower"][index]), int(labels["forbidden"][index]), position, form) for position, (index, form) in enumerate(zip(indices.tolist(), forms))]
        token_ids, lengths = tensorize(texts)
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * ((step - 1) / 4999)
        optimizer.zero_grad(set_to_none=True)
        condition = writer(token_ids, lengths).sum(dim=1)
        last_loss = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices])
        last_loss.backward()
        optimizer.step()
    return {"updates": 5000, "seed": seed, "batch_size": 128, "trainable_parameters": parameter_count(writer), "supervisor_trainable_parameters": sum(parameter.numel() for parameter in supervisor.parameters() if parameter.requires_grad), "final_loss": float(last_loss.detach()), "one_clause_fraction": 0.5, "two_clause_fraction": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output_root = output_for_seed(args.seed) if args.output_root is None else args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    writer = SemanticWriter()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    observations, labels = load_source("train")
    training = train(writer, supervisor, observations, labels, args.seed)
    torch.save({"writer": writer.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(writer), "and_trained": False}, checkpoint_for_seed(args.seed) if args.output_root is None else output_root / "final.pt")
    report = architecture_report(writer)
    (output_root / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I2", "seed": args.seed, "training": training, "checkpoint": sha256(output_root / "final.pt"), "architecture_report": report, "heldout_real_operator_combinations": True}
    (output_root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
