"""Evaluate sequential-update free-running versus teacher-forced trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_u0a import (  # noqa: E402
    ExampleDataset,
    VALUE_CLASS_IDS,
    build_canonical_data,
    collate,
    immediate_vectors,
    materialize,
    save_json,
)
from t1_trainability.unified import SLOT_R, UnifiedT1U0  # noqa: E402


HOPS = (3, 4, 5, 6)
ROUNDS = (1, 2, 4, 6)


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: UnifiedT1U0, examples: list[object], teacher_forced: bool) -> dict[str, dict[str, float]]:
    hits = {str(hop): {str(rounds): 0 for rounds in ROUNDS} for hop in HOPS}
    counts = {str(hop): 0 for hop in HOPS}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(6):
            if teacher_forced and round_index > 0:
                targets = data["intermediate_target_ids"][:, round_index - 1]
                active = targets >= 0
                state[:, SLOT_R] = torch.where(active.unsqueeze(-1), model.token_embedding(targets.clamp_min(0)), state[:, SLOT_R])
            state, _, _ = model.step(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                data["opcodes"][:, round_index],
                immediate_vectors(model, data["immediates"][:, round_index]),
                data["source_slots"][:, round_index],
                data["destination_slots"][:, round_index],
                data["presence"],
            )
            if round_index + 1 in ROUNDS:
                logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
                predicted = class_ids[logits.argmax(-1)]
                for hop in HOPS:
                    selected = batch["hops"] == hop
                    hits[str(hop)][str(round_index + 1)] += int((predicted[selected] == batch["target_ids"][selected]).sum())
        for hop in HOPS:
            counts[str(hop)] += int((batch["hops"] == hop).sum())
    return {hop: {rounds: values / counts[hop] for rounds, values in rounds_map.items()} for hop, rounds_map in hits.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parent)
    model = load_model(args.checkpoint)
    result = {
        "checkpoint": str(args.checkpoint),
        "retrained": False,
        "free_running": evaluate(model, datasets["sequential_update"]["test"], False),
        "teacher_forced": evaluate(model, datasets["sequential_update"]["test"], True),
    }
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
