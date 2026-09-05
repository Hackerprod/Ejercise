"""List free-running composition failures at each exact H frontier."""

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
    VALUE_BASE,
    VALUE_CLASS_IDS,
    build_canonical_data,
    collate,
    immediate_vectors,
    materialize,
    save_json,
)
from t1_trainability.unified import OPCODES, SLOT_R, UnifiedT1U0  # noqa: E402


TARGET_HOPS = (3, 4, 6)


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def inspect(model: UnifiedT1U0, examples: list[object]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    offset = 0
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        predictions_by_round: dict[int, torch.Tensor] = {}
        for round_index in range(6):
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
            logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
            predictions_by_round[round_index + 1] = class_ids[logits.argmax(-1)]
        for local_index, example in enumerate(examples[offset : offset + len(batch["hops"])]):
            hop = int(example.hop_count)
            if hop not in TARGET_HOPS:
                continue
            predicted = int(predictions_by_round[hop][local_index])
            target = int(batch["target_ids"][local_index])
            if predicted == target:
                continue
            active_rounds = []
            for round_index, opcode_id in enumerate(example.opcodes[:hop]):
                active_rounds.append({"opcode": OPCODES[opcode_id], "operand": int(example.immediates[round_index] - VALUE_BASE)})
            failures.append(
                {
                    "test_index": offset + local_index,
                    "hop": hop,
                    "initial": int(example.initial_ids[SLOT_R] - VALUE_BASE),
                    "operations": active_rounds,
                    "target": int(target - VALUE_BASE),
                    "predicted": predicted - VALUE_BASE,
                    "predictions_by_round": {str(rounds): int(values[local_index] - VALUE_BASE) for rounds, values in predictions_by_round.items() if rounds <= hop},
                }
            )
        offset += len(batch["hops"])
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parent)
    output = {"checkpoint": str(args.checkpoint), "retrained": False, "failures": inspect(load_model(args.checkpoint), datasets["sequential_update"]["test"])}
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
