"""Evaluate sequential-update checkpoint by final operation and hop count."""

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
    build_canonical_data,
    class_ids_for_task,
    collate,
    decode,
    run_rounds,
    save_json,
)
from t1_trainability.unified import SLOT_R, UnifiedT1U0  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    examples = build_canonical_data(args.checkpoint.parent)["sequential_update"]["test"]
    class_ids = class_ids_for_task("sequential_update", torch.device("cpu"))
    names = {2: "ALU_ADD", 3: "ALU_SUB", 4: "ALU_MUL"}
    hits: dict[str, dict[str, list[bool]]] = {name: {str(h): [] for h in (3, 4, 5, 6)} for name in names.values()}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            state = run_rounds(model, batch, 6)
            logits, _ = decode(model, state[:, SLOT_R], class_ids, model.register_decoder)
            predicted = class_ids[logits.argmax(-1)]
            final_index = (batch["hops"] - 1).clamp_min(0)
            final_opcode = batch["opcodes"].gather(1, final_index.unsqueeze(1)).squeeze(1)
            for opcode, name in names.items():
                for hop in (3, 4, 5, 6):
                    selected = (final_opcode == opcode) & (batch["hops"] == hop)
                    hits[name][str(hop)].extend((predicted[selected] == batch["target_ids"][selected]).tolist())
    result = {
        "checkpoint": str(args.checkpoint),
        "by_final_operation_and_hop": {name: {hop: (sum(values) / len(values) if values else None) for hop, values in rows.items()} for name, rows in hits.items()},
    }
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
