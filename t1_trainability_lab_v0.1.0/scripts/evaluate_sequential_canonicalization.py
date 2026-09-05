"""Post-hoc sequential-update canonicalization audit; never updates weights."""

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
    save_json,
)
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    SLOT_R,
    UnifiedT1U0,
)


ROUNDS = (1, 2, 4, 6)
HOPS = (3, 4, 5, 6)
VARIANTS = ("raw", "soft_canonical", "hard_canonical")


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def evaluate(model: UnifiedT1U0, examples: list[object]) -> dict[str, dict[str, dict[str, float]]]:
    result = {
        variant: {str(hop): {str(rounds): 0.0 for rounds in ROUNDS} for hop in HOPS}
        for variant in VARIANTS
    }
    counts = {str(hop): 0 for hop in HOPS}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            data = {**batch}
            # Materialize here so every variant starts from identical state.
            from train_u0a import materialize, immediate_vectors  # local import avoids import-cycle tooling noise

            data = materialize(model, data)
            states = {variant: data["state"].clone() for variant in VARIANTS}
            for variant in VARIANTS:
                state = states[variant]
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
                    if variant != "raw" and torch.isin(
                        data["opcodes"][:, round_index],
                        torch.tensor((OPCODE_IDS["ALU_ADD"], OPCODE_IDS["ALU_SUB"], OPCODE_IDS["ALU_MUL"])),
                    ).any():
                        alu = torch.isin(
                            data["opcodes"][:, round_index],
                            torch.tensor((OPCODE_IDS["ALU_ADD"], OPCODE_IDS["ALU_SUB"], OPCODE_IDS["ALU_MUL"])),
                        )
                        logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
                        if variant == "soft_canonical":
                            canonical = torch.softmax(logits, dim=-1) @ model.token_embedding(class_ids)
                        else:
                            canonical = model.token_embedding(class_ids[logits.argmax(dim=-1)])
                        state[:, SLOT_R] = torch.where(alu.unsqueeze(-1), canonical, state[:, SLOT_R])
                    rounds_done = round_index + 1
                    if rounds_done in ROUNDS:
                        logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
                        predicted = class_ids[logits.argmax(dim=-1)]
                        for hop in HOPS:
                            selected = batch["hops"] == hop
                            result[variant][str(hop)][str(rounds_done)] += float((predicted[selected] == batch["target_ids"][selected]).sum())
            for hop in HOPS:
                counts[str(hop)] += int((batch["hops"] == hop).sum())
    return {
        variant: {
            hop: {rounds: values / counts[hop] for rounds, values in hop_values.items()}
            for hop, hop_values in variant_values.items()
        }
        for variant, variant_values in result.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parent)
    model = load_model(args.checkpoint)
    output = {"checkpoint": str(args.checkpoint), "retrained": False, "matrix_h_by_round": evaluate(model, datasets["sequential_update"]["test"])}
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
