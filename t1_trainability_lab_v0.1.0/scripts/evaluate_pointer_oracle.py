"""Evaluate pointer canary with oracle READ_P payloads, no retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_u0a import ExampleDataset, POINTER_CLASS_IDS, build_canonical_data, collate, decode, materialize, save_json  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, CandidateState, ReadResult, SLOT_P, UnifiedT1U0  # noqa: E402


def load_model(checkpoint: Path) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: UnifiedT1U0, examples: list[object], rounds: int, *, direct_decoder: bool = False) -> dict[str, float]:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    hits: dict[int, list[bool]] = {hop: [] for hop in (1, 2, 3, 4)}
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(rounds):
            active = data["hops"] > round_index
            payload = torch.zeros_like(state[:, SLOT_P, :])
            for row_index in torch.where(active)[0].tolist():
                current = int(data["initial_ids"][row_index, SLOT_P]) if round_index == 0 else int(data["state_pointer_ids"][row_index])
                mapping = {
                    int(data["key_ids"][row_index, memory_index]): int(data["value_ids"][row_index, memory_index])
                    for memory_index in torch.where(data["row_mask"][row_index])[0].tolist()
                }
                destination = mapping[current]
                payload[row_index] = model.token_embedding(torch.tensor(destination))
                data.setdefault("state_pointer_ids", {})
                data["state_pointer_ids"][row_index] = destination
            opcode = torch.where(active, torch.full_like(data["hops"], OPCODE_IDS["READ_P"]), torch.full_like(data["hops"], OPCODE_IDS["EMIT"]))
            read_result = ReadResult(
                payload=payload,
                attention=torch.zeros(data["state"].shape[0], data["memory_types"].shape[1]),
                margin=torch.zeros(data["state"].shape[0]),
                valid=active,
            )
            candidates = model.core(
                model.normalize_state(state, data["presence"]),
                model.opcode_embedding(opcode),
                model.token_embedding(data["immediates"][:, round_index]),
                payload,
                model.slot_type_embeddings,
                data["presence"],
            )
            state = model.commit(
                state,
                CandidateState(candidates.values),
                read_result,
                opcode,
                torch.full_like(data["hops"], SLOT_P),
                data["presence"],
            )
        if direct_decoder:
            class_ids = torch.arange(256, dtype=torch.long) + 0
            basis = F.normalize(model.token_embedding(class_ids), dim=-1)
            logits = F.normalize(state[:, SLOT_P], dim=-1) @ basis.transpose(0, 1) * 20.0
        else:
            class_ids = torch.tensor(POINTER_CLASS_IDS, dtype=torch.long)
            logits, class_ids = decode(model, state[:, SLOT_P], class_ids, model.pointer_decoder)
        predictions = class_ids[logits.argmax(dim=-1)]
        for hop in (1, 2, 3, 4):
            selected = data["hops"] == hop
            hits[hop].extend((predictions[selected] == data["target_ids"][selected]).tolist())
    return {str(hop): sum(values) / len(values) for hop, values in hits.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0a_canaries_seed101_sameinit" / "pointer_chasing" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0a_canaries_seed101_sameinit" / "pointer_chasing" / "oracle_eval.json")
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parents[1])
    model = load_model(args.checkpoint)
    learned = {str(rounds): evaluate(model, datasets["pointer_chasing"]["test"], rounds) for rounds in (1, 2, 4)}
    direct = {str(rounds): evaluate(model, datasets["pointer_chasing"]["test"], rounds, direct_decoder=True) for rounds in (1, 2, 4)}
    output = {"checkpoint": str(args.checkpoint), "reader": "oracle destination from canonical REL mapping", "retrained": False, "decoder_ablation": {"learned_unified": learned, "direct_tied_codebook": direct}}
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
