"""Evaluate Sol's four ASSIGN -> ATTR variable-binding ablations."""

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

from train_u0a import (  # noqa: E402
    COLOR_BASE,
    OBJECT_BASE,
    ExampleDataset,
    build_canonical_data,
    collate,
    evaluate_matrix,
    immediate_vectors,
    materialize,
    save_json,
)
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    CandidateState,
    ReadResult,
    ROW_ASSIGN,
    ROW_ATTR,
    SLOT_E,
    SLOT_P,
    UnifiedT1U0,
)


CASES = ("A_normal_normal", "B_assign_oracle_attr_normal", "C_assign_normal_attr_oracle", "D_both_oracle")


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def entropy(attention: torch.Tensor) -> torch.Tensor:
    safe = attention.clamp_min(1e-12)
    return -(attention * safe.log()).sum(dim=-1)


def direct_read_result(payload: torch.Tensor, memory_width: int, valid: torch.Tensor) -> ReadResult:
    return ReadResult(
        payload=payload,
        attention=torch.zeros(payload.shape[0], memory_width),
        margin=torch.zeros(payload.shape[0]),
        valid=valid,
    )


@torch.no_grad()
def evaluate_case(model: UnifiedT1U0, examples: list[object], *, assign_oracle: bool, attr_oracle: bool) -> dict[str, object]:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    assign_hits: list[bool] = []
    attr_hits: list[bool] = []
    attr_conditional_hits: list[bool] = []
    assign_margins: list[float] = []
    assign_entropies: list[float] = []
    attr_margins: list[float] = []
    attr_entropies: list[float] = []
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        object_ids = torch.arange(OBJECT_BASE, OBJECT_BASE + 32)
        color_ids = torch.arange(COLOR_BASE, COLOR_BASE + 8)
        assign_target = data["value_ids"][(data["memory_types"] == ROW_ASSIGN) & data["row_mask"]].view(-1)
        # Each homogeneous variable-binding row has exactly one ASSIGN row.
        assign_target = assign_target.reshape(state.shape[0], -1)[:, 0]
        attr_target = data["target_ids"]

        assign_opcode = torch.full_like(data["hops"], OPCODE_IDS["READ_P"])
        assign_valid = torch.ones_like(data["hops"], dtype=torch.bool)
        if assign_oracle:
            assign_payload = model.token_embedding(assign_target)
            assign_read = direct_read_result(assign_payload, data["memory_types"].shape[1], assign_valid)
        else:
            assign_read = model.memory_reader(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                assign_opcode,
                immediate_vectors(model, data["immediates"][:, 0]),
                data["source_slots"][:, 0],
            )
            assign_margins.extend(assign_read.margin.tolist())
            assign_entropies.extend(entropy(assign_read.attention).tolist())
        candidates = model.core(
            model.normalize_state(state, data["presence"]),
            model.opcode_embedding(assign_opcode),
            immediate_vectors(model, data["immediates"][:, 0]),
            assign_read.payload,
            model.slot_type_embeddings,
            data["presence"],
        )
        state = model.commit(state, candidates, assign_read, assign_opcode, torch.full_like(data["hops"], SLOT_P), data["presence"])
        assign_logits = F.normalize(state[:, SLOT_P], dim=-1) @ F.normalize(model.token_embedding(object_ids), dim=-1).transpose(0, 1)
        assign_prediction = object_ids[assign_logits.argmax(-1)]
        assign_ok = assign_prediction == assign_target
        assign_hits.extend(assign_ok.tolist())

        attr_opcode = torch.full_like(data["hops"], OPCODE_IDS["READ_E"])
        if attr_oracle:
            attr_payload = model.token_embedding(attr_target)
            attr_read = direct_read_result(attr_payload, data["memory_types"].shape[1], assign_valid)
        else:
            attr_read = model.memory_reader(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                attr_opcode,
                immediate_vectors(model, data["immediates"][:, 1]),
                data["source_slots"][:, 1],
            )
            attr_margins.extend(attr_read.margin.tolist())
            attr_entropies.extend(entropy(attr_read.attention).tolist())
        candidates = model.core(
            model.normalize_state(state, data["presence"]),
            model.opcode_embedding(attr_opcode),
            immediate_vectors(model, data["immediates"][:, 1]),
            attr_read.payload,
            model.slot_type_embeddings,
            data["presence"],
        )
        state = model.commit(state, candidates, attr_read, attr_opcode, torch.full_like(data["hops"], SLOT_E), data["presence"])
        color_logits = F.normalize(state[:, SLOT_E], dim=-1) @ F.normalize(model.token_embedding(color_ids), dim=-1).transpose(0, 1)
        attr_prediction = color_ids[color_logits.argmax(-1)]
        attr_ok = attr_prediction == attr_target
        attr_hits.extend(attr_ok.tolist())
        attr_conditional_hits.extend(attr_ok[assign_ok].tolist())
    return {
        "assign_accuracy": sum(assign_hits) / len(assign_hits),
        "attr_accuracy": sum(attr_hits) / len(attr_hits),
        "attr_accuracy_conditioned_on_assign_correct": None if not attr_conditional_hits else sum(attr_conditional_hits) / len(attr_conditional_hits),
        "assign_correct_count": sum(assign_hits),
        "sample_count": len(assign_hits),
        "reader_margin": {"assign": None if assign_oracle else sum(assign_margins) / len(assign_margins), "attr": None if attr_oracle else sum(attr_margins) / len(attr_margins)},
        "reader_entropy": {"assign": None if assign_oracle else sum(assign_entropies) / len(assign_entropies), "attr": None if attr_oracle else sum(attr_entropies) / len(attr_entropies)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0a_variable_real_seed101" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0a_variable_real_seed101" / "ablations.json")
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parent)
    model = load_model(args.checkpoint)
    results = {
        "A_normal_normal": evaluate_case(model, datasets["variable_binding"]["test"], assign_oracle=False, attr_oracle=False),
        "B_assign_oracle_attr_normal": evaluate_case(model, datasets["variable_binding"]["test"], assign_oracle=True, attr_oracle=False),
        "C_assign_normal_attr_oracle": evaluate_case(model, datasets["variable_binding"]["test"], assign_oracle=False, attr_oracle=True),
        "D_both_oracle": evaluate_case(model, datasets["variable_binding"]["test"], assign_oracle=True, attr_oracle=True),
    }
    output = {"checkpoint": str(args.checkpoint), "retrained": False, "cases": results}
    output["canonical_eval"] = evaluate_matrix(model, "variable_binding", datasets["variable_binding"]["test"], (1, 2, 4), (2,))
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
