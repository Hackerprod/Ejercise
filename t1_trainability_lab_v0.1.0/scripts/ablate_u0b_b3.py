"""Run U0-B3 WRITE_E-disabled ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import pointer_diagnostics, summary  # noqa: E402
from train_u0a import (  # noqa: E402
    ExampleDataset,
    build_canonical_data,
    build_sequential_h1_table,
    collate,
    evaluate_accuracy,
    evaluate_all,
    materialize,
)
from t1_trainability.unified import OPCODE_IDS, ROW_ATTR, SLOT_E, SLOT_P, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)
OBJECT_CLASS_IDS = tuple(range(320, 352))
COLOR_CLASS_IDS = tuple(range(352, 360))


class B3WriteEFrozenModel(UnifiedT1U0):
    """Runtime-only B3 variant; preserve E on every READ_E commit."""

    def step(self, state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate, source_slot, destination_slot, presence_mask):  # type: ignore[no-untyped-def]
        next_state, candidates, read_result = super().step(
            state,
            memory_keys,
            memory_values,
            memory_types,
            row_mask,
            opcode,
            immediate,
            source_slot,
            destination_slot,
            presence_mask,
        )
        active = (opcode == OPCODE_IDS["READ_E"]) & (destination_slot == SLOT_E)
        evidence = torch.where(active.unsqueeze(-1), state[:, SLOT_E, :], next_state[:, SLOT_E, :])
        next_state = torch.cat((next_state[:, :SLOT_E, :], evidence.unsqueeze(1), next_state[:, SLOT_E + 1 :, :]), dim=1)
        return next_state, candidates, read_result


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model: UnifiedT1U0 = B3WriteEFrozenModel(64) if ablated else UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def variable_binding_metrics(model: UnifiedT1U0, examples) -> dict[str, float]:  # type: ignore[no-untyped-def]
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    assign_correct = 0
    attr_reference_correct = 0
    attr_reference_mass = 0.0
    attr_output_correct = 0
    count = 0
    for offset in range(0, len(examples), 256):
        batch_examples = examples[offset : offset + 256]
        batch = collate(batch_examples)
        data = materialize(model, batch)
        state = data["state"]
        second_read = None
        for round_index in range(2):
            round_immediate = model.memory_reader._batch_vector(
                data["immediates"][:, round_index],
                dimension=model.dimension,
                embedding=model.immediate_embedding,
            )
            state, _, read_result = model.step(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                data["opcodes"][:, round_index],
                round_immediate,
                data["source_slots"][:, round_index],
                data["destination_slots"][:, round_index],
                data["presence"],
            )
            if round_index == 1:
                second_read = read_result
        class_ids = torch.tensor(OBJECT_CLASS_IDS, dtype=torch.long)
        assign_logits = model.register_decoder(state[:, SLOT_P, :], model.token_embedding(class_ids))
        assign_predictions = class_ids[assign_logits.argmax(dim=-1)]
        color_ids = torch.tensor(COLOR_CLASS_IDS, dtype=torch.long)
        attr_logits = model.evidence_decoder(state[:, SLOT_E, :], model.token_embedding(color_ids))
        attr_predictions = color_ids[attr_logits.argmax(dim=-1)]
        assert second_read is not None
        for row_index, example in enumerate(batch_examples):
            assignment_target = example.memory_values[0]
            attr_candidates = [
                column
                for column, (key, row_type) in enumerate(zip(example.memory_keys, example.memory_types))
                if row_type == ROW_ATTR and key == assignment_target
            ]
            if int(assign_predictions[row_index]) == assignment_target:
                assign_correct += 1
            if attr_candidates:
                target_mass = sum(float(second_read.attention[row_index, column]) for column in attr_candidates)
                attr_reference_mass += target_mass
                if int(second_read.attention[row_index].argmax()) in attr_candidates:
                    attr_reference_correct += 1
            if int(attr_predictions[row_index]) == example.target_id:
                attr_output_correct += 1
            count += 1
    attr_count = sum(any(row_type == ROW_ATTR for row_type in example.memory_types) for example in examples)
    return {
        "assign_accuracy": assign_correct / count,
        "attr_reference_correct": attr_reference_correct / attr_count,
        "attr_reference_target_mass": attr_reference_mass / attr_count,
        "attr_end_to_end_accuracy": attr_output_correct / count,
        "examples": count,
    }


def b3_summary(metrics: dict[str, object], variable: dict[str, float]) -> dict[str, object]:
    output = summary(metrics)
    output["variable_binding_end_to_end"] = metrics["variable_binding"]["2"]["2"]
    output["variable_binding_breakdown"] = variable
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b3_write_e_disabled.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    runs: dict[str, object] = {}
    for seed in args.seeds:
        run_dir = ROOT / "campaign" / f"u0a_iso_clean_seed{seed}_12000"
        checkpoint = run_dir / "best.pt"
        datasets = build_canonical_data(run_dir)
        baseline_model = load_model(checkpoint, ablated=False)
        ablated_model = load_model(checkpoint, ablated=True)
        baseline_metrics = evaluate_all(baseline_model, datasets, "test")
        baseline_metrics["sequential_update_h1_table"] = evaluate_accuracy(baseline_model, "sequential_update", build_sequential_h1_table(), rounds=1)
        ablated_metrics = evaluate_all(ablated_model, datasets, "test")
        ablated_metrics["sequential_update_h1_table"] = evaluate_accuracy(ablated_model, "sequential_update", build_sequential_h1_table(), rounds=1)
        baseline_variable = variable_binding_metrics(baseline_model, datasets["variable_binding"]["test"])
        ablated_variable = variable_binding_metrics(ablated_model, datasets["variable_binding"]["test"])
        baseline = b3_summary(baseline_metrics, baseline_variable)
        ablation = b3_summary(ablated_metrics, ablated_variable)
        runs[str(seed)] = {
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "baseline": baseline,
            "ablation": ablation,
            "delta_ablation_minus_baseline": {
                "associative_final": ablation["associative_final"] - baseline["associative_final"],
                "pointer_final_h4": ablation["pointer_final_h4"] - baseline["pointer_final_h4"],
                "multi_hop_final_h3": ablation["multi_hop_final_h3"] - baseline["multi_hop_final_h3"],
                "sequential_h1_table": ablation["sequential_h1_table"] - baseline["sequential_h1_table"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
                "variable_binding_end_to_end": ablation["variable_binding_end_to_end"] - baseline["variable_binding_end_to_end"],
            },
            "pointer_diagnostics": {
                "baseline": pointer_diagnostics(baseline_model, datasets["pointer_chasing"]["test"]),
                "ablation": pointer_diagnostics(ablated_model, datasets["pointer_chasing"]["test"]),
            },
        }
    result = {
        "phase": "T1-U0-B3",
        "ablation": "E_{r+1}=E_r for READ_E (WRITE_E disabled)",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "associative_recall": "collapse",
            "variable_binding": "ATTR output collapses while ASSIGN remains functional; report reader reference separately",
            "unrelated": "pointer, ALU, workspace unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
