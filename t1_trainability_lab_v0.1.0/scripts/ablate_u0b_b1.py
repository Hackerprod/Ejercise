"""Run U0-B1 pointer replacement-to-residual ablation without training."""

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
    ExampleDataset,
    build_canonical_data,
    build_sequential_h1_table,
    collate,
    evaluate_all,
    evaluate_accuracy,
    materialize,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_P, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)


class B1PointerResidualModel(UnifiedT1U0):
    """Runtime-only B1 variant; frozen checkpoint weights remain unchanged."""

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
        active = (opcode == OPCODE_IDS["READ_P"]) & (destination_slot == SLOT_P)
        residual = F.normalize(state[:, SLOT_P, :] + read_result.payload, dim=-1)
        residual = residual * presence_mask[:, SLOT_P].to(dtype=residual.dtype).unsqueeze(-1)
        pointer = torch.where(active.unsqueeze(-1), residual, next_state[:, SLOT_P, :])
        next_state = torch.cat((pointer.unsqueeze(1), next_state[:, 1:, :]), dim=1)
        return next_state, candidates, read_result


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model: UnifiedT1U0 = B1PointerResidualModel(64) if ablated else UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def pointer_diagnostics(model: UnifiedT1U0, examples) -> dict[str, object]:  # type: ignore[no-untyped-def]
    rows = [{"round": round_index + 1, "pointer_acc": 0.0, "reader_entropy": 0.0, "reader_margin": 0.0, "old_pointer_mass": 0.0, "correct_key_probability": 0.0, "second_candidate_probability": 0.0} for round_index in range(4)]
    count = 0
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    offset = 0
    for batch in loader:
        batch_examples = examples[offset : offset + len(batch["target_ids"])]
        offset += len(batch["target_ids"])
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(4):
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
            for row_index, example in enumerate(batch_examples):
                mapping = dict(zip(example.memory_keys, example.memory_values))
                current = example.initial_ids[0]
                path = [current]
                for _ in range(4):
                    current = mapping[current]
                    path.append(current)
                destination_ids = example.memory_values
                attention = read_result.attention[row_index]
                probabilities: dict[int, float] = {}
                for column, destination in enumerate(destination_ids):
                    probabilities[destination] = probabilities.get(destination, 0.0) + float(attention[column])
                correct = probabilities.get(path[round_index + 1], 0.0)
                old = probabilities.get(path[round_index], 0.0)
                ordered = sorted(probabilities.values(), reverse=True)
                second = ordered[1] if len(ordered) > 1 else 0.0
                entropy = float(-(attention * attention.clamp_min(1e-12).log()).sum())
                rows[round_index]["pointer_acc"] += float(max(probabilities, key=probabilities.get) == path[round_index + 1])
                rows[round_index]["reader_entropy"] += entropy
                rows[round_index]["reader_margin"] += correct - second
                rows[round_index]["old_pointer_mass"] += old
                rows[round_index]["correct_key_probability"] += correct
                rows[round_index]["second_candidate_probability"] += second
                count += 1
    for row in rows:
        for key in row:
            if key != "round":
                row[key] /= count // 4
    return {"examples": count // 4, "rounds": rows}


def summary(metrics: dict[str, object]) -> dict[str, object]:
    sequential = metrics["sequential_update"]
    return {
        "pointer_final_h4": metrics["pointer_chasing"]["4"]["4"],
        "multi_hop_final_h3": metrics["multi_hop"]["3"]["4"],
        "multi_hop_final_h4": metrics["multi_hop"]["4"]["4"],
        "associative_final": metrics["associative_recall"]["1"]["1"],
        "sequential_h1_table": metrics["sequential_update_h1_table"],
        "sequential_composition_final_round6": {hop: sequential[hop]["6"] for hop in ("3", "4", "5", "6")},
        "workspace_h6_error": metrics["workspace_accumulation"]["6"]["6"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b1_pointer_residual.json")
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
        baseline_summary = summary(baseline_metrics)
        ablated_summary = summary(ablated_metrics)
        runs[str(seed)] = {
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "baseline": baseline_summary,
            "ablation": ablated_summary,
            "delta_ablation_minus_baseline": {
                "pointer_final_h4": ablated_summary["pointer_final_h4"] - baseline_summary["pointer_final_h4"],
                "multi_hop_final_h3": ablated_summary["multi_hop_final_h3"] - baseline_summary["multi_hop_final_h3"],
                "multi_hop_final_h4": ablated_summary["multi_hop_final_h4"] - baseline_summary["multi_hop_final_h4"],
                "associative_final": ablated_summary["associative_final"] - baseline_summary["associative_final"],
                "workspace_h6_error": ablated_summary["workspace_h6_error"] - baseline_summary["workspace_h6_error"],
            },
            "pointer_diagnostics": {
                "baseline": pointer_diagnostics(baseline_model, datasets["pointer_chasing"]["test"]),
                "ablation": pointer_diagnostics(ablated_model, datasets["pointer_chasing"]["test"]),
            },
        }
    result = {
        "phase": "T1-U0-B1",
        "ablation": "P_{r+1}=Canon(Y_r) -> P_{r+1}=F.normalize(P_r+Y_r, dim=-1)",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "pointer_chasing": "deep frontier loss",
            "multi_hop": "H3/H4 degradation",
            "pointer_diagnostics": "old_pointer_mass increases",
            "unrelated": "associative, ALU, workspace nearly unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
