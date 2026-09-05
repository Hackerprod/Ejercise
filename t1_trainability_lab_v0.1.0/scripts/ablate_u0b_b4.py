"""Run U0-B4 raw continuous-register ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import summary  # noqa: E402
from train_u0a import (  # noqa: E402
    ExampleDataset,
    VALUE_CLASS_IDS,
    build_canonical_data,
    build_sequential_h1_table,
    collate,
    evaluate_accuracy,
    evaluate_all,
    immediate_vectors,
    materialize,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_R, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)
HOPS = (3, 4, 5, 6)
ROUNDS = (1, 2, 4, 6)
ALU_NAMES = ("ALU_ADD", "ALU_SUB", "ALU_MUL")


class B4RawRegisterModel(UnifiedT1U0):
    """Runtime-only B4 variant; keep continuous core candidate in R."""

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
        alu = torch.isin(opcode, torch.tensor(tuple(OPCODE_IDS[name] for name in ALU_NAMES), device=opcode.device))
        # Current U0-A heads emit 32 logits while historical pre-fix heads
        # emitted D-dimensional continuous vectors.  This is equivalent raw
        # continuous output in current parameterization: project logits onto
        # the value basis without softmax or canonical normalization.
        codebook = self.token_embedding(torch.arange(288, 320, device=opcode.device))
        register = torch.where(alu.unsqueeze(-1), candidates.alu_logits @ codebook, next_state[:, SLOT_R, :])
        next_state = torch.cat((next_state[:, :SLOT_R, :], register.unsqueeze(1), next_state[:, SLOT_R + 1 :, :]), dim=1)
        return next_state, candidates, read_result


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model: UnifiedT1U0 = B4RawRegisterModel(64) if ablated else UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_sequential(model: UnifiedT1U0, examples: list[object], teacher_forced: bool) -> dict[str, dict[str, float]]:
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
                active = (targets >= 0) & (data["opcodes"][:, round_index] != OPCODE_IDS["EMIT"])
                state[:, SLOT_R, :] = torch.where(active.unsqueeze(-1), model.token_embedding(targets.clamp_min(0)), state[:, SLOT_R, :])
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
            rounds_done = round_index + 1
            if rounds_done in ROUNDS:
                logits = model.register_decoder(state[:, SLOT_R, :], model.token_embedding(class_ids))
                predicted = class_ids[logits.argmax(dim=-1)]
                for hop in HOPS:
                    selected = batch["hops"] == hop
                    hits[str(hop)][str(rounds_done)] += int((predicted[selected] == batch["target_ids"][selected]).sum())
        for hop in HOPS:
            counts[str(hop)] += int((batch["hops"] == hop).sum())
    return {hop: {rounds: values / counts[hop] for rounds, values in rounds_map.items()} for hop, rounds_map in hits.items()}


@torch.no_grad()
def evaluate_by_final_operation(model: UnifiedT1U0, examples: list[object], teacher_forced: bool) -> dict[str, dict[str, float]]:
    hits = {name: {str(hop): 0 for hop in HOPS} for name in ALU_NAMES}
    counts = {name: {str(hop): 0 for hop in HOPS} for name in ALU_NAMES}
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    for offset in range(0, len(examples), 256):
        batch_examples = examples[offset : offset + 256]
        batch = collate(batch_examples)
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(6):
            if teacher_forced and round_index > 0:
                targets = data["intermediate_target_ids"][:, round_index - 1]
                active = (targets >= 0) & (data["opcodes"][:, round_index] != OPCODE_IDS["EMIT"])
                state[:, SLOT_R, :] = torch.where(active.unsqueeze(-1), model.token_embedding(targets.clamp_min(0)), state[:, SLOT_R, :])
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
        logits = model.register_decoder(state[:, SLOT_R, :], model.token_embedding(class_ids))
        predicted = class_ids[logits.argmax(dim=-1)]
        for row_index, example in enumerate(batch_examples):
            hop = example.hop_count
            name = ALU_NAMES[example.opcodes[hop - 1] - OPCODE_IDS["ALU_ADD"]]
            counts[name][str(hop)] += 1
            hits[name][str(hop)] += int(predicted[row_index] == example.target_id)
    return {name: {hop: hits[name][hop] / counts[name][hop] for hop in counts[name]} for name in ALU_NAMES}


def b4_summary(metrics: dict[str, object], h1: float, free_running: dict[str, dict[str, float]], teacher_forced: dict[str, dict[str, float]], by_op_free: dict[str, dict[str, float]], by_op_teacher: dict[str, dict[str, float]]) -> dict[str, object]:
    output = summary(metrics)
    output["sequential_update_h1_table"] = h1
    output["sequential_free_running"] = free_running
    output["sequential_teacher_forced"] = teacher_forced
    output["final_operation_free_running"] = by_op_free
    output["final_operation_teacher_forced"] = by_op_teacher
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b4_raw_register.json")
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
        ablated_metrics = evaluate_all(ablated_model, datasets, "test")
        baseline_free = evaluate_sequential(baseline_model, datasets["sequential_update"]["test"], False)
        ablated_free = evaluate_sequential(ablated_model, datasets["sequential_update"]["test"], False)
        baseline_teacher = evaluate_sequential(baseline_model, datasets["sequential_update"]["test"], True)
        ablated_teacher = evaluate_sequential(ablated_model, datasets["sequential_update"]["test"], True)
        baseline_op_free = evaluate_by_final_operation(baseline_model, datasets["sequential_update"]["test"], False)
        ablated_op_free = evaluate_by_final_operation(ablated_model, datasets["sequential_update"]["test"], False)
        baseline_op_teacher = evaluate_by_final_operation(baseline_model, datasets["sequential_update"]["test"], True)
        ablated_op_teacher = evaluate_by_final_operation(ablated_model, datasets["sequential_update"]["test"], True)
        h1_examples = build_sequential_h1_table()
        baseline_h1 = evaluate_accuracy(baseline_model, "sequential_update", h1_examples, rounds=1)
        ablation_h1 = evaluate_accuracy(ablated_model, "sequential_update", h1_examples, rounds=1)
        baseline_metrics["sequential_update_h1_table"] = baseline_h1
        ablated_metrics["sequential_update_h1_table"] = ablation_h1
        baseline = b4_summary(baseline_metrics, baseline_h1, baseline_free, baseline_teacher, baseline_op_free, baseline_op_teacher)
        ablation = b4_summary(ablated_metrics, ablation_h1, ablated_free, ablated_teacher, ablated_op_free, ablated_op_teacher)
        runs[str(seed)] = {
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "baseline": baseline,
            "ablation": ablation,
            "delta_ablation_minus_baseline": {
                "pointer_final_h4": ablation["pointer_final_h4"] - baseline["pointer_final_h4"],
                "multi_hop_final_h3": ablation["multi_hop_final_h3"] - baseline["multi_hop_final_h3"],
                "associative_final": ablation["associative_final"] - baseline["associative_final"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
                "h1_table": ablation["sequential_update_h1_table"] - baseline["sequential_update_h1_table"],
            },
        }
    result = {
        "phase": "T1-U0-B4",
        "ablation": "remove ALU softmax normalization; write unnormalized logits @ value codebook as continuous R",
        "implementation_note": "U0-A operation heads emit 32 logits; unnormalized logits @ D-dimensional value basis is the checkpoint-compatible analogue of historical D-dimensional continuous heads.",
        "evaluation_note": "Teacher forcing injects prior intermediate targets only before non-EMIT rounds; padded EMIT rounds preserve the model's preceding output.",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "h1": "may remain high",
            "teacher_forcing": "may remain high",
            "free_running": "falls with horizon",
            "operation_signature": "ADD/SUB degrade more than MUL",
            "unrelated": "non-ALU tasks unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
