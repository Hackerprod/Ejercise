"""Diagnose B5 head permutation on one fixed teacher-forced ALU_ADD batch."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b5 import load_model  # noqa: E402
from train_u0a import (  # noqa: E402
    VALUE_CLASS_IDS,
    build_canonical_data,
    collate,
    immediate_vectors,
    materialize,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_R  # noqa: E402


def run_fixed_batch(model, rows, final_round: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[no-untyped-def]
    batch = collate(rows)
    data = materialize(model, batch)
    state = data["state"]
    captured_head_logits = None
    captured_candidate_register = None
    for round_index in range(final_round + 1):
        if round_index > 0:
            targets = data["intermediate_target_ids"][:, round_index - 1]
            active = targets >= 0
            state[:, SLOT_R, :] = torch.where(
                active.unsqueeze(-1),
                model.token_embedding(targets.clamp_min(0)),
                state[:, SLOT_R, :],
            )
        state, candidates, _ = model.step(
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
        if round_index == final_round:
            captured_head_logits = candidates.alu_logits
            captured_candidate_register = candidates.values[:, SLOT_R, :]
    assert captured_head_logits is not None
    assert captured_candidate_register is not None
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    output_logits = model.register_decoder(state[:, SLOT_R, :], model.token_embedding(class_ids))
    return captured_head_logits, output_logits, captured_candidate_register


@torch.no_grad()
def main() -> int:
    run_dir = ROOT / "campaign" / "u0a_iso_clean_seed101_12000"
    checkpoint = run_dir / "best.pt"
    datasets = build_canonical_data(run_dir)
    rows = [
        row
        for row in datasets["sequential_update"]["test"]
        if row.hop_count == 3 and row.opcodes[2] == OPCODE_IDS["ALU_ADD"]
    ][:256]
    if len(rows) < 2:
        raise RuntimeError(f"fixed ALU_ADD batch too small: {len(rows)}")

    baseline = load_model(checkpoint, ablated=False)
    ablated = load_model(checkpoint, ablated=True)
    baseline_head, baseline_output, baseline_register = run_fixed_batch(baseline, rows, final_round=2)
    ablated_head, ablated_output, ablated_register = run_fixed_batch(ablated, rows, final_round=2)
    _, baseline_after_padding, _ = run_fixed_batch(baseline, rows, final_round=5)
    _, ablated_after_padding, _ = run_fixed_batch(ablated, rows, final_round=5)
    class_ids = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)
    baseline_predictions = class_ids[baseline_output.argmax(dim=-1)]
    ablated_predictions = class_ids[ablated_output.argmax(dim=-1)]
    targets = torch.tensor([row.target_id for row in rows], dtype=torch.long)
    baseline_head_predictions = class_ids[baseline_head.argmax(dim=-1)]
    ablated_head_predictions = class_ids[ablated_head.argmax(dim=-1)]
    direct_add = baseline.commit.operation_heads["ALU_ADD"](baseline_register)
    direct_sub = baseline.commit.operation_heads["ALU_SUB"](baseline_register)
    direct_head_max_delta = float((direct_add - direct_sub).detach().abs().max())

    add_head = ablated.commit.operation_heads["ALU_ADD"]
    sub_head = ablated.commit.operation_heads["ALU_SUB"]
    parameter_pairs = list(zip(add_head.parameters(), sub_head.parameters()))
    parameter_max_delta = max(float((add - sub).detach().abs().max()) for add, sub in parameter_pairs)
    parameter_data_ptr_equal = any(add.data_ptr() == sub.data_ptr() for add, sub in parameter_pairs)

    result = {
        "seed": 101,
        "hop": 3,
        "opcode": "ALU_ADD",
        "batch_size": len(rows),
        "same_input_examples": True,
        "head_logits": {
            "max_abs_delta": float((baseline_head - ablated_head).detach().abs().max()),
            "mean_abs_delta": float((baseline_head - ablated_head).detach().abs().mean()),
            "tensor_equal": bool(torch.equal(baseline_head, ablated_head)),
            "argmax_equal_fraction": float((baseline_head_predictions == ablated_head_predictions).float().mean()),
            "baseline_target_accuracy": float((baseline_head_predictions == targets).float().mean()),
            "ablated_target_accuracy": float((ablated_head_predictions == targets).float().mean()),
        },
        "pre_head_candidate_register": {
            "max_abs_delta": float((baseline_register - ablated_register).detach().abs().max()),
            "mean_abs_delta": float((baseline_register - ablated_register).detach().abs().mean()),
            "tensor_equal": bool(torch.equal(baseline_register, ablated_register)),
        },
        "decoded_output_logits": {
            "max_abs_delta": float((baseline_output - ablated_output).detach().abs().max()),
            "mean_abs_delta": float((baseline_output - ablated_output).detach().abs().mean()),
            "tensor_equal": bool(torch.equal(baseline_output, ablated_output)),
            "argmax_equal_fraction": float((baseline_predictions == ablated_predictions).float().mean()),
            "baseline_target_accuracy": float((baseline_predictions == targets).float().mean()),
            "ablated_target_accuracy": float((ablated_predictions == targets).float().mean()),
        },
        "artifact_style_after_six_rounds": {
            "baseline_target_accuracy": float((class_ids[baseline_after_padding.argmax(dim=-1)] == targets).float().mean()),
            "ablated_target_accuracy": float((class_ids[ablated_after_padding.argmax(dim=-1)] == targets).float().mean()),
            "note": "Post-final padded EMIT rounds re-inject final intermediate target under current teacher-forced evaluator.",
        },
        "head_parameters": {
            "add_sub_max_abs_delta": parameter_max_delta,
            "any_parameter_data_ptr_equal": parameter_data_ptr_equal,
            "objects_distinct": add_head is not sub_head,
        },
        "same_input_direct_head_probe": {
            "add_sub_max_abs_delta": direct_head_max_delta,
            "add_sub_tensor_equal": bool(torch.equal(direct_add, direct_sub)),
        },
        "interpretation": "head logits and decoded logits differ; near-equal argmax is not tensor aliasing or identical head output",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
