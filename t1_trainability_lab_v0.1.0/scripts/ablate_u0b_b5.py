"""Run U0-B5 cyclically permuted ALU-head ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b4 import (  # noqa: E402
    ALU_NAMES,
    HOPS,
    ROUNDS,
    SEEDS,
    b4_summary,
    evaluate_by_final_operation,
    evaluate_sequential,
)
from ablate_u0b_b1 import summary  # noqa: E402
from train_u0a import (  # noqa: E402
    build_canonical_data,
    build_sequential_h1_table,
    evaluate_accuracy,
    evaluate_all,
)
from t1_trainability.unified import OPCODE_IDS, TypedCommit, UnifiedT1U0  # noqa: E402


class B5CyclicHeadCommit(TypedCommit):
    """Dispatch each ALU opcode to next operation's trained head."""

    HEAD_PERMUTATION = {
        "ALU_ADD": "ALU_SUB",
        "ALU_SUB": "ALU_MUL",
        "ALU_MUL": "ALU_ADD",
    }

    def select_alu_logits(self, register: torch.Tensor, opcode: torch.Tensor) -> torch.Tensor:
        if register.ndim != 2 or register.shape[-1] != self.dimension or opcode.shape != (register.shape[0],):
            raise ValueError("register/opcode shapes invalid")
        selected = torch.zeros(register.shape[0], 32, dtype=register.dtype, device=register.device)
        for target_name, source_name in self.HEAD_PERMUTATION.items():
            indices = (opcode == OPCODE_IDS[target_name]).nonzero(as_tuple=False).flatten()
            if indices.numel():
                selected = selected.index_copy(
                    0,
                    indices,
                    self.operation_heads[source_name](register.index_select(0, indices)),
                )
        return selected


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    if ablated:
        model.commit = B5CyclicHeadCommit(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def b5_summary(
    metrics: dict[str, object],
    h1: float,
    free_running: dict[str, dict[str, float]],
    teacher_forced: dict[str, dict[str, float]],
    by_op_free: dict[str, dict[str, float]],
    by_op_teacher: dict[str, dict[str, float]],
) -> dict[str, object]:
    output = summary(metrics)
    output["sequential_update_h1_table"] = h1
    output["sequential_free_running"] = free_running
    output["sequential_teacher_forced"] = teacher_forced
    output["final_operation_free_running"] = by_op_free
    output["final_operation_teacher_forced"] = by_op_teacher
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b5_head_permutation.json")
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
        h1_examples = build_sequential_h1_table()
        baseline_h1 = evaluate_accuracy(baseline_model, "sequential_update", h1_examples, rounds=1)
        ablated_h1 = evaluate_accuracy(ablated_model, "sequential_update", h1_examples, rounds=1)
        baseline_metrics["sequential_update_h1_table"] = baseline_h1
        ablated_metrics["sequential_update_h1_table"] = ablated_h1

        sequential_test = datasets["sequential_update"]["test"]
        baseline_free = evaluate_sequential(baseline_model, sequential_test, False)
        ablated_free = evaluate_sequential(ablated_model, sequential_test, False)
        baseline_teacher = evaluate_sequential(baseline_model, sequential_test, True)
        ablated_teacher = evaluate_sequential(ablated_model, sequential_test, True)
        baseline_op_free = evaluate_by_final_operation(baseline_model, sequential_test, False)
        ablated_op_free = evaluate_by_final_operation(ablated_model, sequential_test, False)
        baseline_op_teacher = evaluate_by_final_operation(baseline_model, sequential_test, True)
        ablated_op_teacher = evaluate_by_final_operation(ablated_model, sequential_test, True)

        baseline = b5_summary(
            baseline_metrics,
            baseline_h1,
            baseline_free,
            baseline_teacher,
            baseline_op_free,
            baseline_op_teacher,
        )
        ablation = b5_summary(
            ablated_metrics,
            ablated_h1,
            ablated_free,
            ablated_teacher,
            ablated_op_free,
            ablated_op_teacher,
        )
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
        "phase": "T1-U0-B5",
        "ablation": "cyclic ALU head permutation: ADD->SUB, SUB->MUL, MUL->ADD",
        "implementation_note": "Opcode semantics and all non-ALU paths remain unchanged; only TypedCommit ALU head dispatch is permuted at runtime.",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "h1": "collapse",
            "sequential_composition": "ALU collapse under teacher forcing and free running",
            "unrelated": "pointer retrieval, associative retrieval, and workspace unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
