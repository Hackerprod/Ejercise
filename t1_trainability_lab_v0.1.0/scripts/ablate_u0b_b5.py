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
from t1_trainability.unified import (  # noqa: E402
    CandidateState,
    OPCODE_IDS,
    READ_OPCODE_IDS,
    ReadResult,
    SLOT_R,
    UnifiedT1U0,
)


class B5AdapterOpcodePermutationModel(UnifiedT1U0):
    """Feed cyclically permuted ALU opcode identity into shared ALU adapter."""

    HEAD_PERMUTATION = {
        "ALU_ADD": "ALU_SUB",
        "ALU_SUB": "ALU_MUL",
        "ALU_MUL": "ALU_ADD",
    }

    def permute_adapter_opcode(self, opcode: torch.Tensor) -> torch.Tensor:
        adapter_opcode = opcode.clone()
        for source_name, target_name in self.HEAD_PERMUTATION.items():
            adapter_opcode = torch.where(
                opcode == OPCODE_IDS[source_name],
                torch.full_like(opcode, OPCODE_IDS[target_name]),
                adapter_opcode,
            )
        return adapter_opcode

    def step(self, state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate, source_slot, destination_slot, presence_mask):  # type: ignore[no-untyped-def]
        """Run shared core under fake ALU opcode, then commit with real opcode."""
        batch = state.shape[0]
        if opcode.shape != (batch,):
            raise ValueError("opcode must have shape [B]")
        adapter_opcode = self.permute_adapter_opcode(opcode)
        read_required = torch.isin(opcode.to(dtype=torch.long), torch.tensor(tuple(READ_OPCODE_IDS), device=opcode.device)).any()
        if read_required:
            read_result = self.memory_reader(
                state,
                memory_keys,
                memory_values,
                memory_types,
                row_mask,
                opcode,
                immediate,
                source_slot,
            )
        else:
            if immediate.ndim == 2:
                payload = torch.zeros((batch, self.dimension), dtype=state.dtype, device=state.device)
            else:
                payload = torch.zeros_like(state[:, 0, :])
            memory_width = memory_types.shape[-1]
            read_result = ReadResult(
                payload=payload,
                attention=torch.zeros((batch, memory_width), dtype=state.dtype, device=state.device),
                margin=torch.zeros(batch, dtype=state.dtype, device=state.device),
                valid=torch.zeros(batch, dtype=torch.bool, device=state.device),
            )
        opcode_embedding = self.opcode_embedding(adapter_opcode.to(dtype=torch.long))
        immediate_embedding = self.memory_reader._batch_vector(
            immediate,
            dimension=self.dimension,
            embedding=self.immediate_embedding,
        )
        normalized = self.normalize_state(state, presence_mask)
        candidates = self.core(
            normalized,
            opcode_embedding,
            immediate_embedding,
            read_result.payload,
            self.slot_type_embeddings,
            presence_mask,
            opcode=adapter_opcode,
        )
        alu_logits = self.commit.select_alu_logits(candidates.values[:, SLOT_R, :], opcode)
        register_codebook = self.token_embedding(torch.arange(288, 320, device=state.device))
        next_state = self.commit(
            state,
            candidates,
            read_result,
            opcode,
            destination_slot,
            presence_mask,
            register_codebook=register_codebook,
            alu_logits=alu_logits,
        )
        return next_state, CandidateState(candidates.values, alu_logits), read_result


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    if ablated:
        model = B5AdapterOpcodePermutationModel(64)
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
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b5_adapter_opcode_permutation.json")
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
        "ablation": "cyclic ALU adapter opcode permutation: ADD->SUB, SUB->MUL, MUL->ADD",
        "implementation_note": "The fake cyclic opcode and its embedding enter SharedRecurrentCore/ALU adapters; TypedCommit retains real opcode and corresponding real operation head. Non-ALU paths remain unchanged.",
        "prior_attempt": "The TypedCommit-only permutation from commit 7b555df was invalidated by design after diagnosis; this artifact replaces it.",
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
