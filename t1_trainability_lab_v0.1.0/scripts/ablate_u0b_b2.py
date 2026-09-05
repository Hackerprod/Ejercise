"""Run U0-B2 pointer-frozen ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import pointer_diagnostics, summary  # noqa: E402
from train_u0a import build_canonical_data, build_sequential_h1_table, evaluate_accuracy, evaluate_all  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, SLOT_P, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)


class B2PointerFrozenModel(UnifiedT1U0):
    """Runtime-only B2 variant; preserve P on every READ_P commit."""

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
        pointer = torch.where(active.unsqueeze(-1), state[:, SLOT_P, :], next_state[:, SLOT_P, :])
        next_state = torch.cat((pointer.unsqueeze(1), next_state[:, 1:, :]), dim=1)
        return next_state, candidates, read_result


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model: UnifiedT1U0 = B2PointerFrozenModel(64) if ablated else UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def b2_summary(metrics: dict[str, object]) -> dict[str, object]:
    output = summary(metrics)
    output["pointer_final_round4_by_hop"] = {hop: metrics["pointer_chasing"][hop]["4"] for hop in ("1", "2", "3", "4")}
    output["multi_hop_final_round4_by_hop"] = {hop: metrics["multi_hop"][hop]["4"] for hop in ("1", "2", "3", "4")}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b2_pointer_frozen.json")
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
        baseline = b2_summary(baseline_metrics)
        ablation = b2_summary(ablated_metrics)
        runs[str(seed)] = {
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "baseline": baseline,
            "ablation": ablation,
            "delta_ablation_minus_baseline": {
                "pointer_final_h4": ablation["pointer_final_h4"] - baseline["pointer_final_h4"],
                "multi_hop_final_h3": ablation["multi_hop_final_h3"] - baseline["multi_hop_final_h3"],
                "multi_hop_final_h4": ablation["multi_hop_final_h4"] - baseline["multi_hop_final_h4"],
                "associative_final": ablation["associative_final"] - baseline["associative_final"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
            },
            "pointer_diagnostics": {
                "baseline": pointer_diagnostics(baseline_model, datasets["pointer_chasing"]["test"]),
                "ablation": pointer_diagnostics(ablated_model, datasets["pointer_chasing"]["test"]),
            },
        }
    result = {
        "phase": "T1-U0-B2",
        "ablation": "P_{r+1}=P_r for READ_P",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "pointer_chasing": "H>0 near impossible or chance",
            "multi_hop": "H>0 near impossible or chance",
            "unrelated": "associative, ALU, workspace without material degradation",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
