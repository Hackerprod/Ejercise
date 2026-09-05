"""Run U0-B7 frozen-workspace ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import summary  # noqa: E402
from ablate_u0b_b6 import evaluate_workspace_cosine  # noqa: E402
from train_u0a import (  # noqa: E402
    build_canonical_data,
    build_sequential_h1_table,
    evaluate_accuracy,
    evaluate_all,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_W, TypedCommit, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)


class B7WorkspaceFrozenCommit(TypedCommit):
    """Runtime-only B7 variant; preserve W before every ACCUM_W commit."""

    def forward(self, state, candidates, read_result, opcode, destination_slot, presence_mask, register_codebook=None, alu_logits=None):  # type: ignore[no-untyped-def]
        next_state = super().forward(
            state,
            candidates,
            read_result,
            opcode,
            destination_slot,
            presence_mask,
            register_codebook=register_codebook,
            alu_logits=alu_logits,
        )
        active = (opcode == OPCODE_IDS["ACCUM_W"]) & (destination_slot == SLOT_W)
        workspace = torch.where(active.unsqueeze(-1), state[:, SLOT_W, :], next_state[:, SLOT_W, :])
        return torch.cat((next_state[:, :SLOT_W, :], workspace.unsqueeze(1), next_state[:, SLOT_W + 1 :, :]), dim=1)


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    if ablated:
        model.commit = B7WorkspaceFrozenCommit(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def b7_summary(metrics: dict[str, object], cosine: dict[str, dict[str, float]]) -> dict[str, object]:
    output = summary(metrics)
    output["workspace_cosine"] = cosine
    output["workspace_final_cosine"] = {hop: cosine[hop]["6"] for hop in cosine}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b7_workspace_frozen.json")
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
        baseline_metrics["sequential_update_h1_table"] = evaluate_accuracy(baseline_model, "sequential_update", h1_examples, rounds=1)
        ablated_metrics["sequential_update_h1_table"] = evaluate_accuracy(ablated_model, "sequential_update", h1_examples, rounds=1)
        workspace_test = datasets["workspace_accumulation"]["test"]
        baseline = b7_summary(baseline_metrics, evaluate_workspace_cosine(baseline_model, workspace_test))
        ablation = b7_summary(ablated_metrics, evaluate_workspace_cosine(ablated_model, workspace_test))
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
                "sequential_composition_final_round6": ablation["sequential_composition_final_round6"]["6"] - baseline["sequential_composition_final_round6"]["6"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
                "workspace_cosine_h2": ablation["workspace_final_cosine"]["2"] - baseline["workspace_final_cosine"]["2"],
                "workspace_cosine_h4": ablation["workspace_final_cosine"]["4"] - baseline["workspace_final_cosine"]["4"],
                "workspace_cosine_h6": ablation["workspace_final_cosine"]["6"] - baseline["workspace_final_cosine"]["6"],
            },
        }

    result = {
        "phase": "T1-U0-B7",
        "ablation": "workspace frozen: W_next=W instead of W+Y",
        "implementation_note": "Only TypedCommit ACCUM_W writes to W are restored to pre-round W; all other opcodes and paths remain unchanged.",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "workspace_cosine": "approximately zero because initial W is zero and frozen",
            "workspace_error": "large",
            "unrelated": "pointer, associative, ALU, and retrieval tasks unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
