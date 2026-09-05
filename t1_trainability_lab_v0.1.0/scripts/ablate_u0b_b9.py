"""Run U0-B9 zero-reader-payload control ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import summary  # noqa: E402
from train_u0a import (  # noqa: E402
    build_canonical_data,
    build_sequential_h1_table,
    evaluate_accuracy,
    evaluate_all,
)
from t1_trainability.unified import READ_OPCODE_IDS, ReadResult, SharedMemoryReader, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)


class B9ZeroPayloadReader(SharedMemoryReader):
    """Runtime-only B9 reader; preserve attention but zero every read payload."""

    def forward(self, state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate, source_slot):  # type: ignore[no-untyped-def]
        result = super().forward(
            state,
            memory_keys,
            memory_values,
            memory_types,
            row_mask,
            opcode,
            immediate,
            source_slot,
        )
        is_read = torch.isin(opcode.to(dtype=torch.long), torch.tensor(tuple(READ_OPCODE_IDS), device=opcode.device))
        payload = torch.where(is_read.unsqueeze(-1), torch.zeros_like(result.payload), result.payload)
        return ReadResult(payload=payload, attention=result.attention, margin=result.margin, valid=result.valid)


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    if ablated:
        model.memory_reader = B9ZeroPayloadReader(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def b9_summary(metrics: dict[str, object]) -> dict[str, object]:
    return summary(metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b9_zero_reader_payload.json")
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
        baseline = b9_summary(baseline_metrics)
        ablation = b9_summary(ablated_metrics)
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
                "sequential_h1_table": ablation["sequential_h1_table"] - baseline["sequential_h1_table"],
                "sequential_composition_final_round6": ablation["sequential_composition_final_round6"]["6"] - baseline["sequential_composition_final_round6"]["6"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
            },
        }

    result = {
        "phase": "T1-U0-B9",
        "ablation": "zero reader payload for READ_P, READ_E, and ACCUM_W",
        "implementation_note": "Reader attention and validity remain observable, but payload Y is replaced with zero for all memory opcodes; non-reader opcodes are unchanged.",
        "training_performed": False,
        "seeds": list(args.seeds),
        "runs": runs,
        "expected_signature": {
            "retrieval": "pointer, multi-hop, associative, and variable binding collapse",
            "workspace": "collapse",
            "sequential_update": "unchanged because operands come from instruction tape",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
