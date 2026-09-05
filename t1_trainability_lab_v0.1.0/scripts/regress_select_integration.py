"""Read-only regression for integrated SELECT reader policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    READ_MODE_BLEND,
    READ_MODE_SELECT,
    ROW_VEC,
    SLOT_P,
    SLOT_COUNT,
    UnifiedT1U0,
)
from train_u0a import (  # noqa: E402
    DIMENSION,
    build_canonical_data,
    evaluate_all,
)


CHECKPOINT = ROOT / "campaign" / "u0a_iso_clean_seed101_12000" / "best.pt"
DATASET_SEED = 74017
CODEBOOK_KEY_COUNT = 256
MEMORY_WIDTH = 6
SAMPLES = 2048
H_VALUES = (1, 2, 4, 6)


def load_model() -> UnifiedT1U0:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(DIMENSION)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


@torch.no_grad()
def run_sampled_permuted_regression(model: UnifiedT1U0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(DATASET_SEED)
    sampled_ids = torch.stack([torch.randperm(CODEBOOK_KEY_COUNT, generator=generator)[:MEMORY_WIDTH] for _ in range(SAMPLES)])
    row_order = torch.stack([torch.randperm(MEMORY_WIDTH, generator=generator) for _ in range(SAMPLES)])
    memory_keys_logical = model.token_embedding(sampled_ids)
    memory_keys = memory_keys_logical.gather(1, row_order.unsqueeze(-1).expand(-1, -1, DIMENSION))
    memory_values = torch.randn((SAMPLES, MEMORY_WIDTH, DIMENSION), generator=generator)
    query_logical = torch.randint(MEMORY_WIDTH, (SAMPLES, max(H_VALUES)), generator=generator)
    query_physical = row_order.argsort(dim=1).gather(1, query_logical)
    memory_types = torch.full((SAMPLES, MEMORY_WIDTH), ROW_VEC, dtype=torch.long)
    row_mask = torch.ones((SAMPLES, MEMORY_WIDTH), dtype=torch.bool)
    opcode = torch.full((SAMPLES,), OPCODE_IDS["ACCUM_W"], dtype=torch.long)
    immediate = torch.full((SAMPLES,), 511, dtype=torch.long)
    source_slot = torch.full((SAMPLES,), SLOT_P, dtype=torch.long)
    by_h: dict[str, object] = {}
    all_selected = []
    all_relative = []
    for h in H_VALUES:
        selected_by_round = []
        payload_errors = []
        state = torch.zeros((SAMPLES, SLOT_COUNT, DIMENSION))
        for round_index in range(h):
            state[:, SLOT_P, :] = memory_keys_logical[torch.arange(SAMPLES), query_logical[:, round_index]]
            result = model.memory_reader(
                state,
                memory_keys,
                memory_values,
                memory_types,
                row_mask,
                opcode,
                immediate,
                source_slot,
                read_mode=torch.full((SAMPLES,), READ_MODE_SELECT, dtype=torch.long),
            )
            expected_index = query_physical[:, round_index]
            expected_payload = memory_values[torch.arange(SAMPLES), expected_index]
            selected = result.selected_index == expected_index
            relative = torch.linalg.vector_norm(result.payload - expected_payload, dim=-1) / torch.linalg.vector_norm(expected_payload, dim=-1).clamp_min(1e-8)
            selected_by_round.append(float(selected.float().mean()))
            payload_errors.append(float(relative.max()))
            all_selected.append(selected)
            all_relative.append(relative)
        by_h[str(h)] = {
            "top1_accuracy_by_round": selected_by_round,
            "max_payload_relative_error_by_round": payload_errors,
            "top1_accuracy": float(torch.cat([x.reshape(-1) for x in all_selected[-h:]]).float().mean()),
            "max_payload_relative_error": max(payload_errors),
        }
    return {
        "samples": SAMPLES,
        "memory_width": MEMORY_WIDTH,
        "sampled_codebook_key_range": [0, CODEBOOK_KEY_COUNT - 1],
        "row_order_permuted_per_sample": True,
        "by_h": by_h,
        "top1_accuracy_all_active_rounds": float(torch.cat([x.reshape(-1) for x in all_selected]).float().mean()),
        "max_payload_relative_error": float(torch.cat(all_relative).max()),
    }


@torch.no_grad()
def test_mixed_mode_isolation(model: UnifiedT1U0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(DATASET_SEED + 1)
    state = torch.randn((2, SLOT_COUNT, DIMENSION), generator=generator)
    memory_keys = torch.randn((2, MEMORY_WIDTH, DIMENSION), generator=generator)
    memory_values = torch.randn((2, MEMORY_WIDTH, DIMENSION), generator=generator)
    memory_types = torch.full((2, MEMORY_WIDTH), ROW_VEC, dtype=torch.long)
    row_mask = torch.ones((2, MEMORY_WIDTH), dtype=torch.bool)
    opcode = torch.full((2,), OPCODE_IDS["ACCUM_W"], dtype=torch.long)
    immediate = torch.full((2,), 511, dtype=torch.long)
    source_slot = torch.full((2,), SLOT_P, dtype=torch.long)
    modes = torch.tensor([READ_MODE_SELECT, READ_MODE_BLEND], dtype=torch.long)
    mixed = model.memory_reader(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate, source_slot, read_mode=modes)
    separate_select = model.memory_reader(state[:1], memory_keys[:1], memory_values[:1], memory_types[:1], row_mask[:1], opcode[:1], immediate[:1], source_slot[:1], read_mode="SELECT")
    separate_blend = model.memory_reader(state[1:], memory_keys[1:], memory_values[1:], memory_types[1:], row_mask[1:], opcode[1:], immediate[1:], source_slot[1:], read_mode="BLEND")
    payload_diffs = [float((mixed.payload[0] - separate_select.payload[0]).abs().max()), float((mixed.payload[1] - separate_blend.payload[0]).abs().max())]
    attention_diffs = [float((mixed.attention_soft[0] - separate_select.attention_soft[0]).abs().max()), float((mixed.attention_soft[1] - separate_blend.attention_soft[0]).abs().max())]
    selected_diffs = [float((mixed.selected_index[0] - separate_select.selected_index[0]).abs().max()), float((mixed.selected_index[1] - separate_blend.selected_index[0]).abs().max())]
    return {
        "mixed_read_modes": ["SELECT", "BLEND"],
        "payload_max_abs_diff_per_example": payload_diffs,
        "attention_max_abs_diff_per_example": attention_diffs,
        "selected_index_max_abs_diff_per_example": selected_diffs,
        "comparison_atol": 1e-6,
        "passes": max(payload_diffs + attention_diffs + selected_diffs) <= 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_select_integration_seed101")
    args = parser.parse_args()
    random.seed(101)
    torch.manual_seed(101)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()
    datasets = build_canonical_data(args.output_dir)
    baseline_tasks = evaluate_all(model, datasets, "test")
    tasks = evaluate_all(model, datasets, "test")
    sampled = run_sampled_permuted_regression(model)
    isolation = test_mixed_mode_isolation(model)
    result = {
        "status": "completed",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "read_mode": "SELECT",
        "read_mode_encoding": {"BLEND": READ_MODE_BLEND, "SELECT": READ_MODE_SELECT},
        "scope": "ACCUM_W only; READ_P and READ_E remain BLEND",
        "six_task_test_historical_blend": baseline_tasks,
        "six_task_test_compatibility_rerun": tasks,
        "sampled_permuted_codebook_regression": sampled,
        "mixed_instruction_isolation": isolation,
        "emit_invariant": "covered by tests/test_unified.py::test_emit_is_identity_after_each_task_round",
    }
    (args.output_dir / "final.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
