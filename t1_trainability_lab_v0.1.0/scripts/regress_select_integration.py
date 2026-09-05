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
    ROW_VEC,
    SLOT_P,
    SLOT_COUNT,
    UnifiedT1U0,
)
from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    DIMENSION,
    ExampleDataset,
    build_canonical_data,
    collate,
    evaluate_all,
    immediate_vectors,
    materialize,
)
from torch.utils.data import DataLoader  # noqa: E402


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
                read_mode="SELECT",
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
def workspace_select_diagnostics(model: UnifiedT1U0, examples: list[object]) -> dict[str, object]:
    loader = DataLoader(ExampleDataset(examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    by_h: dict[str, dict[str, list[float]]] = {str(h): {"selection": [], "payload_error": []} for h in (2, 4, 6)}
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(6):
            opcode = data["opcodes"][:, round_index]
            state, _, result = model.step(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                opcode,
                immediate_vectors(model, data["immediates"][:, round_index]),
                data["source_slots"][:, round_index],
                data["destination_slots"][:, round_index],
                data["presence"],
                read_mode="SELECT",
            )
            active = data["hops"] > round_index
            for h in (2, 4, 6):
                selected = active & (data["hops"] == h)
                if not selected.any():
                    continue
                hit = (result.selected_index[selected] == round_index).float()
                expected = data["raw_values"][selected, round_index]
                relative = torch.linalg.vector_norm(result.payload[selected] - expected, dim=-1) / torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-8)
                by_h[str(h)]["selection"].extend(hit.tolist())
                by_h[str(h)]["payload_error"].extend(relative.tolist())
    return {
        str(h): {
            "top1_accuracy": sum(values["selection"]) / len(values["selection"]),
            "max_payload_relative_error": max(values["payload_error"]),
        }
        for h, values in by_h.items()
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
    baseline_tasks = evaluate_all(model, datasets, "test", read_mode="BLEND")
    tasks = evaluate_all(model, datasets, "test", read_mode="SELECT")
    sampled = run_sampled_permuted_regression(model)
    workspace_reader = workspace_select_diagnostics(model, datasets["workspace_accumulation"]["test"])
    result = {
        "status": "completed",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "read_mode": "SELECT",
        "scope": "ACCUM_W only; READ_P and READ_E remain BLEND",
        "six_task_test_blend_baseline": baseline_tasks,
        "six_task_test_select": tasks,
        "sampled_permuted_codebook_regression": sampled,
        "workspace_select_reader_diagnostics": workspace_reader,
        "emit_invariant": "covered by tests/test_unified.py::test_emit_is_identity_after_each_task_round",
    }
    (args.output_dir / "final.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
