"""Diagnose B6 replacement payload against Gaussian cosine expectation."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b6 import load_model  # noqa: E402
from train_u0a import INDEX_BASE, ROW_VEC, build_canonical_data, collate, immediate_vectors, materialize  # noqa: E402
from t1_trainability.unified import SLOT_W  # noqa: E402


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hop", type=int, nargs="+", default=[4, 6])
    args = parser.parse_args()
    run_dir = ROOT / "campaign" / f"u0a_iso_clean_seed{args.seed}_12000"
    datasets = build_canonical_data(run_dir)
    examples = [row for row in datasets["workspace_accumulation"]["test"] if row.hop_count in args.hop]
    probe = collate(examples[:1])
    model = load_model(run_dir / "best.pt", ablated=True)
    rows: dict[str, list[float]] = {str(hop): [] for hop in args.hop}
    payload_last_cosines: dict[str, list[float]] = {str(hop): [] for hop in args.hop}
    effective_components: dict[str, list[float]] = {str(hop): [] for hop in args.hop}
    for offset in range(0, len(examples), 256):
        batch_examples = examples[offset : offset + 256]
        batch = collate(batch_examples)
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(6):
            state, _, read = model.step(
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
            active = batch["hops"] == round_index + 1
            if active.any():
                for index in active.nonzero(as_tuple=False).flatten().tolist():
                    hop = str(int(batch["hops"][index]))
                    rows[hop].append(float(F.cosine_similarity(read.payload[index], batch["target_vectors"][index], dim=0)))
                    payload_last_cosines[hop].append(float(F.cosine_similarity(data["raw_values"][index, round_index], batch["target_vectors"][index], dim=0)))
                    weights = read.attention[index][batch["row_mask"][index]]
                    effective_components[hop].append(float(1.0 / weights.square().sum().clamp_min(1e-12)))
    result = {
        "seed": args.seed,
        "requested_hops": args.hop,
        "probe_memory_layout": {
            "memory_width": int(probe["row_mask"].shape[1]),
            "row_count": int(probe["row_mask"][0].sum()),
            "key_ids": probe["key_ids"][0][probe["row_mask"][0]].tolist(),
            "expected_key_ids": [INDEX_BASE + index for index in range(args.hop[0])],
            "all_row_types_vec": bool(torch.all(probe["memory_types"][0][probe["row_mask"][0]] == ROW_VEC)),
        },
        "payload_cosine_to_full_sum": {hop: sum(values) / len(values) for hop, values in rows.items()},
        "raw_last_vector_cosine_to_full_sum": {hop: sum(values) / len(values) for hop, values in payload_last_cosines.items()},
        "attention_effective_component_count": {hop: sum(values) / len(values) for hop, values in effective_components.items()},
        "theory": {"4": 0.5, "6": 1 / (6**0.5)},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
