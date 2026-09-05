"""Audit workspace reader/normalization/opcode paths in isolated and joint models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_u0a import (  # noqa: E402
    ExampleDataset,
    TASKS,
    build_canonical_data,
    collate,
    immediate_vectors,
    materialize,
    run_rounds,
    save_json,
    train_one_step,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_W, UnifiedT1U0  # noqa: E402


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    current = model.state_dict()
    compatible = {name: value for name, value in payload["model"].items() if name in current and value.shape == current[name].shape}
    model.load_state_dict(compatible, strict=False)
    return model.eval()


@torch.no_grad()
def audit_workspace(model: UnifiedT1U0, examples: list[object]) -> dict[str, object]:
    samples: dict[int, list[dict[str, float]]] = defaultdict(list)
    final_errors: dict[int, list[float]] = defaultdict(list)
    reader_rows = 0
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    for batch in loader:
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
            active = batch["hops"] > round_index
            if not active.any():
                continue
            target = data["raw_values"][:, round_index]
            payload_error = torch.linalg.vector_norm(read.payload - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8)
            target_attention = read.attention[:, round_index]
            entropy = -(read.attention.clamp_min(1e-12) * read.attention.clamp_min(1e-12).log()).sum(dim=-1)
            for index in active.nonzero(as_tuple=False).flatten().tolist():
                samples[int(batch["hops"][index])].append(
                    {
                        "payload_relative_error": float(payload_error[index]),
                        "attention_top1": float(read.attention[index].argmax() == round_index),
                        "attention_target": float(target_attention[index]),
                        "reader_margin": float(read.margin[index]),
                        "reader_entropy": float(entropy[index]),
                    }
                )
                reader_rows += 1
        final_error = torch.linalg.vector_norm(state[:, SLOT_W] - data["target_vectors"], dim=-1) / torch.linalg.vector_norm(data["target_vectors"], dim=-1).clamp_min(1e-8)
        for index, hop in enumerate(batch["hops"].tolist()):
            final_errors[int(hop)].append(float(final_error[index]))
    output: dict[str, object] = {"reader_rows": reader_rows, "workspace_correction_weight_norm": sum(float(parameter.norm()) for parameter in model.core.workspace_correction.parameters()), "by_h": {}}
    for hop, values in sorted(samples.items()):
        output["by_h"][str(hop)] = {
            "reader_payload_relative_error": sum(row["payload_relative_error"] for row in values) / len(values),
            "reader_attention_top1": sum(row["attention_top1"] for row in values) / len(values),
            "reader_attention_target": sum(row["attention_target"] for row in values) / len(values),
            "reader_margin": sum(row["reader_margin"] for row in values) / len(values),
            "reader_entropy": sum(row["reader_entropy"] for row in values) / len(values),
            "final_workspace_relative_error": sum(final_errors[hop]) / len(final_errors[hop]),
        }
    return output


def audit_opcode_rows(model: UnifiedT1U0, datasets: dict[str, dict[str, list[object]]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for task in TASKS:
        model.train()
        model.zero_grad(set_to_none=True)
        loader = DataLoader(ExampleDataset(datasets[task]["train"]), batch_size=128, shuffle=False, collate_fn=collate)
        batch = next(iter(loader))
        loss = train_one_step(model, task, batch)
        loss.backward()
        rows: dict[str, object] = {}
        for label, embedding in (("core_opcode", model.opcode_embedding), ("reader_opcode", model.memory_reader.opcode_embedding)):
            gradient = embedding.weight.grad
            rows[label] = {
                "grad_is_none": gradient is None,
                "accum_w_row_norm": None if gradient is None else float(gradient[OPCODE_IDS["ACCUM_W"]].norm()),
                "accum_w_parameter_requires_grad": embedding.weight.requires_grad,
            }
        result[task] = rows
    model.eval()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--isolated-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joint = load_model(args.joint_checkpoint)
    isolated = load_model(args.isolated_checkpoint)
    datasets = build_canonical_data(args.joint_checkpoint.parent)
    output = {
        "joint_checkpoint": str(args.joint_checkpoint),
        "isolated_checkpoint": str(args.isolated_checkpoint),
        "workspace_correction_frozen_joint": all(not parameter.requires_grad for parameter in joint.core.workspace_correction.parameters()),
        "reader": {"joint": audit_workspace(joint, datasets["workspace_accumulation"]["test"]), "isolated": audit_workspace(isolated, datasets["workspace_accumulation"]["test"])},
        "opcode_accum_grad_rows_joint": audit_opcode_rows(joint, datasets),
    }
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
