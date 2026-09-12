"""Train T2-I3 COMP-0 with frozen R3 Writer and CTRL-7."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import train_t2_i2_r2 as r2
from ctrl2_common import load_base_manifests
from evaluate_u0c_ctrl7_preflight import ACTION_NAMES
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl2_o import sha256
from t2_i3_common import MANIFEST_PATH, encode_writer, load_writer, build_calibration_manifest
from t2_i3_comp0 import Composition0, parameter_count
from train_t2_i3 import calibration_rows, verify_execution_targets, sample_atomic_indices


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"


def output_for_seed(seed: int) -> Path:
    return CAMPAIGN / f"t2_i3_comp0_seed{seed}"


def checkpoint_for_seed(seed: int) -> Path:
    return output_for_seed(seed) / "final.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = output_for_seed(args.seed) if args.output_root is None else args.output_root
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    observations, labels = r2.load_source()
    panels = r2.build_panels(labels)
    manifest = build_calibration_manifest(MANIFEST_PATH)
    writer = load_writer()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    comp = Composition0()
    with torch.no_grad():
        p_empty = torch.softmax(supervisor(observations["features"], torch.zeros((len(observations["features"]), 32))), dim=-1).detach()
    calibration = calibration_rows(load_base_manifests()["train"], manifest["calibration"])
    verify_execution_targets(load_base_manifests()["train"], calibration)
    optimizer = torch.optim.AdamW(comp.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed + 1)
    final_main = final_card = final_joint = torch.tensor(0.0)
    for _step in range(1, 5001):
        indices, forms = sample_atomic_indices(labels, generator)
        texts = [r2.instruction_for_label(tuple(labels["constraints"][index].tolist()), int(labels["lower"][index]), int(labels["forbidden"][index]), position, form) for position, (index, form) in enumerate(zip(indices.tolist(), forms))]
        raw = torch.cat([encode_writer(writer, text) for text in texts]).reshape(len(texts), 2, 32)
        condition = comp(raw)
        final_main = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices])
        final_card = r2.cardinality_loss(supervisor, raw.detach(), indices, labels, observations, panels, p_empty)
        joint_indices = torch.randint(len(calibration), (32,), generator=generator)
        joint_pick = [calibration[index] for index in joint_indices.tolist()]
        joint_texts = [f"AT_LEAST VALUE_{item['pair'][0]} AND AVOID VALUE_{item['pair'][1]}" for item in joint_pick]
        joint_raw = torch.cat([encode_writer(writer, text) for text in joint_texts]).reshape(len(joint_pick), 2, 32)
        joint_features = torch.tensor([item["features"] for item in joint_pick], dtype=torch.float32)
        joint_targets = torch.tensor([item["target"] for item in joint_pick], dtype=torch.long)
        final_joint = F.cross_entropy(supervisor(joint_features, comp(joint_raw)), joint_targets)
        loss = final_main + final_joint
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    checkpoint = checkpoint_for_seed(args.seed) if args.output_root is None else output / "final.pt"
    torch.save({"composition": comp.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(comp), "writer_checkpoint": sha256(WRITER_CHECKPOINT), "calibration_manifest_sha256": manifest["sha256"], "writer_frozen": True, "ctrl_frozen": True, "z_zero_initialized": True, "variant": "COMP-0"}, checkpoint)
    report = {"module": "Composition0", "task": "T2-I3-COMP-0", "input": "[batch,2,32]", "phi": "Linear(32,16)+SiLU per slot", "symmetric_pooling": "phi(E0)+phi(E1)", "rho": "SiLU", "w_out": "Linear(16,32), zero initialized", "trainable_parameters": parameter_count(comp), "writer_checkpoint_sha256": sha256(WRITER_CHECKPOINT), "calibration_manifest_sha256": manifest["sha256"], "lambda_legs": None, "frozen_inputs": True}
    (output / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I3-COMP-0", "seed": args.seed, "training": {"updates": 5000, "final_main_loss": float(final_main.detach()), "final_cardinality_loss": float(final_card.detach()), "final_joint_loss": float(final_joint.detach()), "trainable_parameters": parameter_count(comp), "optimized_loss": "L_main + L_J", "cardinality_loss_grad": False}, "checkpoint": sha256(checkpoint), "calibration_manifest_sha256": manifest["sha256"], "architecture_report": report}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
