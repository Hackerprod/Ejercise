"""Train VIEW-0 gates while freezing Writer, COMP-0, and CTRL-7."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import train_t2_i2_r2 as r2
from ctrl2_common import load_base_manifests
from t2_i3_common import MANIFEST_PATH, encode_writer, load_writer
from t2_i3_comp0 import Composition0
from t2_i3_view0 import View0, parameter_count
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i3 import calibration_rows, sample_atomic_indices, verify_execution_targets
from train_u0c_ctrl2_o import sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
COMP_CHECKPOINT = CAMPAIGN / "t2_i3_comp0_seed6401" / "final.pt"
WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    output = CAMPAIGN / f"t2_i3_view0_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    observations, labels = r2.load_source()
    panels = r2.build_panels(labels)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    writer = load_writer()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    view = View0()
    optimizer = torch.optim.AdamW(view.parameters(), lr=1e-3, weight_decay=0.0)
    with torch.no_grad():
        p_empty = torch.softmax(supervisor(observations["features"], torch.zeros((len(observations["features"]), 32))), dim=-1).detach()
    calibration = calibration_rows(load_base_manifests()["train"], manifest["calibration"])
    verify_execution_targets(load_base_manifests()["train"], calibration)
    generator = torch.Generator().manual_seed(args.seed + 1)
    final = {"main": torch.tensor(0.0), "card": torch.tensor(0.0), "joint": torch.tensor(0.0), "views": torch.tensor(0.0)}
    for _step in range(1, 5001):
        indices, forms = sample_atomic_indices(labels, generator)
        texts = [r2.instruction_for_label(tuple(labels["constraints"][index].tolist()), int(labels["lower"][index]), int(labels["forbidden"][index]), position, form) for position, (index, form) in enumerate(zip(indices.tolist(), forms))]
        raw = torch.cat([encode_writer(writer, text) for text in texts])
        vf, va = view(raw)
        floor_mask = labels["constraints"][indices, 0].bool()
        condition = torch.where(floor_mask.unsqueeze(-1), vf, va)
        final["main"] = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices])
        role_slots = torch.stack((condition, torch.zeros_like(condition)), dim=1)
        final["card"] = r2.cardinality_loss(supervisor, role_slots, indices, labels, observations, panels, p_empty)
        joint_indices = torch.randint(len(calibration), (32,), generator=generator)
        joint_pick = [calibration[index] for index in joint_indices.tolist()]
        joint_texts = [f"AT_LEAST VALUE_{item['pair'][0]} AND AVOID VALUE_{item['pair'][1]}" for item in joint_pick]
        joint_raw = torch.cat([encode_writer(writer, text) for text in joint_texts])
        joint_vf, joint_va = view(joint_raw)
        joint_features = torch.tensor([item["features"] for item in joint_pick], dtype=torch.float32)
        joint_target = torch.tensor([item["target"] for item in joint_pick], dtype=torch.long)
        floor_target = torch.tensor([item["target_floor"] for item in joint_pick], dtype=torch.long)
        avoid_target = torch.tensor([item["target_avoid"] for item in joint_pick], dtype=torch.long)
        final["views"] = (F.cross_entropy(supervisor(joint_features, joint_vf), floor_target) + F.cross_entropy(supervisor(joint_features, joint_va), avoid_target)) / 2
        with torch.no_grad():
            joint_c = joint_raw.sum(dim=1) + 0.25 * (comp(joint_raw) - joint_raw.sum(dim=1))
        final["joint"] = F.cross_entropy(supervisor(joint_features, joint_c), joint_target)
        loss = final["main"] + final["card"] + final["views"] + final["joint"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    checkpoint = output / "final.pt"
    torch.save({"view": view.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(view), "writer_checkpoint_sha256": sha256(WRITER_CHECKPOINT), "comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "alpha": 0.25, "writer_frozen": True, "comp_frozen": True, "ctrl_frozen": True, "forbidden_feature": False}, checkpoint)
    report = {"module": "View0", "task": "T2-I3-VIEW-0", "input": "E_i(32)+B(32)+role_onehot(2)+slot_onehot(2)", "hidden": 15, "activation": "SiLU", "gate": "sigmoid independent per role/slot", "trainable_parameters": parameter_count(view), "alpha": 0.25, "forbidden_feature": False, "residual_target_special_case": False}
    (output / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I3-VIEW-0", "seed": args.seed, "training": {"updates": 5000, "final_main_loss": float(final["main"].detach()), "final_cardinality_loss": float(final["card"].detach()), "final_view_loss": float(final["views"].detach()), "final_joint_loss": float(final["joint"].detach()), "optimized_loss": "L_main + L_card + L_views + L_J", "trainable_parameters": parameter_count(view)}, "checkpoint": sha256(checkpoint), "architecture_report": report}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
