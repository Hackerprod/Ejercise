"""Train T2-I3 THINK-0 with frozen R3 writer and CTRL-7 components."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import train_t2_i2_r2 as r2
from ctrl2_common import load_base_manifests, load_ctrl1, load_executor, pair_value
from evaluate_u0c_ctrl7_preflight import supervisor_features_ctrl7
from evaluate_u0c_ctrl7_trained import DIMENSION, KEY_BASE, materialize_graph_batch
from evaluate_u0c_ctrl7_preflight import run_real
from evaluate_u0c_ctrl7_preflight import ACTION_NAMES, oracle_action
from t1_trainability.unified import SLOT_COUNT, SLOT_P, SLOT_R
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, READ_E, READ_P
from t1_trainability.unified import ROW_REL
from evaluate_u0c_ctrl7_preflight import oracle_action, target_value
from t2_i3_common import CALIBRATION_SALT, MANIFEST_PATH, build_calibration_manifest, encode_writer, load_writer
from t2_i3_think0 import architecture_report, parameter_count
from t2_i3_think2 import Think2
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl2_o import sha256

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"


def output_for_seed(seed: int) -> Path: return CAMPAIGN / f"t2_i3_think0_seed{seed}"
def checkpoint_for_seed(seed: int) -> Path: return output_for_seed(seed) / "final.pt"


def calibration_rows(manifest: dict, calibration: list[dict]) -> list[dict]:
    """Generate train-only calibration snapshots with the frozen oracle runtime."""
    from ctrl2_common import load_ctrl1, load_executor
    from train_u0c_ctrl2_o import OrdinalSharedScorer
    model = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = sorted(manifest["episodes"], key=lambda item: item["episode"]); records = []
    for pair in calibration:
        lower, forbidden = pair["lower"], pair["forbidden"]
        episode = episodes[int(pair["digest"][16:32], 16) % len(episodes)]
        outcome = run_real(model, ctrl1, scorer, manifest, episode, lower, forbidden, (1, 1))
        snapshots = [event for event in outcome["events"] if event["action_name"] != "EMIT"]
        snapshot = snapshots[int(pair["digest"][32:48], 16) % len(snapshots)]
        physical = snapshot["oracle_state"]
        target_joint = oracle_action(physical["pointer"], episode["goal_key"], read_e_done=physical["read_e_done"], copied=physical["copied"], value=physical["value"], lower=lower, forbidden=forbidden, constraints=(1, 1))
        if target_joint != snapshot["action"]:
            raise RuntimeError(f"runtime joint target mismatch for pair {(lower, forbidden)}")
        target_floor = oracle_action(physical["pointer"], episode["goal_key"], read_e_done=physical["read_e_done"], copied=physical["copied"], value=physical["value"], lower=lower, forbidden=0, constraints=(1, 0))
        target_avoid = oracle_action(physical["pointer"], episode["goal_key"], read_e_done=physical["read_e_done"], copied=physical["copied"], value=physical["value"], lower=0, forbidden=forbidden, constraints=(0, 1))
        records.append({"pair": [lower, forbidden], "episode": episode["episode"], "graph": episode["graph"], "x0": outcome["x0"], "decision": snapshot["decision"], "features": snapshot["features"], "target": target_joint, "target_name": ACTION_NAMES[target_joint], "target_floor": target_floor, "target_floor_name": ACTION_NAMES[target_floor], "target_avoid": target_avoid, "target_avoid_name": ACTION_NAMES[target_avoid], "oracle_state": physical, "trajectory_length": len(outcome["events"])})
    return records


def verify_execution_targets(manifest: dict, records: list[dict]) -> None:
    """Rerun fresh executor simulation and compare selected snapshots exactly."""
    from ctrl2_common import load_ctrl1, load_executor
    from train_u0c_ctrl2_o import OrdinalSharedScorer
    model = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {item["episode"]: item for item in manifest["episodes"]}
    for record in records:
        lower, forbidden = record["pair"]; episode = episodes[record["episode"]]; digest = hashlib.sha256(f"{CALIBRATION_SALT}|AT_LEAST|{lower}|AVOID|{forbidden}|NORMAL".encode("utf-8")).hexdigest(); outcome = run_real(model, ctrl1, scorer, manifest, episode, lower, forbidden, (1, 1)); snapshots = [event for event in outcome["events"] if event["action_name"] != "EMIT"]; snapshot = snapshots[int(digest[32:48], 16) % len(snapshots)]
        if snapshot["action"] != record["target"] or snapshot["features"] != record["features"]: raise RuntimeError(f"runtime calibration cross-check mismatch for pair {(lower, forbidden)}")


def leg_loss(supervisor: LatentConditionedSupervisor, features: torch.Tensor, slots: torch.Tensor, target_floor: torch.Tensor, target_avoid: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant one-step FLOOR/AVOID calibration loss."""
    logits0 = supervisor(features, slots[:, 0])
    logits1 = supervisor(features, slots[:, 1])
    floor0 = F.cross_entropy(logits0, target_floor, reduction="none")
    floor1 = F.cross_entropy(logits1, target_floor, reduction="none")
    avoid0 = F.cross_entropy(logits0, target_avoid, reduction="none")
    avoid1 = F.cross_entropy(logits1, target_avoid, reduction="none")
    return torch.minimum(floor0 + avoid1, floor1 + avoid0).mean()


def sample_atomic_indices(labels: dict[str, torch.Tensor], generator: torch.Generator) -> tuple[torch.Tensor, list[int]]:
    counts = { (0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16} }; selected = []; forms = []
    for goal, action_counts in counts.items():
        for action, count in action_counts.items():
            bucket = torch.where((labels["constraints"][:, 0] == goal[0]) & (labels["constraints"][:, 1] == goal[1]) & (labels["action"] == action))[0]; half = count // 2; selected.extend((bucket[torch.randint(len(bucket), (half,), generator=generator)], bucket[torch.randint(len(bucket), (count - half,), generator=generator)])); two = count - half; forms.extend((1,) * half + (2,) * (two // 2) + (3,) * (two - two // 2))
    return torch.cat(selected), forms


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6401); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); torch.manual_seed(args.seed); random.seed(args.seed); manifest = build_calibration_manifest(MANIFEST_PATH); observations, labels = r2.load_source(); panels = r2.build_panels(labels); writer = load_writer(); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); supervisor.eval(); think = Think2(); optimizer = torch.optim.AdamW(think.parameters(), lr=1e-3, weight_decay=0.0); with_no_grad = torch.no_grad()
    with with_no_grad: p_empty = torch.softmax(supervisor(observations["features"], torch.zeros((len(observations["features"]), 32))), dim=-1).detach()
    calibration = calibration_rows(load_base_manifests()["train"], manifest["calibration"]); verify_execution_targets(load_base_manifests()["train"], calibration); generator = torch.Generator().manual_seed(args.seed + 1); final_main = final_card = final_joint = final_legs = torch.tensor(0.0)
    for step in range(1, 5001):
        indices, forms = sample_atomic_indices(labels, generator); texts = [r2.instruction_for_label(tuple(labels["constraints"][index].tolist()), int(labels["lower"][index]), int(labels["forbidden"][index]), position, form) for position, (index, form) in enumerate(zip(indices.tolist(), forms))]; raw = torch.cat([encode_writer(writer, text) for text in texts]); slots = think(raw); condition = slots.sum(dim=1); final_main = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices]); final_card = r2.cardinality_loss(supervisor, slots, indices, labels, observations, panels, p_empty); joint_indices = torch.randint(len(calibration), (32,), generator=generator); joint_pick = [calibration[index] for index in joint_indices.tolist()]; joint_texts = [f"AT_LEAST VALUE_{item['pair'][0]} AND AVOID VALUE_{item['pair'][1]}" for item in joint_pick]; joint_raw = torch.cat([encode_writer(writer, text) for text in joint_texts]); joint_slots = think(joint_raw); joint_features = torch.tensor([item["features"] for item in joint_pick], dtype=torch.float32); joint_targets = torch.tensor([item["target"] for item in joint_pick], dtype=torch.long); floor_targets = torch.tensor([item["target_floor"] for item in joint_pick], dtype=torch.long); avoid_targets = torch.tensor([item["target_avoid"] for item in joint_pick], dtype=torch.long); final_joint = F.cross_entropy(supervisor(joint_features, joint_slots.sum(dim=1)), joint_targets); final_legs = leg_loss(supervisor, joint_features, joint_slots, floor_targets, avoid_targets); loss = final_main + final_card + final_joint + final_legs; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    checkpoint = checkpoint_for_seed(args.seed) if args.output_root is None else output / "final.pt"; torch.save({"think": think.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(think), "writer_checkpoint": sha256(WRITER_CHECKPOINT), "calibration_manifest_sha256": manifest["sha256"], "writer_frozen": True, "k": 2, "lambda_legs": 1.0, "think_variant": "THINK-2"}, checkpoint); report = architecture_report(think) | {"module": "Think2", "dynamics": "anchored proposal interpolation with other-slot context", "context": "state.flip(1)", "writer_checkpoint": str(WRITER_CHECKPOINT), "writer_checkpoint_sha256": sha256(WRITER_CHECKPOINT), "calibration_manifest_sha256": manifest["sha256"], "lambda_legs": 1.0, "seeds": [6401, 6402, 6403]}; (output / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); result = {"status": "trained", "task": "T2-I3-THINK-2", "seed": args.seed, "training": {"updates": 5000, "final_main_loss": float(final_main.detach()), "final_cardinality_loss": float(final_card.detach()), "final_joint_loss": float(final_joint.detach()), "final_legs_loss": float(final_legs.detach()), "lambda_legs": 1.0, "trainable_parameters": parameter_count(think)}, "checkpoint": sha256(checkpoint), "calibration_manifest_sha256": manifest["sha256"], "architecture_report": report}; (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
