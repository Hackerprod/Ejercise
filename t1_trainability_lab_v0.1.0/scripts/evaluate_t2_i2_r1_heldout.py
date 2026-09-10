"""T2-I2-R1 held-out JOINT and RECONSTRUCTED batteries."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_heldout import run_canonical_learned
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r1 import checkpoint_for_seed, output_for_seed
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, tensorize

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; REALIZATIONS = (("AT_LEAST_AVOID", "AT_LEAST", "AVOID"), ("AT_LEAST_EXCLUDE", "AT_LEAST", "EXCLUDE"), ("MINIMUM_AVOID", "MINIMUM", "AVOID"), ("MINIMUM_EXCLUDE", "MINIMUM", "EXCLUDE")); ORDERS = ("NORMAL", "INVERTED")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text]);
    with torch.no_grad(): return writer(token_ids, lengths)[0]
def instruction(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> str:
    clauses = (f"{floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden}"); return " AND ".join(clauses if order == "NORMAL" else clauses[::-1])
def post_copy(result: dict[str, Any]) -> list[str]:
    actions = [event["action_name"] for event in result["events"]]; return actions[next(index for index, action in enumerate(actions) if action == "COPY_E_R") + 1 :]


def battery(core: LatentConditionedSupervisor, writer: CompetitiveSemanticWriter, executor: Any, ctrl1: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], episodes: dict[int, Any], floor: str, avoid: str, order: str) -> dict[str, Any]:
    joint = {"canonical": {"samples": 31744, "success": 0}, "real": [], "interaction_gate": {"samples": 32, "success": 0}, "crossing_literal": {"samples": 1, "success": 0, "post_copy_actions": []}}; reconstructed = json.loads(json.dumps(joint))
    for x0 in range(32):
        for lower in range(32):
            for forbidden in range(31):
                text = instruction(floor, avoid, order, lower, forbidden); joint_condition = encode(writer, text).sum(0); atomic_condition = encode(writer, f"{floor} VALUE_{lower}").sum(0) + encode(writer, f"{avoid} VALUE_{forbidden}").sum(0)
                joint["canonical"]["success"] += int(run_canonical_learned(executor, ctrl1, scorer, Adapter(core, joint_condition), manifest, episodes[x0], x0, lower, forbidden, (1, 1))["success"]); reconstructed["canonical"]["success"] += int(run_canonical_learned(executor, ctrl1, scorer, Adapter(core, atomic_condition), manifest, episodes[x0], x0, lower, forbidden, (1, 1))["success"])
    for category, x0, lower, forbidden in real_cases():
        text = instruction(floor, avoid, order, lower, forbidden); joint_result = run_learned(executor, ctrl1, scorer, Adapter(core, encode(writer, text).sum(0)), manifest, episodes[x0], lower, forbidden, (1, 1)); atomic = encode(writer, f"{floor} VALUE_{lower}").sum(0) + encode(writer, f"{avoid} VALUE_{forbidden}").sum(0); reconstructed_result = run_learned(executor, ctrl1, scorer, Adapter(core, atomic), manifest, episodes[x0], lower, forbidden, (1, 1)); joint["real"].append({"category": category, **joint_result}); reconstructed["real"].append({"category": category, **reconstructed_result})
    for result in (joint, reconstructed):
        interaction = [row for row in result["real"] if row["category"] == "x_lt_L_eq_F"]; result["interaction_gate"]["success"] = sum(post_copy(row) == ["INCREASE"] * (row["forbidden"] - row["x0"] + 1) + ["EMIT"] for row in interaction); crossing = next(row for row in result["real"] if row["x0"] == 10 and row["lower"] == 14 and row["forbidden"] == 12); result["crossing_literal"]["success"] = int(post_copy(crossing) == ["INCREASE"] * 4 + ["EMIT"]); result["crossing_literal"]["post_copy_actions"] = post_copy(crossing); result["real"] = {"samples": 128, "success": sum(row["success"] for row in result["real"]), "categories": {category: {"samples": sum(row["category"] == category for row in result["real"]), "success": sum(row["category"] == category and row["success"] for row in result["real"])} for category in sorted({row["category"] for row in result["real"]})}}
    return {"JOINT": joint, "RECONSTRUCTED": reconstructed}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) / "heldout" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); checkpoint = checkpoint_for_seed(args.seed)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; results = {f"{name}_{order}": battery(core, writer, executor, ctrl1, scorer, manifest, episodes, floor, avoid, order) for name, floor, avoid in REALIZATIONS for order in ORDERS}; passed = all(item[condition]["canonical"]["success"] == 31744 and item[condition]["real"]["success"] == 128 and item[condition]["interaction_gate"]["success"] == 32 and item[condition]["crossing_literal"]["success"] == 1 for item in results.values() for condition in ("JOINT", "RECONSTRUCTED")); result = {"status": "passed" if passed else "failed", "task": "T2-I2-R1", "phase": "heldout_joint_and_reconstructed", "training": False, "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)}, "dispatcher_guard": "passed", "conditions": results}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(path), "sha256": file_sha(path), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
