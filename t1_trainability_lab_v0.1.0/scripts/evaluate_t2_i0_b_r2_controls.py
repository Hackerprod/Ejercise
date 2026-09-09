"""Pre-held-out R2 controls for shared clause additive conditioning."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_b_r2 import SharedClauseEncoder, clause_condition
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i0_instruction_r1_1 import instruction_for_r1_1
from t2_i0_instruction_r1 import parse_instruction_r1

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; CHECKPOINT = CAMPAIGN_ROOT / "t2_i0_b_r2_seed5701" / "final.pt"; CTRL7 = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"; SCORER = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; OUTPUT = CAMPAIGN_ROOT / "t2_i0_b_r2_controls_seed5701"


class Adapter(torch.nn.Module):
    def __init__(self, core, condition): super().__init__(); self.core = core; self.condition = condition
    def forward(self, features, _constraints): return self.core(features, self.condition.expand(features.shape[0], -1))


def behavior(result: dict[str, Any]) -> tuple[Any, ...]: return (tuple(event["action_name"] for event in result["events"]), result["target"], result["final_value"], result["success"])
def file_sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def text(constraints, value, variant, dummy=0):
    if constraints == (0, 0): return instruction_for_r1_1(constraints, 0, variant=variant, dummy=dummy, dummy2=(dummy + 11) % 32)
    return instruction_for_r1_1(constraints, value, variant=variant, dummy=dummy)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    encoder = SharedClauseEncoder(); encoder.load_state_dict(torch.load(CHECKPOINT, weights_only=False)["encoder"], strict=True); encoder.eval(); core = LatentConditionedSupervisor(CTRL7); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cache = {}
    def run(instruction_text, constraints, lower, forbidden):
        if instruction_text not in cache: cache[instruction_text] = Adapter(core, clause_condition(encoder, [instruction_text]))
        return run_learned(executor, ctrl1, scorer, cache[instruction_text], manifest, episode[10], lower, forbidden, constraints)
    position = {"samples": 0, "success": 0}; memory = {"samples": 0, "success": 0}; noop = {name: {"samples": 0, "success": 0} for name in ("NONE", "FLOOR", "AVOID")}; commutative = {"samples": 0, "identical": 0}
    for value in range(32):
        for constraints in ((1, 0), (0, 1)):
            forms = [text(constraints, value, 0), text(constraints, value, 1), text(constraints, value, 2)]; runs = [run(item, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0) for item in forms]; position["success"] += int(all(item["success"] for item in runs) and len({behavior(item) for item in runs}) == 1); position["samples"] += 1
        left = text((1, 0), value, 1); right = text((1, 0), value, 2); lrun = run(left, (1, 0), value, 0); rrun = run(right, (1, 0), value, 0); memory["success"] += int(lrun["success"] and rrun["success"] and behavior(lrun) == behavior(rrun)); memory["samples"] += 1
        for constraints, name in (((0, 0), "NONE"), ((1, 0), "FLOOR"), ((0, 1), "AVOID")):
            variants = (0, 1) if constraints == (0, 0) else (1, 2)
            for variant in variants:
                for dummy in range(32):
                    instruction_text = text(constraints, value, variant, dummy); lower = value if constraints == (1, 0) else 0; forbidden = value if constraints == (0, 1) else 0; result = run(instruction_text, constraints, lower, forbidden); noop[name]["success"] += int(result["success"]); noop[name]["samples"] += 1
        floor_text = text((1, 0), value, 1); avoid_text = text((0, 1), value, 1); floor_condition = clause_condition(encoder, [floor_text]); avoid_condition = clause_condition(encoder, [avoid_text]); commutative["identical"] += int(torch.equal(floor_condition + avoid_condition, avoid_condition + floor_condition)); commutative["samples"] += 1
    summary = {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop, "conditioning_commutativity": commutative}; result = {"status": "passed" if position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values()) and commutative["identical"] == commutative["samples"] else "failed", "task": "T2-I0-B-R2", "phase": "pre_heldout_controls", "training": False, "checkpoint": sha256(CHECKPOINT), "ctrl7_checkpoint": sha256(CTRL7), "dispatcher_guard": "passed", "summary": summary}; output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
