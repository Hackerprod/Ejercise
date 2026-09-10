"""Permutation-invariant behavioral matching audit for T2-I2-R1."""

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
from evaluate_u0c_ctrl7_trained import run_learned
from evaluate_u0c_ctrl7_preflight import real_cases
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r1 import checkpoint_for_seed, output_for_seed
from train_u0c_ctrl2_o import OrdinalSharedScorer
from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, tensorize

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; REALIZATIONS = (("AT_LEAST_AVOID", "AT_LEAST", "AVOID"), ("AT_LEAST_EXCLUDE", "AT_LEAST", "EXCLUDE"), ("MINIMUM_AVOID", "MINIMUM", "AVOID"), ("MINIMUM_EXCLUDE", "MINIMUM", "EXCLUDE")); ORDERS = ("NORMAL", "INVERTED"); PERMUTATIONS = ((0, 1), (1, 0))


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text]);
    with torch.no_grad(): return writer(token_ids, lengths)[0]
def instruction(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> str:
    clauses = (f"{floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden}"); return " AND ".join(clauses if order == "NORMAL" else clauses[::-1])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) / "matching" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); checkpoint = checkpoint_for_seed(args.seed)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; results: dict[str, Any] = {}
    for realization, floor, avoid in REALIZATIONS:
        for order in ORDERS:
            key = f"{realization}_{order}"; permutation_counts = {"identity": 0, "swap": 0}; matching_success = 0; categories: dict[str, dict[str, int]] = {}
            for category, x0, lower, forbidden in real_cases():
                slots = encode(writer, instruction(floor, avoid, order, lower, forbidden)); outcomes = []
                for permutation in PERMUTATIONS:
                    floor_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots[permutation[0]]), manifest, episodes[x0], lower, 0, (1, 0)); avoid_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots[permutation[1]]), manifest, episodes[x0], 0, forbidden, (0, 1)); joint_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots.sum(0)), manifest, episodes[x0], lower, forbidden, (1, 1)); outcomes.append(floor_result["success"] and avoid_result["success"] and joint_result["success"])
                passed = any(outcomes); matching_success += int(passed); categories.setdefault(category, {"samples": 0, "success": 0}); categories[category]["samples"] += 1; categories[category]["success"] += int(passed); permutation_counts["identity"] += int(outcomes[0]); permutation_counts["swap"] += int(outcomes[1])
            results[key] = {"scope": {"real": 128, "interaction": 32, "crossing": 1}, "permutations": permutation_counts, "matching": {"samples": 128, "success": matching_success, "fraction": matching_success / 128}, "categories": categories}
    null_inputs = ["NOOP VALUE_0", "NOOP VALUE_0 AND NOOP VALUE_11", "AT_LEAST VALUE_12 AND NOOP VALUE_19", "NOOP VALUE_19 AND AVOID VALUE_23"]; null_behavior = {}
    for text in null_inputs:
        details = writer(*tensorize([text]), return_details=True); null_behavior[text] = {"semantic_mass": details["semantic_mass"][0].tolist(), "null_mass": float(details["routing_probabilities"][0, 2].sum())}
    result = {"status": "passed" if all(item["matching"]["success"] == 128 for item in results.values()) else "failed", "task": "T2-I2-R1", "phase": "behavioral_matching_audit", "training": False, "checkpoint": str(checkpoint), "dispatcher_guard": "passed", "matching_scope": "128 real + 32 interaction + 1 crossing per realization/order; no canonical sweep", "permutation_invariant": True, "results": results, "null_behavior": null_behavior}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
