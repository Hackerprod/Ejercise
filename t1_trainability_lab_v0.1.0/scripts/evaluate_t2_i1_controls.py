"""T2-I1 controls for four lexical operator realizations."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i1 import SharedClauseEncoderI1, clause_condition, instruction_for
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i1_instruction import parse_instruction_i1

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; CHECKPOINT = CAMPAIGN / "t2_i1_b_seed5801" / "final.pt"; CTRL7 = CAMPAIGN / "u0c_ctrl7_pilot_seed4701" / "final.pt"; SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; OUTPUT = CAMPAIGN / "t2_i1_controls_seed5801"


class Adapter(torch.nn.Module):
    def __init__(self, core, condition): super().__init__(); self.core = core; self.condition = condition
    def forward(self, features, _constraints): return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    encoder = SharedClauseEncoderI1(); encoder.load_state_dict(torch.load(CHECKPOINT, weights_only=False)["encoder"], strict=True); encoder.eval(); core = LatentConditionedSupervisor(CTRL7); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cache = {}
    def run(text, constraints, lower, forbidden):
        if text not in cache: cache[text] = Adapter(core, clause_condition(encoder, [text]))
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episodes[10], lower, forbidden, constraints)
    aliases = (("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1)))
    position = {"samples": 0, "success": 0}; memory = {"samples": 0, "success": 0}; noop = {"NONE": {"samples": 0, "success": 0}, **{alias: {"samples": 0, "success": 0} for alias, _ in aliases}}; comm = {"samples": 0, "identical": 0}
    for value in range(32):
        for alias, constraints in aliases:
            operator = f"{alias} VALUE_{value}"; forms = [operator, f"{operator} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {operator}"]; runs = [run(form, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0) for form in forms]; position["success"] += int(all(item["success"] for item in runs) and len({tuple(event["action_name"] for event in item["events"]) for item in runs}) == 1); position["samples"] += 1
            left = forms[1]; right = forms[2]; lrun = run(left, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); rrun = run(right, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); memory["success"] += int(lrun["success"] and rrun["success"] and tuple(event["action_name"] for event in lrun["events"]) == tuple(event["action_name"] for event in rrun["events"])); memory["samples"] += 1
            c1 = clause_condition(encoder, [left]); c2 = clause_condition(encoder, [right]); comm["identical"] += int(torch.equal(c1, c2)); comm["samples"] += 1
        for name, constraints in (("NONE", (0, 0)),) + aliases:
            variants = (0, 1) if name == "NONE" else (1, 2); value_arg = value
            for variant in variants:
                for dummy in range(32):
                    alias = None if name == "NONE" else name; text = instruction_for(constraints, value_arg, variant, dummy=dummy, alias=alias, dummy2=(dummy + 11) % 32); lower = value if constraints == (1, 0) else 0; forbidden = value if constraints == (0, 1) else 0; result = run(text, constraints, lower, forbidden); noop[name]["samples"] += 1; noop[name]["success"] += int(result["success"])
    summary = {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop, "conditioning_commutativity": comm}; result = {"status": "passed" if position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values()) and comm["identical"] == comm["samples"] else "failed", "task": "T2-I1", "phase": "lexical_controls", "training": False, "checkpoint": sha256(CHECKPOINT), "ctrl7": sha256(CTRL7), "summary": summary, "dispatcher_guard": "passed"}; output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
