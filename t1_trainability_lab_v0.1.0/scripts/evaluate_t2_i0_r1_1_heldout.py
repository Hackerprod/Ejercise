"""Frozen T2-I0-R1.1 held-out FLOOR+AVOID evaluation for both clause orders."""

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
from train_t2_i0_r1 import BaselineAR1Classifier, SharedInstructionEncoderR1, tensorize
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from t2_i0_instruction_r1 import parse_instruction_r1
from t2_i0_instruction_r1_1 import instruction_for_r1_1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT_A = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_a_seed5501" / "final.pt"
CHECKPOINT_B = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_b_seed5601" / "final.pt"
CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_r1_1_heldout_seed5501_5601"


class LatentAdapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__(); self.core = core; self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def behavior(result: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(result.get("actions", [event["action_name"] for event in result.get("events", [])])), result["target"], result["final_value"], result["success"]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def predict(classifier: BaselineAR1Classifier, text: str) -> tuple[int, int]:
    parsed = parse_instruction_r1(text); data = tensorize([(text, parsed.constraints)])
    with torch.no_grad(): floor, avoid = classifier(data["token_ids"], data["lengths"])
    return int(floor.argmax(-1).item()), int(avoid.argmax(-1).item())


def order_instruction(order: str, lower: int, forbidden: int, dummy: int = 0) -> str:
    if order == "AT_LEAST_THEN_AVOID": return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}"
    return f"AVOID VALUE_{forbidden} AND AT_LEAST VALUE_{lower}"


def post_copy_actions(result: dict[str, Any]) -> list[str]:
    actions = result.get("actions", [event["action_name"] for event in result.get("events", [])])
    return actions[next(index for index, action in enumerate(actions) if action == "COPY_E_R") + 1:]


def exact_interaction(row: dict[str, Any], key: str) -> bool:
    expected = ["INCREASE"] * (row["forbidden"] - row["x0"] + 1) + ["EMIT"]
    return row[key]["success"] and post_copy_actions(row[key]) == expected


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    classifier = BaselineAR1Classifier(); classifier.load_state_dict(torch.load(CHECKPOINT_A, weights_only=False)["model"], strict=True); classifier.eval(); latent_encoder = SharedInstructionEncoderR1(); latent_encoder.load_state_dict(torch.load(CHECKPOINT_B, weights_only=False)["encoder"], strict=True); latent_encoder.eval(); latent_core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); latent_core.eval(); explicit_core = GoalConditionedSupervisor614(); explicit_core.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); explicit_core.eval()
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; latent_cache: dict[str, LatentAdapter] = {}; bit_cache: dict[str, tuple[int, int]] = {}
    def bits(text: str) -> tuple[int, int]:
        if text not in bit_cache: bit_cache[text] = predict(classifier, text)
        return bit_cache[text]
    def latent(text: str) -> LatentAdapter:
        if text not in latent_cache:
            parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)]); latent_cache[text] = LatentAdapter(latent_core, latent_encoder(tokens, lengths))
        return latent_cache[text]
    results: dict[str, Any] = {"orders": {}}
    for order in ("AT_LEAST_THEN_AVOID", "AVOID_THEN_AT_LEAST"):
        canonical = {"samples": 0, "baseline_a_success": 0, "baseline_b_success": 0, "baseline_a_bit_correct": 0, "a_failures": [], "b_failures": []}; bit_correct_pairs = set()
        for x0 in range(32):
            for lower in range(32):
                for forbidden in range(31):
                    text = order_instruction(order, lower, forbidden); parsed = parse_instruction_r1(text); predicted_bits = bits(text); true_target = max(x0, lower) + (1 if max(x0, lower) == forbidden else 0)
                    a_result = run_canonical_learned(executor, ctrl1, scorer, explicit_core, manifest, episode_by_x[x0], x0, parsed.lower, parsed.forbidden, predicted_bits); a_ok = predicted_bits == (1, 1) and a_result["success"] and a_result["final_value"] == true_target; canonical["samples"] += 1; bit_correct_pairs.add((lower, forbidden)) if predicted_bits == (1, 1) else None; canonical["baseline_a_success"] += int(a_ok)
                    if not a_ok and len(canonical["a_failures"]) < 5: canonical["a_failures"].append({"x0": x0, "lower": lower, "forbidden": forbidden, "predicted_bits": predicted_bits, "target": true_target, "actions": a_result["actions"]})
                    b_result = run_canonical_learned(executor, ctrl1, scorer, latent(text), manifest, episode_by_x[x0], x0, parsed.lower, parsed.forbidden, (1, 1)); canonical["baseline_b_success"] += int(b_result["success"] and b_result["final_value"] == true_target)
                    if not b_result["success"] and len(canonical["b_failures"]) < 5: canonical["b_failures"].append({"x0": x0, "lower": lower, "forbidden": forbidden, "target": true_target, "actions": b_result["actions"]})
        canonical["baseline_a_bit_correct"] = len(bit_correct_pairs); real_rows = []
        for category, x0, lower, forbidden in real_cases():
            text = order_instruction(order, lower, forbidden); parsed = parse_instruction_r1(text); predicted_bits = bits(text); true_target = max(x0, lower) + (1 if max(x0, lower) == forbidden else 0); a_result = run_learned(executor, ctrl1, scorer, explicit_core, manifest, episode_by_x[x0], parsed.lower, parsed.forbidden, predicted_bits); b_result = run_learned(executor, ctrl1, scorer, latent(text), manifest, episode_by_x[x0], parsed.lower, parsed.forbidden, (1, 1)); real_rows.append({"category": category, "x0": x0, "lower": lower, "forbidden": forbidden, "a": a_result, "b": b_result, "predicted_bits": predicted_bits, "target": true_target})
        canonical["real"] = {"samples": len(real_rows), "baseline_a_success": sum(int(row["predicted_bits"] == (1, 1) and row["a"]["success"] and row["a"]["final_value"] == row["target"]) for row in real_rows), "baseline_b_success": sum(int(row["b"]["success"] and row["b"]["final_value"] == row["target"]) for row in real_rows), "categories": {category: {"samples": sum(row["category"] == category for row in real_rows), "baseline_a_success": sum(row["category"] == category and row["predicted_bits"] == (1, 1) and row["a"]["success"] for row in real_rows), "baseline_b_success": sum(row["category"] == category and row["b"]["success"] for row in real_rows)} for category in sorted({row["category"] for row in real_rows})}}
        interaction = [row for row in real_rows if row["category"] == "x_lt_L_eq_F"]
        canonical["interaction_gate"] = {"samples": len(interaction), "baseline_a_success": sum(1 for row in interaction if row["predicted_bits"] == (1, 1) and exact_interaction(row, "a")), "baseline_b_success": sum(1 for row in interaction if exact_interaction(row, "b"))}
        crossing = next(row for row in real_rows if row["x0"] == 10 and row["lower"] == 14 and row["forbidden"] == 12); expected_suffix = ["INCREASE"] * 4 + ["EMIT"]; canonical["crossing_literal"] = {"samples": 1, "baseline_a_success": int(crossing["predicted_bits"] == (1, 1) and post_copy_actions(crossing["a"]) == expected_suffix and crossing["a"]["success"]), "baseline_b_success": int(post_copy_actions(crossing["b"]) == expected_suffix and crossing["b"]["success"]), "baseline_a_actions": [event["action_name"] for event in crossing["a"]["events"]], "baseline_b_actions": [event["action_name"] for event in crossing["b"]["events"]]}
        results["orders"][order] = canonical
    result = {"status": "passed" if all(order["baseline_a_success"] == order["samples"] and order["baseline_b_success"] == order["samples"] and order["baseline_a_bit_correct"] == 992 and order["real"]["baseline_a_success"] == order["real"]["samples"] and order["real"]["baseline_b_success"] == order["real"]["samples"] and order["interaction_gate"]["baseline_a_success"] == 32 and order["interaction_gate"]["baseline_b_success"] == 32 and order["crossing_literal"]["baseline_a_success"] == 1 and order["crossing_literal"]["baseline_b_success"] == 1 for order in results["orders"].values()) else "failed", "task": "T2-I0-R1.1", "phase": "heldout_both_orders", "training": False, "checkpoints": {"a": sha256(CHECKPOINT_A), "b": sha256(CHECKPOINT_B), "ctrl7": sha256(CTRL7_CHECKPOINT)}, "dispatcher_guard": "passed", "results": results}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
