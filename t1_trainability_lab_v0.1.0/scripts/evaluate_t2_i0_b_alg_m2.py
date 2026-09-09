"""Corrected M2 audit on systematically selected real failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, INCREASE, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_preflight import oracle_action, real_cases, supervisor_features_ctrl7
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_r1 import SharedInstructionEncoderR1
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from t2_i0_instruction_r1 import parse_instruction_r1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT_B = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_b_seed5601" / "final.pt"
CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_b_alg_m2_seed5601"


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: Tensor) -> None:
        super().__init__(); self.core = core; self.condition = condition

    def forward(self, features: Tensor, _constraints: Tensor) -> Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def instruction(order: str, lower: int, forbidden: int) -> str:
    return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}" if order == "AT_LEAST_THEN_AVOID" else f"AVOID VALUE_{forbidden} AND AT_LEAST VALUE_{lower}"


def encode(encoder: SharedInstructionEncoderR1, text: str) -> Tensor:
    parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)])
    with torch.no_grad(): return encoder(tokens, lengths).squeeze(0)


def post_copy_state(executor: Any, ctrl1: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], episode: dict[str, Any], x0: int, lower: int, forbidden: int) -> tuple[Tensor, Tensor]:
    graph = manifest["graphs"][episode["graph"]]; keys, values, types, row_mask = materialize_graph_batch(executor, [graph]); state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = executor.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); goal = executor.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); pointer = episode["start_key"]; value = x0; read_e_done = copied = v_e = v_r = False
    for _ in range(64):
        action = oracle_action(pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, value=value, lower=lower, forbidden=forbidden, constraints=(1, 1)); state, operation = dispatch_unified_action(executor, keys, values, types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        if action == READ_P: pointer = graph["rows"][operation["selected_row"]]["value"]; v_e = False
        elif action == READ_E: read_e_done = True; v_e = True
        elif action == COPY_E_R: copied = True; v_r = True
        elif action == INCREASE: value += 1; v_r = True
        elif action == DECREASE: value -= 1; v_r = True
        if copied and value == forbidden: break
    while value != forbidden:
        action = INCREASE if value < forbidden else DECREASE; state, _ = dispatch_unified_action(executor, keys, values, types, row_mask, state, presence, action, v_e=True, v_r=True); value += 1 if action == INCREASE else -1
    features = supervisor_features_ctrl7(executor, ctrl1, scorer, state, goal, lower, forbidden, v_e=True, v_r=True)
    return state, features


def margin(core: LatentConditionedSupervisor, features: Tensor, condition: Tensor) -> float:
    with torch.no_grad(): logits = core(features, condition.unsqueeze(0))
    return float(logits[0, INCREASE] - logits[0, 5])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    encoder = SharedInstructionEncoderR1(); encoder.load_state_dict(torch.load(CHECKPOINT_B, weights_only=False)["encoder"], strict=True); encoder.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); original = GoalConditionedSupervisor614(); original.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); original.eval(); executor = load_executor(); ctrl1 = load_ctrl1(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; z_floor = [encode(encoder, f"AT_LEAST VALUE_{value}") for value in range(32)]; z_avoid = [encode(encoder, f"AVOID VALUE_{value}") for value in range(31)]; goal_weight = original.goal_projection.weight.detach(); g_sum = goal_weight[:, 0] + goal_weight[:, 1]
    report: dict[str, Any] = {"orders": {}}
    for order in ("AT_LEAST_THEN_AVOID", "AVOID_THEN_AT_LEAST"):
        failures = []
        for category, x0, lower, forbidden in real_cases():
            text = instruction(order, lower, forbidden); condition = encode(encoder, text); candidate = run_learned(executor, ctrl1, scorer, Adapter(core, condition), manifest, episode_by_x[x0], lower, forbidden, (1, 1))
            if not candidate["success"]: failures.append({"category": category, "x0": x0, "lower": lower, "forbidden": forbidden, "candidate": candidate})
        failures = list({(row["category"], row["x0"], row["lower"], row["forbidden"]): row for row in failures}.values())
        failures.sort(key=lambda row: (row["x0"], row["lower"], row["forbidden"]))
        selected = [row for row in failures if row["category"] == "x_lt_L_eq_F"][:10] + [row for row in failures if row["category"] == "x_lt_L_ne_F"][:5]
        if len(selected) < 15: selected += [row for row in failures if row not in selected][:15 - len(selected)]
        cases = []
        for row in selected:
            text = instruction(order, row["lower"], row["forbidden"]); _, features = post_copy_state(executor, ctrl1, scorer, manifest, episode_by_x[row["x0"]], row["x0"], row["lower"], row["forbidden"]); real_condition = encode(encoder, text); atomic_condition = z_floor[row["lower"]] + z_avoid[row["forbidden"]]; cases.append({"category": row["category"], "x0": row["x0"], "lower": row["lower"], "forbidden": row["forbidden"], "original_success": row["candidate"]["success"], "margins": {"real_latent": margin(core, features, real_condition), "ground_truth_g_sum": margin(core, features, g_sum), "atomic_sum": margin(core, features, atomic_condition)}})
        report["orders"][order] = {"total_real_failures": len(failures), "selected_failures": len(cases), "cases": cases}
    result = {"status": "completed", "task": "T2-I0-B-ALG", "phase": "corrected_m2_failed_real_cases", "training": False, "checkpoint": {"path": str(CHECKPOINT_B), "sha256": sha256(CHECKPOINT_B)}, "ctrl7_checkpoint": sha256(CTRL7_CHECKPOINT), "results": report}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
