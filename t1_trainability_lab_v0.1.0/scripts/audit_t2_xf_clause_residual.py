"""Frozen micro-audit of the 27 XF-CLAUSE residual cases."""

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
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import SharedClauseEncoder, clause_condition as r2_clause_condition
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from t2_xf_transformer import MatchedTransformerEncoder, encode_clauses

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; XF_CHECKPOINT = CAMPAIGN_ROOT / "t2_xf_clause_r1_1_seed6001" / "final.pt"; R2_CHECKPOINT = CAMPAIGN_ROOT / "t2_i0_b_r2_seed5701" / "final.pt"; CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"; SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_xf_clause_residual_audit_seed6001"; ORDERS = ("AT_LEAST_THEN_AVOID", "AVOID_THEN_AT_LEAST")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def instruction(order: str, lower: int, forbidden: int) -> str:
    return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}" if order == ORDERS[0] else f"AVOID VALUE_{forbidden} AND AT_LEAST VALUE_{lower}"


def margin(logits: torch.Tensor) -> float: return float((logits[0, 3] - logits[0, 5]).item())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    xf = MatchedTransformerEncoder(); xf.load_state_dict(torch.load(XF_CHECKPOINT, weights_only=False)["encoder"], strict=True); xf.eval(); r2 = SharedClauseEncoder(); r2.load_state_dict(torch.load(R2_CHECKPOINT, weights_only=False)["encoder"], strict=True); r2.eval(); ctrl7_core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); ctrl7_core.eval(); ctrl7_full = GoalConditionedSupervisor614(); ctrl7_full.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); ctrl7_full.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    records: list[dict[str, Any]] = []
    for lower, forbidden in ((12, 23), (15, 23)):
        atomic_floor = f"AT_LEAST VALUE_{lower}"; atomic_avoid = f"AVOID VALUE_{forbidden}"
        with torch.no_grad():
            z_floor_xf = encode_clauses(xf, [atomic_floor]); z_avoid_xf = encode_clauses(xf, [atomic_avoid]); z_floor_r2 = r2_clause_condition(r2, [atomic_floor]); z_avoid_r2 = r2_clause_condition(r2, [atomic_avoid]); condition_xf = z_floor_xf + z_avoid_xf; condition_r2 = z_floor_r2 + z_avoid_r2; ground_truth = ctrl7_full.goal_projection.weight[:, 0:1].T + ctrl7_full.goal_projection.weight[:, 1:2].T
        pair_record: dict[str, Any] = {"lower": lower, "forbidden": forbidden, "conditioning": {"xf_clause": {"z_floor": z_floor_xf[0].tolist(), "z_avoid": z_avoid_xf[0].tolist(), "sum": condition_xf[0].tolist()}, "r2_clause": {"z_floor": z_floor_r2[0].tolist(), "z_avoid": z_avoid_r2[0].tolist(), "sum": condition_r2[0].tolist()}, "ctrl7_ground_truth": ground_truth[0].tolist()}, "distances": {"floor_xf_vs_r2_l2": float(torch.linalg.vector_norm(z_floor_xf - z_floor_r2).item()), "avoid_xf_vs_r2_l2": float(torch.linalg.vector_norm(z_avoid_xf - z_avoid_r2).item())}, "orders": {}}
        for order in ORDERS:
            text = instruction(order, lower, forbidden); adapter = Adapter(ctrl7_core, condition_xf); result = run_learned(executor, ctrl1, scorer, adapter, manifest, episode_by_x[0], lower, forbidden, (1, 1)); wrong = next(event for event in result["events"] if event["expected_action_name"] == "INCREASE" and event["action_name"] != "INCREASE"); features = torch.tensor([wrong["features"]], dtype=torch.float32)
            with torch.no_grad(): logits_xf = ctrl7_core(features, condition_xf); logits_r2 = ctrl7_core(features, condition_r2); logits_gt = ctrl7_core(features, ground_truth)
            pair_record["orders"][order] = {"instruction": text, "trajectory_success": result["success"], "first_wrong_decision": wrong["decision"], "expected_action": wrong["expected_action_name"], "actual_action": wrong["action_name"], "state_marker": "R=L-1 before missing INCREASE", "xf_clause": {"margin_increase_minus_emit": margin(logits_xf), "logits": logits_xf[0].tolist()}, "r2_clause": {"margin_increase_minus_emit": margin(logits_r2), "logits": logits_r2[0].tolist()}, "ctrl7_ground_truth": {"margin_increase_minus_emit": margin(logits_gt), "logits": logits_gt[0].tolist()}}
        records.append(pair_record)
    result = {"status": "completed", "task": "T2-XF", "phase": "residual_micro_audit", "training": False, "cases": records, "checkpoints": {"xf_clause": {"path": str(XF_CHECKPOINT), "sha256": sha256(XF_CHECKPOINT)}, "r2": {"path": str(R2_CHECKPOINT), "sha256": sha256(R2_CHECKPOINT)}, "ctrl7": sha256(CTRL7_CHECKPOINT)}, "manifest": {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH)}, "dispatcher_guard": "passed"}; output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
