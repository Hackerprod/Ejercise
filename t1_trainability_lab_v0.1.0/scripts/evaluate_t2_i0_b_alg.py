"""Frozen algebra audit for T2-I0 Baseline B; no training or evaluator edits."""

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
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value
from evaluate_u0c_ctrl7_heldout import run_canonical_learned
from evaluate_u0c_ctrl7_preflight import real_cases, supervisor_features_ctrl7
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_r1 import SharedInstructionEncoderR1
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, materialize_graph_batch
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
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_b_alg_seed5601"
ORDERS = ("AT_LEAST_THEN_AVOID", "AVOID_THEN_AT_LEAST")


class ConditionAdapter(nn.Module):
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
    return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}" if order == ORDERS[0] else f"AVOID VALUE_{forbidden} AND AT_LEAST VALUE_{lower}"


def vectors(encoder: SharedInstructionEncoderR1) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    z_floor = torch.stack([encode(encoder, f"AT_LEAST VALUE_{value}") for value in range(32)])
    z_avoid = torch.stack([encode(encoder, f"AVOID VALUE_{value}") for value in range(31)])
    z_order1 = torch.stack([torch.stack([encode(encoder, instruction(ORDERS[0], lower, forbidden)) for forbidden in range(31)]) for lower in range(32)])
    z_order2 = torch.stack([torch.stack([encode(encoder, instruction(ORDERS[1], lower, forbidden)) for forbidden in range(31)]) for lower in range(32)])
    z_none = torch.stack([encode(encoder, f"NOOP VALUE_{dummy}") for dummy in range(32)])
    return z_floor, z_avoid, z_order1, z_order2, z_none


def encode(encoder: SharedInstructionEncoderR1, text: str) -> Tensor:
    parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)])
    with torch.no_grad(): return encoder(tokens, lengths).squeeze(0)


def projection_metrics(z: Tensor, g_floor: Tensor, g_avoid: Tensor) -> dict[str, float]:
    design = torch.stack((g_floor, g_avoid), dim=1); coefficients = torch.linalg.lstsq(design, z).solution; residual = z - design @ coefficients; target = g_floor + g_avoid
    return {"alpha": float(coefficients[0]), "beta": float(coefficients[1]), "residual_norm": float(residual.norm()), "distance_to_g_sum": float((z - target).norm())}


def aggregate(metrics: list[dict[str, float]]) -> dict[str, Any]:
    return {key: {"mean": float(torch.tensor([item[key] for item in metrics]).mean()), "std": float(torch.tensor([item[key] for item in metrics]).std(unbiased=False)), "min": min(item[key] for item in metrics), "max": max(item[key] for item in metrics)} for key in metrics[0]}


def critical_margin(executor: Any, ctrl1: Any, scorer: OrdinalSharedScorer, core: LatentConditionedSupervisor, manifest: dict[str, Any], episode: dict[str, Any], lower: int, forbidden: int, condition: Tensor) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]; keys, values, types, mask = materialize_graph_batch(executor, [graph]); state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = executor.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); goal = executor.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    for action, v_e, v_r in ((READ_P, False, False), (READ_E, False, False), (COPY_E_R, True, False)):
        state, _ = dispatch_unified_action(executor, keys, values, types, mask, state, presence, action, v_e=v_e, v_r=v_r)
    for _ in range(forbidden - 10): state, _ = dispatch_unified_action(executor, keys, values, types, mask, state, presence, INCREASE, v_e=True, v_r=True)
    features = supervisor_features_ctrl7(executor, ctrl1, scorer, state, goal, lower, forbidden, v_e=True, v_r=True)
    learned = core(features, condition)
    return {"learned_margin": float(learned[0, INCREASE] - learned[0, EMIT]), "state_value": forbidden, "features": features.squeeze(0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    encoder = SharedInstructionEncoderR1(); encoder.load_state_dict(torch.load(CHECKPOINT_B, weights_only=False)["encoder"], strict=True); encoder.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); original = GoalConditionedSupervisor614(); original.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); original.eval(); executor = load_executor(); ctrl1 = load_ctrl1(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    z_floor, z_avoid, z_order1, z_order2, z_none = vectors(encoder); goal_weight = original.goal_projection.weight.detach(); g_floor = goal_weight[:, 0]; g_avoid = goal_weight[:, 1]; g_sum = g_floor + g_avoid; z_atomic = z_floor[:, None, :] + z_avoid[None, :, :]; z_atomic_reverse = z_avoid[None, :, :] + z_floor[:, None, :]; z_centered = z_atomic - z_none.mean(0); commutative = bool(torch.equal(z_atomic, z_atomic_reverse))
    projection = {"AT_LEAST_THEN_AVOID": aggregate([projection_metrics(z_order1[lower, forbidden], g_floor, g_avoid) for lower in range(32) for forbidden in range(31)]), "AVOID_THEN_AT_LEAST": aggregate([projection_metrics(z_order2[lower, forbidden], g_floor, g_avoid) for lower in range(32) for forbidden in range(31)])}
    episode = episode_by_x[10]; critical = {"x0": 10, "lower": 14, "forbidden": 14, "real_latent_order1": critical_margin(executor, ctrl1, scorer, core, manifest, episode, 14, 14, encode(encoder, instruction(ORDERS[0], 14, 14))), "real_latent_order2": critical_margin(executor, ctrl1, scorer, core, manifest, episode, 14, 14, encode(encoder, instruction(ORDERS[1], 14, 14))), "ground_truth": critical_margin(executor, ctrl1, scorer, core, manifest, episode, 14, 14, g_sum), "order2_interaction_non_equal": critical_margin(executor, ctrl1, scorer, core, manifest, episode, 14, 12, encode(encoder, instruction(ORDERS[1], 14, 12))), "note": "same state/features within each comparison; only conditioning differs"}
    results: dict[str, Any] = {"vector_commutativity": {"atomic_sum_equal_under_order_swap": commutative}, "projection": projection, "critical_margin": critical, "batteries": {}}
    strategies = {"atomic_sum": z_atomic, "centered_sum": z_centered, "ground_truth": g_sum.expand(32, 31, 32)}
    for strategy, table in strategies.items():
        canonical_success = 0; real_rows = []
        for x0 in range(32):
            for lower in range(32):
                for forbidden in range(31):
                    condition = table[lower, forbidden] if table.ndim == 3 else table; adapter = ConditionAdapter(core, condition); candidate = run_canonical_learned(executor, ctrl1, scorer, adapter, manifest, episode_by_x[x0], x0, lower, forbidden, (1, 1)); canonical_success += int(candidate["success"])
        for category, x0, lower, forbidden in real_cases():
            condition = table[lower, forbidden] if table.ndim == 3 else table; candidate = run_learned(executor, ctrl1, scorer, ConditionAdapter(core, condition), manifest, episode_by_x[x0], lower, forbidden, (1, 1)); real_rows.append({"category": category, **candidate})
        interaction = [row for row in real_rows if row["category"] == "x_lt_L_eq_F"]; interaction_gate = sum(1 for row in interaction if [event["action_name"] for event in row["events"]][next(index for index, event in enumerate(row["events"]) if event["action_name"] == "COPY_E_R") + 1:] == ["INCREASE"] * (row["forbidden"] - row["x0"] + 1) + ["EMIT"]); crossing = next(row for row in real_rows if row["x0"] == 10 and row["lower"] == 14 and row["forbidden"] == 12); crossing_gate = int([event["action_name"] for event in crossing["events"]][next(index for index, event in enumerate(crossing["events"]) if event["action_name"] == "COPY_E_R") + 1:] == ["INCREASE"] * 4 + ["EMIT"])
        summary = {"canonical": {"samples": 31744, "success": canonical_success}, "real": {"samples": 128, "success": sum(row["success"] for row in real_rows), "categories": {category: {"samples": sum(row["category"] == category for row in real_rows), "success": sum(row["category"] == category and row["success"] for row in real_rows)} for category in sorted({row["category"] for row in real_rows})}}, "interaction_gate": {"samples": 32, "success": interaction_gate}, "crossing_literal": {"samples": 1, "success": crossing_gate}}
        results["batteries"][strategy] = {"same_for_both_orders": True, "orders": {order: summary for order in ORDERS}}
    result = {"status": "completed", "task": "T2-I0-B-ALG", "training": False, "checkpoint": {"path": str(CHECKPOINT_B), "sha256": sha256(CHECKPOINT_B)}, "ctrl7_checkpoint": sha256(CTRL7_CHECKPOINT), "results": results}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), "status": result["status"], "checkpoint": result["checkpoint"], "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
