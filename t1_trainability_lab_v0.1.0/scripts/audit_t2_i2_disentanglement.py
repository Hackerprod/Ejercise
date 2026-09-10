"""Frozen T2-I2 slot disentanglement audit; no learned probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest
from evaluate_t2_i2_heldout import Adapter
from evaluate_u0c_ctrl7_preflight import supervisor_features_ctrl7
from evaluate_u0c_ctrl7_trained import run_learned
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, materialize_graph_batch
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT, SharedClauseEncoder, clause_condition
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i2 import checkpoint_for_seed, output_for_seed, dummy_values
from t2_i2_semantic_writer import SemanticWriter, tensorize
from train_u0c_ctrl2_o import OrdinalSharedScorer

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; R2_CHECKPOINT = CAMPAIGN / "t2_i0_b_r2_seed5701" / "final.pt"
SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"


def encode(writer: SemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad(): return writer(token_ids, lengths)[0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) / "disentanglement" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); checkpoint = checkpoint_for_seed(args.seed)
    writer = SemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval()
    supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); supervisor.eval(); weight = supervisor.goal_projection.weight.detach(); c_gt = float(F.cosine_similarity(weight[:, 0], weight[:, 1], dim=0)); threshold = c_gt + 0.05
    reference_norms = []; inactive_ratios = []; noop_norms = []; active_variations = []; noop_variations = []; active_consistency = []
    for alias in ("AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE"):
        for value in range(32):
            atomic = encode(writer, f"{alias} VALUE_{value}"); norms = torch.linalg.vector_norm(atomic, dim=-1); active = int(norms.argmax()); reference_norms.append(float(norms[active]))
            for dummy in dummy_values(value):
                slots = encode(writer, f"{alias} VALUE_{value} AND NOOP VALUE_{dummy}"); variant_active = int(torch.linalg.vector_norm(slots, dim=-1).argmax()); active_consistency.append(int(variant_active == active)); slot_norms = torch.linalg.vector_norm(slots, dim=-1); inactive_ratios.append(float(slot_norms[variant_active ^ 1] / max(float(slot_norms[variant_active]), 1e-6))); reference = encode(writer, f"{alias} VALUE_{value} AND NOOP VALUE_{dummy_values(value)[0]}"); reference_active = int(torch.linalg.vector_norm(reference, dim=-1).argmax()); active_consistency.append(int(reference_active == active)); active_variations.append(float(torch.linalg.vector_norm(slots[variant_active] - reference[reference_active]) / max(float(norms[active]), 1e-6)))
    R = float(torch.tensor(reference_norms).median());
    for dummy in range(32):
        slots = encode(writer, f"NOOP VALUE_{dummy}"); noop_norms.extend(float(item) for item in torch.linalg.vector_norm(slots, dim=-1))
    noop_baseline = encode(writer, "NOOP VALUE_0")
    for dummy in range(1, 32): noop_variations.extend(float(item) for item in torch.linalg.vector_norm(encode(writer, f"NOOP VALUE_{dummy}") - noop_baseline, dim=-1))
    atomic_pair_cosines = []
    for lower in range(32):
        for forbidden in range(31):
            slots = encode(writer, f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}"); atomic_pair_cosines.append(float(F.cosine_similarity(slots[0:1], slots[1:2], dim=-1)[0]))
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    interchange_samples = interchange_success = 0
    for alias, constraints in (("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1))):
        for value in range(32):
            atomic = encode(writer, f"{alias} VALUE_{value}"); active = int(torch.linalg.vector_norm(atomic, dim=-1).argmax()); variants = (f"{alias} VALUE_{value} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {alias} VALUE_{value}")
            for variant in variants:
                variant_slots = encode(writer, variant); condition_a = atomic[active]; condition_b = variant_slots[active]; lower = value if constraints == (1, 0) else 0; forbidden = value if constraints == (0, 1) else 0
                run_a = run_learned(executor, ctrl1, scorer, Adapter(core, condition_a), manifest, episodes[10], lower, forbidden, constraints); run_b = run_learned(executor, ctrl1, scorer, Adapter(core, condition_b), manifest, episodes[10], lower, forbidden, constraints); interchange_samples += 1; interchange_success += int(run_a["success"] and run_b["success"] and [event["action_name"] for event in run_a["events"]] == [event["action_name"] for event in run_b["events"]])
    episode = episodes[10]; graph = manifest["graphs"][episode["graph"]]; keys, values, types, mask = materialize_graph_batch(executor, [graph]); state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = executor.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); goal = executor.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    for action, v_e, v_r in ((READ_P, False, False), (READ_E, False, False), (COPY_E_R, True, False)): state, _ = dispatch_unified_action(executor, keys, values, types, mask, state, presence, action, v_e=v_e, v_r=v_r)
    for _ in range(4): state, _ = dispatch_unified_action(executor, keys, values, types, mask, state, presence, INCREASE, v_e=True, v_r=True)
    features = supervisor_features_ctrl7(executor, ctrl1, scorer, state, goal, 14, 12, v_e=True, v_r=True); writer_sum = encode(writer, "AT_LEAST VALUE_14").sum(0) + encode(writer, "AVOID VALUE_12").sum(0); r2 = SharedClauseEncoder(); r2.load_state_dict(torch.load(R2_CHECKPOINT, weights_only=False)["encoder"], strict=True); r2.eval(); r2_sum = clause_condition(r2, ["AT_LEAST VALUE_14", "AVOID VALUE_12"])[0]; original = GoalConditionedSupervisor614(); original.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); original.eval(); g_sum = original.goal_projection.weight.detach()[:, 0] + original.goal_projection.weight.detach()[:, 1]
    with torch.no_grad():
        writer_logits = core(features, writer_sum.unsqueeze(0))[0]; r2_logits = core(features, r2_sum.unsqueeze(0))[0]; gt_logits = core(features, g_sum.unsqueeze(0))[0]
    balg = {"condition_norms": {"writer_atomic_sum": float(writer_sum.norm()), "r2_atomic_sum": float(r2_sum.norm()), "ctrl7_g_sum": float(g_sum.norm())}, "logits": {"writer_atomic_sum": writer_logits.tolist(), "r2_atomic_sum": r2_logits.tolist(), "ctrl7_g_sum": gt_logits.tolist()}, "margins_increase_minus_emit": {"writer_atomic_sum": float(writer_logits[INCREASE] - writer_logits[EMIT]), "r2_atomic_sum": float(r2_logits[INCREASE] - r2_logits[EMIT]), "ctrl7_g_sum": float(gt_logits[INCREASE] - gt_logits[EMIT])}, "probe": {"x0": 10, "lower": 14, "forbidden": 12}}
    metrics = {"reference_active_norm_median_R": R, "inactive_ratio_max": max(inactive_ratios), "none_norm_over_R_max": max(noop_norms) / max(R, 1e-6), "active_dummy_relative_variation_max": max(active_variations), "noop_dummy_absolute_variation_over_R_max": max(noop_variations) / max(R, 1e-6), "active_slot_consistency": {"samples": len(active_consistency), "success": sum(active_consistency), "fraction": sum(active_consistency) / len(active_consistency)}, "reconstructed_active_slot_cosine_max": max(atomic_pair_cosines), "c_gt": c_gt, "cosine_threshold": threshold, "interchange_preservation": {"samples": interchange_samples, "success": interchange_success, "fraction": interchange_success / interchange_samples}, "b_alg": balg}
    gates = {"inactive": metrics["inactive_ratio_max"] <= 0.10 and metrics["none_norm_over_R_max"] <= 0.10, "active_dummy": metrics["active_dummy_relative_variation_max"] <= 0.05, "noop_dummy": metrics["noop_dummy_absolute_variation_over_R_max"] <= 0.10, "active_slot_consistency": sum(active_consistency) == len(active_consistency), "reconstruction_cosine": metrics["reconstructed_active_slot_cosine_max"] <= threshold, "interchange": interchange_success == interchange_samples}; result = {"status": "passed" if all(gates.values()) else "failed", "task": "T2-I2", "phase": "frozen_disentanglement_audit", "training": False, "checkpoint": str(checkpoint), "metrics": metrics, "gates": gates, "thresholds": {"inactive_ratio_max": 0.10, "none_norm_over_R_max": 0.10, "active_dummy_relative_variation_max": 0.05, "noop_dummy_absolute_variation_over_R_max": 0.10, "interchange_fraction": 1.0, "cosine": "c_gt + 0.05"}}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
