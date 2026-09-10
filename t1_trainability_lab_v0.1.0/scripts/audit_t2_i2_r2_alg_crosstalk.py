"""Frozen R2 cross-talk audit; routing diagnostics only, never training."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r2 import checkpoint_for_seed
from train_u0c_ctrl2_o import OrdinalSharedScorer
from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, tensorize

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; REALIZATIONS = (("AT_LEAST_AVOID", "AT_LEAST", "AVOID"), ("AT_LEAST_EXCLUDE", "AT_LEAST", "EXCLUDE"), ("MINIMUM_AVOID", "MINIMUM", "AVOID"), ("MINIMUM_EXCLUDE", "MINIMUM", "EXCLUDE")); ORDERS = ("NORMAL", "INVERTED"); CONDITIONS = ("SOFT", "HARD-ARGMAX", "NULL-HARD", "ORACLE-PRUNE-CROSS", "ORACLE-PRUNE-STRICT")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def encode(writer: CompetitiveSemanticWriter, text: str) -> dict[str, torch.Tensor]:
    token_ids, lengths = tensorize([text])
    with torch.no_grad(): return writer(token_ids, lengths, return_details=True)


def instruction(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> str:
    clauses = (f"{floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden}"); return " AND ".join(clauses if order == "NORMAL" else clauses[::-1])


def slots_for(writer: CompetitiveSemanticWriter, details: dict[str, torch.Tensor], condition: str, order: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = details["routing_probabilities"][:, :3, :].clone(); values = details["values"]; valid = probabilities.sum(dim=1, keepdim=True) > 0
    if condition == "HARD-ARGMAX":
        winners = probabilities.argmax(dim=1); probabilities = F.one_hot(winners, 3).permute(0, 2, 1).to(values.dtype) * valid
    elif condition == "NULL-HARD":
        winners = probabilities.argmax(dim=1); null = winners == 2; hard = F.one_hot(winners, 3).permute(0, 2, 1).to(values.dtype); probabilities = torch.where(null.unsqueeze(1), hard, probabilities)
    elif condition.startswith("ORACLE-PRUNE"):
        if order == "NORMAL": floor_indices, avoid_indices, and_index = (0, 1), (3, 4), 2
        else: floor_indices, avoid_indices, and_index = (3, 4), (0, 1), 2
        probabilities[:, 0, list(avoid_indices)] = 0; probabilities[:, 1, list(floor_indices)] = 0
        if condition == "ORACLE-PRUNE-STRICT": probabilities[:, 0, and_index] = 0; probabilities[:, 1, and_index] = 0
    semantic = probabilities[:, :2, :]; mass = semantic.sum(dim=-1, keepdim=True); pooled = torch.einsum("bst,btd->bsd", semantic, values) / (1e-8 + mass); slots = writer.slot_output(pooled) * (mass / (1e-8 + mass)); return slots[0], probabilities[0], pooled[0]


def post_copy(result: dict[str, Any]) -> list[str]:
    actions = [event["action_name"] for event in result["events"]]; return actions[next(index for index, action in enumerate(actions) if action == "COPY_E_R") + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6201); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); checkpoint = checkpoint_for_seed(args.seed); output = (CAMPAIGN / f"t2_i2_r2_alg_seed{args.seed}") if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; results: dict[str, Any] = {}; soft_max_diff = 0.0
    for realization, floor, avoid in REALIZATIONS:
        for order in ORDERS:
            summary = {name: {"matching": {"samples": 128, "success": 0}, "permutations": {"identity": 0, "swap": 0}, "mass": [], "projected_contribution_norm": [], "and_routing": [], "local_l2": [], "local_cosine": []} for name in CONDITIONS}
            for category, x0, lower, forbidden in real_cases():
                text = instruction(floor, avoid, order, lower, forbidden); details = encode(writer, text); soft_slots, soft_prob, _ = slots_for(writer, details, "SOFT", order); soft_max_diff = max(soft_max_diff, float((soft_slots - details["slots"][0]).detach().abs().max()))
                if order == "NORMAL": floor_indices, avoid_indices = (0, 1), (3, 4)
                else: floor_indices, avoid_indices = (3, 4), (0, 1)
                floor_atomic_text = f"{floor} VALUE_{lower} AND NOOP VALUE_0" if order == "NORMAL" else f"NOOP VALUE_0 AND {floor} VALUE_{lower}"
                avoid_atomic_text = f"NOOP VALUE_0 AND {avoid} VALUE_{forbidden}" if order == "NORMAL" else f"{avoid} VALUE_{forbidden} AND NOOP VALUE_0"
                floor_atomic = encode(writer, floor_atomic_text); avoid_atomic = encode(writer, avoid_atomic_text); floor_ref = floor_atomic["local_states"][0, list(floor_indices)]; avoid_ref = avoid_atomic["local_states"][0, list(avoid_indices)]
                for condition in CONDITIONS:
                    slots, probabilities, pooled = slots_for(writer, details, condition, order); floor_mass = probabilities[:, list(floor_indices)].sum(dim=-1); avoid_mass = probabilities[:, list(avoid_indices)].sum(dim=-1); summary[condition]["mass"].append({"F_to_0": float(floor_mass[0]), "F_to_1": float(floor_mass[1]), "F_to_NULL": float(floor_mass[2]), "A_to_0": float(avoid_mass[0]), "A_to_1": float(avoid_mass[1]), "A_to_NULL": float(avoid_mass[2])}); summary[condition]["and_routing"].append(probabilities[:, 2].tolist()); summary[condition]["projected_contribution_norm"].append({"F_to_0": float((torch.einsum("td,t->d", details["values"][0, list(floor_indices)], probabilities[0, list(floor_indices)]) / (1e-8 + probabilities[0].sum())).norm()), "F_to_1": float((torch.einsum("td,t->d", details["values"][0, list(floor_indices)], probabilities[1, list(floor_indices)]) / (1e-8 + probabilities[1].sum())).norm()), "A_to_0": float((torch.einsum("td,t->d", details["values"][0, list(avoid_indices)], probabilities[0, list(avoid_indices)]) / (1e-8 + probabilities[0].sum())).norm()), "A_to_1": float((torch.einsum("td,t->d", details["values"][0, list(avoid_indices)], probabilities[1, list(avoid_indices)]) / (1e-8 + probabilities[1].sum())).norm())}); floor_local = details["local_states"][0, list(floor_indices)]; avoid_local = details["local_states"][0, list(avoid_indices)]; summary[condition]["local_l2"].append({"F": float((floor_local - floor_ref).norm()), "A": float((avoid_local - avoid_ref).norm())}); summary[condition]["local_cosine"].append({"F": float(F.cosine_similarity(floor_local, floor_ref, dim=-1).mean()), "A": float(F.cosine_similarity(avoid_local, avoid_ref, dim=-1).mean())})
                    outcomes = []
                    for permutation_name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
                        floor_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots[permutation[0]]), manifest, episodes[x0], lower, 0, (1, 0)); avoid_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots[permutation[1]]), manifest, episodes[x0], 0, forbidden, (0, 1)); joint_result = run_learned(executor, ctrl1, scorer, Adapter(core, slots.sum(0)), manifest, episodes[x0], lower, forbidden, (1, 1)); ok = floor_result["success"] and avoid_result["success"] and joint_result["success"]; outcomes.append(ok); summary[condition]["permutations"][permutation_name] += int(ok)
                    summary[condition]["matching"]["success"] += int(any(outcomes))
            for condition in CONDITIONS:
                for field in ("mass", "projected_contribution_norm", "and_routing", "local_l2", "local_cosine"): summary[condition][field] = {"samples": len(summary[condition][field]), "mean": summary[condition][field]}
            results[f"{realization}_{order}"] = summary
    if soft_max_diff >= 1e-6: raise RuntimeError(f"SOFT reconstruction mismatch: {soft_max_diff}")
    result = {"status": "diagnostic", "task": "T2-I2-R2-ALG", "training": False, "checkpoint": str(checkpoint), "soft_reconstruction_max_abs_diff": soft_max_diff, "soft_sanity": "passed", "matching_scope": "8 realization/order cases × 128 real probes × both permutations; no canonical sweep", "results": results}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
