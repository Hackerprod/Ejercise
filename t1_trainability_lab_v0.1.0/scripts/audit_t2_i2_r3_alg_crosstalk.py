"""Frozen R3 residual decomposition; diagnostic only, never training."""

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
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r3 import checkpoint_for_seed
from train_u0c_ctrl2_o import OrdinalSharedScorer
from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; REALIZATIONS = (("AT_LEAST_AVOID", "AT_LEAST", "AVOID"), ("AT_LEAST_EXCLUDE", "AT_LEAST", "EXCLUDE"), ("MINIMUM_AVOID", "MINIMUM", "AVOID"), ("MINIMUM_EXCLUDE", "MINIMUM", "EXCLUDE")); ORDERS = ("NORMAL", "INVERTED"); PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM", "ATOMIC/FLOOR", "ATOMIC/AVOID", "ATOMIC-SUM", "REPLACE-F", "REPLACE-A", "REPLACE-BOTH")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def encode(writer: CompetitiveSemanticWriter, text: str) -> dict[str, torch.Tensor]:
    with torch.no_grad(): return writer(*tensorize([text]), return_details=True)


def instruction(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> str:
    clauses = (f"{floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden}"); return " AND ".join(clauses if order == "NORMAL" else clauses[::-1])


def atomic_texts(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> tuple[str, str]:
    if order == "NORMAL": return f"{floor} VALUE_{lower} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {avoid} VALUE_{forbidden}"
    return f"NOOP VALUE_0 AND {floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden} AND NOOP VALUE_0"


def additive_slots(writer: CompetitiveSemanticWriter, details: dict[str, torch.Tensor], probabilities: torch.Tensor | None = None) -> torch.Tensor:
    routing = details["routing_probabilities"] if probabilities is None else probabilities; writes = writer.slot_output(details["values"]); return torch.einsum("bst,btd->bsd", routing[:, :2, :], writes)


def run_probe(executor: Any, ctrl1: Any, scorer: OrdinalSharedScorer, core: LatentConditionedSupervisor, slots: torch.Tensor, manifest: dict[str, Any], episode: dict[str, Any], lower: int, forbidden: int, constraints: tuple[int, int] = (1, 1)) -> bool:
    result = run_learned(executor, ctrl1, scorer, Adapter(core, slots), manifest, episode, lower, forbidden, constraints); return bool(result["success"])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6301); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = CAMPAIGN / f"t2_i2_r3_alg_seed{args.seed}" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); checkpoint = checkpoint_for_seed(args.seed)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cases: list[dict[str, Any]] = []; soft_max_diff = 0.0
    for realization, floor, avoid in REALIZATIONS:
        for order in ORDERS:
            for category, x0, lower, forbidden in real_cases():
                joint = encode(writer, instruction(floor, avoid, order, lower, forbidden)); joint_slots = additive_slots(writer, joint); soft_max_diff = max(soft_max_diff, float((joint_slots - joint["slots"]).detach().abs().max())); floor_atomic_text, avoid_atomic_text = atomic_texts(floor, avoid, order, lower, forbidden); floor_atomic = encode(writer, floor_atomic_text); avoid_atomic = encode(writer, avoid_atomic_text); floor_slots = additive_slots(writer, floor_atomic)[0]; avoid_slots = additive_slots(writer, avoid_atomic)[0]; joint_slots = joint_slots[0]
                permutations: list[dict[str, Any]] = []
                for name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
                    floor_ok = run_probe(executor, ctrl1, scorer, core, joint_slots[permutation[0]], manifest, episodes[x0], lower, 0, (1, 0)); avoid_ok = run_probe(executor, ctrl1, scorer, core, joint_slots[permutation[1]], manifest, episodes[x0], 0, forbidden, (0, 1)); sum_ok = run_probe(executor, ctrl1, scorer, core, joint_slots.sum(0), manifest, episodes[x0], lower, forbidden); permutations.append({"name": name, "slots": list(permutation), "floor": floor_ok, "avoid": avoid_ok, "sum": sum_ok, "matching": floor_ok and avoid_ok and sum_ok})
                winner = next((item for item in permutations if item["matching"]), None); winner_found = winner is not None; selected = (0, 1) if winner is None else tuple(winner["slots"]); floor_slot, avoid_slot = joint_slots[selected[0]], joint_slots[selected[1]]
                atomic_floor = [(name, run_probe(executor, ctrl1, scorer, core, floor_slots[index], manifest, episodes[x0], lower, 0, (1, 0))) for name, index in (("slot0", 0), ("slot1", 1))]; atomic_avoid = [(name, run_probe(executor, ctrl1, scorer, core, avoid_slots[index], manifest, episodes[x0], 0, forbidden, (0, 1))) for name, index in (("slot0", 0), ("slot1", 1))]; floor_active = next((index for index, (_, ok) in enumerate(atomic_floor) if ok), 0); avoid_active = next((index for index, (_, ok) in enumerate(atomic_avoid) if ok), 0); atomic_floor_found = any(ok for _, ok in atomic_floor); atomic_avoid_found = any(ok for _, ok in atomic_avoid); atomic_floor_slot, atomic_avoid_slot = floor_slots[floor_active], avoid_slots[avoid_active]; probe_slots = {"JOINT/FLOOR": floor_slot, "JOINT/AVOID": avoid_slot, "JOINT/SUM": floor_slot + avoid_slot, "ATOMIC/FLOOR": atomic_floor_slot, "ATOMIC/AVOID": atomic_avoid_slot, "ATOMIC-SUM": atomic_floor_slot + atomic_avoid_slot, "REPLACE-F": atomic_floor_slot + avoid_slot, "REPLACE-A": floor_slot + atomic_avoid_slot, "REPLACE-BOTH": atomic_floor_slot + atomic_avoid_slot}; probe_results = {"JOINT/FLOOR": permutations[selected[0]]["floor"] if winner is not None else run_probe(executor, ctrl1, scorer, core, floor_slot, manifest, episodes[x0], lower, 0, (1, 0)), "JOINT/AVOID": permutations[selected[0]]["avoid"] if winner is not None else run_probe(executor, ctrl1, scorer, core, avoid_slot, manifest, episodes[x0], 0, forbidden, (0, 1)), "JOINT/SUM": permutations[selected[0]]["sum"] if winner is not None else run_probe(executor, ctrl1, scorer, core, floor_slot + avoid_slot, manifest, episodes[x0], lower, forbidden), "ATOMIC/FLOOR": atomic_floor_found, "ATOMIC/AVOID": atomic_avoid_found, "ATOMIC-SUM": run_probe(executor, ctrl1, scorer, core, probe_slots["ATOMIC-SUM"], manifest, episodes[x0], lower, forbidden), "REPLACE-F": run_probe(executor, ctrl1, scorer, core, probe_slots["REPLACE-F"], manifest, episodes[x0], lower, forbidden), "REPLACE-A": run_probe(executor, ctrl1, scorer, core, probe_slots["REPLACE-A"], manifest, episodes[x0], lower, forbidden), "REPLACE-BOTH": run_probe(executor, ctrl1, scorer, core, probe_slots["REPLACE-BOTH"], manifest, episodes[x0], lower, forbidden)}; cases.append({"realization": realization, "order": order, "category": category, "x0": x0, "lower": lower, "forbidden": forbidden, "winner_found": winner_found, "winner_permutation": None if winner is None else winner["name"], "winner_slots": list(selected), "atomic_floor_found": atomic_floor_found, "atomic_avoid_found": atomic_avoid_found, "permutations": permutations, "probes": probe_results})
    if soft_max_diff != 0.0: raise RuntimeError(f"SOFT reconstruction mismatch: {soft_max_diff}")
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]: return {probe: {"samples": len(items), "success": sum(bool(item["probes"][probe]) for item in items)} for probe in PROBES}
    by_block: dict[str, Any] = {}; by_fallback: dict[str, Any] = {}; by_winner: dict[str, Any] = {}
    for realization, _, _ in REALIZATIONS:
        for order in ORDERS:
            block = [item for item in cases if item["realization"] == realization and item["order"] == order]; by_block[f"{realization}_{order}"] = summarize(block)
    by_fallback["winner_found_false"] = summarize([item for item in cases if not item["winner_found"]]); by_winner["winner_found_true"] = summarize([item for item in cases if item["winner_found"]]); result = {"status": "diagnostic", "task": "T2-I2-R3-ALG", "training": False, "checkpoint": str(checkpoint), "soft_reconstruction_max_abs_diff": soft_max_diff, "soft_sanity": "passed_exact", "scope": "8 realization/order blocks × 128 Gate3 real_cases = 1024", "probes": list(PROBES), "summary_total": summarize(cases), "summary_by_realization_order": by_block, "summary_by_winner_found": {**by_fallback, **by_winner}, "cases": cases}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": result["status"], "soft_reconstruction_max_abs_diff": soft_max_diff, "summary_total": result["summary_total"], "fallback_cases": sum(not item["winner_found"] for item in cases)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
