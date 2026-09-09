"""Frozen T1-CTRL-7 FLOOR_AND_AVOID held-out composition evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value, state_hash
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl7_preflight import GOALS, oracle_action, primitive_check, real_cases, supervisor_features_ctrl7, target_value
from evaluate_u0c_ctrl7_trained import run_learned
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_heldout_seed4701"
CONSTRAINTS = (1, 1)


@torch.no_grad()
def run_canonical_learned(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor614, manifest: dict[str, Any], episode: dict[str, Any], x0: int, lower: int, forbidden: int) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + x0]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    value = x0; target = target_value(x0, lower, forbidden, CONSTRAINTS); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        expected = oracle_action(episode["goal_key"], episode["goal_key"], read_e_done=True, copied=True, value=value, lower=lower, forbidden=forbidden, constraints=CONSTRAINTS)
        features = supervisor_features_ctrl7(model, ctrl1, scorer, state, navigation_goal, lower, forbidden, v_e=True, v_r=True)
        action = int(supervisor(features, torch.tensor([[1.0, 1.0]])).argmax(-1).item())
        before = state.clone()
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=True, v_r=True)
        check = primitive_check(model, before, state, action, {**operation, "available_e": True, "available_r": True}, -1, graph, target)
        valid = action == expected and check
        events.append({"decision": decision, "action": action, "action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[action], "expected_action": expected, "expected_action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[expected], "choice_correct": action == expected, "primitive_correct": check, "valid": valid, "r_after_sha256": state_hash(state, SLOT_R)})
        first_bad = decision if not valid and first_bad is None else first_bad
        if action == INCREASE and not operation.get("rejected", False): value += 1
        elif action == EMIT and not operation.get("rejected", False): emitted = True
        if emitted: break
    final = int(canonical_value_view(model, state[:, SLOT_R])[1].item())
    return {"x0": x0, "lower": lower, "forbidden": forbidden, "constraints": list(CONSTRAINTS), "goal": GOALS[CONSTRAINTS], "target": target, "actions": [event["action_name"] for event in events], "events": events, "decisions": len(events), "alu_steps": sum(event["action"] in (INCREASE, 4) for event in events), "final_value": final, "first_bad": first_bad, "timeout": not emitted, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden CTRL-7 inputs")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); model = load_executor(); ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    checkpoint_payload = torch.load(CHECKPOINT, weights_only=False); supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(checkpoint_payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    triples = [(x, lower, forbidden) for x in range(32) for lower in range(32) for forbidden in range(31)]
    canonical = [run_canonical_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], x, lower, forbidden) for x, lower, forbidden in triples]
    cases = real_cases()
    real = [{**run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], lower, forbidden, CONSTRAINTS), "category": category} for category, x, lower, forbidden in cases]
    categories = {}
    for category, *_ in cases:
        rows = [item for item in real if item["category"] == category]
        categories[category] = {"samples": len(rows), "success": sum(item["success"] for item in rows)}
    interaction = [item for item in real if item["category"] == "x_lt_L_eq_F"]
    interaction_gate = {"samples": len(interaction), "success": sum(item["success"] and [event["action_name"] for event in item["events"]][next(index for index, event in enumerate(item["events"]) if event["action_name"] == "COPY_E_R") + 1:] == ["INCREASE"] * (item["forbidden"] - item["x0"] + 1) + ["EMIT"] for item in interaction)}
    crossing = next(item for item in real if item["x0"] == 10 and item["lower"] == 14 and item["forbidden"] == 12)
    crossing_actions = [event["action_name"] for event in crossing["events"]]
    crossing_suffix = crossing_actions[next(index for index, action in enumerate(crossing_actions) if action == "COPY_E_R") + 1:]
    crossing_gate = {"samples": 1, "success": int(crossing["success"] and crossing_suffix == ["INCREASE"] * 4 + ["EMIT"]), "actions": crossing_actions, "post_copy_actions": crossing_suffix}
    summary = {"domain": {"triples": len(triples), "formula": "32 x 32 x 31", "f_inclusive": False}, "canonical_post_copy": {"samples": len(canonical), "success": sum(item["success"] for item in canonical), "max_decisions": max(item["decisions"] for item in canonical)}, "real_memory": {"scenarios": len(cases), "samples": len(real), "success": sum(item["success"] for item in real), "categories": categories}, "required_gates": {"x_lt_L_eq_F_exact_sequence": interaction_gate, "crossing_literal_exact_sequence": crossing_gate}}
    result = {"status": "passed" if summary["canonical_post_copy"]["success"] == 31744 and summary["real_memory"]["success"] == 128 and interaction_gate["success"] == interaction_gate["samples"] and crossing_gate["success"] == 1 else "failed", "task": "T1-CTRL-7", "phase": "held_out_floor_avoid", "training": False, "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT), "seed": checkpoint_payload.get("seed"), "updates": checkpoint_payload.get("updates"), "trainable_parameters": checkpoint_payload.get("trainable_parameters"), "clamp_trained": checkpoint_payload.get("clamp_trained", False)}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)}, "summary": summary, "canonical_cases": canonical, "real_memory_cases": real}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "checkpoint": result["checkpoint"], "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
