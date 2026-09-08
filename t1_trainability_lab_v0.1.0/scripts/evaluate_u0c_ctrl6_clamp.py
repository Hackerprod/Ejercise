"""Held-out CTRL-6 CLAMP evaluation. No training or checkpoint mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value, state_hash
from evaluate_u0c_ctrl6_preflight import GOALS, oracle_action, primitive_check, real_cases, supervisor_features_ctrl6, target_value
from evaluate_u0c_ctrl6_trained import run_learned
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl6 import GoalConditionedSupervisor614, TRAIN_GOALS
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, materialize_graph_batch
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
PILOT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_pilot_seed4601"
SUPERVISOR_CHECKPOINT = PILOT_ROOT / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_clamp_evaluation_seed4601"
CLAMP = (1, 1)


@torch.no_grad()
def run_canonical_supervisor(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor614, manifest: dict[str, Any], episode: dict[str, Any], x0: int, lower: int, upper: int, memory_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]; memory_keys, memory_values, memory_types, row_mask = memory_cache[x0]; state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); state[:, SLOT_E] = model.token_embedding(torch.tensor([VALUE_BASE + x0])); state[:, SLOT_R] = state[:, SLOT_E].clone(); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); pointer, symbolic_x = episode["goal_key"], x0; target = target_value(x0, lower, upper, CLAMP); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        expected = oracle_action(pointer, episode["goal_key"], read_e_done=True, copied=True, value=symbolic_x, lower=lower, upper=upper, constraints=CLAMP); features = supervisor_features_ctrl6(model, ctrl1, scorer, state, model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])), lower, upper, v_e=True, v_r=True); action = int(supervisor(features, torch.tensor([[1.0, 1.0]])).argmax(-1).item()); before = state.clone(); state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=True, v_r=True); check = primitive_check(model, before, state, action, {**operation, "available_e": True, "available_r": True}, -1, graph, pointer, target); valid = action == expected and check; events.append({"action": action, "action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[action], "expected_action": expected, "valid": valid}); first_bad = decision if not valid and first_bad is None else first_bad
        if action == INCREASE: symbolic_x += 1
        elif action == DECREASE: symbolic_x -= 1
        elif action == EMIT: emitted = True
        if emitted: break
    final = int(__import__("evaluate_u0c_ctrl2_o_canon").canonical_value_view(model, state[:, SLOT_R])[1].item())
    return {"x0": x0, "lower": lower, "upper": upper, "target": target, "actions": [event["action_name"] for event in events], "decisions": len(events), "alu_steps": sum(event["action"] in (INCREASE, DECREASE) for event in events), "final_value": final, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT, "first_bad": first_bad}


@torch.no_grad()
def build_post_copy(model: Any, manifest: dict[str, Any], episode: dict[str, Any], memory_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]; memory_keys, memory_values, memory_types, row_mask = memory_cache[pair_value(manifest, episode)]; state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); pointer = episode["start_key"]; v_e = v_r = False
    for _ in range(8):
        action = READ_P if pointer != episode["goal_key"] else (READ_E if not v_e else COPY_E_R); expected_row = next(index for index, row in enumerate(graph["rows"]) if (action == READ_P and row["kind"] == ROW_REL and row["key"] == pointer) or (action == READ_E and row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])) if action != COPY_E_R else -1; state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        if action == READ_P: pointer = graph["rows"][expected_row]["value"]
        elif action == READ_E: v_e = True
        elif action == COPY_E_R: v_r = True; return {"state": state, "flags": (v_e, v_r), "pointer": pointer, "read_e_done": True, "copied": True, "x0": pair_value(manifest, episode)}
    raise RuntimeError("failed to build real post-COPY snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if "CLAMP" in TRAIN_GOALS.values(): raise RuntimeError("CLAMP must not be in TRAIN_GOALS")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); model = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); payload = torch.load(SUPERVISOR_CHECKPOINT, map_location="cpu", weights_only=False); supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(payload["supervisor"], strict=True); supervisor.eval(); episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]: episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    memory_cache = {x: materialize_graph_batch(model, [manifest["graphs"][episode_by_x[x]["graph"]]]) for x in range(32)}
    canonical = [run_canonical_supervisor(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], x, lower, upper, memory_cache) for x in range(32) for lower in range(32) for upper in range(lower, 32)]
    scenarios = real_cases(); real = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], lower, upper, CLAMP) for x, lower, upper in scenarios]
    snapshot = build_post_copy(model, manifest, episode_by_x[10], memory_cache); first = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[10], 12, 20, CLAMP, initial_state=snapshot["state"], initial_flags=snapshot["flags"], initial_pointer=episode_by_x[10]["goal_key"], initial_symbolic_x=10, initial_read_e_done=True, initial_copied=True); second = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[10], 15, 20, CLAMP, initial_state=snapshot["state"], initial_flags=snapshot["flags"], initial_pointer=episode_by_x[10]["goal_key"], initial_symbolic_x=10, initial_read_e_done=True, initial_copied=True); control = {"runs": [{key: value for key, value in item.items() if key != "state"} for item in (first, second)], "pass": first["success"] and second["success"] and first["final_value"] == 12 and second["final_value"] == 15 and first["events"] != second["events"]}
    summary = {"canonical_post_copy": {"samples": len(canonical), "success": sum(item["success"] for item in canonical), "max_decisions": max(item["decisions"] for item in canonical), "triples": 16896}, "real_memory": {"samples": len(real), "success": sum(item["success"] for item in real), "regions": {"x0<L": sum(x < lower for x, lower, upper in scenarios), "L<=x0<=U": sum(lower <= x <= upper for x, lower, upper in scenarios), "x0>U": sum(x > upper for x, lower, upper in scenarios)}, "boundaries": {"L_equals_U": sum(lower == upper for x, lower, upper in scenarios), "x0_equals_L": sum(x == lower for x, lower, upper in scenarios), "x0_equals_U": sum(x == upper for x, lower, upper in scenarios)}}}
    result = {"status": "passed" if summary["canonical_post_copy"]["success"] == 16896 and summary["real_memory"]["success"] == len(real) and control["pass"] else "failed", "task": "T1-CTRL-6", "phase": "held_out_clamp_composition", "training": False, "checkpoint_supervisor": {"path": str(SUPERVISOR_CHECKPOINT), "sha256": sha256(SUPERVISOR_CHECKPOINT), "seed": payload.get("seed"), "updates": payload.get("updates"), "parameters": payload.get("trainable_parameters"), "clamp_trained": payload.get("clamp_trained", False)}, "summary": summary, "control_both_active_limit_determines": control, "canonical_cases": canonical, "real_memory_cases": [{key: value for key, value in item.items() if key != "state"} for item in real]}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "summary": summary, "control": control["pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
