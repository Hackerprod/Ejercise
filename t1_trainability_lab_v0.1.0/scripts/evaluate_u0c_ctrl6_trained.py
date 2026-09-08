"""Pre-CLAMP evaluation for the trained T1-CTRL-6 objectives."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_COUNT
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value, state_hash
from evaluate_u0c_ctrl6_preflight import GOALS, oracle_action, primitive_check, real_cases, supervisor_features_ctrl6, target_value
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl6 import GOAL_IDS, GoalConditionedSupervisor614, TRAIN_GOALS
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
PILOT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_pilot_seed4601"
SUPERVISOR_CHECKPOINT = PILOT_ROOT / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_trained_evaluation_seed4601"


@torch.no_grad()
def run_learned(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor614, manifest: dict[str, Any], episode: dict[str, Any], lower: int, upper: int, constraints: tuple[int, int]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]; memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph]); state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); x0 = pair_value(manifest, episode); pointer, symbolic_x = episode["start_key"], x0; read_e_done = copied = v_e = v_r = False; target = target_value(x0, lower, upper, constraints); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        expected_action = oracle_action(pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, value=symbolic_x, lower=lower, upper=upper, constraints=constraints); before = state.clone(); features = supervisor_features_ctrl6(model, ctrl1, scorer, state, navigation_goal, lower, upper, v_e=v_e, v_r=v_r); action = int(supervisor(features, torch.tensor([[float(constraints[0]), float(constraints[1])]])).argmax(-1).item()); expected_row = next((index for index, row in enumerate(graph["rows"]) if (expected_action == READ_P and row["kind"] == ROW_REL and row["key"] == pointer) or (expected_action == READ_E and row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])), -1); state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r); check = primitive_check(model, before, state, action, {**operation, "available_e": v_e, "available_r": v_r}, expected_row, graph, pointer, target); valid = action == expected_action and check; events.append({"decision": decision, "action": action, "action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[action], "expected_action": expected_action, "expected_action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[expected_action], "features": features.squeeze(0).tolist(), "primitive_correct": check, "choice_correct": action == expected_action, "valid": valid, "operation": operation["operation"], "rejected": operation.get("rejected", False), "r_after_sha256": state_hash(state, SLOT_R)}); first_bad = decision if not valid and first_bad is None else first_bad
        if action == READ_P: v_e = False
        elif action == READ_E and not operation.get("rejected", False): v_e = True; read_e_done = True
        elif action == COPY_E_R and not operation.get("rejected", False): v_r = True; copied = True
        elif action in (INCREASE, DECREASE) and not operation.get("rejected", False): v_r = True
        elif action == EMIT: emitted = True
        if expected_action == READ_P: pointer = graph["rows"][expected_row]["value"]
        elif expected_action == READ_E: read_e_done = True
        elif expected_action == COPY_E_R: copied = True
        elif expected_action == INCREASE: symbolic_x += 1
        elif expected_action == DECREASE: symbolic_x -= 1
        if emitted: break
    final = int(__import__("evaluate_u0c_ctrl2_o_canon").canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"episode": episode["episode"], "graph": episode["graph"], "x0": x0, "lower": lower, "upper": upper, "constraints": list(constraints), "goal": GOALS[constraints], "target": target, "events": events, "decisions": len(events), "final_value": final, "alu_steps": sum(event["action"] in (INCREASE, DECREASE) for event in events), "timeout": not emitted, "first_bad": first_bad, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT}


def trace_metrics(root: Path, supervisor: GoalConditionedSupervisor614) -> dict[str, Any]:
    observations = torch.load(root / "test" / "observations.pt", weights_only=False)["features"]; labels = torch.load(root / "test" / "labels.pt", weights_only=False); result = {}
    for descriptor, goal_name in TRAIN_GOALS.items():
        mask = (labels["constraints"][:, 0] == descriptor[0]) & (labels["constraints"][:, 1] == descriptor[1]); prediction = supervisor(observations[mask], labels["constraints"][mask]).argmax(-1); correct = prediction == labels["action"][mask]; result[goal_name] = {"samples": int(mask.sum()), "correct": int(correct.sum()), "accuracy": float(correct.float().mean())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in parameters for name in ("goal", "constraints", "lower", "upper")): raise RuntimeError("dispatcher received forbidden CTRL-6 inputs")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); model = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); payload = torch.load(SUPERVISOR_CHECKPOINT, map_location="cpu", weights_only=False); supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(payload["supervisor"], strict=True); supervisor.eval(); episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]: episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    scenarios = real_cases(); trajectories = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], lower, upper, constraints) for x, lower, upper in scenarios for constraints in TRAIN_GOALS]
    lower_control = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[10], 12, upper, (1, 0)) for upper in (20, 31)]; lower_control_result = {"runs": [{key: value for key, value in item.items() if key != "events"} for item in lower_control], "pass": lower_control[0]["success"] and lower_control[1]["success"] and lower_control[0]["final_value"] == lower_control[1]["final_value"] and [event["action"] for event in lower_control[0]["events"]] == [event["action"] for event in lower_control[1]["events"]]}
    oracle_summary = json.loads((CAMPAIGN_ROOT / "u0c_ctrl6_preflight_seed2201" / "results.json").read_text(encoding="utf-8"))["summary"]
    free = {goal: {"samples": sum(item["goal"] == goal for item in trajectories), "success": sum(item["goal"] == goal and item["success"] for item in trajectories), "timeouts": sum(item["goal"] == goal and item["timeout"] for item in trajectories), "first_bad_count": sum(item["goal"] == goal and item["first_bad"] is not None for item in trajectories)} for goal in TRAIN_GOALS.values()}
    result = {"status": "passed" if all(item["success"] for item in trajectories) and lower_control_result["pass"] else "failed", "task": "T1-CTRL-6", "training": {"seed": payload.get("seed"), "updates": payload.get("updates"), "parameters": payload.get("trainable_parameters"), "clamp_trained": payload.get("clamp_trained", False)}, "checkpoint_supervisor": {"path": str(SUPERVISOR_CHECKPOINT), "sha256": sha256(SUPERVISOR_CHECKPOINT)}, "measurements": {"oracle_executor": oracle_summary, "supervisor_reference_traces": trace_metrics(PILOT_ROOT, supervisor), "supervisor_free_execution": {"samples": len(trajectories), "success": sum(item["success"] for item in trajectories), "by_goal": free}}, "controls": {"vary_inactive_upper_under_lower": lower_control_result}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)}, "trajectories": trajectories}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "measurements": result["measurements"], "controls": {key: value["pass"] for key, value in result["controls"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
