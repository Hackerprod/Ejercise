"""Evaluate frozen T1-CTRL-7 three-objective pilot before held-out CLAMP."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value, state_hash
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl7_preflight import GOALS, oracle_action, primitive_check, real_cases, supervisor_features_ctrl7, target_value
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614, TRAIN_GOALS
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
PILOT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
SUPERVISOR_CHECKPOINT = PILOT_ROOT / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_trained_evaluation_seed4701"


@torch.no_grad()
def run_learned(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor614, manifest: dict[str, Any], episode: dict[str, Any], lower: int, forbidden: int, constraints: tuple[int, int]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    x0 = pair_value(manifest, episode); pointer, symbolic_x = episode["start_key"], x0; read_e_done = copied = v_e = v_r = False
    target = target_value(x0, lower, forbidden, constraints); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        expected = oracle_action(pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, value=symbolic_x, lower=lower, forbidden=forbidden, constraints=constraints)
        features = supervisor_features_ctrl7(model, ctrl1, scorer, state, navigation_goal, lower, forbidden, v_e=v_e, v_r=v_r)
        action = int(supervisor(features, torch.tensor([[float(constraints[0]), float(constraints[1])]])).argmax(-1).item())
        before = state.clone()
        expected_row = next((index for index, row in enumerate(graph["rows"]) if (expected == READ_P and row["kind"] == ROW_REL and row["key"] == pointer) or (expected == READ_E and row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])), -1)
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        check = primitive_check(model, before, state, action, {**operation, "available_e": v_e, "available_r": v_r}, expected_row, graph, target)
        valid = action == expected and check
        events.append({"decision": decision, "action": action, "action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[action], "expected_action": expected, "expected_action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[expected], "choice_correct": action == expected, "primitive_correct": check, "valid": valid, "features": features.squeeze(0).tolist(), "r_after_sha256": state_hash(state, SLOT_R)})
        first_bad = decision if not valid and first_bad is None else first_bad
        rejected = operation.get("rejected", False)
        if action == READ_P and not rejected: v_e = False
        elif action == READ_E and not rejected: v_e = True; read_e_done = True
        elif action == COPY_E_R and not rejected: v_r = True; copied = True
        elif action == INCREASE and not rejected: v_r = True; symbolic_x += 1
        elif action == EMIT and not rejected: emitted = True
        if action == READ_P and not rejected and operation.get("selected_row", -1) >= 0:
            pointer = graph["rows"][operation["selected_row"]]["value"]
        if emitted: break
    final = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"episode": episode["episode"], "graph": episode["graph"], "x0": x0, "lower": lower, "forbidden": forbidden, "constraints": list(constraints), "goal": GOALS[constraints], "target": target, "events": events, "decisions": len(events), "final_value": final, "first_bad": first_bad, "timeout": not emitted, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT}


def trace_metrics(root: Path, supervisor: GoalConditionedSupervisor614) -> dict[str, Any]:
    observations = torch.load(root / "test" / "observations.pt", weights_only=False)["features"]
    labels = torch.load(root / "test" / "labels.pt", weights_only=False)
    result = {}
    for descriptor, goal_name in TRAIN_GOALS.items():
        mask = (labels["constraints"][:, 0] == descriptor[0]) & (labels["constraints"][:, 1] == descriptor[1])
        prediction = supervisor(observations[mask], labels["constraints"][mask]).argmax(-1)
        correct = prediction == labels["action"][mask]
        result[goal_name] = {"samples": int(mask.sum()), "correct": int(correct.sum()), "accuracy": float(correct.float().mean())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden CTRL-7 inputs")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); model = load_executor(); ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    payload = torch.load(SUPERVISOR_CHECKPOINT, weights_only=False); supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    scenarios = real_cases(); trajectories = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], lower, forbidden, constraints) for _, x, lower, forbidden in scenarios for constraints in TRAIN_GOALS]
    floor_control = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[10], 20, forbidden, (1, 0)) for forbidden in (12, 29)]
    interaction_control = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[10], lower, 12, (1, 1)) for lower in (12, 14)]
    oracle_summary = json.loads((CAMPAIGN_ROOT / "u0c_ctrl7_preflight_seed2201" / "results.json").read_text(encoding="utf-8"))["summary"]
    free = {goal: {"samples": sum(item["goal"] == goal for item in trajectories), "success": sum(item["goal"] == goal and item["success"] for item in trajectories), "timeouts": sum(item["goal"] == goal and item["timeout"] for item in trajectories), "first_bad_count": sum(item["goal"] == goal and item["first_bad"] is not None for item in trajectories)} for goal in TRAIN_GOALS.values()}
    controls = {"vary_inactive_forbidden_under_floor": {"runs": [{key: value for key, value in item.items() if key != "events"} for item in floor_control], "pass": all(item["success"] for item in floor_control) and [event["action"] for event in floor_control[0]["events"]] == [event["action"] for event in floor_control[1]["events"]] and floor_control[0]["final_value"] == floor_control[1]["final_value"]}, "vary_lower_at_interaction": {"runs": [{key: value for key, value in item.items() if key != "events"} for item in interaction_control], "pass": all(item["success"] for item in interaction_control) and [event["action"] for event in interaction_control[0]["events"]] != [event["action"] for event in interaction_control[1]["events"]]}}
    result = {"status": "passed" if all(item["success"] for item in trajectories) and all(item["pass"] for item in controls.values()) else "failed", "task": "T1-CTRL-7", "training": {"seed": payload.get("seed"), "updates": payload.get("updates"), "parameters": payload.get("trainable_parameters"), "clamp_trained": payload.get("clamp_trained", False)}, "checkpoint_supervisor": {"path": str(SUPERVISOR_CHECKPOINT), "sha256": sha256(SUPERVISOR_CHECKPOINT)}, "measurements": {"oracle_executor": oracle_summary, "supervisor_reference_traces": trace_metrics(PILOT_ROOT, supervisor), "supervisor_free_execution": {"samples": len(trajectories), "success": sum(item["success"] for item in trajectories), "by_goal": free}}, "controls": controls, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)}, "trajectories": trajectories}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "checkpoint": result["checkpoint_supervisor"], "measurements": result["measurements"], "controls": {key: value["pass"] for key, value in controls.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
