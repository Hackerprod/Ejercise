"""T1-CTRL-5 conditioned supervisor evaluation and causal controls."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import validate_transition
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, FETCH, FETCH_AND_ADJUST, GOAL_NAMES, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, oracle_action, pair_value, state_hash, strict_primitive_check
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, SLOT_W, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl4 import supervisor_features
from train_u0c_ctrl5 import GOAL_IDS, GoalConditionedSupervisor
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
PILOT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl5_pilot_seed4501"
SUPERVISOR_CHECKPOINT = PILOT_ROOT / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl5_evaluation_seed4501"


def expected_row(graph: dict[str, Any], pointer: int, action: int, goal_key: int) -> int:
    if action == READ_P:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == pointer)
    if action == READ_E:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == goal_key)
    return -1


def update_oracle(action: int, graph: dict[str, Any], pointer: int, goal_key: int, symbolic_x: int, read_e_done: bool, copied: bool) -> tuple[int, int, bool, bool]:
    if action == READ_P:
        pointer = graph["rows"][expected_row(graph, pointer, action, goal_key)]["value"]
    elif action == READ_E:
        read_e_done = True
    elif action == COPY_E_R:
        copied = True
    elif action == INCREASE:
        symbolic_x += 1
    elif action == DECREASE:
        symbolic_x -= 1
    return pointer, symbolic_x, read_e_done, copied


def goal_descriptor(goal: str, count: int = 1) -> torch.Tensor:
    value = torch.zeros((count, 2))
    value[:, GOAL_IDS[goal]] = 1.0
    return value


@torch.no_grad()
def run_learned(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor, manifest: dict[str, Any], episode: dict[str, Any], reference: int, goal: str, *, initial_state: torch.Tensor | None = None, initial_flags: tuple[bool, bool] = (False, False), initial_pointer: int | None = None, initial_symbolic_x: int | None = None, initial_read_e_done: bool = False, initial_copied: bool = False, stop_after_copy: bool = False) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = initial_state.clone() if initial_state is not None else torch.zeros((1, SLOT_COUNT, DIMENSION))
    if initial_state is None:
        state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    x0 = pair_value(manifest, episode)
    symbolic_pointer = episode["start_key"] if initial_pointer is None else initial_pointer
    symbolic_x = x0 if initial_symbolic_x is None else initial_symbolic_x
    read_e_done, copied = initial_read_e_done, initial_copied
    v_e, v_r = initial_flags
    events: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    emitted = False
    alu_steps = 0
    for decision in range(MAX_DECISIONS):
        oracle = oracle_action(symbolic_pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, symbolic_x=symbolic_x, reference=reference, goal=goal)
        before = state.clone()
        features = supervisor_features(model, ctrl1, scorer, state, navigation_goal, reference, v_e=v_e, v_r=v_r)
        logits = supervisor(features, goal_descriptor(goal))
        action = int(logits.argmax(-1).item())
        expected = expected_row(graph, symbolic_pointer, oracle, episode["goal_key"])
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        detail = {**operation, "available_e": v_e, "available_r": v_r}
        if (action in (INCREASE, DECREASE, EMIT)) and not v_r:
            check = {"exact": False, "primitive_correct": False, "choice_correct": action == oracle, "result_correct": False, "reason": "R unavailable", "conservation": {"P": True, "E": True, "R": True, "W": torch.equal(before[:, SLOT_W], state[:, SLOT_W])}}
        else:
            check = strict_primitive_check(model, manifest, episode, before, state, action, detail, symbolic_pointer, symbolic_x, reference, expected, goal=goal, x0=x0, expected_action=oracle, alu_steps=alu_steps)
        valid = bool(check["primitive_correct"] and check["choice_correct"] and check["result_correct"])
        event = {"decision": decision, "goal": goal, "action": action, "action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[action], "expected_action": oracle, "expected_action_name": ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")[oracle], "action_correct": check["choice_correct"], "operation": operation["operation"], "rejected": operation.get("rejected", False), "features": features.squeeze(0).tolist(), "selected_row": operation["selected_row"], "p_sha256": state_hash(before, SLOT_P), "e_sha256": state_hash(before, SLOT_E), "r_sha256": state_hash(before, SLOT_R), "w_sha256": state_hash(before, SLOT_W), "p_after_sha256": state_hash(state, SLOT_P), "e_after_sha256": state_hash(state, SLOT_E), "r_after_sha256": state_hash(state, SLOT_R), "w_after_sha256": state_hash(state, SLOT_W), **check, "valid": valid}
        events.append(event)
        if not valid and first_bad is None:
            first_bad = {"category": "supervisor_choice" if not check["choice_correct"] else "component_execution", **event}
        if action == READ_P:
            v_e = False
        elif action == READ_E and not operation.get("rejected", False):
            v_e = True
        elif action == COPY_E_R and not operation.get("rejected", False):
            v_r = True
        elif action in (INCREASE, DECREASE) and not operation.get("rejected", False):
            v_r = True; alu_steps += 1
        elif action == EMIT:
            emitted = True
        symbolic_pointer, symbolic_x, read_e_done, copied = update_oracle(oracle, graph, symbolic_pointer, episode["goal_key"], symbolic_x, read_e_done, copied)
        if emitted or (stop_after_copy and action == COPY_E_R):
            break
    final_value = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    result_target = x0 if goal == FETCH else reference
    result_success = emitted and final_value == result_target and events[-1]["result_correct"]
    return {"episode": episode["episode"], "graph": episode["graph"], "hops": episode["distance"], "goal": goal, "reference": reference, "x0": x0, "events": events, "decisions": len(events), "timeout": not emitted and not stop_after_copy, "emitted": emitted, "final_value": final_value, "result_target": result_target, "final_success": result_success, "trajectory_success": emitted and first_bad is None and result_success and events[-1]["action"] == EMIT, "first_bad": first_bad, "state": state, "flags": (v_e, v_r), "symbolic_pointer": symbolic_pointer, "symbolic_x": symbolic_x, "read_e_done": read_e_done, "copied": copied, "alu_steps": alu_steps}


def trace_metrics(root: Path, supervisor: GoalConditionedSupervisor) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for goal in GOAL_NAMES:
        observations = torch.load(root / "test" / "observations.pt", weights_only=False)
        labels = torch.load(root / "test" / "labels.pt", weights_only=False)
        mask = labels["goal"][:, GOAL_IDS[goal]] == 1
        with torch.no_grad():
            predictions = supervisor(observations["features"][mask], labels["goal"][mask]).argmax(-1)
        correct = predictions == labels["action"][mask]
        output[goal] = {"samples": int(mask.sum()), "correct": int(correct.sum()), "accuracy": float(correct.float().mean())}
    return output


def run_controls(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: GoalConditionedSupervisor, manifest: dict[str, Any], episode_by_x: dict[int, dict[str, Any]]) -> dict[str, Any]:
    episode = episode_by_x[10]
    prefix = run_learned(model, ctrl1, scorer, supervisor, manifest, episode, 12, FETCH_AND_ADJUST, stop_after_copy=True)
    fetch_cont = run_learned(model, ctrl1, scorer, supervisor, manifest, episode, 12, FETCH, initial_state=prefix["state"], initial_flags=prefix["flags"], initial_pointer=prefix["symbolic_pointer"], initial_symbolic_x=prefix["symbolic_x"], initial_read_e_done=prefix["read_e_done"], initial_copied=prefix["copied"])
    adjust_cont = run_learned(model, ctrl1, scorer, supervisor, manifest, episode, 12, FETCH_AND_ADJUST, initial_state=prefix["state"], initial_flags=prefix["flags"], initial_pointer=prefix["symbolic_pointer"], initial_symbolic_x=prefix["symbolic_x"], initial_read_e_done=prefix["read_e_done"], initial_copied=prefix["copied"])
    switch = {"prefix": {key: value for key, value in prefix.items() if key != "state"}, "fetch_continuation": {key: value for key, value in fetch_cont.items() if key != "state"}, "adjust_continuation": {key: value for key, value in adjust_cont.items() if key != "state"}, "pass": prefix["first_bad"] is None and fetch_cont["trajectory_success"] and adjust_cont["trajectory_success"] and fetch_cont["final_value"] == 10 and fetch_cont["alu_steps"] == 0 and adjust_cont["final_value"] == 12 and adjust_cont["alu_steps"] > 0}
    fetch_refs = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode, reference, FETCH) for reference in (12, 5)]
    fetch_b = {"references": [12, 5], "runs": [{key: value for key, value in run.items() if key != "state"} for run in fetch_refs], "pass": all(run["trajectory_success"] and run["final_value"] == 10 and run["alu_steps"] == 0 for run in fetch_refs)}
    adjust_refs = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode, reference, FETCH_AND_ADJUST) for reference in (12, 5)]
    adjust_b = {"references": [12, 5], "runs": [{key: value for key, value in run.items() if key != "state"} for run in adjust_refs], "pass": all(run["trajectory_success"] and run["final_value"] == run["reference"] for run in adjust_refs)}
    return {"change_goal_after_copy": switch, "vary_reference_under_fetch": fetch_b, "vary_reference_under_adjust": adjust_b}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--supervisor-checkpoint", type=Path, default=SUPERVISOR_CHECKPOINT); parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); manifests = __import__("ctrl2_common").load_base_manifests(); manifest = manifests["test"]; model = load_executor(); ctrl1 = load_ctrl1()
    scorer_payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    payload = torch.load(args.supervisor_checkpoint, map_location="cpu", weights_only=False); supervisor = GoalConditionedSupervisor(); supervisor.load_state_dict(payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]:
        episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    trajectories = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], reference, goal) for goal in GOAL_NAMES for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    serializable = [{key: value for key, value in trajectory.items() if key != "state"} for trajectory in trajectories]
    oracle_summary = json.loads((CAMPAIGN_ROOT / "u0c_ctrl5_preflight_seed2201" / "results.json").read_text(encoding="utf-8"))["summary"]
    goal_summaries = {goal: {"samples": sum(item["goal"] == goal for item in trajectories), "trajectory_success": sum(item["goal"] == goal and item["trajectory_success"] for item in trajectories), "final_success": sum(item["goal"] == goal and item["final_success"] for item in trajectories), "timeouts": sum(item["goal"] == goal and item["timeout"] for item in trajectories), "first_bad_count": sum(item["goal"] == goal and item["first_bad"] is not None for item in trajectories), "max_decisions": max(item["decisions"] for item in trajectories if item["goal"] == goal)} for goal in GOAL_NAMES}
    controls = run_controls(model, ctrl1, scorer, supervisor, manifest, episode_by_x)
    result = {"status": "completed", "task": "T1-CTRL-5", "training": {"seed": payload.get("seed"), "updates": payload.get("updates"), "parameters": payload.get("trainable_parameters")}, "checkpoint_supervisor": {"path": str(args.supervisor_checkpoint), "sha256": sha256(args.supervisor_checkpoint)}, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint)}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)}, "measurements": {"oracle_executor": oracle_summary, "supervisor_reference_traces": trace_metrics(PILOT_ROOT, supervisor), "supervisor_free_execution": {"samples": len(trajectories), "trajectory_success": sum(item["trajectory_success"] for item in trajectories), "final_success": sum(item["final_success"] for item in trajectories), "timeouts": sum(item["timeout"] for item in trajectories), "first_bad_count": sum(item["first_bad"] is not None for item in trajectories), "max_decisions": max(item["decisions"] for item in trajectories), "by_goal": goal_summaries, "adjust_ctrl4_1024": {"samples": 1024, "trajectory_success": goal_summaries[FETCH_AND_ADJUST]["trajectory_success"], "pass": goal_summaries[FETCH_AND_ADJUST]["trajectory_success"] == 1024}}}, "controls": controls, "trajectories": serializable}
    result["status"] = "passed" if result["measurements"]["supervisor_free_execution"]["trajectory_success"] == 2048 and all(value["pass"] for value in controls.values()) else "failed"
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "measurements": result["measurements"], "controls": {key: value["pass"] for key, value in controls.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
