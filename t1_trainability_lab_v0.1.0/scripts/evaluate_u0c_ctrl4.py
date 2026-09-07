"""T1-CTRL-4 learned supervisor measurements and causal controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, adjustment_action, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import dispatch_adjustment_iterative, validate_transition
from evaluate_u0c_ctrl4_preflight import ACTION_NAMES, COPY_E_R, DECREASE, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, oracle_action, pair_value, state_hash, strict_primitive_check
from train_u0a import immediate_vectors
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, SLOT_W, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl4 import SupervisorMLP, supervisor_features
from t1_trainability.unified import OPCODE_IDS, ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SUPERVISOR_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_evaluation_seed4401"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"


def expected_row(graph: dict[str, Any], pointer: int, action: int, goal_key: int) -> int:
    if action == READ_P:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == pointer)
    if action == READ_E:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == goal_key)
    return -1


def update_oracle(action: int, graph: dict[str, Any], pointer: int, goal_key: int, symbolic_x: int, reference: int, read_e_done: bool, copied: bool) -> tuple[int, int, bool, bool, bool]:
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
    return pointer, symbolic_x, read_e_done, copied, action == EMIT


@torch.no_grad()
def run_learned(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: SupervisorMLP, manifest: dict[str, Any], episode: dict[str, Any], reference: int, *, initial_state: torch.Tensor | None = None, initial_flags: tuple[bool, bool] = (False, False), initial_pointer: int | None = None, initial_symbolic_x: int | None = None, initial_read_e_done: bool = False, initial_copied: bool = False, stop_after_first_alu: bool = False) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = initial_state.clone() if initial_state is not None else torch.zeros((1, SLOT_COUNT, DIMENSION))
    if initial_state is None:
        state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
    symbolic_pointer = episode["start_key"] if initial_pointer is None else initial_pointer
    symbolic_x = pair_value(manifest, episode) if initial_symbolic_x is None else initial_symbolic_x
    read_e_done, copied = initial_read_e_done, initial_copied
    v_e, v_r = initial_flags[0], initial_flags[1]
    events: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    emitted = False
    for decision in range(MAX_DECISIONS):
        oracle = oracle_action(symbolic_pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, symbolic_x=symbolic_x, reference=reference)
        before = state.clone()
        features = supervisor_features(model, ctrl1, scorer, state, goal, reference, v_e=v_e, v_r=v_r)
        logits = supervisor(features)
        action = int(logits.argmax(-1).item())
        expected = expected_row(graph, symbolic_pointer, oracle, episode["goal_key"])
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        # Validate actual selected primitive; no correction follows a bad choice.
        if action in (INCREASE, DECREASE) and not v_r:
            check = {"exact": False, "reason": "ALU selected while R unavailable", "conservation": {"P": True, "E": True, "R": True, "W": torch.equal(before[:, SLOT_W], state[:, SLOT_W])}}
        elif action == EMIT and not v_r:
            check = {"exact": False, "reason": "EMIT selected while R unavailable", "conservation": {"P": torch.equal(before[:, SLOT_P], state[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], state[:, SLOT_E]), "R": torch.equal(before[:, SLOT_R], state[:, SLOT_R]), "W": torch.equal(before[:, SLOT_W], state[:, SLOT_W])}}
        else:
            check = strict_primitive_check(model, manifest, episode, before, state, action, {**operation, "available_e": v_e, "available_r": v_r}, symbolic_pointer, symbolic_x, reference, expected)
        action_correct = action == oracle
        valid = action_correct and bool(check["exact"])
        event = {"decision": decision, "action": action, "action_name": ACTION_NAMES[action], "expected_action": oracle, "expected_action_name": ACTION_NAMES[oracle], "action_correct": action_correct, "operation": operation["operation"], "rejected": operation.get("rejected", False), "v_E_before": v_e, "v_R_before": v_r, "features": features.squeeze(0).tolist(), "selected_row": operation["selected_row"], "p_sha256": state_hash(before, SLOT_P), "e_sha256": state_hash(before, SLOT_E), "r_sha256": state_hash(before, SLOT_R), "w_sha256": state_hash(before, SLOT_W), "p_after_sha256": state_hash(state, SLOT_P), "e_after_sha256": state_hash(state, SLOT_E), "r_after_sha256": state_hash(state, SLOT_R), "w_after_sha256": state_hash(state, SLOT_W), **check, "valid": valid}
        events.append(event)
        if not valid and first_bad is None:
            first_bad = {"category": "supervisor_choice" if not action_correct else "component_execution", **event}
        # Availability follows actual operations, never expected semantics.
        if action == READ_P:
            v_e = False
        elif action == READ_E and not operation.get("rejected", False):
            v_e = True
        elif action == COPY_E_R and not operation.get("rejected", False):
            v_r = True
        elif action in (INCREASE, DECREASE) and not operation.get("rejected", False):
            v_r = True
        elif action == EMIT:
            emitted = True
        # Oracle state advances independently to label future choices.
        symbolic_pointer, symbolic_x, read_e_done, copied, _ = update_oracle(oracle, graph, symbolic_pointer, episode["goal_key"], symbolic_x, reference, read_e_done, copied)
        if emitted:
            break
        if stop_after_first_alu and action in (INCREASE, DECREASE):
            break
    final_value = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"episode": episode["episode"], "graph": episode["graph"], "hops": episode["distance"], "reference": reference, "events": events, "decisions": len(events), "timeout": not emitted and not stop_after_first_alu, "emitted": emitted, "final_value": final_value, "final_success": final_value == reference, "trajectory_success": emitted and first_bad is None and final_value == reference and events[-1]["action"] == EMIT, "first_bad": first_bad, "state": state, "flags": (v_e, v_r), "symbolic_pointer": symbolic_pointer, "symbolic_x": symbolic_x, "read_e_done": read_e_done, "copied": copied}


def trace_metrics(root: Path, supervisor_checkpoint: Path) -> dict[str, Any]:
    observations = torch.load(root / "test" / "observations.pt", weights_only=False)
    labels = torch.load(root / "test" / "labels.pt", weights_only=False)
    model = SupervisorMLP(); payload = torch.load(supervisor_checkpoint, weights_only=False); model.load_state_dict(payload["supervisor"], strict=True); model.eval()
    with torch.no_grad():
        predictions = model(observations["features"]).argmax(-1)
    correct = predictions == labels["action"]
    per_action = {ACTION_NAMES[action]: {"samples": int((labels["action"] == action).sum()), "correct": int(correct[labels["action"] == action].sum()), "accuracy": float(correct[labels["action"] == action].float().mean())} for action in range(6)}
    return {"samples": len(labels["action"]), "correct": int(correct.sum()), "accuracy": float(correct.float().mean()), "per_action": per_action}


def run_controls(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, supervisor: SupervisorMLP, manifest: dict[str, Any], episode_by_x: dict[int, dict[str, Any]]) -> dict[str, Any]:
    first_episode = episode_by_x[10]
    switch_prefix = run_learned(model, ctrl1, scorer, supervisor, manifest, first_episode, 12, stop_after_first_alu=True)
    switch_cont = run_learned(model, ctrl1, scorer, supervisor, manifest, first_episode, 5, initial_state=switch_prefix["state"], initial_flags=switch_prefix["flags"], initial_pointer=switch_prefix["symbolic_pointer"], initial_symbolic_x=switch_prefix["symbolic_x"], initial_read_e_done=switch_prefix["read_e_done"], initial_copied=switch_prefix["copied"])
    controls: dict[str, Any] = {"reference_switch_mid_adjustment": {"prefix": {key: value for key, value in switch_prefix.items() if key != "state"}, "continuation": {key: value for key, value in switch_cont.items() if key != "state"}, "pass": switch_prefix["first_bad"] is None and switch_cont["trajectory_success"]}}
    graph = manifest["graphs"][first_episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([first_episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    symbolic_pointer = first_episode["start_key"]
    for _ in range(first_episode["distance"]):
        row = expected_row(graph, symbolic_pointer, READ_P, first_episode["goal_key"])
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), immediate_vectors(model, torch.tensor([511])), torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode="BLEND", read_set="explicit")
        symbolic_pointer = graph["rows"][row]["value"]
    state_after_read_e, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), immediate_vectors(model, torch.tensor([511])), torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
    read_e = run_learned(model, ctrl1, scorer, supervisor, manifest, first_episode, 12, initial_state=state_after_read_e, initial_flags=(True, False), initial_pointer=first_episode["goal_key"], initial_symbolic_x=pair_value(manifest, first_episode), initial_read_e_done=True, initial_copied=False)
    state_after_copy = state_after_read_e.clone(); state_after_copy[:, SLOT_R] = state_after_copy[:, SLOT_E].clone()
    copy = run_learned(model, ctrl1, scorer, supervisor, manifest, first_episode, 12, initial_state=state_after_copy, initial_flags=(True, True), initial_pointer=first_episode["goal_key"], initial_symbolic_x=pair_value(manifest, first_episode), initial_read_e_done=True, initial_copied=True)
    controls["snapshot_after_read_e"] = {**{key: value for key, value in read_e.items() if key != "state"}, "pass": read_e["trajectory_success"]}
    controls["snapshot_after_copy"] = {**{key: value for key, value in copy.items() if key != "state"}, "pass": copy["trajectory_success"]}
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--supervisor-checkpoint", type=Path, default=SUPERVISOR_CHECKPOINT)
    parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_manifest(args.manifest, args.manifest_sha256)
    manifests = load_base_manifests(); manifest = manifests["test"]
    model = load_executor(); ctrl1 = load_ctrl1()
    scorer_payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    supervisor_payload = torch.load(args.supervisor_checkpoint, map_location="cpu", weights_only=False); supervisor = SupervisorMLP(); supervisor.load_state_dict(supervisor_payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]:
        episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    trajectories = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], reference) for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    # Serializable copies omit transient tensors while retaining complete traces.
    serializable = [{key: value for key, value in trajectory.items() if key not in ("state",)} for trajectory in trajectories]
    oracle_preflight_path = CAMPAIGN_ROOT / "u0c_ctrl4_preflight_seed2201" / "results.json"
    oracle_summary = json.loads(oracle_preflight_path.read_text(encoding="utf-8"))["summary"]
    controls = run_controls(model, ctrl1, scorer, supervisor, manifest, episode_by_x)
    result = {"status": "completed", "task": "T1-CTRL-4", "training": False, "checkpoint_supervisor": {"path": str(args.supervisor_checkpoint), "sha256": sha256(args.supervisor_checkpoint), "seed": supervisor_payload.get("seed")}, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint)}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "reused_existing": True}, "measurements": {"oracle_supervisor_executor": oracle_summary, "learned_on_reference_traces": trace_metrics(args.supervisor_checkpoint.parent, args.supervisor_checkpoint), "learned_free_execution": {"samples": len(trajectories), "trajectory_success": sum(t["trajectory_success"] for t in trajectories), "final_success": sum(t["final_success"] for t in trajectories), "timeouts": sum(t["timeout"] for t in trajectories), "first_bad_count": sum(t["first_bad"] is not None for t in trajectories), "max_decisions": max(t["decisions"] for t in trajectories)}}, "controls": controls, "trajectories": serializable}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "measurements": result["measurements"], "controls": {key: value.get("pass", value.get("trajectory_success")) for key, value in controls.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
