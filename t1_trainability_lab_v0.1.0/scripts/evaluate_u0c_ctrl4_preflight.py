"""T1-CTRL-4 unified six-action oracle-supervisor preflight.

This is a runtime refactor check only. No supervisor network is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, INCREASE as CTRL_INCREASE, DECREASE as CTRL_DECREASE, KEEP as CTRL_KEEP, adjustment_action, load_base_manifests, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import dispatch_adjustment_iterative, validate_transition
from train_u0a import immediate_vectors
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, SLOT_W, materialize_graph_batch, pointer_decoder_id
from train_u0c_ctrl2_o import OrdinalSharedScorer, action_from_difference, sha256
from t1_trainability.unified import OPCODE_IDS, ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_preflight_seed2201"
MAX_DECISIONS = 38
READ_P, READ_E, COPY_E_R, INCREASE, DECREASE, EMIT = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")


def state_hash(state: torch.Tensor, slot: int) -> str:
    return hashlib.sha256(state[:, slot].detach().cpu().numpy().tobytes()).hexdigest()


def pair_value(manifest: dict[str, Any], episode: dict[str, Any]) -> int:
    graph = manifest["graphs"][episode["graph"]]
    return next(row["value"] for row in graph["rows"] if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])


def load_fixed_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"fixed real-R manifest hash mismatch: expected {expected_sha256}, got {actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if len(value.get("entries", [])) != 236:
        raise RuntimeError("fixed real-R manifest must contain exactly 236 entries")
    return value


@torch.no_grad()
def dispatch_unified_action(model: Any, memory_keys: torch.Tensor, memory_values: torch.Tensor, memory_types: torch.Tensor, row_mask: torch.Tensor, state: torch.Tensor, presence: torch.Tensor, action: int, *, v_e: bool, v_r: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mechanical six-action dispatch; no phase masks or implicit follow-up."""
    zero = immediate_vectors(model, torch.tensor([511]))
    if action == READ_P:
        next_state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode="BLEND", read_set="explicit")
        return next_state, {"operation": "READ_P", "selected_row": int(read.selected_index.item())}
    if action == READ_E:
        next_state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
        return next_state, {"operation": "READ_E", "selected_row": int(read.selected_index.item())}
    if action == COPY_E_R:
        if not v_e:
            return state.clone(), {"operation": "COPY_E_R_REJECTED_UNAVAILABLE_E", "selected_row": -1, "rejected": True}
        next_state = state.clone()
        next_state[:, SLOT_R] = state[:, SLOT_E].clone()
        return next_state, {"operation": "COPY_E_R", "selected_row": -1}
    if action == INCREASE:
        if not v_r:
            return state.clone(), {"operation": "INCREASE_REJECTED_UNAVAILABLE_R", "selected_row": -1, "rejected": True}
        next_state, operation, _ = dispatch_adjustment_iterative(model, memory_keys, memory_values, memory_types, row_mask, state, presence, CTRL_INCREASE)
        return next_state, {"operation": operation, "selected_row": -1}
    if action == DECREASE:
        if not v_r:
            return state.clone(), {"operation": "DECREASE_REJECTED_UNAVAILABLE_R", "selected_row": -1, "rejected": True}
        next_state, operation, _ = dispatch_adjustment_iterative(model, memory_keys, memory_values, memory_types, row_mask, state, presence, CTRL_DECREASE)
        return next_state, {"operation": operation, "selected_row": -1}
    next_state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
    return next_state, {"operation": "EMIT", "selected_row": -1}


def oracle_action(symbolic_pointer: int, goal_key: int, *, read_e_done: bool, copied: bool, symbolic_x: int, reference: int) -> int:
    if symbolic_pointer != goal_key:
        return READ_P
    if not read_e_done:
        return READ_E
    if not copied:
        return COPY_E_R
    if symbolic_x < reference:
        return INCREASE
    if symbolic_x > reference:
        return DECREASE
    return EMIT


def strict_primitive_check(model: Any, manifest: dict[str, Any], episode: dict[str, Any], before: torch.Tensor, after: torch.Tensor, action: int, detail: dict[str, Any], symbolic_pointer: int, symbolic_x: int, reference: int, expected_row: int) -> dict[str, Any]:
    after_value = int(canonical_value_view(model, after[:, SLOT_R])[1].item()) if action in (COPY_E_R, INCREASE, DECREASE, EMIT) else None
    before_value = int(canonical_value_view(model, before[:, SLOT_R])[1].item()) if action in (INCREASE, DECREASE, EMIT) else None
    conservation = {"P": True, "E": True, "R": True, "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])}
    # Explicitly check slots that must not be written for each primitive.
    if action == READ_P:
        conservation.update({"E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])})
    elif action == READ_E:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])})
    elif action == COPY_E_R:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W]), "R_exact": torch.equal(after[:, SLOT_R], after[:, SLOT_E])})
    elif action in (INCREASE, DECREASE):
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])})
    else:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])})
    exact = all(bool(value) for value in conservation.values())
    if action == READ_P:
        graph = manifest["graphs"][episode["graph"]]
        expected_pointer = graph["rows"][expected_row]["value"]
        exact = exact and detail["selected_row"] == expected_row and int(pointer_decoder_id(model, after).item()) == expected_pointer
    elif action == READ_E:
        graph = manifest["graphs"][episode["graph"]]
        expected_value = next(row["value"] for row in graph["rows"] if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])
        exact = exact and detail["selected_row"] == expected_row and int(canonical_value_view(model, after[:, SLOT_E])[1].item()) == expected_value
    elif action == COPY_E_R:
        exact = exact and bool(detail.get("available_e", False))
    elif action in (INCREASE, DECREASE):
        expected_action = adjustment_action(before_value, reference)
        ctrl_action = CTRL_INCREASE if action == INCREASE else CTRL_DECREASE
        transition = validate_transition(x_before=before_value, x_after=after_value, reference=reference, selected_action=ctrl_action, expected_action=expected_action, emitted=False, raw_before_sha256=state_hash(before, SLOT_R), raw_after_sha256=state_hash(after, SLOT_R))
        exact = exact and transition["valid"]
    elif action == EMIT:
        expected_action = EMIT
        exact = exact and action == expected_action and before_value == reference and after_value == before_value and state_hash(before, SLOT_R) == state_hash(after, SLOT_R)
    return {"exact": exact, "conservation": conservation, "decoded_before": before_value, "decoded_after": after_value}


@torch.no_grad()
def run_preflight(model: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], episode: dict[str, Any], reference: int) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    symbolic_pointer = episode["start_key"]
    symbolic_x = pair_value(manifest, episode)
    read_e_done = False
    copied = False
    v_e = False
    v_r = False
    events: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    emitted = False
    for decision in range(MAX_DECISIONS):
        expected = oracle_action(symbolic_pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, symbolic_x=symbolic_x, reference=reference)
        before = state.clone()
        expected_row = -1
        if expected == READ_P:
            expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == symbolic_pointer)
        elif expected == READ_E:
            expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])
        # All six actions remain available; oracle selection is the only source of dispatch.
        action = expected
        if action in (INCREASE, DECREASE, EMIT) and not v_r:
            detail = {"available_e": v_e, "available_r": v_r}
        else:
            detail = {"available_e": v_e, "available_r": v_r}
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        if action == READ_P:
            symbolic_pointer = graph["rows"][expected_row]["value"]
            v_e = False
        elif action == READ_E:
            read_e_done = True
            v_e = True
        elif action == COPY_E_R:
            copied = True
            v_r = True
        elif action in (INCREASE, DECREASE):
            v_r = True
            symbolic_x += 1 if action == INCREASE else -1
        elif action == EMIT:
            emitted = True
        check = strict_primitive_check(model, manifest, episode, before, state, action, operation | detail, symbolic_pointer, symbolic_x, reference, expected_row)
        event = {"decision": decision, "action": action, "action_name": ACTION_NAMES[action], "expected_action": expected, "expected_action_name": ACTION_NAMES[expected], "action_correct": action == expected, "v_E_before": detail["available_e"], "v_R_before": detail["available_r"], "v_E_after": v_e, "v_R_after": v_r, "operation": operation["operation"], "selected_row": operation["selected_row"], "p_sha256": state_hash(before, SLOT_P), "e_sha256": state_hash(before, SLOT_E), "r_sha256": state_hash(before, SLOT_R), "w_sha256": state_hash(before, SLOT_W), "p_after_sha256": state_hash(state, SLOT_P), "e_after_sha256": state_hash(state, SLOT_E), "r_after_sha256": state_hash(state, SLOT_R), "w_after_sha256": state_hash(state, SLOT_W), **check}
        events.append(event)
        if not check["exact"] and first_bad is None:
            first_bad = event
        if emitted:
            break
    final_value = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"episode": episode["episode"], "graph": episode["graph"], "hops": episode["distance"], "reference": reference, "symbolic_x": pair_value(manifest, episode), "events": events, "decisions": len(events), "timeout": not emitted, "emitted": emitted, "final_value": final_value, "final_success": final_value == reference, "trajectory_success": emitted and first_bad is None and final_value == reference and events[-1]["action"] == EMIT, "first_bad": first_bad}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--manifest-sha256", default="d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = load_base_manifests()["test"]
    fixed = load_fixed_manifest(args.manifest, args.manifest_sha256)
    model = load_executor()
    payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval()
    episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]:
        episode_by_x.setdefault(int(entry["x"]), base["episodes"][entry["episode"]])
    if set(episode_by_x) != set(range(VALUE_COUNT)):
        raise RuntimeError("fixed manifest does not provide one recovered episode for every x")
    trajectories = [run_preflight(model, scorer, base, episode_by_x[x], reference) for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    subset = [item for item in trajectories if item["symbolic_x"] in (0, 10, 20, 31) and item["reference"] in {item["symbolic_x"], max(0, item["symbolic_x"] - 1), min(31, item["symbolic_x"] + 1)}]
    summary = {"samples": len(trajectories), "trajectory_success": sum(item["trajectory_success"] for item in trajectories), "final_success": sum(item["final_success"] for item in trajectories), "timeouts": sum(item["timeout"] for item in trajectories), "first_bad_count": sum(item["first_bad"] is not None for item in trajectories), "max_decisions": max(item["decisions"] for item in trajectories), "by_hops": {str(hops): {"samples": sum(item["hops"] == hops for item in trajectories), "trajectory_success": sum(item["hops"] == hops and item["trajectory_success"] for item in trajectories)} for hops in range(5)}, "real_navigation_subset": {"samples": len(subset), "trajectory_success": sum(item["trajectory_success"] for item in subset)}}
    result = {"status": "completed", "task": "T1-CTRL-4", "phase": "unified_runtime_oracle_preflight", "training": False, "actions": ACTION_NAMES, "budget": MAX_DECISIONS, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint), "training_seed": payload.get("controller_seed")}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "reused_existing": True}, "protocol": {"all_actions_available": True, "phase_masks": False, "read_e_does_not_copy": True, "adjustments_do_not_emit": True, "q_for_scorer_only": True, "runtime_never_repairs_copy": True, "premature_emit_fails": True, "timeout_has_no_rescue_emit": True, "oracle_labels_not_used_as_learned_dispatch": True}, "summary": summary, "trajectories": trajectories}
    output = args.output_root / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
