"""T1-CTRL-3 iterative frozen evaluation, beginning with canonical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, adjustment_action, load_base_manifests, load_ctrl1, load_executor, navigate_collect
from evaluate_u0c_c1_e_r_alu import DIMENSION, VALUE_BASE, VALUE_COUNT
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_R
from train_u0a import immediate_vectors
from train_u0c_ctrl2_o import OrdinalSharedScorer, action_from_difference, sha256
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from t1_trainability.unified import OPCODE_IDS


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl3_canonical_seed2201"
MAX_ADJUSTMENT_DECISIONS = 32
INCREASE, KEEP, DECREASE = 0, 1, 2


@torch.no_grad()
def dispatch_adjustment_iterative(model: Any, memory_keys: torch.Tensor, memory_values: torch.Tensor, memory_types: torch.Tensor, row_mask: torch.Tensor, state: torch.Tensor, presence: torch.Tensor, action: int) -> tuple[torch.Tensor, str, bool]:
    """Execute one ALU adjustment, or KEEP→EMIT; never emits after an adjustment."""
    zero = immediate_vectors(model, torch.tensor([511]))
    if action == INCREASE:
        next_state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["ALU_ADD"]]), immediate_vectors(model, torch.tensor([VALUE_BASE + 1])), torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        return next_state, "ALU_ADD(1)", False
    if action == DECREASE:
        next_state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["ALU_SUB"]]), immediate_vectors(model, torch.tensor([VALUE_BASE + 1])), torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        return next_state, "ALU_SUB(1)", False
    next_state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
    return next_state, "EMIT", True


def scorer_detail(scorer: OrdinalSharedScorer, register: torch.Tensor, reference: torch.Tensor, expected: int) -> dict[str, Any]:
    difference = scorer.difference(register, reference)
    tau = scorer.tau()
    logits = scorer.logits(register, reference).squeeze(0)
    action = int(action_from_difference(difference, tau).item())
    values = [float(value) for value in logits.tolist()]
    return {"scores": {"s_R": float(scorer.score(register).item()), "s_b": float(scorer.score(reference).item())}, "difference": float(difference.item()), "tau": float(tau.item()), "logits": values, "d_order": values[0] - values[2], "d_keep": values[1] - max(values[0], values[2]), "predicted_action": action, "predicted_action_name": ADJUSTMENT_NAMES[action], "expected_action": expected, "expected_action_name": ADJUSTMENT_NAMES[expected], "action_correct": action == expected}


def validate_transition(*, x_before: int, x_after: int, reference: int, selected_action: int, expected_action: int, emitted: bool, raw_before_sha256: str, raw_after_sha256: str) -> dict[str, Any]:
    """Strict evaluator-only transition contract; dispatcher remains policy-driven."""
    action_correct = selected_action == expected_action
    if selected_action == INCREASE:
        expected_value = (x_before + 1) % VALUE_COUNT
        exact_operation = x_after == expected_value and not emitted
    elif selected_action == DECREASE:
        expected_value = (x_before - 1) % VALUE_COUNT
        exact_operation = x_after == expected_value and not emitted
    else:
        expected_value = x_before
        exact_operation = x_after == expected_value and emitted and raw_before_sha256 == raw_after_sha256
    distance_decreased = x_before == reference or abs(x_after - reference) == abs(x_before - reference) - 1
    return {"action_correct": action_correct, "expected_value": expected_value, "exact_operation": exact_operation, "distance_decreased": distance_decreased, "valid": action_correct and exact_operation and distance_decreased}


def canonical_trajectory(model: Any, scorer: OrdinalSharedScorer, base_navigation: dict[str, Any], x: int, reference: int) -> dict[str, Any]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    codebook = model.token_embedding(class_ids)
    reference_raw = codebook[reference:reference + 1]
    state = base_navigation["state"].clone()
    state[:, SLOT_R] = codebook[x:x + 1]
    events: list[dict[str, Any]] = []
    first_bad_transition: dict[str, Any] | None = None
    emitted = False
    for decision in range(MAX_ADJUSTMENT_DECISIONS):
        raw = state[:, SLOT_R].clone()
        q_r, q_index = canonical_value_view(model, raw)
        q_b, b_index = canonical_value_view(model, reference_raw)
        decoded_before = int(q_index.item())
        expected = adjustment_action(decoded_before, reference)
        detail = scorer_detail(scorer, q_r, q_b, expected)
        next_state, operation, emitted = dispatch_adjustment_iterative(model, base_navigation["memory_keys"], base_navigation["memory_values"], base_navigation["memory_types"], base_navigation["row_mask"], state, base_navigation["presence"], detail["predicted_action"])
        _, after_index = canonical_value_view(model, next_state[:, SLOT_R])
        decoded_after = int(after_index.item())
        transition = validate_transition(x_before=decoded_before, x_after=decoded_after, reference=reference, selected_action=detail["predicted_action"], expected_action=expected, emitted=emitted, raw_before_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(), raw_after_sha256=hashlib.sha256(next_state[:, SLOT_R].numpy().tobytes()).hexdigest())
        terminal_ok = transition["valid"]
        if not terminal_ok and first_bad_transition is None:
            first_bad_transition = {"decision": decision, "decoded_before": decoded_before, "decoded_after": decoded_after, "reference": reference, "predicted_action": detail["predicted_action_name"], "expected_action": detail["expected_action_name"], **transition, "emitted": emitted}
        events.append({"decision": decision, "decoded_before": decoded_before, "decoded_after": decoded_after, "reference": reference, "raw_r_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(), "q_r_value": int(q_index.item()), "q_reference_value": int(b_index.item()), "operation": operation, "emitted": emitted, **transition, **detail})
        state = next_state
        if emitted:
            break
    timeout = not emitted
    _, final_index = canonical_value_view(model, state[:, SLOT_R])
    final_value = int(final_index.item())
    return {"x": x, "reference": reference, "events": events, "decisions": len(events), "timeout": timeout, "emitted": emitted, "final_value": final_value, "final_success": final_value == reference, "trajectory_success": not timeout and first_bad_transition is None and final_value == reference and events[-1]["predicted_action"] == KEEP, "first_bad_transition": first_bad_transition, "action_sequence": [event["predicted_action_name"] for event in events]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    model = load_executor()
    ctrl1 = load_ctrl1()
    payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(payload["controller"], strict=True)
    scorer.eval()
    first_episode = manifests["test"]["episodes"][0]
    base_navigation = navigate_collect(model, ctrl1, manifests["test"], first_episode, trace=False)
    if not base_navigation["collected"]:
        raise RuntimeError("CTRL-1 failed to collect base state for canonical CTRL-3 trajectories")
    trajectories = [canonical_trajectory(model, scorer, base_navigation, x, reference) for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    summary = {"samples": len(trajectories), "trajectory_success": sum(item["trajectory_success"] for item in trajectories), "final_success": sum(item["final_success"] for item in trajectories), "timeouts": sum(item["timeout"] for item in trajectories), "first_bad_transition_count": sum(item["first_bad_transition"] is not None for item in trajectories), "max_decisions": max(item["decisions"] for item in trajectories), "by_distance": {str(distance): {"samples": sum(abs(item["x"] - item["reference"]) == distance for item in trajectories), "trajectory_success": sum(abs(item["x"] - item["reference"]) == distance and item["trajectory_success"] for item in trajectories)} for distance in range(VALUE_COUNT)}}
    result = {"status": "completed", "task": "T1-CTRL-3", "phase": "canonical_trajectories", "training": False, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint), "training_seed": payload.get("controller_seed")}, "protocol": {"max_adjustment_decisions": MAX_ADJUSTMENT_DECISIONS, "q_for_scorer_only": True, "raw_r_for_alu": True, "diagnostic_read_e_select": False, "reference_not_reinjected": True, "distance_not_used_to_direct_execution": True, "decoded_integers_used_for_evaluation_only": True, "dispatch_adjustment_unchanged": True}, "summary": summary, "trajectories": trajectories}
    path = args.output_root / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "summary": summary, "checkpoint_ctrl2_training_seed": payload.get("controller_seed")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
