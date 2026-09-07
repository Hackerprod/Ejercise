"""T1-CTRL-3 frozen evaluation on recovered real R states and causal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, DECREASE, INCREASE, KEEP, adjustment_action, load_base_manifests, load_ctrl1, load_executor, navigate_collect
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import MAX_ADJUSTMENT_DECISIONS, dispatch_adjustment_iterative, scorer_detail, validate_transition
from train_u0c_ctrl1 import SLOT_R
from train_u0c_ctrl2 import generate_dataset
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201"
MANIFEST_PATH = OUTPUT_ROOT / "real_r_manifest.json"


def raw_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def load_fixed_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load, never regenerate, the approved real-R manifest."""
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"fixed real-R manifest hash mismatch: expected {expected_sha256}, got {actual}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if len(manifest.get("entries", [])) != 236:
        raise RuntimeError("fixed real-R manifest must contain exactly 236 entries")
    if not manifest.get("runtime_recovered") or not manifest.get("embedding_ideal_not_used_for_R"):
        raise RuntimeError("fixed manifest is not marked as recovered real-R data")
    return manifest


@torch.no_grad()
def run_loop(model: Any, scorer: OrdinalSharedScorer, navigation: dict[str, Any], reference: int, *, action_selector: Callable[[dict[str, Any]], int] | None = None, max_decisions: int = MAX_ADJUSTMENT_DECISIONS) -> dict[str, Any]:
    state = navigation["state"].clone()
    events: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    emitted = False
    for decision in range(max_decisions):
        before = state[:, SLOT_R].clone()
        q_r, q_index = canonical_value_view(model, before)
        reference_raw = model.token_embedding(torch.tensor([VALUE_BASE + reference]))
        q_b, b_index = canonical_value_view(model, reference_raw)
        decoded_before = int(q_index.item())
        expected = adjustment_action(decoded_before, reference)
        detail = scorer_detail(scorer, q_r, q_b, expected)
        selected = int(action_selector(detail)) if action_selector is not None else detail["predicted_action"]
        next_state, operation, emitted = dispatch_adjustment_iterative(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], state, navigation["presence"], selected)
        _, after_index = canonical_value_view(model, next_state[:, SLOT_R])
        decoded_after = int(after_index.item())
        transition = validate_transition(x_before=decoded_before, x_after=decoded_after, reference=reference, selected_action=selected, expected_action=expected, emitted=emitted, raw_before_sha256=raw_hash(before), raw_after_sha256=raw_hash(next_state[:, SLOT_R]))
        transition_ok = transition["valid"]
        if not transition_ok and first_bad is None:
            first_bad = {"decision": decision, "decoded_before": decoded_before, "decoded_after": decoded_after, "reference": reference, "predicted_action": ADJUSTMENT_NAMES[selected], "expected_action": ADJUSTMENT_NAMES[expected], **transition, "emitted": emitted}
        events.append({"decision": decision, "decoded_before": decoded_before, "decoded_after": decoded_after, "reference": reference, "raw_r_sha256": raw_hash(before), "raw_after_sha256": raw_hash(next_state[:, SLOT_R]), "q_r_value": int(q_index.item()), "q_reference_value": int(b_index.item()), "selected_action": selected, "selected_action_name": ADJUSTMENT_NAMES[selected], "operation": operation, "emitted": emitted, **transition, "transition_ok": transition_ok, **detail})
        state = next_state
        if emitted:
            break
    _, final_index = canonical_value_view(model, state[:, SLOT_R])
    final_value = int(final_index.item())
    return {"events": events, "decisions": len(events), "timeout": not emitted, "emitted": emitted, "final_value": final_value, "final_success": final_value == reference, "trajectory_success": emitted and first_bad is None and final_value == reference and events[-1]["selected_action"] == 1, "first_bad_transition": first_bad}


@torch.no_grad()
def policy_prefix_step(model: Any, scorer: OrdinalSharedScorer, navigation: dict[str, Any], reference: int) -> tuple[dict[str, Any], torch.Tensor]:
    """Execute one prefix decision selected by the scorer, returning its state and receipt."""
    state = navigation["state"].clone()
    before = state[:, SLOT_R].clone()
    q_r, q_index = canonical_value_view(model, before)
    reference_raw = model.token_embedding(torch.tensor([VALUE_BASE + reference]))
    q_b, b_index = canonical_value_view(model, reference_raw)
    expected = adjustment_action(int(q_index.item()), reference)
    detail = scorer_detail(scorer, q_r, q_b, expected)
    selected = detail["predicted_action"]
    next_state, operation, emitted = dispatch_adjustment_iterative(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], state, navigation["presence"], selected)
    _, after_index = canonical_value_view(model, next_state[:, SLOT_R])
    before_value = int(q_index.item())
    after_value = int(after_index.item())
    transition = validate_transition(x_before=before_value, x_after=after_value, reference=reference, selected_action=selected, expected_action=expected, emitted=emitted, raw_before_sha256=raw_hash(before), raw_after_sha256=raw_hash(next_state[:, SLOT_R]))
    event = {"decision": 0, "decoded_before": before_value, "decoded_after": after_value, "reference": reference, "selected_action": selected, "selected_action_name": ADJUSTMENT_NAMES[selected], "operation": operation, "emitted": emitted, **transition, **detail}
    return event, next_state


def real_r_evaluation(model: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], fixed: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for entry in fixed["entries"]:
        navigation = runtime[entry["episode"]]
        initial_raw = navigation["state"][:, SLOT_R].clone()
        if raw_hash(initial_raw) != entry["raw_r_sha256"]:
            raise RuntimeError(f"runtime R hash changed for manifest pair {entry['pair']}")
        predicted = run_loop(model, scorer, navigation, entry["reference"])
        oracle = run_loop(model, scorer, navigation, entry["reference"], action_selector=lambda detail: detail["expected_action"])
        records.append({**entry, "initial_r_recovered": True, "initial_r_sha256": raw_hash(initial_raw), "predicted": predicted, "oracle": oracle, "oracle_same_initial_r_sha256": raw_hash(initial_raw) == entry["raw_r_sha256"]})
    summary = {"samples": len(records), "trajectory_success": sum(record["predicted"]["trajectory_success"] for record in records), "final_success": sum(record["predicted"]["final_success"] for record in records), "oracle_trajectory_success": sum(record["oracle"]["trajectory_success"] for record in records), "oracle_final_success": sum(record["oracle"]["final_success"] for record in records), "timeouts": sum(record["predicted"]["timeout"] for record in records), "first_bad_transition_count": sum(record["predicted"]["first_bad_transition"] is not None for record in records), "oracle_same_initial_state": all(record["oracle_same_initial_r_sha256"] for record in records), "max_decisions": max(record["predicted"]["decisions"] for record in records)}
    return summary, records


def causal_controls(model: Any, scorer: OrdinalSharedScorer, runtime: dict[int, dict[str, Any]], fixed: dict[str, Any]) -> dict[str, Any]:
    base = next(entry for entry in fixed["entries"] if entry["x"] == 10 and entry["reference"] == 12)
    donor = next(entry for entry in fixed["entries"] if entry["x"] == 13 and entry["reference"] == 13)
    base_nav = runtime[base["episode"]]
    donor_nav = runtime[donor["episode"]]
    prefix_event, first_state = policy_prefix_step(model, scorer, base_nav, 12)
    intervened = first_state.clone()
    intervened[:, SLOT_R] = donor_nav["state"][:, SLOT_R].clone()
    intervention_navigation = {**base_nav, "state": intervened}
    intervention = run_loop(model, scorer, intervention_navigation, 12)
    intervention_event = intervention["events"][0]
    direction_control = {"base_x": 10, "reference": 12, "prefix": prefix_event, "post_alu_raw_r_sha256": raw_hash(first_state[:, SLOT_R]), "intervention_source_x": 13, "intervened_raw_r_sha256": raw_hash(intervened[:, SLOT_R]), "expected_after_intervention": "DECREASE", "observed_action": intervention_event["selected_action_name"], "action_inverted": intervention_event["selected_action"] == DECREASE, "trajectory": intervention, "pass": prefix_event["valid"] and intervention_event["selected_action"] == DECREASE and intervention["trajectory_success"]}

    switch_base = next(entry for entry in fixed["entries"] if entry["x"] == 10 and entry["reference"] == 12)
    switch_nav = runtime[switch_base["episode"]]
    switch_prefix_event, state_after_first = policy_prefix_step(model, scorer, switch_nav, 12)
    switched_nav = {**switch_nav, "state": state_after_first}
    switched = run_loop(model, scorer, switched_nav, 5)
    switch_event = switched["events"][0]
    reference_control = {"initial_x": 10, "old_reference": 12, "new_reference": 5, "prefix": switch_prefix_event, "state_after_old_reference_step_decoded": int(canonical_value_view(model, state_after_first[:, SLOT_R])[1].item()), "observed_action_after_switch": switch_event["selected_action_name"], "expected_action_after_switch": "DECREASE", "followed_new_reference": switch_event["selected_action"] == DECREASE, "trajectory": switched, "pass": switch_prefix_event["valid"] and switch_event["selected_action"] == DECREASE and switched["trajectory_success"]}

    local_first_state, _, _ = dispatch_adjustment_iterative(model, base_nav["memory_keys"], base_nav["memory_values"], base_nav["memory_types"], base_nav["row_mask"], base_nav["state"].clone(), base_nav["presence"], adjustment_action(10, 12))
    local_intervened = local_first_state.clone(); local_intervened[:, SLOT_R] = donor_nav["state"][:, SLOT_R].clone()
    local_direction = run_loop(model, scorer, {**base_nav, "state": local_intervened}, 12, max_decisions=1)
    local_switch_state, _, _ = dispatch_adjustment_iterative(model, base_nav["memory_keys"], base_nav["memory_values"], base_nav["memory_types"], base_nav["row_mask"], base_nav["state"].clone(), base_nav["presence"], adjustment_action(10, 12))
    local_switch = run_loop(model, scorer, {**base_nav, "state": local_switch_state}, 5)
    return {"r_post_alu_crosses_reference": direction_control, "reference_switch_mid_trajectory": reference_control, "controls_local_oracle_prefix": {"label": "local-only; oracle-directed prefix retained for historical comparison", "r_post_alu_crosses_reference": local_direction, "reference_switch_mid_trajectory": local_switch}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--manifest-sha256", default="d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    model = load_executor()
    ctrl1 = load_ctrl1()
    _, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    fixed = load_fixed_manifest(args.manifest, args.manifest_sha256)
    payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval()
    summary, records = real_r_evaluation(model, scorer, manifests["test"], runtime, fixed)
    controls = causal_controls(model, scorer, runtime, fixed)
    result = {"status": "completed", "task": "T1-CTRL-3", "phase": "real_r_and_causal_controls", "training": False, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint), "training_seed": payload.get("controller_seed")}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "samples": len(fixed["entries"]), "reused_existing": True}, "protocol": {"max_adjustment_decisions": MAX_ADJUSTMENT_DECISIONS, "q_for_scorer_only": True, "raw_r_for_alu": True, "reference_not_reinjected": True, "distance_not_used_to_direct_execution": True, "decoded_integers_used_for_evaluation_only": True, "oracle_uses_same_initial_real_state": True, "oracle_repairs_state": False, "dispatch_adjustment_unchanged": True}, "summary": summary, "controls": controls, "records": records}
    output = args.output_root / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "manifest_sha256": sha256(args.manifest), "summary": summary, "controls": {key: {name: value for name, value in control.items() if name not in ("trajectory",)} for key, control in controls.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
