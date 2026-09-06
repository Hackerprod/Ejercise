"""Shared CTRL-2 data, navigation, and arithmetic dispatch helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from evaluate_u0c_c1_e_r_alu import DIMENSION, VALUE_BASE, VALUE_COUNT, C1JointModel, load_approved_model
from train_u0a import immediate_vectors
from train_u0c_ctrl1 import ADVANCE, COLLECT, DECISION_LIMIT, DISTANCES, Ctrl1MLP, materialize_graph_batch, pointer_decoder_id, trace_success, update_causal_accounting
from t1_trainability.unified import OPCODE_IDS, ROW_PAIR, ROW_REL, SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R


INCREASE = 0
KEEP = 1
DECREASE = 2
ADJUSTMENT_NAMES = ("INCREASE", "KEEP", "DECREASE")
CTRL1_PILOT = Path(__file__).resolve().parents[1] / "campaign" / "u0c_ctrl1_pilot_seed101_frozen"
CTRL1_CHECKPOINT = CTRL1_PILOT / "selected.pt"
BASE_CHECKPOINT = Path(__file__).resolve().parents[1] / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt"


def load_ctrl1() -> Ctrl1MLP:
    payload = torch.load(CTRL1_CHECKPOINT, map_location="cpu", weights_only=False)
    controller = Ctrl1MLP()
    controller.load_state_dict(payload["controller"], strict=True)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def load_executor() -> C1JointModel:
    model = load_approved_model()
    payload = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_base_manifests() -> dict[str, dict[str, Any]]:
    return {split: json.loads((CTRL1_PILOT / f"{split}_manifest.json").read_text(encoding="utf-8")) for split in ("train", "val", "test")}


def pair_value(manifest: dict[str, Any], episode: dict[str, Any]) -> int:
    graph = manifest["graphs"][episode["graph"]]
    return next(row["value"] for row in graph["rows"] if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])


def adjustment_action(x: int, reference: int) -> int:
    if x < reference:
        return INCREASE
    if x > reference:
        return DECREASE
    return KEEP


def adjusted_target(x: int, reference: int) -> int:
    action = adjustment_action(x, reference)
    if action == INCREASE:
        return x + 1
    if action == DECREASE:
        return x - 1
    return x


def reference_values(x: int) -> list[int]:
    # Anchor references create inverse pairs: same b, different recovered x.
    values = {0, 8, 16, 20, 24, 31, x}
    if x > 0:
        values.add(x - 1)
    if x < VALUE_COUNT - 1:
        values.add(x + 1)
    return sorted(values)


def reference_curriculum_metadata() -> dict[str, Any]:
    return {"anchors": [0, 8, 16, 20, 24, 31], "include_current_x": True, "include_adjacent_valid": True, "admitted_pair_count": sum(len(reference_values(x)) for x in range(VALUE_COUNT)), "admitted_pairs": [{"x": x, "references": reference_values(x)} for x in range(VALUE_COUNT)]}


def build_examples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    example_id = 0
    for episode in manifest["episodes"]:
        x = pair_value(manifest, episode)
        for reference in reference_values(x):
            action = adjustment_action(x, reference)
            examples.append({**episode, "example": example_id, "x": x, "reference": reference, "action": action, "action_name": ADJUSTMENT_NAMES[action], "target_value": adjusted_target(x, reference), "target_id": VALUE_BASE + adjusted_target(x, reference)})
            example_id += 1
    return examples


def augment_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    augmented = copy.deepcopy(manifest)
    augmented["ctrl2_examples"] = build_examples(manifest)
    augmented["reference_curriculum"] = reference_curriculum_metadata()
    return augmented


def decode_value(model: C1JointModel, state: Tensor) -> int:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    return int(class_ids[model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids)).argmax(-1)].item()) - VALUE_BASE


def update_adjustment_accounting(accounting: dict[str, Any], *, decision: int, action: int, expected_action: int, execution_ok: bool | None) -> dict[str, Any]:
    aligned_before = bool(accounting["aligned"])
    action_correct = action == expected_action
    control_error = aligned_before and not action_correct
    execution_error = aligned_before and action_correct and execution_ok is False
    if control_error and accounting["first_control_error"] is None:
        accounting["first_control_error"] = {"stage": "CTRL-2", "decision": decision, "predicted": ADJUSTMENT_NAMES[action], "expected": ADJUSTMENT_NAMES[expected_action]}
        accounting["aligned"] = False
    elif execution_error and accounting["first_execution_error"] is None:
        accounting["first_execution_error"] = {"stage": "CTRL-2", "decision": decision, "instruction": "adjustment_step"}
        accounting["aligned"] = False
    return {"aligned_before": aligned_before, "action_correct": action_correct if aligned_before else None, "control_error": control_error, "execution_error": execution_error if aligned_before else False}


@torch.no_grad()
def dispatch_adjustment(model: C1JointModel, memory_keys: Tensor, memory_values: Tensor, memory_types: Tensor, row_mask: Tensor, state: Tensor, presence: Tensor, action: int) -> tuple[Tensor, str]:
    zero = immediate_vectors(model, torch.tensor([511]))
    if action == INCREASE:
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["ALU_ADD"]]), immediate_vectors(model, torch.tensor([VALUE_BASE + 1])), torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        operation = "ALU_ADD(1)"
    elif action == DECREASE:
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["ALU_SUB"]]), immediate_vectors(model, torch.tensor([VALUE_BASE + 1])), torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        operation = "ALU_SUB(1)"
    else:
        operation = "KEEP"
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
    return state, operation


@torch.no_grad()
def navigate_collect(model: C1JointModel, controller: Ctrl1MLP, manifest: dict[str, Any], episode: dict[str, Any], *, trace: bool = True) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"]]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    zero = immediate_vectors(model, torch.tensor([511]))
    goal = model.token_embedding(torch.tensor([episode["goal_key"]]))
    symbolic_pointer = episode["start_key"]
    accounting: dict[str, Any] = {"aligned": True, "first_control_error": None, "first_execution_error": None}
    events: list[dict[str, Any]] = []
    collected = False
    for decision in range(DECISION_LIMIT):
        action = int(controller(state[:, SLOT_P], goal).argmax(-1).item())
        expected_action = COLLECT if symbolic_pointer == episode["goal_key"] else ADVANCE
        decision_event = update_causal_accounting(accounting, decision=decision, action=action, expected_action=expected_action, execution_ok=None)
        entry: dict[str, Any] = {"decision": decision, "stage": "CTRL-1", "predicted_action": ("ADVANCE", "COLLECT")[action], "expected_action": ("ADVANCE", "COLLECT")[expected_action], "action_correct": decision_event["action_correct"], "trajectory_aligned_before": decision_event["aligned_before"], "pointer_decoder_id": int(pointer_decoder_id(model, state).item()), "goal_key": int(episode["goal_key"])}
        if action == ADVANCE:
            expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == symbolic_pointer)
            state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode="BLEND", read_set="explicit")
            selected = int(read.selected_index.item())
            row_ok = selected == expected_row
            pointer_ok = int(pointer_decoder_id(model, state).item()) == graph["rows"][expected_row]["value"]
            event = update_causal_accounting(accounting, decision=decision, action=action, expected_action=expected_action, execution_ok=row_ok and pointer_ok)
            if accounting["first_execution_error"] is not None and accounting["first_execution_error"].get("decision") == decision:
                accounting["first_execution_error"].update({"stage": "CTRL-1", "instruction": "READ_P", "selected_row": selected, "expected_row": expected_row})
            entry.update({"instruction": "READ_P", "selected_row": selected, "expected_row": expected_row, "row_match": row_ok, "pointer_ok": pointer_ok, "action_correct": event["action_correct"], "execution_error": event["execution_error"], "trajectory_aligned_after": accounting["aligned"]})
            symbolic_pointer = graph["rows"][expected_row]["value"]
        else:
            expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])
            state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
            selected = int(read.selected_index.item())
            state[:, SLOT_R] = state[:, SLOT_E].clone()
            row_ok = selected == expected_row
            event = update_causal_accounting(accounting, decision=decision, action=action, expected_action=expected_action, execution_ok=row_ok)
            if accounting["first_execution_error"] is not None and accounting["first_execution_error"].get("decision") == decision:
                accounting["first_execution_error"].update({"stage": "CTRL-1", "instruction": "READ_E", "selected_row": selected, "expected_row": expected_row})
            entry.update({"instruction": "READ_E→COPY", "selected_row": selected, "expected_row": expected_row, "row_match": row_ok, "r_real": state[:, SLOT_R].squeeze(0).tolist(), "action_correct": event["action_correct"], "execution_error": event["execution_error"], "trajectory_aligned_after": accounting["aligned"]})
            collected = True
        if trace:
            events.append(entry)
        if collected:
            break
    timeout = not collected
    return {"state": state, "memory_keys": memory_keys, "memory_values": memory_values, "memory_types": memory_types, "row_mask": row_mask, "presence": presence, "collected": collected, "timeout": timeout, "first_control_error": accounting["first_control_error"], "first_execution_error": accounting["first_execution_error"], "aligned": accounting["aligned"], "trace": events, "r": state[:, SLOT_R].clone() if collected else None, "symbolic_x": pair_value(manifest, episode)}


def example_groups(manifest: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {episode["episode"]: [example for example in build_examples(manifest) if example["episode"] == episode["episode"]] for episode in manifest["episodes"]}
