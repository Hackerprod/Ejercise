"""T1-CTRL-6 four-objective constraint-composition preflight.

No supervisor is trained here. The objective indicators reach only the
symbolic oracle; the shared six-action dispatcher receives action and state.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, adjustment_action, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import dispatch_adjustment_iterative, validate_transition
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, load_fixed_manifest, pair_value, state_hash
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, SLOT_W, materialize_graph_batch, pointer_decoder_id
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_preflight_seed2201"

GOALS = {(0, 0): "NONE", (1, 0): "LOWER", (0, 1): "UPPER", (1, 1): "CLAMP"}
ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")


def state_hash_slot(state: torch.Tensor, slot: int) -> str:
    return hashlib.sha256(state[:, slot].detach().cpu().numpy().tobytes()).hexdigest()


def target_value(x0: int, lower: int, upper: int, constraints: tuple[int, int]) -> int:
    c_lower, c_upper = constraints
    value = x0
    if c_lower:
        value = max(value, lower)
    if c_upper:
        value = min(value, upper)
    return value


def oracle_action(pointer: int, goal_key: int, *, read_e_done: bool, copied: bool, value: int, lower: int, upper: int, constraints: tuple[int, int]) -> int:
    if constraints not in GOALS:
        raise ValueError(f"unknown constraint descriptor: {constraints}")
    if pointer != goal_key:
        return READ_P
    if not read_e_done:
        return READ_E
    if not copied:
        return COPY_E_R
    target = target_value(value, lower, upper, constraints)
    if value < target:
        return INCREASE
    if value > target:
        return DECREASE
    return EMIT


@torch.no_grad()
def supervisor_features_ctrl6(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, state: torch.Tensor, goal: torch.Tensor, lower: int, upper: int, *, v_e: bool, v_r: bool) -> torch.Tensor:
    """Return ten features using one shared scorer instance twice."""
    navigation = F.softmax(ctrl1(state[:, SLOT_P], goal), dim=-1)
    if v_r:
        q_r, _ = canonical_value_view(model, state[:, SLOT_R])
        q_lower, _ = canonical_value_view(model, model.token_embedding(torch.tensor([VALUE_BASE + lower])))
        q_upper, _ = canonical_value_view(model, model.token_embedding(torch.tensor([VALUE_BASE + upper])))
        lower_comparison = F.softmax(scorer.logits(q_r, q_lower), dim=-1)
        upper_comparison = F.softmax(scorer.logits(q_r, q_upper), dim=-1)
    else:
        lower_comparison = torch.zeros((state.shape[0], 3))
        upper_comparison = torch.zeros((state.shape[0], 3))
    availability = torch.tensor([[float(v_e), float(v_r)]])
    return torch.cat((navigation, lower_comparison, upper_comparison, availability), dim=-1)


def primitive_check(model: Any, before: torch.Tensor, after: torch.Tensor, action: int, detail: dict[str, Any], expected_row: int, graph: dict[str, Any], pointer: int, target: int) -> bool:
    conservation = {"P": True, "E": True, "R": True, "W": torch.equal(before[:, SLOT_W], after[:, SLOT_W])}
    if action == READ_P:
        expected_pointer = graph["rows"][expected_row]["value"]
        conservation.update({"E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "selected": detail["selected_row"] == expected_row, "decoded": int(pointer_decoder_id(model, after).item()) == expected_pointer})
    elif action == READ_E:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "selected": detail["selected_row"] == expected_row})
    elif action == COPY_E_R:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "R_exact": torch.equal(after[:, SLOT_R], after[:, SLOT_E]), "available": bool(detail.get("available_e", False))})
    elif action in (INCREASE, DECREASE):
        before_value = int(canonical_value_view(model, before[:, SLOT_R])[1].item())
        after_value = int(canonical_value_view(model, after[:, SLOT_R])[1].item())
        expected = adjustment_action(before_value, target)
        selected = 0 if action == INCREASE else 2
        transition = validate_transition(x_before=before_value, x_after=after_value, reference=target, selected_action=selected, expected_action=expected, emitted=False, raw_before_sha256=state_hash_slot(before, SLOT_R), raw_after_sha256=state_hash_slot(after, SLOT_R))
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "transition": transition["valid"], "available": bool(detail.get("available_r", False))})
    elif action == EMIT:
        conservation.update({"P": torch.equal(before[:, SLOT_P], after[:, SLOT_P]), "E": torch.equal(before[:, SLOT_E], after[:, SLOT_E]), "R": torch.equal(before[:, SLOT_R], after[:, SLOT_R]), "available": bool(detail.get("available_r", False))})
    return all(bool(value) for value in conservation.values())


@torch.no_grad()
def run_canonical(model: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], episode: dict[str, Any], x0: int, lower: int, upper: int, constraints: tuple[int, int], memory_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = memory_cache[x0]
    state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); state[:, SLOT_E] = model.token_embedding(torch.tensor([VALUE_BASE + x0])); state[:, SLOT_R] = state[:, SLOT_E].clone()
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
    pointer = episode["goal_key"]; value = x0; target = target_value(x0, lower, upper, constraints); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        action = oracle_action(pointer, episode["goal_key"], read_e_done=True, copied=True, value=value, lower=lower, upper=upper, constraints=constraints)
        before = state.clone(); state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=True, v_r=True)
        check = primitive_check(model, before, state, action, {**operation, "available_e": True, "available_r": True}, -1, graph, pointer, target)
        events.append({"action": action, "action_name": ACTION_NAMES[action], "primitive_correct": check, "r_after": int(canonical_value_view(model, state[:, SLOT_R])[1].item())})
        if not check and first_bad is None: first_bad = decision
        if action == INCREASE: value += 1
        elif action == DECREASE: value -= 1
        elif action == EMIT: emitted = True
        if emitted: break
    final = int(canonical_value_view(model, state[:, SLOT_R])[1].item())
    return {"x0": x0, "lower": lower, "upper": upper, "constraints": list(constraints), "goal": GOALS[constraints], "target": target, "actions": [event["action_name"] for event in events], "decisions": len(events), "alu_steps": sum(event["action"] in (INCREASE, DECREASE) for event in events), "final_value": final, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT, "first_bad": first_bad}


@torch.no_grad()
def run_real(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], episode: dict[str, Any], lower: int, upper: int, constraints: tuple[int, int]) -> dict[str, Any]:
    graph = manifest["graphs"][episode["graph"]]; memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph]); state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE])); presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); x0 = pair_value(manifest, episode); pointer, value = episode["start_key"], x0; read_e_done = copied = v_e = v_r = False; target = target_value(x0, lower, upper, constraints); events: list[dict[str, Any]] = []; first_bad = None; emitted = False
    for decision in range(MAX_DECISIONS):
        action = oracle_action(pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, value=value, lower=lower, upper=upper, constraints=constraints); before = state.clone(); features = supervisor_features_ctrl6(model, ctrl1, scorer, state, navigation_goal, lower, upper, v_e=v_e, v_r=v_r); expected_row = next((index for index, row in enumerate(graph["rows"]) if (action == READ_P and row["kind"] == ROW_REL and row["key"] == pointer) or (action == READ_E and row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])), -1); state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r); check = primitive_check(model, before, state, action, {**operation, "available_e": v_e, "available_r": v_r}, expected_row, graph, pointer, target); events.append({"decision": decision, "action": action, "action_name": ACTION_NAMES[action], "features": features.squeeze(0).tolist(), "primitive_correct": check, "p_sha256": state_hash_slot(before, SLOT_P), "r_after_sha256": state_hash_slot(state, SLOT_R), "rejected": operation.get("rejected", False)}); bad = not check or operation.get("rejected", False)
        if bad and first_bad is None: first_bad = decision
        if action == READ_P: pointer = graph["rows"][expected_row]["value"]; v_e = False
        elif action == READ_E: read_e_done = True; v_e = True
        elif action == COPY_E_R: copied = True; v_r = True
        elif action == INCREASE: value += 1; v_r = True
        elif action == DECREASE: value -= 1; v_r = True
        elif action == EMIT: emitted = True
        if emitted: break
    final = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"episode": episode["episode"], "graph": episode["graph"], "x0": x0, "lower": lower, "upper": upper, "constraints": list(constraints), "goal": GOALS[constraints], "target": target, "events": events, "decisions": len(events), "alu_steps": sum(event["action"] in (INCREASE, DECREASE) for event in events), "final_value": final, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT, "first_bad": first_bad}


def real_cases() -> list[tuple[int, int, int]]:
    cases: list[tuple[int, int, int]] = []
    for index in range(32):
        lower_x = index % 31; cases.append((lower_x, lower_x + 1, 31))
        upper_x = (index % 31) + 1; cases.append((upper_x, 0, upper_x - 1))
        if index % 2 == 0: cases.append((index, index, index))
        else: cases.append((index, max(0, index - 1), min(31, index + 1)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH); parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if "goal" in inspect.signature(dispatch_unified_action).parameters or "constraints" in inspect.signature(dispatch_unified_action).parameters or "lower" in inspect.signature(dispatch_unified_action).parameters or "upper" in inspect.signature(dispatch_unified_action).parameters: raise RuntimeError("dispatcher must remain constraint-agnostic")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(args.manifest, args.manifest_sha256); model = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER_CHECKPOINT, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval()
    episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]: episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    memory_cache = {x: materialize_graph_batch(model, [manifest["graphs"][episode_by_x[x]["graph"]]]) for x in range(32)}
    canonical = [run_canonical(model, scorer, manifest, episode_by_x[x], x, lower, upper, constraints, memory_cache) for x in range(32) for lower in range(32) for upper in range(lower, 32) for constraints in GOALS]
    cases = real_cases(); real = [run_real(model, ctrl1, scorer, manifest, episode_by_x[x], lower, upper, constraints) for x, lower, upper in cases for constraints in GOALS]
    canonical_by_goal = {name: {"samples": sum(item["goal"] == name for item in canonical), "success": sum(item["goal"] == name and item["success"] for item in canonical), "max_decisions": max(item["decisions"] for item in canonical if item["goal"] == name)} for name in GOALS.values()}
    region_counts = {"x0<L": sum(x < lower for x, lower, upper in cases), "L<=x0<=U": sum(lower <= x <= upper for x, lower, upper in cases), "x0>U": sum(x > upper for x, lower, upper in cases)}
    boundaries = {"L_equals_U": sum(lower == upper for x, lower, upper in cases), "x0_equals_L": sum(x == lower for x, lower, upper in cases), "x0_equals_U": sum(x == upper for x, lower, upper in cases)}
    summary = {"canonical": {"triples": 16896, "samples": len(canonical), "success": sum(item["success"] for item in canonical), "max_decisions": max(item["decisions"] for item in canonical), "by_goal": canonical_by_goal}, "real_memory": {"scenario_count": len(cases), "samples": len(real), "success": sum(item["success"] for item in real), "regions": region_counts, "boundaries": boundaries, "by_goal": {name: {"samples": sum(item["goal"] == name for item in real), "success": sum(item["goal"] == name and item["success"] for item in real)} for name in GOALS.values()}}, "dispatcher": {"goal_agnostic_signature": True, "all_six_actions_available": True}, "training": False}
    status = "passed" if summary["canonical"]["samples"] == 16896 * 4 and summary["canonical"]["success"] == 16896 * 4 and summary["real_memory"]["samples"] == 96 * 4 and summary["real_memory"]["success"] == 96 * 4 and len(set(region_counts.values())) == 1 else "failed"
    result = {"status": status, "task": "T1-CTRL-6", "phase": "constraint_composition_oracle_preflight", "summary": summary, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2": {"path": str(SCORER_CHECKPOINT), "sha256": sha256(SCORER_CHECKPOINT)}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)}, "canonical_cases": canonical, "real_memory_cases": real}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": status, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
