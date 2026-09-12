"""P0 plumbing: CTRL-7 condition references come only from parsed text."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, pair_value, state_hash
from evaluate_u0c_ctrl7_preflight import GOALS, oracle_action, primitive_check, target_value
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R, materialize_graph_batch
from t2_i1_instruction import parse_instruction_i1


ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")


@torch.no_grad()
def supervisor_features_nobypass(model: Any, ctrl1: Any, scorer: Any, state: torch.Tensor, goal: torch.Tensor, value_tokens: tuple[int, int], *, v_e: bool, v_r: bool) -> torch.Tensor:
    """Build same CTRL-7 features, but references derive from parsed token IDs."""
    navigation = F.softmax(ctrl1(state[:, SLOT_P], goal), dim=-1)
    if v_r:
        q_r, _ = canonical_value_view(model, state[:, SLOT_R])
        q_lower, _ = canonical_value_view(model, model.token_embedding(torch.tensor([VALUE_BASE + value_tokens[0]])))
        q_forbidden, _ = canonical_value_view(model, model.token_embedding(torch.tensor([VALUE_BASE + value_tokens[1]])))
        floor_comparison = F.softmax(scorer.logits(q_r, q_lower), dim=-1)
        avoid_comparison = F.softmax(scorer.logits(q_r, q_forbidden), dim=-1)
    else:
        floor_comparison = torch.zeros((state.shape[0], 3))
        avoid_comparison = torch.zeros((state.shape[0], 3))
    return torch.cat((navigation, floor_comparison, avoid_comparison, torch.tensor([[float(v_e), float(v_r)]])), dim=-1)


@torch.no_grad()
def run_learned_nobypass(model: Any, ctrl1: Any, scorer: Any, supervisor: Any, manifest: dict[str, Any], episode: dict[str, Any], instruction: str) -> dict[str, Any]:
    parsed = parse_instruction_i1(instruction)
    condition_lower = model.token_embedding(torch.tensor([VALUE_BASE + parsed.lower]))
    condition_forbidden = model.token_embedding(torch.tensor([VALUE_BASE + parsed.forbidden]))
    graph = manifest["graphs"][episode["graph"]]
    memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION)); state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
    presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool); navigation_goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE])); x0 = pair_value(manifest, episode); pointer, symbolic_x = episode["start_key"], x0; read_e_done = copied = v_e = v_r = False; target = target_value(x0, parsed.lower, parsed.forbidden, parsed.constraints); events = []; emitted = False; first_bad = None
    for decision in range(MAX_DECISIONS):
        expected = oracle_action(pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, value=symbolic_x, lower=parsed.lower, forbidden=parsed.forbidden, constraints=parsed.constraints)
        features = supervisor_features_nobypass(model, ctrl1, scorer, state, navigation_goal, (parsed.lower, parsed.forbidden), v_e=v_e, v_r=v_r)
        action = int(supervisor(features, torch.tensor([[float(parsed.constraints[0]), float(parsed.constraints[1])]])).argmax(-1).item())
        before = state.clone(); expected_row = next((index for index, row in enumerate(graph["rows"]) if (expected == READ_P and row["kind"] == "REL" and row["key"] == pointer) or (expected == READ_E and row["kind"] == "PAIR" and row["key"] == episode["goal_key"])), -1)
        state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
        check = primitive_check(model, before, state, action, {**operation, "available_e": v_e, "available_r": v_r}, expected_row, graph, target)
        events.append({"decision": decision, "action": action, "action_name": ACTION_NAMES[action], "expected_action": expected, "expected_action_name": ACTION_NAMES[expected], "features": features.squeeze(0).tolist(), "primitive_correct": check, "r_after_sha256": state_hash(state, SLOT_R)})
        first_bad = decision if not check and first_bad is None else first_bad
        rejected = operation.get("rejected", False)
        if action == READ_P and not rejected: v_e = False
        elif action == READ_E and not rejected: v_e = True; read_e_done = True
        elif action == COPY_E_R and not rejected: v_r = True; copied = True
        elif action == INCREASE and not rejected: v_r = True; symbolic_x += 1
        elif action == EMIT and not rejected: emitted = True
        if action == READ_P and not rejected and operation.get("selected_row", -1) >= 0: pointer = graph["rows"][operation["selected_row"]]["value"]
        if emitted: break
    final = int(canonical_value_view(model, state[:, SLOT_R])[1].item()) if v_r else None
    return {"instruction": instruction, "parsed": {"tokens": list(parsed.tokens), "token_ids": list(parsed.token_ids), "constraints": list(parsed.constraints), "lower": parsed.lower, "forbidden": parsed.forbidden}, "condition_hashes": {"C_L": hashlib.sha256(condition_lower.detach().cpu().numpy().tobytes()).hexdigest(), "C_F": hashlib.sha256(condition_forbidden.detach().cpu().numpy().tobytes()).hexdigest()}, "actions": [event["action_name"] for event in events], "action_ids": [event["action"] for event in events], "state_hashes": [event["r_after_sha256"] for event in events], "events": events, "target": target, "final_value": final, "first_bad": first_bad, "timeout": not emitted, "success": emitted and first_bad is None and final == target and events[-1]["action"] == EMIT}


def action_digest(result: dict) -> str:
    return hashlib.sha256((json_bytes(result["action_ids"]) + json_bytes(result["state_hashes"]))).hexdigest()


def json_bytes(value: Any) -> bytes:
    import json
    return json.dumps(value, separators=(",", ":")).encode()
