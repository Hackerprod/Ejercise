"""Frozen THINK-ALG diagnosis for T2-I3 calibration pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_t2_i2_r2 as r2
from audit_t2_i2_r1_matching import Adapter
from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import load_fixed_manifest
from evaluate_u0c_ctrl7_preflight import supervisor_features_ctrl7
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest
from t2_i3_think0 import Think0
from train_t2_i3 import calibration_rows, verify_execution_targets


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
REAL_MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
REAL_MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
THINK_ROUNDS = 2


def encode(writer: CompetitiveSemanticWriter, text: str) -> Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def cosine_distance(left: Tensor, right: Tensor) -> float:
    return float((1.0 - F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1)).item())


def l2_distance(left: Tensor, right: Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right).item())


def atomic_text(role: str, value: int) -> str:
    if role == "FLOOR":
        return f"AT_LEAST VALUE_{value} AND NOOP VALUE_0"
    if role == "AVOID":
        return f"NOOP VALUE_0 AND AVOID VALUE_{value}"
    return "NOOP VALUE_0 AND NOOP VALUE_11"


def panel_slot_selection(
    writer: CompetitiveSemanticWriter,
    supervisor: LatentConditionedSupervisor,
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
    panels: dict[str, dict[str, Any]],
    role: str,
    value: int,
) -> dict[str, Any]:
    slots = encode(writer, atomic_text(role, value))
    panel = panels[f"{role}:{value}"]
    row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long)
    features = observations["features"][row_ids]
    targets = labels["action"][row_ids]
    empty = supervisor(features, torch.zeros((len(row_ids), 32))).argmax(dim=-1)
    candidates: list[int] = []
    for active in (0, 1):
        inactive = 1 - active
        active_prediction = supervisor(features, slots[active].expand(len(row_ids), -1)).argmax(dim=-1)
        inactive_prediction = supervisor(features, slots[inactive].expand(len(row_ids), -1)).argmax(dim=-1)
        if torch.equal(active_prediction, targets) and torch.equal(inactive_prediction, empty):
            candidates.append(active)
    return {"role": role, "value": value, "candidates": candidates, "selected": candidates[0] if len(candidates) == 1 else None, "ambiguous": len(candidates) != 1}


def build_references(
    writer: CompetitiveSemanticWriter,
    think: Think0,
    supervisor: LatentConditionedSupervisor,
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
    panels: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selections: dict[str, Any] = {}
    references: dict[str, list[dict[str, Any]]] = {str(round_index): [] for round_index in range(THINK_ROUNDS + 1)}
    for role in ("FLOOR", "AVOID"):
        for value in range(32):
            selection = panel_slot_selection(writer, supervisor, observations, labels, panels, role, value)
            selections[f"{role}:{value}"] = selection
            if selection["selected"] is None:
                continue
            raw = encode(writer, atomic_text(role, value)).unsqueeze(0)
            for round_index in range(THINK_ROUNDS + 1):
                state = think.run_rounds(raw, round_index)[0, selection["selected"]]
                references[str(round_index)].append({"role": role, "value": value, "vector": state})
    null_raw = encode(writer, atomic_text("NULL", 0)).unsqueeze(0)
    for round_index in range(THINK_ROUNDS + 1):
        state = think.run_rounds(null_raw, round_index)[0]
        for slot in (0, 1):
            references[str(round_index)].append({"role": "NULL", "value": None, "slot": slot, "vector": state[slot]})
    return references, selections


def nearest_decodes(state: Tensor, references: list[dict[str, Any]]) -> dict[str, Any]:
    distances = sorted(
        ({"role": item["role"], "value": item["value"], "l2": l2_distance(state, item["vector"]), "cosine": cosine_distance(state, item["vector"])} for item in references),
        key=lambda item: item["l2"],
    )
    top = distances[0]
    second = distances[1] if len(distances) > 1 else None
    return {"top1": top, "top2": second, "margin_l2": None if second is None else second["l2"] - top["l2"]}


def downstream_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in ("success", "first_bad", "final_value", "decisions", "timeout")}


def run_leg(
    executor: Any,
    ctrl1: Any,
    scorer: OrdinalSharedScorer,
    core: LatentConditionedSupervisor,
    slots: Tensor,
    manifest: dict[str, Any],
    episode: dict[str, Any],
    lower: int,
    forbidden: int,
    constraints: tuple[int, int],
) -> dict[str, Any]:
    return downstream_summary(run_learned(executor, ctrl1, scorer, Adapter(core, slots), manifest, episode, lower, forbidden, constraints))


def pair_record(
    pair: dict[str, Any],
    row: dict[str, Any],
    writer: CompetitiveSemanticWriter,
    think: Think0,
    references: dict[str, list[dict[str, Any]]],
    supervisor: LatentConditionedSupervisor,
    executor: Any,
    ctrl1: Any,
    scorer: OrdinalSharedScorer,
    core: LatentConditionedSupervisor,
    manifest: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    lower, forbidden = pair["lower"], pair["forbidden"]
    joint = encode(writer, f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}").unsqueeze(0)
    states = {str(round_index): think.run_rounds(joint, round_index)[0] for round_index in range(THINK_ROUNDS + 1)}
    targets = {}
    for round_index in range(THINK_ROUNDS + 1):
        floor_reference = next((item for item in references[str(round_index)] if item["role"] == "FLOOR" and item["value"] == lower), None)
        avoid_reference = next((item for item in references[str(round_index)] if item["role"] == "AVOID" and item["value"] == forbidden), None)
        targets[str(round_index)] = {"T0": None if floor_reference is None else floor_reference["vector"], "T1": None if avoid_reference is None else avoid_reference["vector"]}
    distance_by_round: dict[str, Any] = {}
    decodes: dict[str, Any] = {}
    for round_index in range(THINK_ROUNDS + 1):
        state = states[str(round_index)]
        target = targets[str(round_index)]
        if target["T0"] is None or target["T1"] is None:
            distance_by_round[str(round_index)] = {"status": "target_unavailable", "missing": [name for name in ("T0", "T1") if target[name] is None]}
        else:
            id_l2 = l2_distance(state[0], target["T0"]) + l2_distance(state[1], target["T1"])
            swap_l2 = l2_distance(state[0], target["T1"]) + l2_distance(state[1], target["T0"])
            id_cos = cosine_distance(state[0], target["T0"]) + cosine_distance(state[1], target["T1"])
            swap_cos = cosine_distance(state[0], target["T1"]) + cosine_distance(state[1], target["T0"])
            distance_by_round[str(round_index)] = {"status": "computed", "D_id_l2": id_l2, "D_swap_l2": swap_l2, "D_id_minus_swap_l2": id_l2 - swap_l2, "D_id_cosine": id_cos, "D_swap_cosine": swap_cos, "D_id_minus_swap_cosine": id_cos - swap_cos}
        decodes[str(round_index)] = {"slot0": nearest_decodes(state[0], references[str(round_index)]), "slot1": nearest_decodes(state[1], references[str(round_index)])}
    state0, state1 = states["2"]
    permutations: dict[str, Any] = {}
    for name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
        selected = states["2"][[permutation[0], permutation[1]]]
        permutations[name] = {"slots": list(permutation), "legs": {"JOINT/FLOOR": run_leg(executor, ctrl1, scorer, core, selected[0], manifest, episode, lower, 0, (1, 0)), "JOINT/AVOID": run_leg(executor, ctrl1, scorer, core, selected[1], manifest, episode, 0, forbidden, (0, 1)), "JOINT/SUM": run_leg(executor, ctrl1, scorer, core, selected.sum(0), manifest, episode, lower, forbidden, (1, 1))}}
    joint_features = torch.tensor(row["features"], dtype=torch.float32).unsqueeze(0)
    joint_target = torch.tensor([row["target"]], dtype=torch.long)
    with torch.no_grad():
        joint_logits = supervisor(joint_features, (state0 + state1).unsqueeze(0))
        joint_loss = float(F.cross_entropy(joint_logits, joint_target).item())
    return {"lower": lower, "forbidden": forbidden, "digest": pair["digest"], "bucket": pair["bucket"], "form": "CALIBRATION_NORMAL", "orientation": "NORMAL", "clause_type": "AT_LEAST+AVOID", "x0": row["x0"], "distance_by_round": distance_by_round, "movement": {"delta1_slot0": l2_distance(states["1"][0], states["0"][0]), "delta1_slot1": l2_distance(states["1"][1], states["0"][1]), "delta2_slot0": l2_distance(states["2"][0], states["1"][0]), "delta2_slot1": l2_distance(states["2"][1], states["1"][1])}, "decodes": decodes, "permutations": permutations, "joint_loss": joint_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--think-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or CAMPAIGN / f"t2_i3_think_alg_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    think = Think0()
    think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True)
    think.eval()
    writer = CompetitiveSemanticWriter()
    writer.load_state_dict(torch.load(writer_checkpoint(6301), weights_only=False)["writer"], strict=True)
    writer.eval()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    observations, labels = r2.load_source()
    panels = r2.build_panels(labels)
    manifest = build_calibration_manifest(MANIFEST_PATH)
    calibration = calibration_rows(load_base_manifests()["train"], manifest["calibration"])
    verify_execution_targets(load_base_manifests()["train"], calibration)
    references, selections = build_references(writer, think, supervisor, observations, labels, panels)
    base_manifest = load_base_manifests()["test"]
    fixed = load_fixed_manifest(REAL_MANIFEST, REAL_MANIFEST_SHA)
    episodes = {int(entry["x"]): base_manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False)
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(scorer_payload["controller"], strict=True)
    scorer.eval()
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    core.eval()
    cases = [pair_record(pair, row, writer, think, references, supervisor, executor, ctrl1, scorer, core, base_manifest, episodes[10]) for pair, row in zip(manifest["calibration"], calibration)]
    def g3_success(item: dict[str, Any], permutation: str) -> bool:
        return all(item["permutations"][permutation]["legs"][probe]["success"] for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM"))

    pass_cases = [item for item in cases if g3_success(item, "identity") or g3_success(item, "swap")]
    fail_cases = [item for item in cases if item not in pass_cases]
    leg_summary = {permutation: {probe: {"samples": len(cases), "success": sum(item["permutations"][permutation]["legs"][probe]["success"] for item in cases)} for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")} for permutation in ("identity", "swap")}
    geometry_values = [item["distance_by_round"][str(THINK_ROUNDS)] for item in cases if item["distance_by_round"][str(THINK_ROUNDS)]["status"] == "computed"]
    movement_values = [item["movement"] for item in cases]
    geometry_summary = {key: {"samples": len(geometry_values), "mean": sum(value[key] for value in geometry_values) / len(geometry_values)} for key in ("D_id_l2", "D_swap_l2", "D_id_minus_swap_l2", "D_id_cosine", "D_swap_cosine", "D_id_minus_swap_cosine")}
    movement_summary = {key: sum(value[key] for value in movement_values) / len(movement_values) for key in ("delta1_slot0", "delta1_slot1", "delta2_slot0", "delta2_slot1")}
    decode_summary: dict[str, Any] = {}
    for round_index in range(THINK_ROUNDS + 1):
        rows: dict[str, int] = {"slot0_role_correct": 0, "slot0_value_correct": 0, "slot1_role_correct": 0, "slot1_value_correct": 0, "available": 0}
        for item in cases:
            for slot, expected_role, expected_value in (("slot0", "FLOOR", item["lower"]), ("slot1", "AVOID", item["forbidden"])):
                top = item["decodes"][str(round_index)][slot]["top1"]
                if top["role"] == expected_role:
                    rows[f"{slot}_role_correct"] += 1
                if top["role"] == expected_role and top["value"] == expected_value:
                    rows[f"{slot}_value_correct"] += 1
                rows["available"] += 1
        decode_summary[str(round_index)] = rows
    result = {"status": "diagnostic", "task": "T2-I3-THINK-ALG", "training": False, "checkpoint": {"path": str(args.think_checkpoint), "sha256": sha256(args.think_checkpoint)}, "writer_checkpoint": {"path": str(writer_checkpoint(6301)), "sha256": sha256(writer_checkpoint(6301))}, "calibration_manifest_sha256": manifest["sha256"], "scope": {"samples": len(cases), "bucket": 0, "orientation": "NORMAL", "x0": 10, "form": "CALIBRATION_NORMAL", "clause_type": "AT_LEAST+AVOID"}, "targets": {"definition": "THINK-processed R3 atomic FLOOR/AVOID references", "distance_primary": "L2", "distance_secondary": "1-cosine", "selections": selections}, "losses": {"joint_loss": {"samples": len(cases), "mean": sum(item["joint_loss"] for item in cases) / len(cases), "max": max(item["joint_loss"] for item in cases), "pass_mean": sum(item["joint_loss"] for item in pass_cases) / len(pass_cases) if pass_cases else None, "fail_mean": sum(item["joint_loss"] for item in fail_cases) / len(fail_cases) if fail_cases else None}, "main_loss": {"status": "not_defined_per_calibration_pair", "reason": "train_t2_i3 main_loss is sampled atomic source-batch CE"}, "card_loss": {"status": "not_defined_per_calibration_pair", "reason": "train_t2_i3 cardinality_loss is indexed by atomic panels; no (1,1) calibration panel"}}, "summary": {"g3_identity": sum(g3_success(item, "identity") for item in cases), "g3_swap": sum(g3_success(item, "swap") for item in cases), "g3_any": len(pass_cases), "legs": leg_summary, "geometry_k2": geometry_summary, "movement_means": movement_summary, "decodes": decode_summary}, "cases": cases}
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, Tensor) else value) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": result["status"], "summary": result["summary"], "joint_loss": result["losses"]["joint_loss"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
