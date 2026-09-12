"""Diagnostic-only THINK-1 movement and K2/K3 idempotence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from audit_t2_i2_r1_matching import Adapter
from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest
from t2_i3_think2 import Think2


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
REAL_MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
REAL_MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in ("success", "first_bad", "final_value", "decisions", "timeout")}


def run_leg(core: LatentConditionedSupervisor, executor: Any, ctrl1: Any, scorer: OrdinalSharedScorer, condition: torch.Tensor, manifest: dict[str, Any], episode: dict[str, Any], lower: int, forbidden: int, constraints: tuple[int, int]) -> dict[str, Any]:
    return result_summary(run_learned(executor, ctrl1, scorer, Adapter(core, condition), manifest, episode, lower, forbidden, constraints))


def proposal(think: Think2, state: torch.Tensor, anchor: torch.Tensor, round_index: int) -> torch.Tensor:
    context = state.flip(1)
    role = think.role_embedding.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
    step = think.step_embedding.weight[round_index].view(1, 1, -1).expand(state.shape[0], state.shape[1], -1)
    features = torch.cat((think.slot_norm(state), think.context_norm(context), think.anchor_norm(anchor), role, step), dim=-1)
    delta = think.w2(F.silu(think.w1(features)))
    return anchor + delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--think-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or CAMPAIGN / f"t2_i3_think1_diagnostic_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    think = Think2()
    think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True)
    think.eval()
    writer = CompetitiveSemanticWriter()
    writer.load_state_dict(torch.load(writer_checkpoint(6301), weights_only=False)["writer"], strict=True)
    writer.eval()
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    core.eval()
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(torch.load(SCORER_CHECKPOINT, weights_only=False)["controller"], strict=True)
    scorer.eval()
    manifest = load_base_manifests()["test"]
    fixed = load_fixed_manifest(REAL_MANIFEST, REAL_MANIFEST_SHA)
    episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    calibration = build_calibration_manifest(MANIFEST_PATH)["calibration"]
    cases: list[dict[str, Any]] = []
    states_at_k1: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    for pair in calibration:
        lower, forbidden = pair["lower"], pair["forbidden"]
        anchor = encode(writer, f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}").unsqueeze(0)
        state1 = think.run_rounds(anchor, 1)
        state2 = think.run_rounds(anchor, 2)
        state3 = think._step(state2, anchor, 1)
        states_at_k1.append(state1.detach())
        anchors.append(anchor.detach())
        movement = {"m1_slot0": float(torch.linalg.vector_norm(state1[:, 0] - anchor[:, 0]).item()), "m1_slot1": float(torch.linalg.vector_norm(state1[:, 1] - anchor[:, 1]).item()), "m2_slot0": float(torch.linalg.vector_norm(state2[:, 0] - state1[:, 0]).item()), "m2_slot1": float(torch.linalg.vector_norm(state2[:, 1] - state1[:, 1]).item()), "a1_slot0": float(torch.linalg.vector_norm(state1[:, 0] - anchor[:, 0]).item()), "a1_slot1": float(torch.linalg.vector_norm(state1[:, 1] - anchor[:, 1]).item()), "a2_slot0": float(torch.linalg.vector_norm(state2[:, 0] - anchor[:, 0]).item()), "a2_slot1": float(torch.linalg.vector_norm(state2[:, 1] - anchor[:, 1]).item())}
        k2 = {"JOINT/FLOOR": run_leg(core, executor, ctrl1, scorer, state2[0, 0], manifest, episodes[10], lower, 0, (1, 0)), "JOINT/AVOID": run_leg(core, executor, ctrl1, scorer, state2[0, 1], manifest, episodes[10], 0, forbidden, (0, 1)), "JOINT/SUM": run_leg(core, executor, ctrl1, scorer, state2[0].sum(0), manifest, episodes[10], lower, forbidden, (1, 1))}
        k3 = {"JOINT/FLOOR": run_leg(core, executor, ctrl1, scorer, state3[0, 0], manifest, episodes[10], lower, 0, (1, 0)), "JOINT/AVOID": run_leg(core, executor, ctrl1, scorer, state3[0, 1], manifest, episodes[10], 0, forbidden, (0, 1)), "JOINT/SUM": run_leg(core, executor, ctrl1, scorer, state3[0].sum(0), manifest, episodes[10], lower, forbidden, (1, 1))}
        cases.append({"lower": lower, "forbidden": forbidden, "digest": pair["digest"], "movement": movement, "k2": k2, "k3": k3})
    cross_cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, state in enumerate(states_at_k1):
            donor = (index + 1) % len(states_at_k1)
            mixed_for_slot0 = state.clone(); mixed_for_slot0[:, 1] = states_at_k1[donor][:, 1]
            mixed_for_slot1 = state.clone(); mixed_for_slot1[:, 0] = states_at_k1[donor][:, 0]
            original = proposal(think, state, anchors[index], 1)
            replaced0 = proposal(think, mixed_for_slot0, anchors[index], 1)
            replaced1 = proposal(think, mixed_for_slot1, anchors[index], 1)
            cross_cases.append({"index": index, "donor": donor, "delta_cross_slot0": float(torch.linalg.vector_norm(original[:, 0] - replaced0[:, 0]).item()), "delta_cross_slot1": float(torch.linalg.vector_norm(original[:, 1] - replaced1[:, 1]).item())})
    probes = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")
    result = {"status": "diagnostic", "task": "T2-I3-THINK-2", "phase": "movement_idempotence_cross_slot", "training": False, "checkpoint": {"path": str(args.think_checkpoint), "sha256": sha256(args.think_checkpoint)}, "writer_checkpoint_sha256": sha256(writer_checkpoint(6301)), "calibration_manifest_sha256": build_calibration_manifest(MANIFEST_PATH)["sha256"], "scope": {"samples": len(cases), "x0": 10, "k_train": 2, "k3_rule": "_step(S2, anchor, round_index=1)", "cross_slot_donor": "(index+1)%139"}, "summary": {"movement_means": {key: sum(item["movement"][key] for item in cases) / len(cases) for key in ("m1_slot0", "m1_slot1", "m2_slot0", "m2_slot1", "a1_slot0", "a1_slot1", "a2_slot0", "a2_slot1")}, "k2_k3_success": {key: {"k2": sum(item["k2"][key]["success"] for item in cases), "k3": sum(item["k3"][key]["success"] for item in cases), "changed": sum(item["k2"][key] != item["k3"][key] for item in cases)} for key in probes}, "cross_slot_sensitivity": {"samples": len(cross_cases), "mean_slot0": sum(item["delta_cross_slot0"] for item in cross_cases) / len(cross_cases), "mean_slot1": sum(item["delta_cross_slot1"] for item in cross_cases) / len(cross_cases), "max_slot0": max(item["delta_cross_slot0"] for item in cross_cases), "max_slot1": max(item["delta_cross_slot1"] for item in cross_cases)}}, "cases": cases, "cross_slot_cases": cross_cases}
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": result["status"], "summary": result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
