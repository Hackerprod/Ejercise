"""Cold K=0..8 depth diagnostic for frozen T2-I3 THINK-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

import audit_t2_i2_r1_matching as g3_base
import audit_t2_i2_r3_cardinality as g2_base
import evaluate_t2_i2_r1_controls as g1_base
import t2_i2_r3_semantic_writer as writer_mod
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
REAL_MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
REAL_MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
ROUNDS = tuple(range(9))
PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def trace(think: Think2, anchor: torch.Tensor) -> list[torch.Tensor]:
    state1 = think.run_rounds(anchor, 1)
    state = think.run_rounds(anchor, 2)
    states = [anchor, state1, state]
    for _ in range(3, 9):
        state = think._step(state, anchor, 1)
        states.append(state)
    return states


def run_base_at_k(base: Any, kind: str, k: int, seed: int, think: Think2, writer_checkpoint_path: Path) -> dict[str, Any]:
    original_encode = base.encode

    def encoded(writer: Any, text: str) -> torch.Tensor:
        raw = original_encode(writer, text).unsqueeze(0)
        return trace(think, raw)[k][0]

    base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter
    base.checkpoint_for_seed = lambda _seed: writer_checkpoint_path
    output = CAMPAIGN / f"t2_i3_depth_{kind.lower()}_k{k}_seed{seed}"
    base.output_for_seed = lambda _seed, output_root=output: output_root
    base.encode = encoded
    if kind == "G2":
        panel_path = CAMPAIGN / "t2_i2_r3_seed6301" / "panel_manifest.json"
        base.load_panels = lambda _seed: (json.loads(panel_path.read_text(encoding="utf-8")), sha256(panel_path))
    argv = sys.argv
    sys.argv = [argv[0], "--seed", str(seed)]
    try:
        base.main()
    finally:
        sys.argv = argv
    result_dir = output / ("controls" if kind == "G1" else "cardinality")
    return json.loads((result_dir / "results.json").read_text(encoding="utf-8"))


def summary(result: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "G1":
        return {"status": result["status"], "position_invariance": result["summary"]["position_invariance"], "early_memory": result["summary"]["early_memory"], "noop_exhaustive": result["summary"]["noop_exhaustive"]}
    return {"status": result["status"], "samples": result["matching"]["samples"], "success": result["matching"]["success"]}


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in ("success", "first_bad", "final_value", "decisions", "timeout")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--think-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or CAMPAIGN / f"t2_i3_think2_depth_alg_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    think = Think2()
    think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True)
    think.eval()
    writer = CompetitiveSemanticWriter()
    writer.load_state_dict(torch.load(writer_checkpoint(6301), weights_only=False)["writer"], strict=True)
    writer.eval()
    calibration = build_calibration_manifest(MANIFEST_PATH)["calibration"]
    base_manifest = load_base_manifests()["test"]
    fixed = load_fixed_manifest(REAL_MANIFEST, REAL_MANIFEST_SHA)
    episodes = {int(entry["x"]): base_manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(torch.load(CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt", weights_only=False)["controller"], strict=True)
    scorer.eval()
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    core.eval()
    states_by_case: list[list[torch.Tensor]] = []
    for pair in calibration:
        raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}").unsqueeze(0)
        states_by_case.append(trace(think, raw))
    g1_series = {str(k): summary(run_base_at_k(g1_base, "G1", k, args.seed, think, writer_checkpoint(6301)), "G1") for k in ROUNDS}
    g2_series = {str(k): summary(run_base_at_k(g2_base, "G2", k, args.seed, think, writer_checkpoint(6301)), "G2") for k in ROUNDS}
    cases: list[dict[str, Any]] = []
    g3_series = {str(k): {"legs": {"identity": {probe: 0 for probe in PROBES}, "swap": {probe: 0 for probe in PROBES}}, "identity": 0, "swap": 0, "any": 0} for k in ROUNDS}
    for index, pair in enumerate(calibration):
        lower, forbidden = pair["lower"], pair["forbidden"]
        per_k: dict[str, Any] = {}
        for k in ROUNDS:
            state = states_by_case[index][k][0]
            permutations: dict[str, Any] = {}
            for name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
                selected = state[[permutation[0], permutation[1]]]
                legs = {"JOINT/FLOOR": compact(run_learned(executor, ctrl1, scorer, Adapter(core, selected[0]), base_manifest, episodes[10], lower, 0, (1, 0))), "JOINT/AVOID": compact(run_learned(executor, ctrl1, scorer, Adapter(core, selected[1]), base_manifest, episodes[10], 0, forbidden, (0, 1))), "JOINT/SUM": compact(run_learned(executor, ctrl1, scorer, Adapter(core, selected.sum(0)), base_manifest, episodes[10], lower, forbidden, (1, 1)))}
                permutations[name] = {"slots": list(permutation), "legs": legs}
                for probe in PROBES:
                    g3_series[str(k)]["legs"][name][probe] += int(legs[probe]["success"])
                passed = all(legs[probe]["success"] for probe in PROBES)
                g3_series[str(k)][name] += int(passed)
            g3_series[str(k)]["any"] += int(any(all(permutations[name]["legs"][probe]["success"] for probe in PROBES) for name in ("identity", "swap")))
            per_k[str(k)] = {"permutations": permutations, "pass": any(all(permutations[name]["legs"][probe]["success"] for probe in PROBES) for name in ("identity", "swap"))}
        pass_values = [per_k[str(k)]["pass"] for k in ROUNDS]
        first = next((k for k, passed in enumerate(pass_values) if passed), None)
        cases.append({"lower": lower, "forbidden": forbidden, "digest": pair["digest"], "pass_by_k": pass_values, "first_pass_k": first, "remains_passing_after_first": first is not None and all(pass_values[k] for k in range(first, 9)), "pass_fail_transition_count": sum(pass_values[k] != pass_values[k - 1] for k in range(1, 9)), "movement": {str(k): {"slot0": float(torch.linalg.vector_norm(states_by_case[index][k + 1][0, 0] - states_by_case[index][k][0, 0]).item()), "slot1": float(torch.linalg.vector_norm(states_by_case[index][k + 1][0, 1] - states_by_case[index][k][0, 1]).item())} for k in range(8)}, "per_k": per_k})
    for k in ROUNDS:
        g3_series[str(k)]["legs"] = {name: {probe: {"samples": len(cases), "success": value} for probe, value in probes.items()} for name, probes in g3_series[str(k)]["legs"].items()}
    result = {"status": "diagnostic", "task": "T2-I3-THINK-2-DEPTH-ALG", "training": False, "checkpoint": {"path": str(args.think_checkpoint), "sha256": sha256(args.think_checkpoint)}, "writer_checkpoint_sha256": sha256(writer_checkpoint(6301)), "calibration_manifest_sha256": build_calibration_manifest(MANIFEST_PATH)["sha256"], "scope": {"samples": len(cases), "rounds": list(ROUNDS), "k_ge_3_rule": "Think2._step(state, anchor, round_index=1)"}, "series": {"g1": g1_series, "g2": g2_series, "g3": g3_series}, "trajectory_summary": {"first_pass_counts": {str(k): sum(item["first_pass_k"] == k for item in cases) for k in ROUNDS}, "never_passed": sum(item["first_pass_k"] is None for item in cases), "remains_passing": sum(item["remains_passing_after_first"] for item in cases), "transition_count_distribution": {str(k): sum(item["pass_fail_transition_count"] == k for item in cases) for k in range(9)}, "movement_means": {str(k): {slot: sum(item["movement"][str(k)][slot] for item in cases) / len(cases) for slot in ("slot0", "slot1")} for k in range(8)}}, "cases": cases}
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "status": result["status"], "g3": {k: {"identity": v["identity"], "swap": v["swap"], "any": v["any"]} for k, v in g3_series.items()}, "trajectory_summary": result["trajectory_summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
