"""COMP-0 G3 aggregate composition evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

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
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
REAL_MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
REAL_MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--composition-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or CAMPAIGN / f"t2_i3_comp0_g3_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    writer = CompetitiveSemanticWriter()
    writer.load_state_dict(torch.load(writer_checkpoint(6301), weights_only=False)["writer"], strict=True)
    writer.eval()
    comp = Composition0()
    if args.composition_checkpoint is not None:
        comp.load_state_dict(torch.load(args.composition_checkpoint, weights_only=False)["composition"], strict=True)
    comp.eval()
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
    calibration = build_calibration_manifest(MANIFEST_PATH)["calibration"]
    cases = []
    counts = {perm: {probe: 0 for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")} for perm in ("identity", "swap")}
    for pair in calibration:
        raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")
        corrected = comp(raw.unsqueeze(0))[0]
        outcomes = {}
        for name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
            legs = {
                "JOINT/FLOOR": run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[0]]), base_manifest, episodes[10], pair["lower"], 0, (1, 0)),
                "JOINT/AVOID": run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[1]]), base_manifest, episodes[10], 0, pair["forbidden"], (0, 1)),
                "JOINT/SUM": run_learned(executor, ctrl1, scorer, Adapter(core, corrected), base_manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1)),
            }
            outcomes[name] = {probe: bool(legs[probe]["success"]) for probe in legs}
            for probe in PROBES:
                counts[name][probe] += int(outcomes[name][probe])
        cases.append({"lower": pair["lower"], "forbidden": pair["forbidden"], "digest": pair["digest"], "permutations": outcomes, "pass": any(all(outcomes[name].values()) for name in outcomes)})
    summary = {name: {probe: {"samples": len(cases), "success": value} for probe, value in probes.items()} for name, probes in counts.items()}
    identity = sum(all(case["permutations"]["identity"].values()) for case in cases)
    swap = sum(all(case["permutations"]["swap"].values()) for case in cases)
    result = {"status": "passed" if sum(case["pass"] for case in cases) == len(cases) else "failed", "task": "T2-I3-COMP-0", "gate": "G3", "training": False, "checkpoint": None if args.composition_checkpoint is None else {"path": str(args.composition_checkpoint), "sha256": sha256(args.composition_checkpoint)}, "writer_checkpoint_sha256": sha256(writer_checkpoint(6301)), "calibration_manifest_sha256": build_calibration_manifest(MANIFEST_PATH)["sha256"], "matching": {"samples": len(cases), "success": sum(case["pass"] for case in cases), "fraction": sum(case["pass"] for case in cases) / len(cases)}, "identity": identity, "swap": swap, "legs": summary, "cases": cases}
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "status": result["status"], "matching": result["matching"], "identity": identity, "swap": swap, "legs": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
