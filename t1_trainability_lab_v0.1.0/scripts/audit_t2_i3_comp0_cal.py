"""T2-I3-COMP-0-CAL fixed-alpha materialization and integrity gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from audit_t2_i3_comp0_reg_alg import COMP_CHECKPOINT, DEPTH_ARTIFACT, RESIDUAL_ARTIFACT, g1_alpha, g2_alpha, g3_case, load_runtime, sha256
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest, encode_writer, load_writer
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
REG_ARTIFACT = CAMPAIGN / "t2_i3_comp0_reg_alg_seed6401" / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    output = CAMPAIGN / f"t2_i3_comp0_cal_alpha0125_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if source["calibration_count"] != 139 or source["heldout_count"] != 853 or source["candidate_count"] != 992:
        raise RuntimeError("unexpected calibration manifest counts")
    calibration = source["calibration"]
    heldout = source["heldout"]
    calibration_digests = {pair["digest"] for pair in calibration}
    heldout_digests = {pair["digest"] for pair in heldout}
    if calibration_digests & heldout_digests or len(calibration_digests | heldout_digests) != 992:
        raise RuntimeError("calibration/heldout overlap or omission")
    reg = json.loads(REG_ARTIFACT.read_text(encoding="utf-8"))
    persisted_norms = {digest: value for group in reg["groups"].values() for digest, value in group["per_case"].items()}
    if len(persisted_norms) != 139:
        raise RuntimeError("REG-ALG artifact lacks 139 per-case norms")
    writer = load_writer()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    rows = {}
    cases = {}
    max_norm_error = 0.0
    max_c1_error = 0.0
    for pair in calibration:
        text = f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"
        raw = encode_writer(writer, text)[0]
        with torch.no_grad():
            b = raw.sum(dim=0)
            z = (comp(raw.unsqueeze(0))[0] - b).detach()
        norm = float(torch.linalg.vector_norm(z).item())
        max_norm_error = max(max_norm_error, abs(norm - persisted_norms[pair["digest"]]))
        max_c1_error = max(max_c1_error, float((b + z - comp(raw.unsqueeze(0))[0]).abs().max().item()))
        cases[pair["digest"]] = {"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "raw": raw, "B": b, "Z": z, "z_l2": norm}
        rows[text] = (raw, z)
    if max_norm_error > 1e-6 or max_c1_error != 0.0:
        raise RuntimeError(f"REG-ALG cross-check failed: max_norm_error={max_norm_error} max_c1_error={max_c1_error}")
    conditions = {"task": "T2-I3-COMP-0-CAL", "alpha": 0.125, "alpha_fixed": True, "digests": [pair["digest"] for pair in calibration], "B": torch.stack([cases[pair["digest"]]["B"] for pair in calibration]), "Z": torch.stack([cases[pair["digest"]]["Z"] for pair in calibration]), "C": torch.stack([cases[pair["digest"]]["B"] + 0.125 * cases[pair["digest"]]["Z"] for pair in calibration])}
    torch.save(conditions, output / "conditions.pt")
    runtime = load_runtime()
    g3_identity = g3_swap = g3_any = 0
    for pair in calibration:
        case = cases[pair["digest"]]
        outcome = g3_case(case["raw"], case["B"] + 0.125 * case["Z"], pair, runtime)
        case["g3"] = outcome
        g3_identity += int(all(outcome["identity"].values()))
        g3_swap += int(all(outcome["swap"].values()))
        g3_any += int(outcome["any"])
    g1 = g1_alpha(0.125, rows, writer, comp, runtime)
    g2 = g2_alpha(calibration, runtime)
    integrity = {"g1": g1, "g2": g2, "g3": {"samples": 139, "identity": g3_identity, "swap": g3_swap, "any": g3_any, "fraction": g3_any / 139}, "all_exact": g1["status"] == "passed" and g2["matching"]["success"] == 386 and g3_any == 139}
    if not integrity["all_exact"]:
        raise RuntimeError(f"integrity confirmation failed: {integrity}")
    result = {"status": "integrity_confirmed", "task": "T2-I3-COMP-0-CAL", "training": False, "alpha": 0.125, "alpha_fixed": True, "integrity_confirmation_not_new_evidence": True, "source_artifacts": {"comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "reg_alg_sha256": sha256(REG_ARTIFACT), "depth_alg_sha256": sha256(DEPTH_ARTIFACT), "residual_alg_sha256": sha256(RESIDUAL_ARTIFACT), "calibration_manifest_sha256": sha256(MANIFEST_PATH)}, "verification": {"calibration_count": 139, "heldout_count": 853, "candidate_count": 992, "disjoint_exhaustive": True, "norm_crosscheck_digests": 139, "max_norm_error": max_norm_error, "max_c1_error": max_c1_error, "checkpoint_untouched": True, "training_updates": 0, "g4_conditional": True, "g5_conditional": True}, "integrity": integrity, "conditions_sha256": sha256(output / "conditions.pt")}
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    g4_output = output / "g4"
    env = None
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "audit_t2_i3_comp0_g4.py"), "--alpha", "0.125", "--output-root", str(g4_output)], check=True, env=env)
    g4 = json.loads((g4_output / "g4_results.json").read_text(encoding="utf-8"))
    result["g4"] = {"status": g4["status"], "matching": g4["matching"], "artifact": str(g4_output / "g4_results.json"), "artifact_sha256": sha256(g4_output / "g4_results.json")}
    result["g5"] = {"executed": False, "reason": "G4 failed; conditional HALT"} if g4["status"] != "passed" else {"executed": "pending", "reason": "G4 passed; execute original G5 battery"}
    result["status"] = "g4_failed_halted" if g4["status"] != "passed" else "g4_passed_g5_required"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "status": result["status"], "integrity": {"g1": g1["status"], "g2": g2["matching"], "g3": integrity["g3"]}, "g4": result["g4"], "g5": result["g5"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
