"""Postmortem alpha sweep over consumed COMP-0 G4 cases; G5 is never touched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_t2_i3_comp0_reg_alg import COMP_CHECKPOINT, DEPTH_ARTIFACT, RESIDUAL_ARTIFACT, encode, g3_case, load_runtime, sha256
from t2_i3_common import MANIFEST_PATH, load_writer
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
REG_ARTIFACT = CAMPAIGN / "t2_i3_comp0_reg_alg_seed6401" / "results.json"
G4_ARTIFACT = CAMPAIGN / "t2_i3_comp0_cal_alpha0125_seed6401" / "g4" / "g4_results.json"
ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def pass_intervals(statuses: list[bool]) -> list[dict]:
    intervals = []
    start = None
    for index, passed in enumerate(statuses + [False]):
        if passed and start is None:
            start = index
        elif not passed and start is not None:
            intervals.append({"start_alpha": ALPHAS[start], "end_alpha": ALPHAS[index - 1], "count": index - start})
            start = None
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    if args.seed != 6401:
        raise ValueError("G4-REG-ALG source lineage is frozen to seed6401")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    calibration = manifest["calibration"]
    heldout = manifest["heldout"]
    calibration_digests = {pair["digest"] for pair in calibration}
    heldout_digests = {pair["digest"] for pair in heldout}
    if (len(calibration) != 139 or len(heldout) != 853 or manifest["candidate_count"] != 992 or len(calibration_digests) != 139 or len(heldout_digests) != 853 or calibration_digests & heldout_digests or len(calibration_digests | heldout_digests) != 992):
        raise RuntimeError("manifest is not exact 139/853 disjoint exhaustive partition")
    reg = json.loads(REG_ARTIFACT.read_text(encoding="utf-8"))
    persisted_norms = {digest: value for group in reg["groups"].values() for digest, value in group["per_case"].items()}
    writer = load_writer()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    max_norm_error = 0.0
    max_c1_error = 0.0
    for pair in calibration:
        raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")
        with torch.no_grad():
            b = raw.sum(dim=0)
            c = comp(raw.unsqueeze(0))[0]
        z = (c - b).detach()
        max_norm_error = max(max_norm_error, abs(float(torch.linalg.vector_norm(z).item()) - persisted_norms[pair["digest"]]))
        max_c1_error = max(max_c1_error, float((c - (b + z)).abs().max().item()))
    if max_norm_error > 1e-6 or max_c1_error != 0.0:
        raise RuntimeError(f"Z cross-check failed: norm_error={max_norm_error} c1_error={max_c1_error}")
    runtime = load_runtime()
    cases = {}
    for pair in heldout:
        raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")
        with torch.no_grad():
            b = raw.sum(dim=0)
            z = (comp(raw.unsqueeze(0))[0] - b).detach()
        cases[pair["digest"]] = {"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "raw": raw, "B": b, "Z": z, "z_l2": float(torch.linalg.vector_norm(z).item()), "trace": []}
    for case in cases.values():
        case["trace"] = []
        for alpha in ALPHAS:
            case["trace"].append({"alpha": alpha, "g3": g3_case(case["raw"], case["B"] + alpha * case["Z"], case, runtime)})
        statuses = [entry["g3"]["any"] for entry in case["trace"]]
        intervals = pass_intervals(statuses)
        first = next((alpha for alpha, passed in zip(ALPHAS, statuses) if passed), None)
        last = next((alpha for alpha, passed in reversed(list(zip(ALPHAS, statuses))) if passed), None)
        case["baseline_alpha0"] = statuses[0]
        case["first_pass_alpha"] = first
        case["last_pass_alpha"] = last
        case["pass_intervals"] = intervals
        case["continuous_pass_interval"] = bool(intervals)
        case.pop("B")
        case.pop("Z")
        case.pop("raw")
    series = []
    for index, alpha in enumerate(ALPHAS):
        entries = [case["trace"][index] for case in cases.values()]
        passed = {case["digest"] for case, entry in zip(cases.values(), entries) if entry["g3"]["any"]}
        baseline_pass = {case["digest"] for case in cases.values() if case["baseline_alpha0"]}
        baseline_fail = set(cases) - baseline_pass
        series.append({"alpha": alpha, "baseline": alpha == 0.0, "preserved": len(baseline_pass & passed), "repaired": len(baseline_fail & passed), "regressions": len(baseline_pass - passed), "g3": {"identity": sum(all(entry["g3"]["identity"].values()) for entry in entries), "swap": sum(all(entry["g3"]["swap"].values()) for entry in entries), "any": len(passed), "samples": 853}})
    g4_prior = json.loads(G4_ARTIFACT.read_text(encoding="utf-8"))
    failed_0125 = {case["digest"] for case in g4_prior["cases"] if not case["g3"]["any"]}
    if len(failed_0125) != 27:
        raise RuntimeError(f"G4 prior alpha=.125 failure set expected 27, got {len(failed_0125)}")
    alpha_index = {alpha: index for index, alpha in enumerate(ALPHAS)}
    alpha025 = {case["digest"] for case in cases.values() if case["trace"][alpha_index[0.25]]["g3"]["any"]}
    any_grid = {case["digest"] for case in cases.values() if any(entry["g3"]["any"] for entry in case["trace"])}
    never = failed_0125 - any_grid
    result = {"status": "completed", "task": "T2-I3-COMP-0-G4-REG-ALG", "phase": "postmortem", "training": False, "alpha_set": list(ALPHAS), "baseline_alpha0_first": series[0], "series": series, "source_artifacts": {"comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "reg_alg_sha256": sha256(REG_ARTIFACT), "g4_prior_sha256": sha256(G4_ARTIFACT), "depth_alg_sha256": sha256(DEPTH_ARTIFACT), "residual_alg_sha256": sha256(RESIDUAL_ARTIFACT), "calibration_manifest_sha256": sha256(MANIFEST_PATH)}, "verification": {"z_norm_crosscheck_digests": 139, "max_norm_error": max_norm_error, "c1_equals_b_plus_z_max_abs": max_c1_error, "manifest_calibration": 139, "manifest_heldout": 853, "manifest_candidate": 992, "manifest_disjoint_exhaustive": True, "g5_touched": False, "g5_executed": False}, "g4_alpha0125_failures": {"samples": 27, "pass_at_alpha025": len(failed_0125 & alpha025), "pass_at_any_other_grid_alpha": len(failed_0125 & any_grid), "never_passes_any_grid": len(never), "never_passes_digests": sorted(never)}, "cases": [{key: value for key, value in case.items() if key != "z_l2"} | {"z_l2": case["z_l2"]} for case in cases.values()], "norm_z_l2": {"samples": 853, "mean": sum(case["z_l2"] for case in cases.values()) / 853, "min": min(case["z_l2"] for case in cases.values()), "max": max(case["z_l2"] for case in cases.values())}}
    output = CAMPAIGN / f"t2_i3_comp0_g4_reg_alg_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "baseline_alpha0": series[0], "series": series, "g4_alpha0125_failures": result["g4_alpha0125_failures"], "verification": result["verification"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
