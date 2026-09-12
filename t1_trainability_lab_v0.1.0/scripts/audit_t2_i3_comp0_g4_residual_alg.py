"""Classify G4-REG-ALG never-pass cases using persisted traces only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_u0c_ctrl2_o import sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
SOURCE = CAMPAIGN / "t2_i3_comp0_g4_reg_alg_seed6401" / "results.json"
ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    if args.seed != 6401:
        raise ValueError("source artifact is frozen to seed6401")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    never_digests = source["g4_alpha0125_failures"]["never_passes_digests"]
    if len(never_digests) != 22:
        raise RuntimeError(f"expected 22 persisted never-pass digests, got {len(never_digests)}")
    by_digest = {case["digest"]: case for case in source["cases"]}
    if set(never_digests) - set(by_digest):
        raise RuntimeError("persisted never-pass digest missing from source cases")
    details = []
    category_counts = {"UNARY_ONLY": 0, "JOINT_HARD": 0, "MIXED": 0}
    for digest in never_digests:
        case = by_digest[digest]
        trace = case["trace"]
        if [entry["alpha"] for entry in trace] != list(ALPHAS):
            raise RuntimeError(f"alpha trace mismatch for {digest}")
        unary_rows = [(entry["g3"]["identity"]["JOINT/FLOOR"], entry["g3"]["identity"]["JOINT/AVOID"], entry["g3"]["swap"]["JOINT/FLOOR"], entry["g3"]["swap"]["JOINT/AVOID"]) for entry in trace]
        if any(row != unary_rows[0] for row in unary_rows[1:]):
            raise RuntimeError(f"FLOOR/AVOID alpha constancy failed for {digest}")
        identity_u = unary_rows[0][0] and unary_rows[0][1]
        swap_u = unary_rows[0][2] and unary_rows[0][3]
        u = identity_u or swap_u
        u_permutations = (["identity"] if identity_u else []) + (["swap"] if swap_u else [])
        j_alphas = [entry["alpha"] for entry in trace if entry["g3"]["identity"]["JOINT/SUM"] or entry["g3"]["swap"]["JOINT/SUM"]]
        j = bool(j_alphas)
        if u and j:
            raise RuntimeError(f"ALERT_INCONSISTENCY U AND J for {digest}: permutations={u_permutations} alphas={j_alphas}")
        category = "UNARY_ONLY" if not u and j else "JOINT_HARD" if u and not j else "MIXED"
        category_counts[category] += 1
        details.append({"digest": digest, "lower": case["lower"], "forbidden": case["forbidden"], "U": u, "U_permutations": u_permutations, "J": j, "J_alphas": j_alphas, "category": category, "unary_constant": True, "unary_signature": {"identity_floor": unary_rows[0][0], "identity_avoid": unary_rows[0][1], "swap_floor": unary_rows[0][2], "swap_avoid": unary_rows[0][3]}})
    result = {"status": "completed", "task": "T2-I3-COMP-0-G4-RESIDUAL-ALG", "training": False, "model_executed": False, "forwards_executed": False, "g5_touched": False, "source_artifact": {"path": str(SOURCE), "sha256": sha256(SOURCE)}, "alpha_grid": list(ALPHAS), "sanity_check": {"cases": 22, "passed": 22, "floor_avoid_constant_across_alphas": True, "sum_checked_both_permutations": True}, "category_counts": category_counts, "alert_u_and_j": False, "cases": details}
    output = CAMPAIGN / f"t2_i3_comp0_g4_residual_alg_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "source_sha256": result["source_artifact"]["sha256"], "sanity_check": result["sanity_check"], "category_counts": category_counts, "alert_u_and_j": False, "cases": details}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
