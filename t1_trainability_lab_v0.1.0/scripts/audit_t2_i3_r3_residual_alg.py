"""Pure K0 residual-failure extraction from the frozen depth artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")
PERMUTATIONS = ("identity", "swap")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("campaign/t2_i3_think2_depth_alg_seed6401/results.json"))
    parser.add_argument("--output", type=Path, default=Path("campaign/t2_i3_r3_residual_alg_seed6401/results.json"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    cases = source["cases"]
    if len(cases) != 139:
        raise AssertionError(f"expected 139 cases, got {len(cases)}")
    failed = [case for case in cases if case["per_k"]["0"]["pass"] is False]
    passed = [case for case in cases if case["per_k"]["0"]["pass"] is True]
    if len(failed) != 10 or len(passed) != 129:
        raise AssertionError(f"K0 cardinality mismatch: failed={len(failed)} passed={len(passed)}")
    details = []
    counts = {"JOINT_ONLY": 0, "UNARY_ONLY": 0, "MIXED": 0}
    for case in failed:
        k0 = case["per_k"]["0"]
        legs = {perm: {probe: bool(k0["permutations"][perm]["legs"][probe]["success"]) for probe in PROBES} for perm in PERMUTATIONS}
        unary_perm = [perm for perm in PERMUTATIONS if legs[perm]["JOINT/FLOOR"] and legs[perm]["JOINT/AVOID"]]
        sum_perm = [perm for perm in PERMUTATIONS if legs[perm]["JOINT/SUM"]]
        if unary_perm:
            category = "JOINT_ONLY"
        elif sum_perm:
            category = "UNARY_ONLY"
        else:
            category = "MIXED"
        counts[category] += 1
        details.append({"digest": case["digest"], "lower": case["lower"], "forbidden": case["forbidden"], "legs": legs, "unary_permutations": unary_perm, "sum_permutations": sum_perm, "same_permutation": sorted(set(unary_perm).intersection(sum_perm)), "category": category})
    if sum(counts.values()) != 10:
        raise AssertionError(f"category cardinality mismatch: {counts}")
    result = {"status": "completed", "task": "T2-I3-R3-RESIDUAL-ALG", "training": False, "source_artifact": {"path": str(args.input), "sha256": file_sha(args.input)}, "cross_check": {"total": len(cases), "failed_k0": len(failed), "passed_k0": len(passed), "expected": {"total": 139, "failed": 10, "passed": 129}}, "counts": counts, "cases": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "sha256": file_sha(args.output), "status": result["status"], "counts": counts, "cross_check": result["cross_check"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
