"""T2-I3-COMP-0-CAL held-out G4 evaluator at fixed alpha=0.125."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_t2_i3_comp0_reg_alg import COMP_CHECKPOINT, REAL_MANIFEST, REAL_MANIFEST_SHA, encode, g3_case, load_runtime, sha256
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest, load_writer
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.alpha != 0.125:
        raise ValueError("COMP-0-CAL G4 accepts fixed alpha=0.125 only")
    source = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if source["calibration_count"] != 139 or source["heldout_count"] != 853 or source["candidate_count"] != 992:
        raise RuntimeError("unexpected calibration manifest counts")
    calibration = {pair["digest"] for pair in source["calibration"]}
    heldout = source["heldout"]
    heldout_digests = {pair["digest"] for pair in heldout}
    if len(heldout_digests) != 853 or calibration & heldout_digests or len(calibration | heldout_digests) != 992:
        raise RuntimeError("held-out manifest is not exact complement of calibration")
    writer = load_writer()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    runtime = load_runtime()
    cases = []
    identity = swap = any_pass = 0
    for pair in heldout:
        raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")
        with torch.no_grad():
            z = (comp(raw.unsqueeze(0))[0] - raw.sum(dim=0)).detach()
        result = g3_case(raw, raw.sum(dim=0) + args.alpha * z, pair, runtime)
        identity += int(all(result["identity"].values()))
        swap += int(all(result["swap"].values()))
        any_pass += int(result["any"])
        cases.append({"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "g3": result})
    result = {"status": "passed" if any_pass == 853 else "failed", "task": "T2-I3-COMP-0-CAL", "gate": "G4", "training": False, "alpha": args.alpha, "alpha_fixed": True, "source_artifacts": {"comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "calibration_manifest_sha256": sha256(MANIFEST_PATH), "real_manifest_sha256": sha256(REAL_MANIFEST)}, "complement_assertion": {"calibration": 139, "heldout": 853, "candidate": 992, "disjoint": True, "exhaustive": True}, "matching": {"samples": 853, "identity": identity, "swap": swap, "any": any_pass, "fraction": any_pass / 853}, "cases": cases}
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "g4_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "status": result["status"], "matching": result["matching"], "complement_assertion": result["complement_assertion"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
