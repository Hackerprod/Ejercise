"""Historical mixed-slot ablation reproducing COMP-0 gate semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from audit_t2_i3_comp0_reg_alg import load_runtime, sha256
from evaluate_u0c_ctrl7_trained import run_learned
from t2_i3_common import MANIFEST_PATH, encode_writer, load_writer
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
COMP_CHECKPOINT = CAMPAIGN / "t2_i3_comp0_seed6401" / "final.pt"
SUMS = ("E0", "E1", "B", "B+0.25Z")
PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")


class Adapter(nn.Module):
    def __init__(self, core: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6401); args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")); calibration = manifest["calibration"]; heldout = manifest["heldout"]; pairs = calibration + heldout
    if len(pairs) != 992 or len(calibration) != 139 or len(heldout) != 853: raise RuntimeError("unexpected manifest split")
    writer = load_writer(); comp = Composition0(); comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True); comp.eval(); runtime = load_runtime(); _m, _f, episodes, base_manifest, core, executor, ctrl1, scorer = runtime
    cases = []
    for pair in pairs:
        raw = encode_writer(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")[0]; b = raw.sum(dim=0)
        with torch.no_grad(): corrected = b + 0.25 * (comp(raw.unsqueeze(0))[0] - b)
        sums = {"E0": raw[0], "E1": raw[1], "B": b, "B+0.25Z": corrected}; result_case = {"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "sums": {}}
        for sum_name, sum_condition in sums.items():
            outcomes = {}
            for permutation_name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
                floor = run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[0]]), base_manifest, episodes[10], pair["lower"], 0, (1, 0)); avoid = run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[1]]), base_manifest, episodes[10], 0, pair["forbidden"], (0, 1)); joint = run_learned(executor, ctrl1, scorer, Adapter(core, sum_condition), base_manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1)); outcomes[permutation_name] = {"JOINT/FLOOR": bool(floor["success"]), "JOINT/AVOID": bool(avoid["success"]), "JOINT/SUM": bool(joint["success"]), "pass": bool(floor["success"] and avoid["success"] and joint["success"])}
            result_case["sums"][sum_name] = outcomes
        cases.append(result_case)
    summaries = {}
    for sum_name in SUMS:
        rows = {"calibration": [case for case in cases if case["digest"] in {p["digest"] for p in calibration}], "heldout": [case for case in cases if case["digest"] in {p["digest"] for p in heldout}], "total": cases}; summaries[sum_name] = {}
        for split, split_rows in rows.items():
            summaries[sum_name][split] = {"samples": len(split_rows), "identity": sum(all(case["sums"][sum_name]["identity"][probe] for probe in PROBES) for case in split_rows), "swap": sum(all(case["sums"][sum_name]["swap"][probe] for probe in PROBES) for case in split_rows), "any": sum(case["sums"][sum_name]["identity"]["pass"] or case["sums"][sum_name]["swap"]["pass"] for case in split_rows), "legs": {"identity": {probe: sum(case["sums"][sum_name]["identity"][probe] for case in split_rows) for probe in PROBES}, "swap": {probe: sum(case["sums"][sum_name]["swap"][probe] for case in split_rows) for probe in PROBES}}}
    sanity = {"B_calibration": summaries["B"]["calibration"]["any"] == 129, "B_heldout": summaries["B"]["heldout"]["any"] == 783, "B_plus_025Z_calibration": summaries["B+0.25Z"]["calibration"]["any"] == 139, "B_plus_025Z_heldout": summaries["B+0.25Z"]["heldout"]["any"] == 831, "E1_guardrail": summaries["E1"]["heldout"]["any"] <= 831}
    if not all(sanity.values()): raise RuntimeError(f"mixed audit sanity failure: {sanity}")
    result = {"status": "completed", "task": "T2-NOBYPASS-B-MIXED-ALG", "training": False, "development_only": True, "g5_touched": False, "sum_semantics": "raw E_p0 FLOOR + raw E_p1 AVOID; only SUM varies", "sums": SUMS, "source_hashes": {"writer": sha256(CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"), "comp": sha256(COMP_CHECKPOINT), "manifest": sha256(MANIFEST_PATH)}, "sanity_checks": sanity, "summaries": summaries, "cases": cases}
    output = CAMPAIGN / f"t2_nobypass_b_mixed_alg_seed{args.seed}" / "results.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "sanity_checks": sanity, "summaries": summaries, "g5_touched": False}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
