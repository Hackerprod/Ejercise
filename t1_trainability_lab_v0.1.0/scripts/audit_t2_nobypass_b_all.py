"""Historical all-domain E0/E1/B/COMP ablation; no training and no G5."""

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
CONDITIONS = ("E0", "E1", "B", "B+0.25Z")


class Adapter(nn.Module):
    def __init__(self, core: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pairs = manifest["calibration"] + manifest["heldout"]
    if len(pairs) != 992 or len({pair["digest"] for pair in pairs}) != 992:
        raise RuntimeError("expected exact 992-pair domain")
    writer = load_writer(); comp = Composition0(); comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True); comp.eval(); runtime = load_runtime(); _m, _f, episodes, base_manifest, core, executor, ctrl1, scorer = runtime
    counts = {name: {"identity": 0, "swap": 0, "any": 0, "legs": {"identity": {probe: 0 for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")}, "swap": {probe: 0 for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")}}} for name in CONDITIONS}
    per_case = []
    for pair in pairs:
        raw = encode_writer(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")[0]
        b = raw.sum(dim=0)
        with torch.no_grad(): z = comp(raw.unsqueeze(0))[0] - b
        conditions = {"E0": raw[0], "E1": raw[1], "B": b, "B+0.25Z": b + 0.25 * z}
        case_result = {"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "conditions": {}}
        for name, condition in conditions.items():
            outcomes = {}
            for permutation_name in ("identity", "swap"):
                floor = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], pair["lower"], 0, (1, 0))
                avoid = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], 0, pair["forbidden"], (0, 1))
                joint = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1))
                outcomes[permutation_name] = {"JOINT/FLOOR": bool(floor["success"]), "JOINT/AVOID": bool(avoid["success"]), "JOINT/SUM": bool(joint["success"]), "pass": bool(floor["success"] and avoid["success"] and joint["success"])}
            counts[name]["identity"] += int(all(outcomes["identity"][probe] for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM"))); counts[name]["swap"] += int(all(outcomes["swap"][probe] for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM"))); counts[name]["any"] += int(outcomes["identity"]["pass"] or outcomes["swap"]["pass"])
            for permutation_name in ("identity", "swap"):
                for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM"):
                    counts[name]["legs"][permutation_name][probe] += int(outcomes[permutation_name][probe])
            case_result["conditions"][name] = outcomes
        per_case.append(case_result)
    summaries = {name: {**value, "samples": 992, "identity_fraction": value["identity"] / 992, "swap_fraction": value["swap"] / 992, "any_fraction": value["any"] / 992} for name, value in counts.items()}
    heldout_digests = {pair["digest"] for pair in manifest["heldout"]}
    heldout_summary = {name: {"samples": 853, "any": sum(case["digest"] in heldout_digests and case["conditions"][name]["identity"]["pass"] or case["digest"] in heldout_digests and case["conditions"][name]["swap"]["pass"] for case in per_case)} for name in CONDITIONS}
    result = {"status": "completed", "task": "T2-NOBYPASS-B-ALL", "training": False, "development_only": True, "g5_touched": False, "conditions": CONDITIONS, "alpha": 0.25, "source_hashes": {"writer": sha256(CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"), "comp": sha256(COMP_CHECKPOINT), "manifest": sha256(MANIFEST_PATH)}, "summary_992": summaries, "summary_heldout_853": heldout_summary, "comparison": {"e1_heldout_vs_comp025_831": {"e1_any": heldout_summary["E1"]["any"], "comp025_any_reference": 831, "e1_at_least_831": heldout_summary["E1"]["any"] >= 831}}, "cases": per_case}
    output = CAMPAIGN / f"t2_nobypass_b_all_seed{args.seed}" / "results.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "summary_992": summaries, "summary_heldout_853": heldout_summary, "comparison": result["comparison"], "g5_touched": False}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
