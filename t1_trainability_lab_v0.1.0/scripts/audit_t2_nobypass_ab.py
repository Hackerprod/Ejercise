"""Development-only NOBYPASS audits A/B; G5 is never touched."""

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
ALPHA = 0.25


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
    calibration = manifest["calibration"]
    writer = load_writer()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    runtime = load_runtime()
    _manifests, _fixed, episodes, base_manifest, core, executor, ctrl1, scorer = runtime
    conditions = {}
    for pair in calibration:
        raw = encode_writer(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")[0]
        with torch.no_grad():
            b = raw.sum(dim=0)
            c = b + ALPHA * (comp(raw.unsqueeze(0))[0] - b)
        conditions[pair["digest"]] = {"raw": raw, "condition": c}

    audit_a = []
    categories = {name: 0 for name in ("follows_false", "follows_real", "invariant_same", "ambiguous")}
    for index, pair in enumerate(calibration):
        false_pair = calibration[(index + 1) % len(calibration)]
        original = conditions[pair["digest"]]["condition"]
        false_condition = conditions[false_pair["digest"]]["condition"]
        real_ref = run_learned(executor, ctrl1, scorer, Adapter(core, original), base_manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1))
        false_ref = run_learned(executor, ctrl1, scorer, Adapter(core, false_condition), base_manifest, episodes[10], false_pair["lower"], false_pair["forbidden"], (1, 1))
        contaminated = run_learned(executor, ctrl1, scorer, Adapter(core, original), base_manifest, episodes[10], false_pair["lower"], false_pair["forbidden"], (1, 1))
        real_actions = [event["action_name"] for event in real_ref["events"]]
        false_actions = [event["action_name"] for event in false_ref["events"]]
        contaminated_actions = [event["action_name"] for event in contaminated["events"]]
        same_refs = real_actions == false_actions
        follows_false = not same_refs and contaminated_actions == false_actions and contaminated["final_value"] == false_ref["final_value"]
        follows_real = not same_refs and contaminated_actions == real_actions and contaminated["final_value"] == real_ref["final_value"]
        category = "invariant_same" if same_refs else "follows_false" if follows_false else "follows_real" if follows_real else "ambiguous"
        categories[category] += 1
        audit_a.append({"digest": pair["digest"], "false_digest": false_pair["digest"], "real_pair": [pair["lower"], pair["forbidden"]], "false_pair": [false_pair["lower"], false_pair["forbidden"]], "real_condition_source": pair["digest"], "false_condition_source": false_pair["digest"], "category": category, "real_ref": real_ref, "false_ref": false_ref, "contaminated": contaminated})

    audit_b = []
    for pair in calibration:
        raw = conditions[pair["digest"]]["raw"]
        b = raw.sum(dim=0)
        variant_results = {}
        for name, condition in (("E0", raw[0]), ("E1", raw[1]), ("B", b)):
            permutations = {}
            for permutation_name in ("identity", "swap"):
                floor = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], pair["lower"], 0, (1, 0))
                avoid = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], 0, pair["forbidden"], (0, 1))
                joint = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1))
                permutations[permutation_name] = {"JOINT/FLOOR": bool(floor["success"]), "JOINT/AVOID": bool(avoid["success"]), "JOINT/SUM": bool(joint["success"]), "pass": bool(floor["success"] and avoid["success"] and joint["success"])}
            variant_results[name] = permutations
        audit_b.append({"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "conditions": variant_results})
    summary_b = {}
    for name in ("E0", "E1", "B"):
        identity = sum(case["conditions"][name]["identity"]["pass"] for case in audit_b)
        swap = sum(case["conditions"][name]["swap"]["pass"] for case in audit_b)
        summary_b[name] = {"samples": len(audit_b), "identity": identity, "swap": swap, "any": sum(bool(case["conditions"][name]["identity"]["pass"] or case["conditions"][name]["swap"]["pass"]) for case in audit_b), "legs": {"identity": {probe: sum(case["conditions"][name]["identity"][probe] for case in audit_b) for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")}, "swap": {probe: sum(case["conditions"][name]["swap"][probe] for case in audit_b) for probe in ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")}}}
    result = {"status": "completed", "task": "T2-NOBYPASS-A-B", "training": False, "development_only": True, "g5_touched": False, "source_artifacts": {"writer_checkpoint_sha256": sha256(CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"), "comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "manifest_sha256": sha256(MANIFEST_PATH)}, "audit_a": {"false_pair_rule": "calibration[(i+1)%139]", "samples": 139, "categories": categories, "percentages": {key: value / 139 * 100 for key, value in categories.items()}, "cases": audit_a}, "audit_b": {"samples": 139, "summary": summary_b, "cases": audit_b}}
    output = CAMPAIGN / f"t2_nobypass_ab_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "audit_a": {"categories": categories, "percentages": result["audit_a"]["percentages"]}, "audit_b": summary_b, "g5_touched": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
