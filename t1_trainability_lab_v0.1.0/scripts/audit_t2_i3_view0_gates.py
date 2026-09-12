"""VIEW-0 G1/G2/G3/G4-development evaluator."""

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
from t2_i3_view0 import View0
import train_t2_i2_r2 as r2


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
VIEW_CHECKPOINT = CAMPAIGN / "t2_i3_view0_seed6401" / "final.pt"
COMP_CHECKPOINT = CAMPAIGN / "t2_i3_comp0_seed6401" / "final.pt"
COMP_G4 = CAMPAIGN / "t2_i3_comp0_g4_reg_alg_seed6401" / "results.json"
RESIDUAL = CAMPAIGN / "t2_i3_comp0_g4_residual_alg_seed6401" / "results.json"
ALPHA = 0.25


class Adapter(nn.Module):
    def __init__(self, core: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def run(core, executor, ctrl1, scorer, manifest, episode, condition, lower, forbidden, constraints):
    return run_learned(executor, ctrl1, scorer, Adapter(core, condition), manifest, episode, lower, forbidden, constraints)


def role_condition(view: View0, raw: torch.Tensor, text: str, comp: Composition0) -> torch.Tensor:
    if "AT_LEAST" in text or "MINIMUM" in text:
        return view(raw.unsqueeze(0))[0][0]
    if "AVOID" in text or "EXCLUDE" in text:
        return view(raw.unsqueeze(0))[1][0]
    return raw.sum(dim=0)


def g1(view: View0, writer, runtime: tuple) -> dict:
    _m, _f, episodes, manifest, core, executor, ctrl1, scorer = runtime
    cache = {}
    def evaluate(text, constraints, value):
        if text not in cache:
            raw = encode_writer(writer, text)[0]
            cache[text] = role_condition(view, raw, text, comp_model)
        return run(core, executor, ctrl1, scorer, manifest, episodes[10], cache[text], value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0, constraints)
    position = {"samples": 0, "success": 0}; memory = {"samples": 0, "success": 0}; noop = {name: {"samples": 0, "success": 0} for name in ("NONE", "AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE")}
    for value in range(32):
        for alias, constraints in (("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1))):
            operator = f"{alias} VALUE_{value}"; forms = (operator, f"{operator} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {operator}"); results = [evaluate(text, constraints, value) for text in forms]; position["samples"] += 1; position["success"] += int(all(item["success"] for item in results) and len({tuple(event["action_name"] for event in item["events"]) for item in results}) == 1); memory["samples"] += 1; memory["success"] += int(results[1]["success"] and results[2]["success"] and tuple(event["action_name"] for event in results[1]["events"]) == tuple(event["action_name"] for event in results[2]["events"]))
        for name, constraints in (("NONE", (0, 0)), ("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1))):
            for variant in ((0, 1) if name == "NONE" else (1, 2)):
                for dummy in range(32):
                    text = f"NOOP VALUE_{dummy}" if name == "NONE" and variant == 0 else f"NOOP VALUE_{dummy} AND NOOP VALUE_{(dummy + 11) % 32}" if name == "NONE" else f"{name} VALUE_{value} AND NOOP VALUE_{dummy}" if variant == 1 else f"NOOP VALUE_{dummy} AND {name} VALUE_{value}"; result = evaluate(text, constraints, value); noop[name]["samples"] += 1; noop[name]["success"] += int(result["success"])
    passed = position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values())
    return {"status": "passed" if passed else "failed", "summary": {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop}}


def g2(view: View0, writer) -> dict:
    observations, labels = r2.load_source(); panels = json.loads((CAMPAIGN / "t2_i2_r3_seed6301" / "panel_manifest.json").read_text(encoding="utf-8"))["panels"]; supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); supervisor.eval(); total = passed = 0
    for kind, aliases in (("FLOOR", ("AT_LEAST", "MINIMUM")), ("AVOID", ("AVOID", "EXCLUDE"))):
        for alias in aliases:
            for value in range(32):
                for form in (1, 2, 3):
                    text = f"{alias} VALUE_{value}" if form == 1 else f"{alias} VALUE_{value} AND NOOP VALUE_0" if form == 2 else f"NOOP VALUE_0 AND {alias} VALUE_{value}"; raw = encode_writer(writer, text)[0]; vf, va = view(raw.unsqueeze(0)); active = vf[0] if kind == "FLOOR" else va[0]; panel = panels[f"{kind}:{value}"]; row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long); features = observations["features"][row_ids]; targets = labels["action"][row_ids]; empty = supervisor(features, torch.zeros((len(row_ids), 32))).argmax(dim=-1); learned = supervisor(features, active.expand(len(row_ids), -1)).argmax(dim=-1); total += 1; passed += int(torch.equal(learned, targets) and torch.equal(empty, empty))
    return {"status": "passed" if passed == total else "failed", "matching": {"samples": total, "success": passed}, "criterion": "role view reproduces active panel action"}


def g3_or_g4(view: View0, comp: Composition0, writer, pairs: list[dict], runtime: tuple) -> tuple[dict, list[dict]]:
    _m, _f, episodes, manifest, core, executor, ctrl1, scorer = runtime; cases = []; identity = swap = any_pass = 0
    for pair in pairs:
        raw = encode_writer(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}")[0]; outputs = {}
        for name, slots in (("identity", raw), ("swap", raw.flip(0))):
            vf, va = view(slots.unsqueeze(0)); b = slots.sum(dim=0); c = b + ALPHA * (comp(slots.unsqueeze(0))[0] - b); floor = run(core, executor, ctrl1, scorer, manifest, episodes[10], vf[0], pair["lower"], 0, (1, 0)); avoid = run(core, executor, ctrl1, scorer, manifest, episodes[10], va[0], 0, pair["forbidden"], (0, 1)); summ = run(core, executor, ctrl1, scorer, manifest, episodes[10], c, pair["lower"], pair["forbidden"], (1, 1)); outputs[name] = {"JOINT/FLOOR": bool(floor["success"]), "JOINT/AVOID": bool(avoid["success"]), "JOINT/SUM": bool(summ["success"])}
        any_ok = any(all(value.values()) for value in outputs.values()); identity += int(all(outputs["identity"].values())); swap += int(all(outputs["swap"].values())); any_pass += int(any_ok); cases.append({"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "pass": any_ok, "permutations": outputs})
    return {"samples": len(pairs), "identity": identity, "swap": swap, "any": any_pass, "fraction": any_pass / len(pairs)}, cases


def main() -> None:
    global comp_model, LatentConditionedSupervisor, CTRL7_CHECKPOINT
    from train_t2_i0_baseline_b import LatentConditionedSupervisor
    from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
    parser = argparse.ArgumentParser(); parser.add_argument("--gate", choices=("G1", "G2", "G3", "G4"), required=True); args = parser.parse_args()
    writer = load_writer(); view = View0(); view.load_state_dict(torch.load(VIEW_CHECKPOINT, weights_only=False)["view"], strict=True); view.eval(); comp_model = Composition0(); comp_model.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True); comp_model.eval(); runtime = load_runtime()
    if args.gate == "G1": result = {"status": "completed", "gate": "G1", "result": g1(view, writer, runtime)}
    elif args.gate == "G2": result = {"status": "completed", "gate": "G2", "result": g2(view, writer)}
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")); pairs = manifest["calibration"] if args.gate == "G3" else manifest["heldout"]; summary, cases = g3_or_g4(view, comp_model, writer, pairs, runtime); baseline = json.loads(COMP_G4.read_text(encoding="utf-8")); baseline_cases = {case["digest"]: case for case in baseline["cases"]}; residual = {case["digest"] for case in json.loads(RESIDUAL.read_text(encoding="utf-8"))["cases"]}; view_pass = {case["digest"] for case in cases if case["pass"]}; base_pass = {digest for digest, case in baseline_cases.items() if case["trace"][2]["g3"]["any"]} if args.gate == "G4" else set(); result = {"status": "completed", "gate": args.gate, "result": summary, "development_decomposition": {"baseline_comp025_pass": len(base_pass), "preserved": len(base_pass & view_pass), "regressions": len(base_pass - view_pass), "residual_total": len(residual) if args.gate == "G4" else None, "residual_fixed": len(residual & view_pass) if args.gate == "G4" else None, "bcb26b5b": next((case for case in cases if case["digest"] == "bcb26b5b9affc8f267a32f53b222beb030102f226427c3480162ccf14fe1c3f9"), None) if args.gate == "G4" else None}, "cases": cases}
    output = CAMPAIGN / "t2_i3_view0_seed6401" / args.gate.lower() / "results.json"; output.parent.mkdir(parents=True, exist_ok=True); result["source_hashes"] = {"view_checkpoint": sha256(VIEW_CHECKPOINT), "comp_checkpoint": sha256(COMP_CHECKPOINT), "manifest": sha256(MANIFEST_PATH)}; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
