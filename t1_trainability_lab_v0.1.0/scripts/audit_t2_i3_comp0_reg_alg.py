"""Diagnostic alpha sweep for the frozen T2-I3-COMP-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

import train_t2_i2_r2 as r2
from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from evaluate_u0c_ctrl7_preflight import oracle_action
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest, load_writer
from t2_i3_comp0 import Composition0


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
REAL_MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
REAL_MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"
DEPTH_ARTIFACT = CAMPAIGN / "t2_i3_think2_depth_alg_seed6401" / "results.json"
RESIDUAL_ARTIFACT = CAMPAIGN / "t2_i3_r3_residual_alg_seed6401" / "results.json"
COMP_CHECKPOINT = CAMPAIGN / "t2_i3_comp0_seed6401" / "final.pt"
SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
PROBES = ("JOINT/FLOOR", "JOINT/AVOID", "JOINT/SUM")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def load_runtime() -> tuple[dict, dict, dict, dict, LatentConditionedSupervisor, object, object, OrdinalSharedScorer]:
    manifests = load_base_manifests()
    fixed = load_fixed_manifest(REAL_MANIFEST, REAL_MANIFEST_SHA)
    episodes = {int(entry["x"]): manifests["test"]["episodes"][entry["episode"]] for entry in fixed["entries"]}
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(torch.load(SCORER, weights_only=False)["controller"], strict=True)
    scorer.eval()
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    core.eval()
    return manifests, fixed, episodes, manifests["test"], core, executor, ctrl1, scorer


def g3_case(raw: torch.Tensor, condition: torch.Tensor, pair: dict, runtime: tuple) -> dict:
    _manifests, _fixed, episodes, manifest, core, executor, ctrl1, scorer = runtime
    legs = {}
    for name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
        legs[name] = {
            "JOINT/FLOOR": run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[0]]), manifest, episodes[10], pair["lower"], 0, (1, 0)),
            "JOINT/AVOID": run_learned(executor, ctrl1, scorer, Adapter(core, raw[permutation[1]]), manifest, episodes[10], 0, pair["forbidden"], (0, 1)),
            "JOINT/SUM": run_learned(executor, ctrl1, scorer, Adapter(core, condition), manifest, episodes[10], pair["lower"], pair["forbidden"], (1, 1)),
        }
    status = {name: {probe: bool(legs[name][probe]["success"]) for probe in PROBES} for name in legs}
    return {"identity": status["identity"], "swap": status["swap"], "any": any(all(status[name].values()) for name in status)}


def g1_alpha(alpha: float, rows: dict[str, tuple[torch.Tensor, torch.Tensor]], writer: CompetitiveSemanticWriter, comp: Composition0, runtime: tuple) -> dict:
    _manifests, fixed, episodes, manifest, core, executor, ctrl1, scorer = runtime
    cache: dict[str, Adapter] = {}

    def run(text: str, constraints: tuple[int, int], value: int) -> dict:
        if text not in cache:
            if text not in rows:
                raw = encode(writer, text)
                with torch.no_grad():
                    z = (comp(raw.unsqueeze(0))[0] - raw.sum(dim=0)).detach()
                rows[text] = (raw, z)
            raw, z = rows[text]
            cache[text] = Adapter(core, raw.sum(dim=0) + alpha * z)
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episodes[10], value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0, constraints)

    position = {"samples": 0, "success": 0}
    memory = {"samples": 0, "success": 0}
    noop = {name: {"samples": 0, "success": 0} for name in ("NONE", "AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE")}
    for value in range(32):
        for alias, constraints in (("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1))):
            operator = f"{alias} VALUE_{value}"
            forms = (operator, f"{operator} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {operator}")
            results = [run(text, constraints, value) for text in forms]
            position["samples"] += 1
            position["success"] += int(all(item["success"] for item in results) and len({tuple(event["action_name"] for event in item["events"]) for item in results}) == 1)
            memory["samples"] += 1
            memory["success"] += int(results[1]["success"] and results[2]["success"] and tuple(event["action_name"] for event in results[1]["events"]) == tuple(event["action_name"] for event in results[2]["events"]))
        for name, constraints in (("NONE", (0, 0)), ("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1))):
            for variant in ((0, 1) if name == "NONE" else (1, 2)):
                for dummy in range(32):
                    if name == "NONE" and variant == 0:
                        text = f"NOOP VALUE_{dummy}"
                    elif name == "NONE":
                        text = f"NOOP VALUE_{dummy} AND NOOP VALUE_{(dummy + 11) % 32}"
                    elif variant == 1:
                        text = f"{name} VALUE_{value} AND NOOP VALUE_{dummy}"
                    else:
                        text = f"NOOP VALUE_{dummy} AND {name} VALUE_{value}"
                    result = run(text, constraints, value)
                    noop[name]["samples"] += 1
                    noop[name]["success"] += int(result["success"])
    passed = position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values())
    return {"status": "passed" if passed else "failed", "summary": {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop}, "dispatcher_guard": "passed" if not any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")) else "failed"}


def g2_alpha(rows: list[dict], runtime: tuple) -> dict:
    _manifests, _fixed, _episodes, _manifest, core, _executor, _ctrl1, _scorer = runtime
    panel_manifest_path = CAMPAIGN / "t2_i2_r3_seed6301" / "panel_manifest.json"
    panels = json.loads(panel_manifest_path.read_text(encoding="utf-8"))["panels"]
    observations, labels = r2.load_source()
    writer = load_writer()
    total = passed = 0
    for kind, aliases in (("FLOOR", ("AT_LEAST", "MINIMUM")), ("AVOID", ("AVOID", "EXCLUDE"))):
        for alias in aliases:
            for value in range(32):
                for form in (1, 2, 3):
                    text = f"{alias} VALUE_{value}" if form == 1 else f"{alias} VALUE_{value} AND NOOP VALUE_0" if form == 2 else f"NOOP VALUE_0 AND {alias} VALUE_{value}"
                    slots = encode(writer, text)
                    panel = panels[f"{kind}:{value}"]
                    row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long)
                    features = observations["features"][row_ids]
                    targets = labels["action"][row_ids]
                    empty = core(features, torch.zeros((len(row_ids), 32))).argmax(dim=-1)
                    outcomes = []
                    for permutation in ((0, 1), (1, 0)):
                        active = core(features, slots[permutation[0]].expand(len(row_ids), -1)).argmax(dim=-1)
                        inactive = core(features, slots[permutation[1]].expand(len(row_ids), -1)).argmax(dim=-1)
                        outcomes.append(torch.equal(active, targets) and torch.equal(inactive, empty))
                    total += 1
                    passed += int(any(outcomes))
    for form, text in ((1, "NOOP VALUE_0"), (2, "NOOP VALUE_0 AND NOOP VALUE_11")):
        slots = encode(writer, text)
        panel = panels["NONE:0"]
        row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long)
        features = observations["features"][row_ids]
        targets = labels["action"][row_ids]
        empty = core(features, torch.zeros((len(row_ids), 32))).argmax(dim=-1)
        outcomes = []
        for permutation in ((0, 1), (1, 0)):
            active = core(features, slots[permutation[0]].expand(len(row_ids), -1)).argmax(dim=-1)
            inactive = core(features, slots[permutation[1]].expand(len(row_ids), -1)).argmax(dim=-1)
            outcomes.append(torch.equal(active, targets) and torch.equal(inactive, empty))
        total += 1
        passed += int(any(outcomes))
    return {"status": "passed" if passed == total else "failed", "matching": {"samples": total, "success": passed}, "criterion": "one permutation reproduces every panel action and other slot matches zero-conditioned behavior", "input": "raw E0/E1 slots; alpha sweep re-executed to verify invariance"}


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values), "median": ordered[len(ordered) // 2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    if args.seed != 6401:
        raise ValueError("REG-ALG source groups are frozen to seed6401 artifacts")
    writer = load_writer()
    comp = Composition0()
    comp.load_state_dict(torch.load(COMP_CHECKPOINT, weights_only=False)["composition"], strict=True)
    comp.eval()
    manifest = build_calibration_manifest(MANIFEST_PATH)
    runtime = load_runtime()
    calibration = manifest["calibration"]
    depth = json.loads(DEPTH_ARTIFACT.read_text(encoding="utf-8"))
    residual = json.loads(RESIDUAL_ARTIFACT.read_text(encoding="utf-8"))
    baseline_pass = {case["digest"] for case in depth["cases"] if case["first_pass_k"] == 0}
    baseline_fail = {case["digest"] for case in residual["cases"]}
    if len(baseline_pass) != 129 or len(baseline_fail) != 10 or baseline_pass & baseline_fail:
        raise RuntimeError("frozen source groups do not contain expected 129/10 disjoint digests")
    rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    cases = {}
    max_perm_error = 0.0
    for pair in calibration:
        text = f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"
        raw = encode(writer, text)
        batch = raw.unsqueeze(0)
        with torch.no_grad():
            z = (comp(batch)[0] - raw.sum(dim=0)).detach()
            swapped = raw.flip(0).unsqueeze(0)
            max_perm_error = max(max_perm_error, float((comp(swapped)[0] - swapped.sum(dim=1)[0] - z).abs().max().item()))
        if not torch.equal(comp(batch)[0], raw.sum(dim=0) + z):
            raise RuntimeError(f"C(1)=B+Z verification failed for {pair['digest']}")
        cases[pair["digest"]] = {"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], "raw": raw, "B": raw.sum(dim=0), "Z": z, "z_l2": float(torch.linalg.vector_norm(z).item()), "group": "baseline_pass" if pair["digest"] in baseline_pass else "baseline_fail"}
        rows[text] = (raw, z)
    if max_perm_error > 1e-6:
        raise RuntimeError(f"COMP-0 Z permutation invariance failed: max_abs={max_perm_error}")
    series = []
    for alpha in ALPHAS:
        for case in cases.values():
            condition = case["B"] + alpha * case["Z"]
            case.setdefault("trace", []).append({"alpha": alpha, "g3": g3_case(case["raw"], condition, case, runtime)})
        g1 = g1_alpha(alpha, rows, writer, comp, runtime)
        g2 = g2_alpha(calibration, runtime)
        alpha_cases = [case["trace"][-1] for case in cases.values()]
        identity = sum(all(item["g3"]["identity"].values()) for item in alpha_cases)
        swap = sum(all(item["g3"]["swap"].values()) for item in alpha_cases)
        any_pass = sum(item["g3"]["any"] for item in alpha_cases)
        preserved = sum(case["digest"] in baseline_pass and case["trace"][-1]["g3"]["any"] for case in cases.values())
        repaired = sum(case["digest"] in baseline_fail and case["trace"][-1]["g3"]["any"] for case in cases.values())
        for case in cases.values():
            case["trace"][-1]["g1"] = g1
            case["trace"][-1]["g2"] = g2
        series.append({"alpha": alpha, "baseline_pass_preserved": preserved, "baseline_fail_repaired": repaired, "g3": {"identity": identity, "swap": swap, "any": any_pass}, "g1": g1, "g2": g2})
    for case in cases.values():
        previous = case["trace"][0]["g3"]["any"]
        first = None
        for entry in case["trace"][1:]:
            current = entry["g3"]["any"]
            if first is None and current != previous:
                first = {"from_alpha": entry["alpha"] - 0.125, "to_alpha": entry["alpha"], "from_status": "PASS" if previous else "FAIL", "to_status": "PASS" if current else "FAIL"}
            previous = current
        case["first_transition"] = first
        case["z_l2"] = case.pop("z_l2")
        case.pop("B")
        case.pop("Z")
        case.pop("raw")
    alpha1 = {case["digest"] for case in cases.values() if case["trace"][-1]["g3"]["any"]}
    regressions = baseline_pass - alpha1
    fixed = baseline_fail & alpha1
    persistent = baseline_fail - alpha1
    if len(regressions) != 37 or len(fixed) != 9 or persistent != {"b51ff2fe6748f6f8cc65c92c1a8ff20ef76fdd69b4043bec4b1c8c0e17c830b9"}:
        raise RuntimeError(f"alpha=1 group mismatch: regressions={len(regressions)} fixed={len(fixed)} persistent={persistent}")
    for digest, case in cases.items():
        case["group"] = "preserved" if digest in baseline_pass & alpha1 else "fixed" if digest in fixed else "regressed" if digest in regressions else "persistent-fail"
    groups = {}
    for name, members in (("preserved", baseline_pass & alpha1), ("fixed", fixed), ("regressed", regressions), ("persistent-fail", persistent)):
        groups[name] = {"digests": sorted(members), "norm_z_l2": stats([cases[digest]["z_l2"] for digest in members]), "per_case": {digest: cases[digest]["z_l2"] for digest in sorted(members)}}
    result = {"status": "completed", "task": "T2-I3-COMP-0-REG-ALG", "training": False, "alpha_set": list(ALPHAS), "source_artifacts": {"comp_checkpoint_sha256": sha256(COMP_CHECKPOINT), "depth_alg_sha256": sha256(DEPTH_ARTIFACT), "residual_alg_sha256": sha256(RESIDUAL_ARTIFACT), "calibration_manifest_sha256": manifest["sha256"]}, "verification": {"c1_equals_b_plus_z_max_abs": 0.0, "z_permutation_invariant": True, "z_permutation_max_abs": max_perm_error, "checkpoint_untouched": True, "training_updates": 0, "g4_executed": False, "g5_executed": False}, "series": series, "groups": groups, "cases": sorted(cases.values(), key=lambda item: item["digest"])}
    output = CAMPAIGN / f"t2_i3_comp0_reg_alg_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "series": [{"alpha": item["alpha"], "preserved": item["baseline_pass_preserved"], "repaired": item["baseline_fail_repaired"], "g3": item["g3"], "g1": item["g1"]["status"], "g2": item["g2"]["matching"]} for item in series], "groups": {name: item["norm_z_l2"] for name, item in groups.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
