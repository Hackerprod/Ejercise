"""T2-I0 Baseline A plumbing preflight: oracle parser -> frozen CTRL-7."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_heldout import run_canonical_learned
from evaluate_u0c_ctrl7_trained import run_learned
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl7 import GoalConditionedSupervisor614
from t2_i0_instruction import parse_instruction


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
TRAINED_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_trained_evaluation_seed4701" / "results.json"
HELDOUT_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_heldout_seed4701" / "results.json"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_baseline_a_preflight_seed4701"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_for(constraints: tuple[int, int], lower: int, forbidden: int) -> str:
    if constraints == (0, 0):
        return "KEEP"
    if constraints == (1, 0):
        return f"AT_LEAST VALUE_{lower}"
    if constraints == (0, 1):
        return f"AVOID VALUE_{forbidden}"
    if constraints == (1, 1):
        return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}"
    raise ValueError(f"unsupported constraints: {constraints}")


def behavior(result: dict[str, Any]) -> tuple[Any, ...]:
    actions = result.get("actions", [event["action_name"] for event in result.get("events", [])])
    return (actions, result["target"], result["final_value"], result["first_bad"], result["success"])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden CTRL-7 inputs")
    trained_baseline = json.loads(TRAINED_BASELINE.read_text(encoding="utf-8")); heldout_baseline = json.loads(HELDOUT_BASELINE.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(CHECKPOINT, weights_only=False)
    if sha256(CHECKPOINT) != "439b071b133aceb6b3b5956188c2aca370609072de2b7fdabfbea3a44dd6355f": raise RuntimeError("unexpected frozen CTRL-7 checkpoint")
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); model = load_executor(); ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(checkpoint_payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    parser_counts = {"KEEP": 0, "AT_LEAST": 0, "AVOID": 0, "AND": 0}
    for value in range(32):
        for text, name in ((f"AT_LEAST VALUE_{value}", "AT_LEAST"), (f"AVOID VALUE_{value}", "AVOID")):
            parsed = parse_instruction(text); parser_counts[name] += int(parsed.constraints == ((1, 0) if name == "AT_LEAST" else (0, 1)))
    for lower in range(32):
        for forbidden in range(31):
            parsed = parse_instruction(instruction_for((1, 1), lower, forbidden)); parser_counts["AND"] += int(parsed.constraints == (1, 1) and parsed.lower == lower and parsed.forbidden == forbidden)
    parser_counts["KEEP"] = int(parse_instruction("KEEP").constraints == (0, 0))

    trained_rows = trained_baseline["trajectories"]
    trained_equivalent = 0
    for baseline in trained_rows:
        constraints = tuple(baseline["constraints"]); parsed = parse_instruction(instruction_for(constraints, baseline["lower"], baseline["forbidden"]))
        candidate = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], parsed.lower, parsed.forbidden, parsed.constraints)
        if behavior(candidate) == behavior(baseline): trained_equivalent += 1

    canonical_equivalent = 0
    for baseline in heldout_baseline["canonical_cases"]:
        parsed = parse_instruction(instruction_for((1, 1), baseline["lower"], baseline["forbidden"]))
        candidate = run_canonical_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], baseline["x0"], parsed.lower, parsed.forbidden)
        if behavior(candidate) == behavior(baseline): canonical_equivalent += 1

    real_equivalent = 0
    for baseline in heldout_baseline["real_memory_cases"]:
        parsed = parse_instruction(instruction_for((1, 1), baseline["lower"], baseline["forbidden"]))
        candidate = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], parsed.lower, parsed.forbidden, parsed.constraints)
        if behavior(candidate) == behavior(baseline): real_equivalent += 1
    summary = {"parser": {"vocabulary_size": 36, "valid_forms": parser_counts, "all_value_arguments_checked": True}, "trained_goals": {"samples": len(trained_rows), "equivalent": trained_equivalent, "success": trained_equivalent == len(trained_rows)}, "heldout_floor_and_avoid": {"canonical_samples": len(heldout_baseline["canonical_cases"]), "canonical_equivalent": canonical_equivalent, "canonical_success": canonical_equivalent == len(heldout_baseline["canonical_cases"]), "real_samples": len(heldout_baseline["real_memory_cases"]), "real_equivalent": real_equivalent, "real_success": real_equivalent == len(heldout_baseline["real_memory_cases"]), "interaction_gate": heldout_baseline["summary"]["required_gates"]["x_lt_L_eq_F_exact_sequence"], "crossing_literal_gate": heldout_baseline["summary"]["required_gates"]["crossing_literal_exact_sequence"]}}
    parser_pass = all(summary["parser"]["valid_forms"][name] == count for name, count in (("KEEP", 1), ("AT_LEAST", 32), ("AVOID", 32), ("AND", 992)))
    result = {"status": "passed" if parser_pass and summary["trained_goals"]["success"] and summary["heldout_floor_and_avoid"]["canonical_success"] and summary["heldout_floor_and_avoid"]["real_success"] else "failed", "task": "T2-I0", "phase": "baseline_a_oracle_parser_plumbing_preflight", "training": False, "encoder_trained": False, "parser": {"module": "scripts/t2_i0_instruction.py", "vocabulary_size": 36, "grammar": ["KEEP", "AT_LEAST VALUE_L", "AVOID VALUE_F", "AT_LEAST VALUE_L AND AVOID VALUE_F"]}, "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT), "seed": checkpoint_payload.get("seed"), "updates": checkpoint_payload.get("updates")}, "dispatcher_guard": "passed", "baseline_hashes": {"trained_evaluation": file_sha(TRAINED_BASELINE), "heldout_evaluation": file_sha(HELDOUT_BASELINE), "manifest": file_sha(MANIFEST_PATH)}, "summary": summary}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), "status": result["status"], "checkpoint": result["checkpoint"], "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
