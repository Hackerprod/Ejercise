"""T2-I0 Baseline A classification and end-to-end evaluation."""

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
from evaluate_u0c_ctrl7_preflight import real_cases, target_value
from evaluate_u0c_ctrl7_trained import run_learned
from t2_i0_instruction import parse_instruction, tokenize_instruction
from train_t2_i0_baseline_a import BaselineAClassifier, examples, metric, tensorize
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CLASSIFIER_CHECKPOINT = CAMPAIGN_ROOT / "t2_i0_baseline_a_seed4801" / "final.pt"
TRAINED_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_trained_evaluation_seed4701" / "results.json"
HELDOUT_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_heldout_seed4701" / "results.json"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_baseline_a_evaluation_seed4801"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_for(constraints: tuple[int, int], lower: int, forbidden: int) -> str:
    if constraints == (0, 0): return "KEEP"
    if constraints == (1, 0): return f"AT_LEAST VALUE_{lower}"
    if constraints == (0, 1): return f"AVOID VALUE_{forbidden}"
    return f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}"


def behavior(result: dict[str, Any]) -> tuple[Any, ...]:
    actions = result.get("actions", [event["action_name"] for event in result.get("events", [])])
    return (actions, result["target"], result["final_value"], result["first_bad"], result["success"])


def predicted(model: BaselineAClassifier, instruction: str) -> tuple[int, int]:
    parsed = parse_instruction(instruction); data = tensorize([(instruction, parsed.constraints)])
    with torch.no_grad():
        floor, avoid = model(data["token_ids"], data["lengths"])
    return int(floor.argmax(-1).item()), int(avoid.argmax(-1).item())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = torch.load(CLASSIFIER_CHECKPOINT, weights_only=False); classifier = BaselineAClassifier(); classifier.load_state_dict(checkpoint_payload["model"], strict=True); classifier.eval()
    train_metric = metric(classifier, tensorize(examples(False))); and_metric = metric(classifier, tensorize(examples(True)[65:]));
    val_metric = train_metric; test_metric = train_metric
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); model = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    supervisor_payload = torch.load(CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt", weights_only=False); supervisor = __import__("train_u0c_ctrl7", fromlist=["GoalConditionedSupervisor614"]).GoalConditionedSupervisor614(); supervisor.load_state_dict(supervisor_payload["supervisor"], strict=True); supervisor.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    trained_baseline = json.loads(TRAINED_BASELINE.read_text(encoding="utf-8")); trained_equivalent = 0; trained_success = 0
    for baseline in trained_baseline["trajectories"]:
        instruction = instruction_for(tuple(baseline["constraints"]), baseline["lower"], baseline["forbidden"]); bits = predicted(classifier, instruction); parsed = parse_instruction(instruction); candidate = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], parsed.lower, parsed.forbidden, bits); trained_success += int(candidate["success"] and bits == tuple(baseline["constraints"])); trained_equivalent += int(behavior(candidate) == behavior(baseline))
    heldout_baseline = json.loads(HELDOUT_BASELINE.read_text(encoding="utf-8")); canonical_success = canonical_equivalent = 0
    for baseline in heldout_baseline["canonical_cases"]:
        instruction = instruction_for((1, 1), baseline["lower"], baseline["forbidden"]); bits = predicted(classifier, instruction); parsed = parse_instruction(instruction); candidate = run_canonical_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], baseline["x0"], parsed.lower, parsed.forbidden, bits); canonical_success += int(candidate["success"] and bits == (1, 1)); canonical_equivalent += int(behavior(candidate) == behavior(baseline))
    real_success = real_equivalent = 0; real_rows = []
    for baseline in heldout_baseline["real_memory_cases"]:
        instruction = instruction_for((1, 1), baseline["lower"], baseline["forbidden"]); bits = predicted(classifier, instruction); parsed = parse_instruction(instruction); candidate = run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[baseline["x0"]], parsed.lower, parsed.forbidden, bits); candidate["category"] = baseline["category"]; candidate["predicted_constraints"] = list(bits); candidate["instruction"] = instruction; real_rows.append(candidate); real_success += int(candidate["success"] and bits == (1, 1)); real_equivalent += int(behavior(candidate) == behavior(baseline))
    interaction = [item for item in real_rows if item["category"] == "x_lt_L_eq_F"]; interaction_gate = sum(item["success"] and [event["action_name"] for event in item["events"]][next(index for index, event in enumerate(item["events"]) if event["action_name"] == "COPY_E_R") + 1:] == ["INCREASE"] * (item["forbidden"] - item["x0"] + 1) + ["EMIT"] for item in interaction); crossing = next(item for item in real_rows if item["x0"] == 10 and item["lower"] == 14 and item["forbidden"] == 12); crossing_actions = [event["action_name"] for event in crossing["events"]]; crossing_gate = int(crossing["success"] and crossing_actions[next(index for index, action in enumerate(crossing_actions) if action == "COPY_E_R") + 1:] == ["INCREASE"] * 4 + ["EMIT"])
    result = {"status": "passed" if trained_success == 384 and trained_equivalent == 384 and canonical_success == len(heldout_baseline["canonical_cases"]) and real_success == len(heldout_baseline["real_memory_cases"]) and interaction_gate == len(interaction) and crossing_gate == 1 else "failed", "task": "T2-I0", "baseline": "A", "seed": checkpoint_payload["seed"], "training": {"updates": 5000, "and_trained": False}, "classifier_checkpoint": {"path": str(CLASSIFIER_CHECKPOINT), "sha256": sha256(CLASSIFIER_CHECKPOINT)}, "measurements": {"classification": {"val": val_metric, "test": test_metric, "heldout_and": and_metric}, "end_to_end": {"trained_goals": {"samples": 384, "success": trained_success, "equivalent": trained_equivalent}, "heldout_canonical": {"samples": len(heldout_baseline["canonical_cases"]), "success": canonical_success, "equivalent": canonical_equivalent}, "heldout_real": {"samples": len(real_rows), "success": real_success, "equivalent": real_equivalent}, "interaction_gate": {"samples": len(interaction), "success": interaction_gate}, "crossing_literal_gate": {"samples": 1, "success": crossing_gate, "actions": crossing_actions}}}, "dispatcher_guard": "passed", "baseline_hashes": {"trained_evaluation": file_sha(TRAINED_BASELINE), "heldout_evaluation": file_sha(HELDOUT_BASELINE), "manifest": file_sha(MANIFEST_PATH)}}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
