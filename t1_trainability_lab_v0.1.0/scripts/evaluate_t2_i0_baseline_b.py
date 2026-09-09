"""T2-I0 Baseline B end-to-end evaluation with latent instruction conditioning."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_heldout import run_canonical_learned
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_baseline_a import SharedInstructionEncoder
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i0_instruction import parse_instruction, tokenize_instruction


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
ENCODER_CHECKPOINT = CAMPAIGN_ROOT / "t2_i0_baseline_b_seed4901" / "final.pt"
SUPERVISOR_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
TRAINED_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_trained_evaluation_seed4701" / "results.json"
HELDOUT_BASELINE = CAMPAIGN_ROOT / "u0c_ctrl7_heldout_seed4701" / "results.json"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_baseline_b_evaluation_seed4901"


class InstructionAdapter(nn.Module):
    def __init__(self, supervisor: LatentConditionedSupervisor, condition: Tensor) -> None:
        super().__init__(); self.supervisor = supervisor; self.condition = condition

    def forward(self, observation: Tensor, _ignored_constraints: Tensor) -> Tensor:
        return self.supervisor(observation, self.condition.expand(observation.shape[0], -1))


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
    return actions, result["target"], result["final_value"], result["first_bad"], result["success"]


def encode(encoder: SharedInstructionEncoder, instruction: str) -> Tensor:
    parsed = parse_instruction(instruction); tokens = torch.tensor([parsed.token_ids], dtype=torch.long); lengths = torch.tensor([len(parsed.token_ids)], dtype=torch.long)
    with torch.no_grad(): return encoder(tokens, lengths)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden CTRL-7 inputs")
    payload = torch.load(ENCODER_CHECKPOINT, weights_only=False); encoder = SharedInstructionEncoder(); encoder.load_state_dict(payload["encoder"], strict=True); encoder.eval()
    supervisor = LatentConditionedSupervisor(SUPERVISOR_CHECKPOINT); supervisor.eval(); frozen_ok = all(not parameter.requires_grad for parameter in supervisor.parameters())
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); model = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    cache: dict[str, Tensor] = {}
    def adapter(instruction: str) -> InstructionAdapter:
        if instruction not in cache: cache[instruction] = encode(encoder, instruction)
        return InstructionAdapter(supervisor, cache[instruction])

    trained_baseline = json.loads(TRAINED_BASELINE.read_text(encoding="utf-8")); trained_success = trained_equivalent = 0
    for baseline in trained_baseline["trajectories"]:
        instruction = instruction_for(tuple(baseline["constraints"]), baseline["lower"], baseline["forbidden"]); candidate = run_learned(model, ctrl1, scorer, adapter(instruction), manifest, episode_by_x[baseline["x0"]], baseline["lower"], baseline["forbidden"], tuple(baseline["constraints"])); trained_success += int(candidate["success"]); trained_equivalent += int(behavior(candidate) == behavior(baseline))
    heldout_baseline = json.loads(HELDOUT_BASELINE.read_text(encoding="utf-8")); canonical_success = canonical_equivalent = 0
    for baseline in heldout_baseline["canonical_cases"]:
        instruction = instruction_for((1, 1), baseline["lower"], baseline["forbidden"]); candidate = run_canonical_learned(model, ctrl1, scorer, adapter(instruction), manifest, episode_by_x[baseline["x0"]], baseline["x0"], baseline["lower"], baseline["forbidden"], (1, 1)); canonical_success += int(candidate["success"]); canonical_equivalent += int(behavior(candidate) == behavior(baseline))
    real_success = real_equivalent = 0; real_rows = []
    for baseline in heldout_baseline["real_memory_cases"]:
        instruction = instruction_for((1, 1), baseline["lower"], baseline["forbidden"]); candidate = run_learned(model, ctrl1, scorer, adapter(instruction), manifest, episode_by_x[baseline["x0"]], baseline["lower"], baseline["forbidden"], (1, 1)); candidate["category"] = baseline["category"]; candidate["instruction"] = instruction; real_rows.append(candidate); real_success += int(candidate["success"]); real_equivalent += int(behavior(candidate) == behavior(baseline))
    interaction = [item for item in real_rows if item["category"] == "x_lt_L_eq_F"]; interaction_gate = sum(item["success"] and [event["action_name"] for event in item["events"]][next(index for index, event in enumerate(item["events"]) if event["action_name"] == "COPY_E_R") + 1:] == ["INCREASE"] * (item["forbidden"] - item["x0"] + 1) + ["EMIT"] for item in interaction); crossing = next(item for item in real_rows if item["x0"] == 10 and item["lower"] == 14 and item["forbidden"] == 12); crossing_actions = [event["action_name"] for event in crossing["events"]]; crossing_gate = int(crossing["success"] and crossing_actions[next(index for index, action in enumerate(crossing_actions) if action == "COPY_E_R") + 1:] == ["INCREASE"] * 4 + ["EMIT"])
    real_categories = {category: {"samples": sum(item["category"] == category for item in real_rows), "success": sum(item["category"] == category and item["success"] for item in real_rows)} for category in sorted({item["category"] for item in real_rows})}
    result = {"status": "passed" if frozen_ok and trained_success == len(trained_baseline["trajectories"]) and trained_equivalent == len(trained_baseline["trajectories"]) and canonical_success == len(heldout_baseline["canonical_cases"]) and real_success == len(heldout_baseline["real_memory_cases"]) and interaction_gate == len(interaction) and crossing_gate == 1 else "failed", "task": "T2-I0", "baseline": "B", "seed": payload["seed"], "training": {"updates": payload["updates"], "and_trained": payload["and_trained"]}, "supervisor_frozen": frozen_ok, "checkpoint": {"path": str(ENCODER_CHECKPOINT), "sha256": sha256(ENCODER_CHECKPOINT)}, "measurements": {"trained_goals": {"samples": len(trained_baseline["trajectories"]), "success": trained_success, "equivalent": trained_equivalent}, "heldout_canonical": {"samples": len(heldout_baseline["canonical_cases"]), "success": canonical_success, "equivalent": canonical_equivalent}, "heldout_real": {"samples": len(real_rows), "success": real_success, "equivalent": real_equivalent, "categories": real_categories}, "interaction_gate": {"samples": len(interaction), "success": interaction_gate}, "crossing_literal_gate": {"samples": 1, "success": crossing_gate, "actions": crossing_actions}}, "dispatcher_guard": "passed", "baseline_hashes": {"trained_evaluation": file_sha(TRAINED_BASELINE), "heldout_evaluation": file_sha(HELDOUT_BASELINE), "manifest": file_sha(MANIFEST_PATH)}}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
