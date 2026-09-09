"""Pre-held-out controls for T2-I0-R1 structural deconfounding."""

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
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_a import BaselineAClassifier
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_baseline_b import SharedInstructionEncoder
from train_t2_i0_r1 import BaselineAR1Classifier, SharedInstructionEncoderR1
from train_t2_i0_r1 import instruction_for as r1_instruction_for
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i0_instruction_r1 import parse_instruction_r1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT_A = CAMPAIGN_ROOT / "t2_i0_r1_baseline_a_seed5301" / "final.pt"
CHECKPOINT_B = CAMPAIGN_ROOT / "t2_i0_r1_baseline_b_seed5401" / "final.pt"
SUPERVISOR_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_r1_controls_seed5301_5401"


class Adapter(torch.nn.Module):
    def __init__(self, supervisor: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__(); self.supervisor = supervisor; self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.supervisor(features, self.condition.expand(features.shape[0], -1))


def instruction(constraints: tuple[int, int], value: int, variant: int, dummy: int = 0) -> str:
    if constraints == (0, 0): return r1_instruction_for(constraints, 0, 0, variant, dummy)
    return r1_instruction_for(constraints, value, value, variant, dummy)


def classify(model: BaselineAR1Classifier, text: str) -> tuple[int, int]:
    parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)])
    with torch.no_grad(): floor, avoid = model(tokens, lengths)
    return int(floor.argmax(-1).item()), int(avoid.argmax(-1).item())


def behavior(result: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(event["action_name"] for event in result["events"]), result["target"], result["final_value"], result["success"])


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    a = BaselineAR1Classifier(); a.load_state_dict(torch.load(CHECKPOINT_A, weights_only=False)["model"], strict=True); a.eval(); b_encoder = SharedInstructionEncoderR1(); b_encoder.load_state_dict(torch.load(CHECKPOINT_B, weights_only=False)["encoder"], strict=True); b_encoder.eval(); b_core = LatentConditionedSupervisor(SUPERVISOR_CHECKPOINT); b_core.eval()
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cache: dict[str, Adapter] = {}
    def run(text: str, constraints: tuple[int, int], lower: int, forbidden: int) -> dict[str, Any]:
        if text not in cache:
            parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)]); cache[text] = Adapter(b_core, b_encoder(tokens, lengths))
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episode_by_x[10], lower, forbidden, constraints)

    position_a = 0; position_b = 0; position_samples = 0
    for value in range(32):
        for constraints in ((1, 0), (0, 1)):
            variants = [instruction(constraints, value, variant) for variant in range(3)]
            predictions = [classify(a, text) for text in variants]; expected = constraints; position_a += int(all(prediction == expected for prediction in predictions) and len(set(predictions)) == 1)
            b_runs = [run(text, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0) for text in variants]; position_b += int(all(item["success"] for item in b_runs) and len({behavior(item) for item in b_runs}) == 1); position_samples += 1
    memory_a = memory_b = noop_a = noop_b = 0; memory_samples = noop_samples = 0; noop_a_by = {"NONE": 0, "FLOOR": 0, "AVOID": 0}; noop_b_by = {"NONE": 0, "FLOOR": 0, "AVOID": 0}; noop_samples_by = {"NONE": 0, "FLOOR": 0, "AVOID": 0}
    for value in range(32):
        left = instruction((1, 0), value, 1); right = instruction((1, 0), value, 2); memory_a += int(classify(a, left) == classify(a, right) == (1, 0)); left_run = run(left, (1, 0), value, 0); right_run = run(right, (1, 0), value, 0); memory_b += int(left_run["success"] and right_run["success"] and behavior(left_run) == behavior(right_run)); memory_samples += 1
        for constraints in ((1, 0), (0, 1)):
            for variant in (1, 2):
                value_arg = value
                text0 = instruction(constraints, value_arg, variant, 0); text5 = instruction(constraints, value_arg, variant, 5); name = "FLOOR" if constraints == (1, 0) else "AVOID"; a_ok = classify(a, text0) == classify(a, text5) == constraints; b0 = run(text0, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); b5 = run(text5, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); b_ok = b0["success"] and b5["success"] and behavior(b0) == behavior(b5); noop_a += int(a_ok); noop_b += int(b_ok); noop_a_by[name] += int(a_ok); noop_b_by[name] += int(b_ok); noop_samples += 1; noop_samples_by[name] += 1
        for variant in (0, 1):
            text0 = instruction((0, 0), value, variant, 0); text5 = instruction((0, 0), value, variant, 5); a_ok = classify(a, text0) == classify(a, text5) == (0, 0); b0 = run(text0, (0, 0), 0, 0); b5 = run(text5, (0, 0), 0, 0); b_ok = b0["success"] and b5["success"] and behavior(b0) == behavior(b5); noop_a += int(a_ok); noop_b += int(b_ok); noop_a_by["NONE"] += int(a_ok); noop_b_by["NONE"] += int(b_ok); noop_samples += 1; noop_samples_by["NONE"] += 1
    summary = {"position_invariance": {"samples": position_samples, "baseline_a": position_a, "baseline_b": position_b}, "early_memory": {"samples": memory_samples, "baseline_a": memory_a, "baseline_b": memory_b}, "noop_causal": {"samples": noop_samples, "baseline_a": noop_a, "baseline_b": noop_b, "by_semantics": {name: {"samples": noop_samples_by[name], "baseline_a": noop_a_by[name], "baseline_b": noop_b_by[name]} for name in noop_samples_by}}}
    result = {"status": "passed" if position_a == position_samples and position_b == position_samples and memory_a == memory_samples and memory_b == memory_samples and noop_a == noop_samples and noop_b == noop_samples else "failed", "task": "T2-I0-R1", "phase": "structural_deconfounding_controls", "training": False, "checkpoints": {"a": sha256(CHECKPOINT_A), "b": sha256(CHECKPOINT_B), "ctrl7": sha256(SUPERVISOR_CHECKPOINT)}, "summary": summary, "latent_cosine_diagnostic": "not a gate", "dispatcher_guard": "passed"}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
