"""Exhaustive pre-heldout controls for T2-I0-R1.1."""

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
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_r1 import BaselineAR1Classifier, SharedInstructionEncoderR1, tensorize
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i0_instruction_r1_1 import instruction_for_r1_1
from t2_i0_instruction_r1 import parse_instruction_r1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT_A = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_a_seed5501" / "final.pt"
CHECKPOINT_B = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_b_seed5601" / "final.pt"
CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_i0_r1_1_controls_seed5501_5601"


class Adapter(torch.nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__(); self.core = core; self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def classify(model: BaselineAR1Classifier, text: str) -> tuple[int, int]:
    parsed = parse_instruction_r1(text); data = tensorize([(text, parsed.constraints)])
    with torch.no_grad(): floor, avoid = model(data["token_ids"], data["lengths"])
    return int(floor.argmax(-1).item()), int(avoid.argmax(-1).item())


def behavior(result: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(event["action_name"] for event in result["events"]), result["target"], result["final_value"], result["success"])


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    a = BaselineAR1Classifier(); a.load_state_dict(torch.load(CHECKPOINT_A, weights_only=False)["model"], strict=True); a.eval(); b_encoder = SharedInstructionEncoderR1(); b_encoder.load_state_dict(torch.load(CHECKPOINT_B, weights_only=False)["encoder"], strict=True); b_encoder.eval(); b_core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); b_core.eval()
    manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cache: dict[str, Adapter] = {}
    def run(text: str, constraints: tuple[int, int], lower: int, forbidden: int) -> dict[str, Any]:
        if text not in cache:
            parsed = parse_instruction_r1(text); tokens = torch.tensor([parsed.token_ids]); lengths = torch.tensor([len(parsed.token_ids)]); cache[text] = Adapter(b_core, b_encoder(tokens, lengths))
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episode[10], lower, forbidden, constraints)

    position = {"samples": 0, "baseline_a": 0, "baseline_b": 0}; memory = {"samples": 0, "baseline_a": 0, "baseline_b": 0}; noop = {name: {"samples": 0, "baseline_a": 0, "baseline_b": 0} for name in ("NONE", "FLOOR", "AVOID")}
    for value in range(32):
        for constraints in ((1, 0), (0, 1)):
            texts = [instruction_for_r1_1(constraints, value, variant=0), instruction_for_r1_1(constraints, value, variant=1, dummy=0), instruction_for_r1_1(constraints, value, variant=2, dummy=0)]; expected = constraints; predictions = [classify(a, text) for text in texts]; position["baseline_a"] += int(all(prediction == expected for prediction in predictions)); b_runs = [run(text, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0) for text in texts]; position["baseline_b"] += int(all(item["success"] for item in b_runs) and len({behavior(item) for item in b_runs}) == 1); position["samples"] += 1
        left = instruction_for_r1_1((1, 0), value, variant=1, dummy=0); right = instruction_for_r1_1((1, 0), value, variant=2, dummy=0); memory["baseline_a"] += int(classify(a, left) == classify(a, right) == (1, 0)); lrun = run(left, (1, 0), value, 0); rrun = run(right, (1, 0), value, 0); memory["baseline_b"] += int(lrun["success"] and rrun["success"] and behavior(lrun) == behavior(rrun)); memory["samples"] += 1
        for constraints, name in (((0, 0), "NONE"), ((1, 0), "FLOOR"), ((0, 1), "AVOID")):
            dummies = range(32)
            variants = (0, 1) if constraints == (0, 0) else (1, 2)
            for variant in variants:
                for dummy in dummies:
                    text0 = instruction_for_r1_1(constraints, value if constraints != (0, 0) else 0, variant=variant, dummy=dummy, dummy2=(dummy + 11) % 32); expected = constraints; a_ok = classify(a, text0) == expected; b0 = run(text0, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); b_ok = b0["success"]; noop[name]["baseline_a"] += int(a_ok); noop[name]["baseline_b"] += int(b_ok); noop[name]["samples"] += 1
    summary = {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop}
    result = {"status": "passed" if position["baseline_a"] == position["samples"] and position["baseline_b"] == position["samples"] and memory["baseline_a"] == memory["samples"] and memory["baseline_b"] == memory["samples"] and all(item["baseline_a"] == item["samples"] and item["baseline_b"] == item["samples"] for item in noop.values()) else "failed", "task": "T2-I0-R1.1", "phase": "structural_deconfounding_controls", "training": False, "checkpoints": {"a": sha256(CHECKPOINT_A), "b": sha256(CHECKPOINT_B), "ctrl7": sha256(CTRL7_CHECKPOINT)}, "summary": summary, "latent_cosine_diagnostic": "not a gate", "dispatcher_guard": "passed"}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
