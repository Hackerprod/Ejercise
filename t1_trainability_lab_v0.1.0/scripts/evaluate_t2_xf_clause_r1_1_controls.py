"""XF-CLAUSE R1.1 seen-task validity controls."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from t2_i0_instruction_r1_1 import instruction_for_r1_1
from t2_xf_transformer import MatchedTransformerEncoder, encode_clauses, parameter_count
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_xf_clause_r1_1 import OUTPUT as TRAINING_OUTPUT
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; CHECKPOINT = TRAINING_OUTPUT / "final.pt"; CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"; SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"; MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"; MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"; OUTPUT_ROOT = CAMPAIGN_ROOT / "t2_xf_clause_r1_1_controls_seed6001"


class Adapter(torch.nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None: super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor: return self.core(features, self.condition.expand(features.shape[0], -1))


def behavior(result: dict[str, Any]) -> tuple[Any, ...]: return (tuple(event["action_name"] for event in result["events"]), result["target"], result["final_value"], result["success"])
def file_sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    encoder = MatchedTransformerEncoder(); encoder.load_state_dict(torch.load(CHECKPOINT, weights_only=False)["encoder"], strict=True); encoder.eval(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST_PATH, MANIFEST_SHA256); executor = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(SCORER_CHECKPOINT, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval(); episode = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}; cache: dict[str, Adapter] = {}
    def run(text: str, constraints: tuple[int, int], lower: int, forbidden: int) -> dict[str, Any]:
        if text not in cache:
            with torch.no_grad(): cache[text] = Adapter(core, encode_clauses(encoder, [text]))
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episode[10], lower, forbidden, constraints)
    position = {"samples": 0, "success": 0}; memory = {"samples": 0, "success": 0}; noop = {name: {"samples": 0, "success": 0} for name in ("NONE", "FLOOR", "AVOID")}
    for value in range(32):
        for constraints in ((1, 0), (0, 1)):
            texts = [instruction_for_r1_1(constraints, value, variant=0), instruction_for_r1_1(constraints, value, variant=1, dummy=0), instruction_for_r1_1(constraints, value, variant=2, dummy=0)]; runs = [run(text, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0) for text in texts]; position["success"] += int(all(item["success"] for item in runs) and len({behavior(item) for item in runs}) == 1); position["samples"] += 1
        left = instruction_for_r1_1((1, 0), value, variant=1, dummy=0); right = instruction_for_r1_1((1, 0), value, variant=2, dummy=0); lrun = run(left, (1, 0), value, 0); rrun = run(right, (1, 0), value, 0); memory["success"] += int(lrun["success"] and rrun["success"] and behavior(lrun) == behavior(rrun)); memory["samples"] += 1
        for constraints, name in (((0, 0), "NONE"), ((1, 0), "FLOOR"), ((0, 1), "AVOID")):
            variants = (0, 1) if constraints == (0, 0) else (1, 2)
            for variant in variants:
                for dummy in range(32):
                    text = instruction_for_r1_1(constraints, value if constraints != (0, 0) else 0, variant=variant, dummy=dummy, dummy2=(dummy + 11) % 32); result = run(text, constraints, value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0); noop[name]["samples"] += 1; noop[name]["success"] += int(result["success"])
    benchmark_texts = [instruction_for_r1_1((1, 0), value, variant=variant, dummy=0) for value in range(32) for variant in (0, 1, 2)]; start = time.perf_counter()
    with torch.no_grad():
        for text in benchmark_texts: encode_clauses(encoder, [text])
    inference_seconds = (time.perf_counter() - start) / len(benchmark_texts)
    summary = {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop, "inference": {"instructions": len(benchmark_texts), "seconds_per_instruction": inference_seconds}, "trainable_parameters": parameter_count(encoder)}; passed = position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values()); result = {"status": "passed" if passed else "failed", "task": "T2-XF", "phase": "XF-CLAUSE_R1.1_controls", "training": False, "checkpoint": sha256(CHECKPOINT), "ctrl7": sha256(CTRL7_CHECKPOINT), "summary": summary, "dispatcher_guard": "passed"}; output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
