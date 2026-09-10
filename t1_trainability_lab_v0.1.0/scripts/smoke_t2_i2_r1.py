"""T2-I2-R1 implementation smoke test; never overwrites v0 or R1 artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from evaluate_u0c_ctrl4_preflight import dispatch_unified_action
from t2_i2_r1_semantic_writer import MAX_POSITIONS, CompetitiveSemanticWriter, architecture_report, tensorize
from train_t2_i2_r1 import checkpoint_for_seed, output_for_seed

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6101); args = parser.parse_args(); forbidden = ("constraints", "lower", "forbidden", "floor", "avoid", "segments", "clauses", "split"); dispatcher_parameters = inspect.signature(dispatch_unified_action).parameters; writer_parameters = inspect.signature(CompetitiveSemanticWriter.forward).parameters
    if any(name in dispatcher_parameters for name in forbidden): raise RuntimeError("dispatcher received forbidden inputs")
    if any(name in writer_parameters for name in forbidden): raise RuntimeError("writer received oracle inputs")
    instructions = ["NOOP VALUE_0", "AT_LEAST VALUE_12", "NOOP VALUE_19 AND EXCLUDE VALUE_23", "MINIMUM VALUE_12 AND NOOP VALUE_19"]; token_ids, lengths = tensorize(instructions); writer = CompetitiveSemanticWriter();
    with torch.no_grad(): details = writer(token_ids, lengths, return_details=True)
    if token_ids.shape[1] > MAX_POSITIONS - 1 or details["slots"].shape != (4, 2, 32) or details["routing_probabilities"].shape != (4, 3, 5): raise RuntimeError("unexpected R1 writer tensor shape")
    report = architecture_report(writer); 
    if report["trainable_parameters"] != 3184: raise RuntimeError(f"writer parameter count changed: {report['trainable_parameters']}")
    r1_output = output_for_seed(args.seed); r1_checkpoint = checkpoint_for_seed(args.seed); v0_checkpoint = CAMPAIGN / f"t2_i2_seed{args.seed}" / "final.pt"
    if r1_checkpoint == v0_checkpoint or r1_checkpoint.exists(): raise RuntimeError("smoke test refuses collision with existing v0/R1 checkpoint")
    result = {"status": "passed", "task": "T2-I2-R1", "phase": "implementation_smoke", "training": False, "writer_oracle_guard": "passed", "dispatcher_guard": "passed", "checkpoint_overwrite_guard": "passed", "v0_collision_guard": "passed", "max_positions": MAX_POSITIONS, "input_max_tokens": int(token_ids.shape[1]), "output_shape": list(details["slots"].shape), "routing_shape": list(details["routing_probabilities"].shape), "architecture": report, "reserved_r1_output": str(r1_output), "reserved_v0_output": str(CAMPAIGN / f"t2_i2_seed{args.seed}")}; output = CAMPAIGN / "t2_i2_r1_smoke.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
