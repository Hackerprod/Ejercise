"""T2-I2-R2 smoke test; no training and no checkpoint overwrite."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from evaluate_u0c_ctrl4_preflight import dispatch_unified_action
from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, MAX_POSITIONS, architecture_report, tensorize
from train_t2_i2_r2 import checkpoint_for_seed, output_for_seed

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6201); args = parser.parse_args(); forbidden = ("constraints", "lower", "forbidden", "floor", "avoid", "segments", "clauses", "split")
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in forbidden): raise RuntimeError("dispatcher received forbidden inputs")
    if any(name in inspect.signature(CompetitiveSemanticWriter.forward).parameters for name in forbidden): raise RuntimeError("writer received oracle inputs")
    token_ids, lengths = tensorize(["NOOP VALUE_0", "AT_LEAST VALUE_12", "NOOP VALUE_19 AND EXCLUDE VALUE_23"]); writer = CompetitiveSemanticWriter(); details = writer(token_ids, lengths, return_details=True); report = architecture_report(writer)
    r2_checkpoint = checkpoint_for_seed(args.seed); v0 = CAMPAIGN / f"t2_i2_seed{args.seed}" / "final.pt"; r1 = CAMPAIGN / f"t2_i2_r1_seed{args.seed}" / "final.pt"
    if r2_checkpoint == v0 or r2_checkpoint == r1 or r2_checkpoint.exists(): raise RuntimeError("R2 smoke collision/overwrite guard failed")
    result = {"status": "passed", "task": "T2-I2-R2", "phase": "implementation_smoke", "training": False, "writer_oracle_guard": "passed", "dispatcher_guard": "passed", "collision_guard": "passed", "output_shape": list(details["slots"].shape), "architecture": report, "reserved_r2_output": str(output_for_seed(args.seed)), "v0_output": str(v0.parent), "r1_output": str(r1.parent), "max_positions": MAX_POSITIONS}; output = CAMPAIGN / "t2_i2_r2_smoke.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
