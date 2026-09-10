"""T2-I2 implementation smoke test; does not train or overwrite checkpoints."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from evaluate_u0c_ctrl4_preflight import dispatch_unified_action
from t2_i2_semantic_writer import MAX_POSITIONS, SemanticWriter, architecture_report, tensorize
from train_t2_i2 import checkpoint_for_seed, output_for_seed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6101)
    args = parser.parse_args()
    forbidden = ("constraints", "lower", "forbidden", "floor", "avoid", "segments", "clauses", "split")
    dispatcher_parameters = inspect.signature(dispatch_unified_action).parameters
    if any(name in dispatcher_parameters for name in forbidden):
        raise RuntimeError("dispatcher received forbidden inputs")
    writer_parameters = inspect.signature(SemanticWriter.forward).parameters
    if any(name in writer_parameters for name in forbidden):
        raise RuntimeError("writer received oracle inputs")
    instructions = [
        "NOOP VALUE_0",
        "AT_LEAST VALUE_12",
        "NOOP VALUE_19 AND EXCLUDE VALUE_23",
        "MINIMUM VALUE_12 AND NOOP VALUE_19",
    ]
    token_ids, lengths = tensorize(instructions)
    writer = SemanticWriter()
    with torch.no_grad():
        details = writer(token_ids, lengths, return_details=True)
    if token_ids.shape[1] > MAX_POSITIONS - 1 or details["slots"].shape != (4, 2, 32):
        raise RuntimeError("unexpected writer tensor shape")
    expected = architecture_report(writer)
    if expected["trainable_parameters"] != 4882:
        raise RuntimeError(f"writer parameter count changed: {expected['trainable_parameters']}")
    seed_output = output_for_seed(args.seed)
    seed_checkpoint = checkpoint_for_seed(args.seed)
    if seed_checkpoint.exists():
        raise RuntimeError(f"smoke test refuses existing checkpoint: {seed_checkpoint}")
    result = {
        "status": "passed",
        "task": "T2-I2",
        "phase": "implementation_smoke",
        "training": False,
        "checkpoint_overwrite_guard": "passed",
        "dispatcher_guard": "passed",
        "writer_oracle_guard": "passed",
        "max_positions": MAX_POSITIONS,
        "input_max_tokens": int(token_ids.shape[1]),
        "output_shape": list(details["slots"].shape),
        "architecture": expected,
        "reserved_seed_output": str(seed_output),
    }
    output = ROOT / "campaign" / "t2_i2_smoke.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
