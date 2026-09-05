"""Evaluate an existing UnifiedT1U0 checkpoint across all canonical tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from train_u0a import build_canonical_data, evaluate_accuracy, evaluate_all, save_json, build_sequential_h1_table
from t1_trainability.unified import UnifiedT1U0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    datasets = build_canonical_data(args.checkpoint.parent)
    test = evaluate_all(model, datasets, "test")
    test["sequential_update_h1_table"] = evaluate_accuracy(model, "sequential_update", build_sequential_h1_table(), rounds=1)
    output = {"checkpoint": str(args.checkpoint), "checkpoint_step": payload.get("step"), "test": test, "retrained": False}
    save_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
