"""Train one U0-A task with the fixed real SharedMemoryReader."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import torch

from run_u0a_canaries import train_canary
from train_u0a import TASKS, build_canonical_data, save_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_canonical_data(args.output_dir)
    result = train_canary(args.task, datasets, args.output_dir, args.seed, args.steps)
    save_json(args.output_dir / "summary.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
